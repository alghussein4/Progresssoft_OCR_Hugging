"""
Top-K text recognizer for high-integrity OCR.

Loads the exported PaddleOCR recognition model directly via paddle.inference
(NOT the high-level PaddleOCR API, which only returns top-1), captures the raw
per-timestep CTC probability matrix [T, C], and runs a lightweight CTC prefix
beam search to return the top-K most probable *sequences* for a crop.

Dependencies: numpy, opencv (cv2), paddle — all already installed. No new deps.
"""

import numpy as np
import cv2
from collections import defaultdict
import paddle.inference as paddle_infer

_NEG_INF = -1e30


# ──────────────────────────────────────────────────────────────────────────────
# CTC prefix beam search (log-domain, pure NumPy)
# ──────────────────────────────────────────────────────────────────────────────
def _logaddexp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_prefix_beam_search(probs, idx_to_char, blank=0,
                           beam_size=10, top_k=5, top_n=10):
    """
    Args:
        probs      : [T, C] softmax probabilities (per timestep).
        idx_to_char: dict mapping class index -> character ("" for blank).
        blank      : blank class index (0 for PaddleOCR CTCLabelDecode).
        beam_size  : beams kept per timestep (speed/accuracy trade-off).
        top_k      : number of final hypotheses to return.
        top_n      : only the top-N chars per timestep are expanded (pruning).

    Returns: list of (text, log_prob) ranked best-first, length <= top_k.
    """
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    T, C = log_probs.shape
    top_n = min(top_n, C)   # can't prune to more classes than exist

    # prefix (tuple of class ids) -> [log p ending in blank, log p ending in non-blank]
    beams = {(): [0.0, _NEG_INF]}

    for t in range(T):
        row = log_probs[t]
        # Prune: expand only the top-N classes this timestep (+ always blank).
        cand = set(np.argpartition(row, -top_n)[-top_n:].tolist())
        cand.add(blank)

        nxt = defaultdict(lambda: [_NEG_INF, _NEG_INF])
        for prefix, (pb, pnb) in beams.items():
            ptot = _logaddexp(pb, pnb)
            last = prefix[-1] if prefix else -1
            for c in cand:
                lp = row[c]
                if c == blank:
                    e = nxt[prefix]
                    e[0] = _logaddexp(e[0], ptot + lp)
                elif c == last:
                    # repeat of last char: stays same prefix from non-blank,
                    # or extends prefix only from a blank-ending path.
                    e = nxt[prefix]
                    e[1] = _logaddexp(e[1], pnb + lp)
                    ext = prefix + (c,)
                    e2 = nxt[ext]
                    e2[1] = _logaddexp(e2[1], pb + lp)
                else:
                    ext = prefix + (c,)
                    e2 = nxt[ext]
                    e2[1] = _logaddexp(e2[1], ptot + lp)

        # Keep the strongest beam_size prefixes.
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                            reverse=True)[:beam_size])

    ranked = sorted(beams.items(),
                    key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                    reverse=True)

    out = []
    for prefix, (pb, pnb) in ranked[:top_k]:
        text = "".join(idx_to_char.get(c, "") for c in prefix)
        out.append((text, float(_logaddexp(pb, pnb))))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Recognizer wrapper around the exported inference model
# ──────────────────────────────────────────────────────────────────────────────
class TopKRecognizer:
    def __init__(self, model_dir: str, dict_path: str,
                 img_shape=(3, 48, 320), use_gpu: bool = False,
                 use_space_char: bool = True):
        """
        model_dir : folder with inference.pdmodel + inference.pdiparams
        dict_path : the character dictionary used in training (en_dict.txt)
        """
        cfg = paddle_infer.Config(f"{model_dir}/inference.pdmodel",
                                  f"{model_dir}/inference.pdiparams")
        if use_gpu:
            cfg.enable_use_gpu(500, 0)
        else:
            cfg.disable_gpu()
            cfg.set_cpu_math_library_num_threads(4)
        cfg.disable_glog_info()
        cfg.switch_ir_optim(True)
        cfg.enable_memory_optim()

        self.predictor = paddle_infer.create_predictor(cfg)
        self._in = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self._out = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        self.C, self.H, self.W = img_shape

        # Build index->char map exactly as PaddleOCR's CTCLabelDecode does:
        #   index 0 = blank, then dict chars, then optional space.
        chars = [""]  # blank
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                chars.append(line.rstrip("\n"))
        if use_space_char:
            chars.append(" ")
        self.idx_to_char = {i: c for i, c in enumerate(chars)}

    def _preprocess(self, img_bgr):
        """Resize-with-ratio to height H, right-pad to width W, normalize."""
        h, w = img_bgr.shape[:2]
        ratio = w / float(h) if h > 0 else 1.0
        resized_w = min(self.W, max(1, int(np.ceil(self.H * ratio))))
        resized = cv2.resize(img_bgr, (resized_w, self.H)).astype("float32")
        resized = resized.transpose(2, 0, 1) / 255.0
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((self.C, self.H, self.W), dtype="float32")
        padded[:, :, :resized_w] = resized
        return padded

    def recognize_topk(self, img_bgr, top_k: int = 5):
        """
        Run inference on one crop and return top-K (text, score) hypotheses.
        img_bgr: HxWx3 uint8 BGR image (a single text-line/field crop).
        """
        x = self._preprocess(img_bgr)[None]            # [1, C, H, W]
        self._in.copy_from_cpu(x)
        self.predictor.run()
        probs = self._out.copy_to_cpu()[0]             # [T, C] (softmax probs)
        return ctc_prefix_beam_search(probs, self.idx_to_char, top_k=top_k)


# ──────────────────────────────────────────────────────────────────────────────
# Self-test of the beam search on a synthetic probability matrix (no model needed)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # tiny alphabet: 0=blank, 1='C', 2='A', 3='T'
    idx = {0: "", 1: "C", 2: "A", 3: "T"}
    T, C = 6, 4
    rng = np.random.default_rng(0)
    probs = rng.random((T, C))
    # bias toward C,A,T over time so "CAT" is the strongest path
    probs[0, 1] += 2; probs[1, 1] += 2
    probs[2, 2] += 2; probs[3, 2] += 2
    probs[4, 3] += 2; probs[5, 3] += 2
    probs = probs / probs.sum(axis=1, keepdims=True)
    for text, score in ctc_prefix_beam_search(probs, idx, top_k=5):
        print(f"{text!r}  logp={score:.3f}")
