"""
Whitelist logic gate + Top-K selection for high-integrity (banking) OCR.

Design goals:
  - ZERO hallucination: we only ever SELECT among the model's own hypotheses.
    We never edit, correct, or generate text.
  - O(1) average validation: frozenset membership + a handful of precompiled
    regexes (compiled once at import).
  - No heavy dependencies — standard library only.

If no hypothesis passes the gate, the top-1 prediction is returned verbatim
with a [REVIEW_REQUIRED] flag for routing to a human operator.
"""

import re

# ──────────────────────────────────────────────────────────────────────────────
# 1. The whitelist  (extend with the bank's controlled vocabulary)
# ──────────────────────────────────────────────────────────────────────────────
# Stored lowercase in a frozenset → membership test is average O(1).
BANKING_TERMS = frozenset({
    "account", "accounts", "balance", "credit", "debit", "interest",
    "principal", "loan", "mortgage", "deposit", "withdrawal", "statement",
    "branch", "iban", "swift", "bic", "currency", "amount", "payment",
    "installment", "instalment", "borrower", "lender", "guarantor",
    "collateral", "maturity", "signature", "date", "total", "subtotal",
    "tax", "vat", "fee", "fees", "usd", "eur", "jod", "gbp", "sar", "aed",
    "annum", "rate", "term", "contract", "agreement", "party", "bank",
    "cheque", "check", "transfer", "reference", "number",
})

# ──────────────────────────────────────────────────────────────────────────────
# 2. Structured numeric / identifier patterns  (compiled ONCE at import)
# ──────────────────────────────────────────────────────────────────────────────
_NUMERIC_PATTERNS = (
    re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d{1,2})?$"),       # money: 1,234.56 / 1,000
    re.compile(r"^-?\d+(\.\d{1,4})?$"),                    # plain number / decimal
    re.compile(r"^\d{1,3}(\.\d{1,2})?%$"),                 # percentage: 4.5%
    re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$"),         # IBAN
    re.compile(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$"),    # SWIFT / BIC
    re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$"),    # date 01/02/2026
    re.compile(r"^[A-Z0-9\-]{6,}$"),                       # account / ref codes
)


def is_valid_token(token: str) -> bool:
    """
    Validate a SINGLE token against the whitelist + numeric patterns.
    Average O(1): one set lookup, then a few precompiled regex matches.
    """
    if not token:
        return False
    t = token.strip()
    if not t:
        return False
    if t.lower() in BANKING_TERMS:          # O(1) set membership
        return True
    for pattern in _NUMERIC_PATTERNS:        # small fixed number of checks
        if pattern.match(t):
            return True
    return False


def is_valid_candidate(text: str, mode: str = "any") -> bool:
    """
    Validate a full candidate STRING (one Top-K hypothesis for a bbox).

    mode="any"  : valid if the whole string is a known term/pattern,
                  OR (for multi-word fields) every token is individually valid.
    mode="all"  : strict — every whitespace-separated token must be valid.
    """
    if not text or not text.strip():
        return False
    text = text.strip()

    # Whole-string check first (covers single-field boxes: an amount, an IBAN…)
    if is_valid_token(text):
        return True

    # Multi-token field: require ALL tokens to be individually valid.
    tokens = text.split()
    if len(tokens) <= 1:
        return False
    return all(is_valid_token(tok) for tok in tokens)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Selection / fallback logic
# ──────────────────────────────────────────────────────────────────────────────
REVIEW_FLAG = "[REVIEW_REQUIRED]"


def select_prediction(candidates, mode: str = "any") -> dict:
    """
    Apply the logic gate to ranked Top-K hypotheses.

    Args:
        candidates : list of (text, score) ordered best-first (top-1 at index 0).
                     `score` is the model's own sequence probability — used only
                     for ordering/reporting, never to alter text.
        mode       : validation strictness passed to is_valid_candidate.

    Returns a dict:
        {
          "text"       : final text (flagged if nothing validated),
          "accepted"   : bool,        # True if a hypothesis passed the gate
          "matched_rank": int | None, # 1-based rank that validated, else None
          "score"      : float,       # score of the chosen/flagged hypothesis
        }

    Guarantee: "text" is ALWAYS one of the model's own hypotheses (optionally
    with the review flag appended). No characters are ever invented or edited.
    """
    if not candidates:
        return {"text": "", "accepted": False, "matched_rank": None, "score": 0.0}

    for rank, (text, score) in enumerate(candidates, start=1):
        if is_valid_candidate(text, mode=mode):
            return {"text": text, "accepted": True,
                    "matched_rank": rank, "score": float(score)}

    # Nothing matched → keep the #1 prediction verbatim, flag for human review.
    top_text, top_score = candidates[0]
    return {"text": f"{top_text} {REVIEW_FLAG}".strip(),
            "accepted": False, "matched_rank": None, "score": float(top_score)}


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated Top-K outputs (text, score) best-first
    cases = [
        [("1,250.00", 0.91), ("1,25O.00", 0.74), ("I,250.00", 0.40)],  # #1 valid
        [("Acc0unt", 0.88), ("Account", 0.71), ("Accaunt", 0.33)],     # #2 valid
        [("qwxz", 0.55), ("qwxs", 0.40), ("qvxz", 0.21)],              # none → flag
    ]
    for c in cases:
        print(c[0][0], "->", select_prediction(c))
