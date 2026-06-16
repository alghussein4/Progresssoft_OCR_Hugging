"""
Post-OCR text correction using SymSpell.

Conservatively fixes misspelled words produced by the OCR models while
preserving punctuation, capitalization, numbers, line breaks, and any word
that is already spelled correctly.

Only purely alphabetic, out-of-dictionary words are corrected — this avoids
mangling proper nouns, codes, and numbers.
"""

import re
import importlib.resources as resources
from symspellpy import SymSpell, Verbosity

_sym = None


def _get_symspell() -> SymSpell:
    """Lazily load the bundled English frequency dictionary once."""
    global _sym
    if _sym is None:
        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        with resources.path("symspellpy", "frequency_dictionary_en_82_765.txt") as p:
            sym.load_dictionary(str(p), term_index=0, count_index=1)
        _sym = sym
    return _sym


def _match_case(original: str, corrected: str) -> str:
    """Apply the original token's capitalization pattern to the correction."""
    if original.isupper():
        return corrected.upper()
    if original[0].isupper():
        return corrected.capitalize()
    return corrected


def correct_text(text: str, max_edit_distance: int = 2) -> str:
    """
    Correct each word in the text conservatively.

    - Words already in the dictionary are left unchanged.
    - Tokens with digits or symbols are left unchanged.
    - Only alphabetic out-of-dictionary words (length >= 2) are corrected.

    Punctuation, spacing, and line breaks are preserved.
    """
    if not text or not text.strip():
        return text

    sym = _get_symspell()

    def fix_token(token: str) -> str:
        if len(token) < 2 or not token.isalpha():
            return token

        lower = token.lower()
        # Already a known word → keep as-is
        if sym.lookup(lower, Verbosity.TOP, max_edit_distance=0):
            return token

        suggestions = sym.lookup(lower, Verbosity.CLOSEST,
                                 max_edit_distance=max_edit_distance)
        if suggestions:
            return _match_case(token, suggestions[0].term)
        return token

    # Preserve line breaks; correct word by word within each line
    out_lines = []
    for line in text.split("\n"):
        # Split into alternating word / non-word chunks, keeping separators
        chunks = re.split(r"(\W+)", line)
        out_lines.append("".join(fix_token(c) for c in chunks))

    return "\n".join(out_lines)
