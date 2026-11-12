"""Text canonicalisation shared by the rule engine and the ML pipeline."""

import re
import unicodedata

# Zero-width characters are a common trick to break keyword matching ("O​TP").
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

# The Devanagari nukta is used inconsistently ("ज़रूरी" vs "जरूरी"), and NFKC
# decomposes precomposed nukta letters, so dropping it makes both spellings match.
_NUKTA = dict.fromkeys([ord("़")], None)

_WHITESPACE = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    """Apply case-preserving Unicode cleanup.

    NFKC folds stylised look-alike characters (fullwidth or mathematical bold
    letters) back to plain ASCII. This is safe to apply to regex patterns too,
    unlike case folding, which would corrupt escapes such as ``\\S`` or ``\\D``.
    """
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_ZERO_WIDTH).translate(_NUKTA)


def normalize_text(text: str) -> str:
    """Canonicalise, case-fold and collapse whitespace for matching."""
    text = canonicalize(text).casefold()
    return _WHITESPACE.sub(" ", text).strip()
