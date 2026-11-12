"""Text preparation shared by model training and inference.

Trained model pipelines reference ``prepare_batch`` by import path, so it must
stay in this module for saved models to keep loading.
"""

import re
from collections.abc import Iterable

from app.detection.anonymize import anonymize
from app.detection.normalize import canonicalize, normalize_text

_WHITESPACE = re.compile(r"\s+")


def prepare_for_model(text: str) -> str:
    """Input for the TF-IDF baseline: anonymised, case-folded, whitespace collapsed."""
    return normalize_text(anonymize(text))


def prepare_batch(texts: Iterable[str]) -> list[str]:
    return [prepare_for_model(str(text)) for text in texts]


def prepare_for_transformer(text: str) -> str:
    """Input for cased transformer models such as MuRIL.

    Same anonymisation and Unicode cleanup as the baseline, but case is kept:
    the model was pre-trained on cased text, and capitals ("URGENT", "KYC") carry signal.
    """
    return _WHITESPACE.sub(" ", canonicalize(anonymize(str(text)))).strip()
