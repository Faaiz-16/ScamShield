"""Loading trained text classifiers and running predictions.

Models are saved with joblib (pickle). Only load model files you trained
yourself: unpickling a file from an untrusted source can execute arbitrary code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = 1
SCAM_LABEL = "scam"


class ClassifierUnavailableError(RuntimeError):
    """The requested model has not been trained or cannot be loaded."""


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]
    model_name: str

    @property
    def scam_probability(self) -> float:
        return self.probabilities.get(SCAM_LABEL, 0.0)


class TextClassifier(Protocol):
    name: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


@dataclass
class BaselineClassifier:
    """TF-IDF + logistic regression pipeline that takes raw message text."""

    pipeline: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = "baseline"

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.pipeline.classes_]

    @classmethod
    def load(cls, path: Path | str) -> BaselineClassifier:
        path = Path(path)
        if not path.exists():
            raise ClassifierUnavailableError(
                f"baseline model not found at {path}; run `python -m ml.train_baseline`"
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise ClassifierUnavailableError(f"could not load model {path}: {exc}") from exc

        if not isinstance(bundle, dict) or bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise ClassifierUnavailableError(f"{path} is not a ScamShield baseline model bundle")

        metadata = bundle.get("metadata", {})
        _warn_on_version_mismatch(metadata)
        return cls(pipeline=bundle["pipeline"], metadata=metadata)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "pipeline": self.pipeline,
            "metadata": self.metadata,
        }
        joblib.dump(bundle, path, compress=3)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if not texts:
            return []
        probabilities = self.pipeline.predict_proba(list(texts))
        labels = self.labels
        predictions = []
        for row in probabilities:
            probs = {label: float(p) for label, p in zip(labels, row, strict=True)}
            best = max(probs, key=probs.__getitem__)
            predictions.append(Prediction(best, probs, self.name))
        return predictions

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]


def _warn_on_version_mismatch(metadata: dict[str, Any]) -> None:
    import sklearn

    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "model was trained with scikit-learn %s but %s is installed; retrain if "
            "predictions look wrong",
            trained_with,
            sklearn.__version__,
        )


def _load_muril(settings: Settings) -> TextClassifier:
    # Imported here so the API does not need torch unless MuRIL is used.
    from app.detection.muril_classifier import MurilClassifier

    return MurilClassifier.load(settings.muril_model_dir, device=settings.model_device)


_LOADERS: dict[str, Callable[[Settings], TextClassifier]] = {
    "baseline": lambda settings: BaselineClassifier.load(settings.baseline_model_path),
    "muril": _load_muril,
}


def available_model_names() -> list[str]:
    return list(_LOADERS)


def load_classifier(name: str = "baseline", settings: Settings | None = None) -> TextClassifier:
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ClassifierUnavailableError(
            f"unknown model {name!r}; choose from {available_model_names()}"
        ) from None
    return loader(settings or get_settings())
