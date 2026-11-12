"""Inference with a fine-tuned MuRIL (or any BERT-style) sequence classifier.

torch and transformers are imported lazily, so the API runs without them
unless CLASSIFIER=muril is configured. Install them from requirements-ml.txt.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.detection.classifier import ClassifierUnavailableError, Prediction
from app.detection.preprocessing import prepare_for_transformer

logger = logging.getLogger(__name__)

METADATA_FILE = "scamshield_metadata.json"


def resolve_device(requested: str = "auto") -> str:
    """Pick "cuda", "mps" or "cpu". "auto" prefers a GPU when one is available."""
    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA GPU is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available on this machine")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device {requested!r}; use auto, cpu, cuda or mps")
    return requested


class MurilClassifier:
    name = "muril"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        metadata: dict[str, Any] | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.metadata = metadata or {}
        self.device = device
        self.batch_size = batch_size
        self.max_length = int(self.metadata.get("max_length", 128))
        id2label = model.config.id2label
        self.labels = [str(id2label[i]) for i in range(len(id2label))]

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> MurilClassifier:
        path = Path(path)
        if not (path / "config.json").exists():
            raise ClassifierUnavailableError(
                f"MuRIL model not found at {path}; run `python -m ml.train_muril`"
            )
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ClassifierUnavailableError(
                "torch and transformers are not installed; run `pip install -r requirements-ml.txt`"
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            # Weights are stored as safetensors, which cannot execute code on load.
            model = AutoModelForSequenceClassification.from_pretrained(path)
            resolved = resolve_device(device)
        except (OSError, ValueError) as exc:
            raise ClassifierUnavailableError(f"could not load MuRIL model {path}: {exc}") from exc

        metadata_path = path / METADATA_FILE
        metadata = json.loads(metadata_path.read_text("utf-8")) if metadata_path.exists() else {}
        return cls(model, tokenizer, metadata=metadata, device=resolved)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        import torch

        predictions: list[Prediction] = []
        prepared = [prepare_for_transformer(t) for t in texts]
        for start in range(0, len(prepared), self.batch_size):
            batch = self.tokenizer(
                prepared[start : start + self.batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(**batch).logits.float(), dim=-1).cpu()
            for row in probabilities.tolist():
                probs = dict(zip(self.labels, row, strict=True))
                best = max(probs, key=probs.__getitem__)
                predictions.append(Prediction(best, probs, self.name))
        return predictions
