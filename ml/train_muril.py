"""Fine-tune MuRIL (google/muril-base-cased) to classify messages.

Usage on a laptop CPU (slow but works; freezing lower layers speeds it up):
    python -m ml.train_muril --device cpu --batch-size 8 --grad-accum 4 --freeze-layers 6

Usage on a GPU, for example in Google Colab (see ml/colab/train_muril.ipynb):
    python -m ml.train_muril --device cuda --batch-size 32 --fp16

The best checkpoint by validation macro F1 is written to --output-dir together
with the tokenizer and a scamshield_metadata.json file.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from app.core.config import get_settings
from app.detection.preprocessing import prepare_for_transformer
from ml.dataset import LABELS, PROCESSED_DIR, DatasetError, describe, load_dataset, split_dataset
from ml.train_baseline import RESULTS_DIR

logger = logging.getLogger("scamshield.train_muril")

DEFAULT_MODEL = "google/muril-base-cased"
METADATA_FILE = "scamshield_metadata.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=get_settings().muril_model_dir)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--model-name", default=DEFAULT_MODEL, help="Hugging Face model id or local path"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--grad-accum", type=int, default=2, help="batches per optimiser step (effective batch)"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="share of training data held out (0 = none)"
    )
    parser.add_argument("--patience", type=int, default=2, help="epochs without improvement")
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="freeze embeddings and the lowest N encoder layers (faster on CPU)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument(
        "--num-threads", type=int, default=0, help="CPU threads (0 = torch default)"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: list[str]) -> list[float]:
    """Inverse-frequency weights so the rarer classes are not ignored."""
    counts = pd.Series(labels).value_counts()
    total = len(labels)
    return [total / (len(LABELS) * counts.get(label, 1)) for label in LABELS]


def freeze_lower_layers(model: Any, n_layers: int) -> int:
    """Freeze the embeddings and the first n encoder layers. Returns layers frozen."""
    if n_layers <= 0:
        return 0
    base = getattr(model, model.base_model_prefix)
    for parameter in base.embeddings.parameters():
        parameter.requires_grad = False
    layers = base.encoder.layer
    n_layers = min(n_layers, len(layers))
    for layer in layers[:n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return n_layers


@dataclass
class EncodedSplit:
    features: list[dict[str, list[int]]]
    labels: list[int]


def encode(tokenizer: Any, df: pd.DataFrame, max_length: int) -> EncodedSplit:
    texts = [prepare_for_transformer(t) for t in df["text"]]
    encodings = tokenizer(texts, truncation=True, max_length=max_length)
    features = [{key: encodings[key][i] for key in encodings.keys()} for i in range(len(texts))]
    return EncodedSplit(features, [LABELS.index(label) for label in df["label"]])


def make_loader(split: EncodedSplit, collator: Any, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    rows = [
        dict(features, labels=label)
        for features, label in zip(split.features, split.labels, strict=True)
    ]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, generator=generator
    )


def evaluate_loader(model: Any, loader: Any, device: str) -> tuple[float, float]:
    """Return (macro F1, accuracy) on a data loader."""
    import torch

    model.eval()
    predicted: list[int] = []
    expected: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            predicted += logits.argmax(dim=-1).cpu().tolist()
            expected += labels.tolist()
    model.train()
    macro_f1 = f1_score(
        expected, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0
    )
    accuracy = float(np.mean(np.array(predicted) == np.array(expected))) if expected else 0.0
    return float(macro_f1), accuracy


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        print(
            "error: install the ML extras first: pip install -r requirements-ml.txt",
            file=sys.stderr,
        )
        return 1

    from app.detection.muril_classifier import resolve_device

    try:
        device = resolve_device(args.device)
        df = load_dataset(args.train)
    except (ValueError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    use_fp16 = args.fp16 and device == "cuda"
    if args.fp16 and not use_fp16:
        logger.warning("--fp16 only applies to CUDA GPUs; training in full precision on %s", device)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    if args.val_size > 0:
        try:
            train_df, val_df = split_dataset(df, test_size=args.val_size, seed=args.seed)
        except (DatasetError, ValueError) as exc:
            logger.warning("training without a validation split: %s", exc)
            train_df, val_df = df, None
    else:
        train_df, val_df = df, None

    logger.info("loading %s on %s", args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)
    frozen = freeze_lower_layers(model, args.freeze_layers)

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = make_loader(
        encode(tokenizer, train_df, args.max_length), collator, args.batch_size, True, args.seed
    )
    val_loader = (
        make_loader(
            encode(tokenizer, val_df, args.max_length), collator, args.batch_size, False, args.seed
        )
        if val_df is not None
        else None
    )

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if "bias" in name or "LayerNorm" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    steps_per_epoch = max(1, -(-len(train_loader) // args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights(train_df["label"].tolist()), dtype=torch.float32, device=device
        )
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    best_f1, best_epoch, stale_epochs = -1.0, 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda" if use_fp16 else "cpu", dtype=torch.float16, enabled=use_fp16
            ):
                logits = model(**inputs).logits
            loss = loss_fn(logits.float(), labels) / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        record = {"epoch": epoch, "train_loss": running_loss / len(train_loader)}
        if val_loader is not None:
            record["val_macro_f1"], record["val_accuracy"] = evaluate_loader(
                model, val_loader, device
            )
        history.append(record)
        logger.info(
            "epoch %d: %s", epoch, {k: round(v, 4) for k, v in record.items() if k != "epoch"}
        )

        score = record.get("val_macro_f1", float(epoch))  # without validation, keep the last epoch
        if score > best_f1:
            best_f1, best_epoch, stale_epochs = score, epoch, 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        else:
            stale_epochs += 1
            if val_loader is not None and stale_epochs >= args.patience:
                logger.info("stopping early: no improvement for %d epochs", stale_epochs)
                break

    metadata = {
        "model": "muril_finetuned",
        "base_model": args.model_name,
        "labels": list(LABELS),
        "max_length": args.max_length,
        "preprocessing": "prepare_for_transformer",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": device,
        "fp16": use_fp16,
        "frozen_layers": frozen,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        },
        "train_summary": describe(train_df),
        "validation_rows": 0 if val_df is None else len(val_df),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1 if val_loader is not None else None,
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    (args.output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), "utf-8")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "muril_training.json").write_text(json.dumps(metadata, indent=2), "utf-8")

    best = f", best validation macro F1 {best_f1:.3f}" if val_loader is not None else ""
    print(f"Saved model from epoch {best_epoch} to {args.output_dir}{best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
