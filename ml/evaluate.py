"""Evaluate trained models on the held-out test set and compare them.

Usage:
    python -m ml.evaluate                      # every model that has been trained
    python -m ml.evaluate --models baseline

Writes to ml/results/:
    <model>_metrics.json            precision / recall / F1 per class, confusion matrix
    <model>_confusion_matrix.csv    and .png when matplotlib is installed
    <model>_errors.csv              misclassified test messages (anonymised)
    comparison.md / comparison.json side-by-side table of all evaluated models
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.detection.classifier import (
    ClassifierUnavailableError,
    available_model_names,
    load_classifier,
)
from ml.dataset import LABELS, PROCESSED_DIR, DatasetError, load_dataset
from ml.metrics import (
    comparison_table,
    compute_metrics,
    confusion_matrix_table,
    language_table,
)
from ml.train_baseline import RESULTS_DIR


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--models", nargs="+", default=available_model_names())
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="resamples for confidence intervals (0 = off)"
    )
    return parser.parse_args(argv)


def save_confusion_matrix_plot(metrics: dict[str, Any], title: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > peak / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def evaluate_model(
    name: str, df: pd.DataFrame, results_dir: Path, bootstrap: int = 1000
) -> dict[str, Any] | None:
    try:
        classifier = load_classifier(name)
    except ClassifierUnavailableError as exc:
        print(f"skipping {name}: {exc}")
        return None

    texts = df["text"].tolist()
    started = time.perf_counter()
    predictions = classifier.predict(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000

    y_true = df["label"].tolist()
    y_pred = [p.label for p in predictions]
    scam_probs = [p.scam_probability for p in predictions]
    metrics = compute_metrics(
        y_true,
        y_pred,
        scam_probs,
        languages=df["language"].tolist(),
        bootstrap_samples=bootstrap,
    )
    ms_per_message = elapsed_ms / max(len(texts), 1)

    payload = {"model": name, "ms_per_message": ms_per_message, "metrics": metrics}
    (results_dir / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2), "utf-8")

    labels = metrics["confusion_matrix"]["labels"]
    pd.DataFrame(metrics["confusion_matrix"]["matrix"], index=labels, columns=labels).to_csv(
        results_dir / f"{name}_confusion_matrix.csv", index_label="true\\predicted"
    )
    save_confusion_matrix_plot(metrics, name, results_dir / f"{name}_confusion_matrix.png")

    errors = df.assign(predicted=y_pred, scam_probability=[round(p, 4) for p in scam_probs])
    errors = errors[errors["label"] != errors["predicted"]]
    errors[["id", "language", "label", "predicted", "scam_probability", "text"]].to_csv(
        results_dir / f"{name}_errors.csv", index=False
    )

    print(f"\n== {name} ({len(texts)} test messages) ==")
    ci = metrics["macro_f1_ci95"]
    ci_text = f" (95% CI {ci[0]:.3f} to {ci[1]:.3f})" if ci else ""
    print(
        f"accuracy {metrics['accuracy']:.3f}, "
        f"macro F1 {metrics['macro_avg']['f1-score']:.3f}{ci_text}"
    )
    for label in LABELS:
        c = metrics["per_class"][label]
        print(
            f"  {label:<14} precision {c['precision']:.3f}  recall {c['recall']:.3f}  "
            f"f1 {c['f1-score']:.3f}  support {int(c['support'])}"
        )
    print(confusion_matrix_table(metrics))
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = load_dataset(args.test)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for name in args.models
        if (r := evaluate_model(name, df, args.results_dir, args.bootstrap))
    ]
    if not rows:
        print("error: no trained model could be evaluated", file=sys.stderr)
        return 1

    table = comparison_table(rows)
    by_language = language_table(rows)
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    report = (
        f"# Model comparison\n\nTest set: `{args.test}` ({len(df)} messages), "
        f"generated {generated}.\n\n## Overall\n\n{table}"
    )
    if by_language:
        report += f"\n## By language\n\n{by_language}"
    report += (
        "\nThe confidence interval comes from bootstrap resampling of the test set. "
        "When two models' intervals overlap heavily, the test set is too small to say "
        "which one is better.\n"
    )
    (args.results_dir / "comparison.md").write_text(report, encoding="utf-8")
    (args.results_dir / "comparison.json").write_text(
        json.dumps(
            {"generated_at": generated, "test_file": str(args.test), "models": rows}, indent=2
        ),
        encoding="utf-8",
    )
    print("\n" + table)
    if by_language:
        print(by_language)
    print(f"Results written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
