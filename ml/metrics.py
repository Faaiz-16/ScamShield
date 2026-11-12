from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from ml.dataset import LABELS, LANGUAGES


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scam_probabilities: Sequence[float] | None = None,
    labels: Sequence[str] = LABELS,
    languages: Sequence[str] | None = None,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    labels = list(labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    legit, scam = labels.index("legit"), labels.index("scam")
    legit_total = int(matrix[legit].sum())
    # The costliest mistake for users: a genuine bank message flagged as a scam.
    legit_as_scam = matrix[legit][scam] / legit_total if legit_total else 0.0

    scam_auc = None
    is_scam = [label == "scam" for label in y_true]
    if scam_probabilities is not None and 0 < sum(is_scam) < len(is_scam):
        scam_auc = float(roc_auc_score(is_scam, scam_probabilities))

    metrics: dict[str, Any] = {
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_avg": _round_dict(report["macro avg"]),
        "weighted_avg": _round_dict(report["weighted avg"]),
        "per_class": {label: _round_dict(report[label]) for label in labels},
        "legit_flagged_as_scam_rate": float(legit_as_scam),
        "scam_roc_auc": scam_auc,
        "confusion_matrix": {"labels": labels, "matrix": matrix.tolist()},
        "macro_f1_ci95": None,
        "by_language": {},
    }
    if bootstrap_samples:
        metrics["macro_f1_ci95"] = bootstrap_macro_f1(
            y_true, y_pred, labels, bootstrap_samples, seed
        )
    if languages is not None:
        metrics["by_language"] = per_language_metrics(y_true, y_pred, languages, labels)
    return metrics


def bootstrap_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    samples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """95% percentile bootstrap interval for macro F1.

    On a small test set the interval is wide, which is the honest picture:
    two models whose intervals overlap heavily cannot be told apart.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        index = rng.integers(0, len(true), len(true))
        scores.append(_macro_f1_present(true[index], pred[index], labels))
    low, high = np.percentile(scores, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def _macro_f1_present(true: np.ndarray, pred: np.ndarray, labels: Sequence[str]) -> float:
    """Macro F1 over the labels that occur in this sample.

    A label absent from both the truth and the predictions (common in a small
    resample or a single-language slice) would otherwise count as an F1 of 0.
    """
    present = [label for label in labels if (true == label).any() or (pred == label).any()]
    return float(f1_score(true, pred, labels=present, average="macro", zero_division=0))


def per_language_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    languages: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> dict[str, dict[str, float | int | None]]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    langs = np.asarray(languages)
    result: dict[str, dict[str, float | int | None]] = {}
    for language in LANGUAGES:
        mask = langs == language
        if not mask.any():
            continue
        has_scam = bool((true[mask] == "scam").any())
        result[language] = {
            "samples": int(mask.sum()),
            "accuracy": round(float(accuracy_score(true[mask], pred[mask])), 4),
            "macro_f1": round(_macro_f1_present(true[mask], pred[mask], labels), 4),
            "scam_recall": (
                round(
                    float(
                        recall_score(
                            true[mask],
                            pred[mask],
                            labels=["scam"],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                )
                if has_scam
                else None
            ),
        }
    return result


def _round_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def comparison_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "| Model | Accuracy | Macro P | Macro R | Macro F1 | Macro F1 95% CI | Scam P | Scam R "
        "| Scam F1 | Legit flagged as scam | Scam ROC-AUC | ms / message |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        m = row["metrics"]
        scam = m["per_class"]["scam"]
        macro = m["macro_avg"]
        ci = m.get("macro_f1_ci95")
        ci_text = f"{ci[0]:.3f} to {ci[1]:.3f}" if ci else "n/a"
        lines.append(
            f"| {row['model']} | {m['accuracy']:.3f} | {macro['precision']:.3f} "
            f"| {macro['recall']:.3f} | {macro['f1-score']:.3f} | {ci_text} "
            f"| {scam['precision']:.3f} | {scam['recall']:.3f} | {scam['f1-score']:.3f} "
            f"| {m['legit_flagged_as_scam_rate']:.1%} | {_fmt(m['scam_roc_auc'])} "
            f"| {row['ms_per_message']:.2f} |"
        )
    return header + "\n".join(lines) + "\n"


def language_table(rows: Sequence[dict[str, Any]]) -> str:
    """Macro F1 and scam recall per language, one row per model."""
    columns = [lang for lang in LANGUAGES if any(lang in r["metrics"]["by_language"] for r in rows)]
    if not columns:
        return ""
    header = "| Model | " + " | ".join(f"{lang} macro F1 | {lang} scam recall" for lang in columns)
    header += " |\n|---" + "|---|---" * len(columns) + "|\n"
    lines = []
    for row in rows:
        cells = []
        for lang in columns:
            stats = row["metrics"]["by_language"].get(lang)
            if stats is None:
                cells += ["n/a", "n/a"]
            else:
                cells += [_fmt(stats["macro_f1"]), _fmt(stats["scam_recall"])]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    samples = ", ".join(
        f"{lang}: {rows[0]['metrics']['by_language'][lang]['samples']}"
        for lang in columns
        if lang in rows[0]["metrics"]["by_language"]
    )
    return header + "\n".join(lines) + f"\n\nTest messages per language: {samples}.\n"


def confusion_matrix_table(metrics: dict[str, Any]) -> str:
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    header = "| true \\ predicted | " + " | ".join(labels) + " |\n"
    header += "|---" * (len(labels) + 1) + "|\n"
    body = "\n".join(
        f"| {label} | " + " | ".join(str(v) for v in row) + " |"
        for label, row in zip(labels, matrix, strict=True)
    )
    return header + body + "\n"
