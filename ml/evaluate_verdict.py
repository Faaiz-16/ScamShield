"""Evaluate the complete system (rules + links + model + verdict engine) on the test set.

Usage:
    python -m ml.evaluate_verdict                         # every available classifier
    python -m ml.evaluate_verdict --classifiers none baseline

Unlike ml.evaluate, which scores the text classifiers alone, this measures what
users actually see: the Safe / Suspicious / Likely Scam verdict. It answers
"is it a scam?", so spam and genuine messages both count as "not a scam".

Link checks run offline (no shortener expansion, WHOIS or reputation lookups)
unless --network is passed, so results are reproducible.

Also sweeps the Likely Scam threshold to show the trade-off between catching
scams and wrongly flagging genuine messages, as a guide for SCAM_THRESHOLD.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.detection.classifier import (
    ClassifierUnavailableError,
    available_model_names,
    load_classifier,
)
from app.detection.rules import RuleEngine
from app.detection.url_analyzer import UrlAnalyzer
from app.detection.verdict import VerdictConfig, VerdictEngine
from app.services.analysis import AnalysisService
from ml.dataset import PROCESSED_DIR, DatasetError, load_dataset
from ml.train_baseline import RESULTS_DIR

LEVELS = ("safe", "suspicious", "likely_scam")
SWEEP = [round(float(x), 2) for x in np.arange(0.40, 0.91, 0.05)]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["none", *available_model_names()],
        help="'none' means rules and links only",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--network", action="store_true", help="enable online link checks")
    return parser.parse_args(argv)


def build_service(classifier_name: str, settings: Settings) -> AnalysisService:
    rules = RuleEngine.from_file(settings.rules_path)
    classifier = None if classifier_name == "none" else load_classifier(classifier_name, settings)
    return AnalysisService(
        rule_engine=rules,
        url_analyzer=UrlAnalyzer(settings),
        verdict_engine=VerdictEngine(VerdictConfig.from_settings(settings), rules.categories),
        classifier=classifier,
    )


async def run_service(service: AnalysisService, texts: list[str]) -> list[tuple[str, float]]:
    try:
        results = [await service.analyze(text) for text in texts]
    finally:
        await service.url_analyzer.aclose()
    return [(r.verdict.level, r.verdict.risk_score) for r in results]


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def binary_metrics(is_scam: np.ndarray, predicted_scam: np.ndarray) -> dict[str, Any]:
    true_positive = int((is_scam & predicted_scam).sum())
    false_positive = int((~is_scam & predicted_scam).sum())
    false_negative = int((is_scam & ~predicted_scam).sum())
    precision = rate(true_positive, true_positive + false_positive)
    recall = rate(true_positive, true_positive + false_negative)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": rate(false_positive, int((~is_scam).sum())),
    }


def score_verdicts(
    labels: list[str], levels: list[str], risks: list[float], config: VerdictConfig
) -> dict[str, Any]:
    is_scam = np.array([label == "scam" for label in labels])
    level_array = np.array(levels)
    risk_array = np.array(risks)

    distribution = {
        label: {
            level: int(((np.array(labels) == label) & (level_array == level)).sum())
            for level in LEVELS
        }
        for label in sorted(set(labels))
    }
    sweep = []
    for threshold in SWEEP:
        stats = binary_metrics(is_scam, risk_array >= threshold)
        sweep.append({"threshold": threshold, **stats})
    best = max(sweep, key=lambda row: (row["f1"], -abs(row["threshold"] - config.scam_threshold)))

    return {
        "samples": len(labels),
        "likely_scam": binary_metrics(is_scam, level_array == "likely_scam"),
        "flagged": binary_metrics(is_scam, level_array != "safe"),
        "distribution": distribution,
        "threshold_sweep": sweep,
        "best_threshold": best["threshold"],
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def build_report(rows: list[dict[str, Any]], test_file: Path, config: VerdictConfig) -> str:
    lines = [
        "# Verdict evaluation",
        "",
        f"Test set: `{test_file}` ({rows[0]['metrics']['samples']} messages). "
        f"Thresholds: suspicious {config.suspicious_threshold}, likely scam "
        f"{config.scam_threshold}. A scam counts as caught when the verdict is Likely Scam; "
        "flagged means Suspicious or Likely Scam.",
        "",
        "| Classifier | Likely Scam precision | Likely Scam recall | Likely Scam F1 "
        "| Non-scams called Likely Scam | Scams flagged | Non-scams flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        likely, flagged = row["metrics"]["likely_scam"], row["metrics"]["flagged"]
        lines.append(
            f"| {row['classifier']} | {_fmt(likely['precision'])} | {_fmt(likely['recall'])} "
            f"| {_fmt(likely['f1'])} | {_pct(likely['false_alarm_rate'])} "
            f"| {_pct(flagged['recall'])} | {_pct(flagged['false_alarm_rate'])} |"
        )

    for row in rows:
        metrics = row["metrics"]
        lines += [
            "",
            f"## {row['classifier']}",
            "",
            "| True label | Safe | Suspicious | Likely Scam |",
            "|---|---|---|---|",
        ]
        for label, counts in metrics["distribution"].items():
            lines.append(
                f"| {label} | {counts['safe']} | {counts['suspicious']} | {counts['likely_scam']} |"
            )
        lines += [
            "",
            "Likely Scam threshold sweep:",
            "",
            "| Threshold | Precision | Recall | F1 | Non-scams called Likely Scam |",
            "|---|---|---|---|---|",
        ]
        for point in metrics["threshold_sweep"]:
            marker = " (current)" if point["threshold"] == config.scam_threshold else ""
            lines.append(
                f"| {point['threshold']:.2f}{marker} | {_fmt(point['precision'])} "
                f"| {_fmt(point['recall'])} | {_fmt(point['f1'])} "
                f"| {_pct(point['false_alarm_rate'])} |"
            )
        lines.append(f"\nBest F1 at threshold {metrics['best_threshold']:.2f}.")
    lines.append(
        "\nA threshold chosen on this test set should be confirmed on fresh data before "
        "changing SCAM_THRESHOLD, otherwise the reported scores become optimistic.\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = load_dataset(args.test)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    base = get_settings()
    settings = base.model_copy(update={"url_network_checks": args.network})
    config = VerdictConfig.from_settings(settings)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    rows = []
    for name in args.classifiers:
        try:
            service = build_service(name, settings)
        except ClassifierUnavailableError as exc:
            print(f"skipping {name}: {exc}")
            continue
        outcomes = asyncio.run(run_service(service, texts))
        metrics = score_verdicts(
            labels, [level for level, _ in outcomes], [risk for _, risk in outcomes], config
        )
        rows.append({"classifier": name, "metrics": metrics})
        likely = metrics["likely_scam"]
        print(
            f"{name:<9} likely-scam precision {_fmt(likely['precision'])}, "
            f"recall {_fmt(likely['recall'])}, non-scams called scam "
            f"{_pct(likely['false_alarm_rate'])}"
        )

    if not rows:
        print("error: no classifier configuration could be evaluated", file=sys.stderr)
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(rows, args.test, config)
    (args.results_dir / "verdict_evaluation.md").write_text(report, encoding="utf-8")
    (args.results_dir / "verdict_evaluation.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "test_file": str(args.test),
                "network_checks": args.network,
                "config": {
                    "suspicious_threshold": config.suspicious_threshold,
                    "scam_threshold": config.scam_threshold,
                    "rule_weight": config.rule_weight,
                    "url_weight": config.url_weight,
                    "ml_weight": config.ml_weight,
                },
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.results_dir / 'verdict_evaluation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
