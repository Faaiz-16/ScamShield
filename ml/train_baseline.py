"""Train the TF-IDF + logistic regression baseline.

Usage:
    python -m ml.train_baseline --train data/processed/train.csv
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from app.core.config import BASE_DIR, get_settings
from app.detection.classifier import BaselineClassifier
from ml.baseline import build_pipeline
from ml.dataset import PROCESSED_DIR, DatasetError, describe, load_dataset

RESULTS_DIR = BASE_DIR / "ml" / "results"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--output", type=Path, default=get_settings().baseline_model_path)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--c-values",
        default="0.5,1,2,5,10",
        help="comma-separated inverse regularisation strengths to search",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = load_dataset(args.train)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    texts = df["text"].tolist()
    labels = df["label"].tolist()
    smallest_class = int(df["label"].value_counts().min())
    n_splits = min(5, smallest_class)
    if n_splits < 2 or df["label"].nunique() < 2:
        print("error: need at least two labels with two examples each", file=sys.stderr)
        return 1

    c_values = [float(v) for v in args.c_values.split(",") if v.strip()]
    search = GridSearchCV(
        build_pipeline(seed=args.seed),
        param_grid={"classifier__C": c_values},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed),
        refit=True,
    )
    search.fit(texts, labels)

    best_index = search.best_index_
    cv_mean = float(search.cv_results_["mean_test_score"][best_index])
    cv_std = float(search.cv_results_["std_test_score"][best_index])

    metadata = {
        "model": "tfidf_word_char_logreg",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_file": str(args.train),
        "train_sha256": file_sha256(args.train),
        "train_summary": describe(df),
        "best_params": {k: float(v) for k, v in search.best_params_.items()},
        "cv_folds": n_splits,
        "cv_f1_macro_mean": cv_mean,
        "cv_f1_macro_std": cv_std,
    }
    BaselineClassifier(pipeline=search.best_estimator_, metadata=metadata).save(args.output)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "baseline_training.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Trained on {len(df)} messages, best C = {search.best_params_['classifier__C']}")
    print(f"{n_splits}-fold CV macro F1: {cv_mean:.3f} +/- {cv_std:.3f}")
    print(f"Saved model to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
