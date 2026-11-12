"""Validate, anonymise, deduplicate and split labelled CSV files.

Usage:
    python -m ml.prepare_data                       # seed sample + every file in data/labels/
    python -m ml.prepare_data --input data/seed_sample.csv data/raw/my_messages.csv
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml.dataset import (
    PROCESSED_DIR,
    SEED_DATA_PATH,
    DatasetError,
    anonymize_dataset,
    deduplicate,
    describe,
    load_dataset,
    reviewed_label_files,
    split_dataset,
    validate_dataset,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[SEED_DATA_PATH, *reviewed_label_files()],
        help="labelled CSV files (default: the seed sample and every file in data/labels/)",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frames = [load_dataset(path) for path in args.input]
        combined = validate_dataset(pd.concat(frames, ignore_index=True), "combined input")
        anonymized = anonymize_dataset(combined)
        deduped, dropped = deduplicate(anonymized)
        train, test = split_dataset(deduped, args.test_size, args.seed)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": [str(p) for p in args.input],
        "duplicates_dropped": dropped,
        "test_size": args.test_size,
        "seed": args.seed,
        "train": describe(train),
        "test": describe(test),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Inputs: " + ", ".join(p.name for p in args.input))
    print(f"Wrote {len(train)} training and {len(test)} test rows to {args.output_dir}")
    print(f"Duplicates dropped: {dropped}")
    print(f"Train labels: {summary['train']['labels']}")
    print(f"Test labels:  {summary['test']['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
