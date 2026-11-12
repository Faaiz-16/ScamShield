"""Loading, validation, anonymisation and splitting of labelled message data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import BASE_DIR
from app.detection.anonymize import anonymize
from app.detection.preprocessing import prepare_for_model
from app.detection.rules import load_rule_set

LABELS = ("scam", "legit", "spam_not_scam")
LANGUAGES = ("en", "hi", "hinglish")
COLUMNS = ["id", "text", "language", "label", "source", "scam_type"]

SEED_DATA_PATH = BASE_DIR / "data" / "seed_sample.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


class DatasetError(ValueError):
    pass


def allowed_scam_types() -> set[str]:
    categories = set(load_rule_set().categories) - {"generic"}
    return categories | {"other"}


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"{path}: file not found")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return validate_dataset(frame, source_name=str(path))


def validate_dataset(frame: pd.DataFrame, source_name: str = "dataset") -> pd.DataFrame:
    """Check the schema and every row. Returns a cleaned copy or raises DatasetError."""
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise DatasetError(f"{source_name}: missing columns {missing}; expected {COLUMNS}")

    df = frame[COLUMNS].copy().fillna("").astype(str)
    for column in COLUMNS:
        df[column] = df[column].str.strip()
    for column in ("language", "label", "scam_type"):
        df[column] = df[column].str.lower()

    scam_types = allowed_scam_types()
    errors: list[str] = []
    for position, row in enumerate(df.itertuples(index=False)):
        line = position + 2  # header is line 1
        if not row.id:
            errors.append(f"line {line}: empty id")
        if not row.text:
            errors.append(f"line {line} ({row.id}): empty text")
        if row.label not in LABELS:
            errors.append(f"line {line} ({row.id}): label {row.label!r} not in {LABELS}")
        if row.language not in LANGUAGES:
            errors.append(f"line {line} ({row.id}): language {row.language!r} not in {LANGUAGES}")
        if not row.source:
            errors.append(f"line {line} ({row.id}): empty source")
        if row.label == "scam" and row.scam_type not in scam_types:
            errors.append(
                f"line {line} ({row.id}): scam_type {row.scam_type!r} is not a known category"
            )
        if row.label != "scam" and row.scam_type:
            errors.append(f"line {line} ({row.id}): scam_type must be empty for {row.label!r}")

    duplicated = df.loc[df["id"].duplicated() & (df["id"] != ""), "id"].unique()
    if len(duplicated):
        errors.append(f"duplicate ids: {', '.join(sorted(duplicated)[:20])}")

    if errors:
        shown = "\n  ".join(errors[:50])
        more = f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""
        raise DatasetError(f"{source_name}: {len(errors)} problem(s):\n  {shown}{more}")
    return df


def anonymize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["text"] = result["text"].map(anonymize)
    return result


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop messages that are identical after model preprocessing.

    The same text with two different labels is a labelling mistake, so it is
    reported instead of silently resolved.
    """
    keys = df["text"].map(prepare_for_model)
    labels_per_key = df.groupby(keys)["label"].nunique()
    conflicting = labels_per_key[labels_per_key > 1].index
    if len(conflicting):
        ids = df.loc[keys.isin(conflicting), "id"].tolist()
        raise DatasetError(f"same text labelled differently, check ids: {', '.join(ids[:20])}")

    keep = ~keys.duplicated()
    return df[keep].reset_index(drop=True), int((~keep).sum())


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["label"].value_counts()
    too_small = counts[counts < 2]
    if len(too_small):
        raise DatasetError(
            f"need at least 2 examples per label to split, got {too_small.to_dict()}"
        )
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(df),
        "labels": df["label"].value_counts().to_dict(),
        "languages": df["language"].value_counts().to_dict(),
        "scam_types": df.loc[df["label"] == "scam", "scam_type"].value_counts().to_dict(),
    }
