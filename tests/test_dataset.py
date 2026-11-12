from pathlib import Path

import pandas as pd
import pytest

from ml.dataset import (
    COLUMNS,
    LABELS,
    LANGUAGES,
    SEED_DATA_PATH,
    DatasetError,
    anonymize_dataset,
    deduplicate,
    load_dataset,
    split_dataset,
    validate_dataset,
)


def row(**overrides: str) -> dict[str, str]:
    values = {
        "id": "r1",
        "text": "Your KYC has expired",
        "language": "en",
        "label": "scam",
        "source": "sms",
        "scam_type": "kyc_scam",
    }
    values.update(overrides)
    return values


def frame(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=COLUMNS)


# --- Seed sample -------------------------------------------------------------


def test_seed_sample_is_valid_and_balanced_enough() -> None:
    df = load_dataset(SEED_DATA_PATH)
    assert 90 <= len(df) <= 120
    assert set(df["label"]) == set(LABELS)
    assert set(df["language"]) == set(LANGUAGES)
    assert (df["source"] == "synthetic_sample").all()
    for language in LANGUAGES:
        assert set(df.loc[df["language"] == language, "label"]) == set(LABELS), language


def test_seed_sample_has_no_duplicate_messages() -> None:
    df = anonymize_dataset(load_dataset(SEED_DATA_PATH))
    _, dropped = deduplicate(df)
    assert dropped == 0


# --- Validation --------------------------------------------------------------


def test_valid_rows_are_cleaned() -> None:
    df = validate_dataset(frame(row(label=" SCAM ", language="EN", scam_type="KYC_SCAM")))
    assert df.iloc[0][["label", "language", "scam_type"]].tolist() == ["scam", "en", "kyc_scam"]


def test_missing_column_is_rejected() -> None:
    with pytest.raises(DatasetError, match="missing columns"):
        validate_dataset(pd.DataFrame([{"id": "1", "text": "hi"}]))


@pytest.mark.parametrize(
    ("bad_row", "message"),
    [
        (row(text=""), "empty text"),
        (row(id=""), "empty id"),
        (row(label="fraud"), "label 'fraud'"),
        (row(language="ta"), "language 'ta'"),
        (row(source=""), "empty source"),
        (row(scam_type=""), "scam_type ''"),
        (row(scam_type="made_up"), "scam_type 'made_up'"),
        (row(label="legit", scam_type="kyc_scam"), "must be empty"),
    ],
)
def test_invalid_rows_are_rejected(bad_row: dict[str, str], message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        validate_dataset(frame(bad_row))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(DatasetError, match="duplicate ids: r1"):
        validate_dataset(frame(row(), row(text="another message")))


def test_errors_report_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    frame(row(id="a"), row(id="b", label="oops")).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="line 3 \\(b\\)"):
        load_dataset(path)


def test_other_is_an_allowed_scam_type() -> None:
    validate_dataset(frame(row(scam_type="other")))


# --- Deduplication and splitting ---------------------------------------------


def test_deduplicate_drops_messages_identical_after_anonymisation() -> None:
    df = frame(
        row(id="a", text="Call 9876543210 to update KYC"),
        row(id="b", text="Call  9123456780 to update KYC"),
        row(id="c", text="Something else"),
    )
    deduped, dropped = deduplicate(df)
    assert dropped == 1
    assert deduped["id"].tolist() == ["a", "c"]


def test_deduplicate_rejects_conflicting_labels() -> None:
    df = frame(
        row(id="a", text="Same message"),
        row(id="b", text="same message", label="legit", scam_type=""),
    )
    with pytest.raises(DatasetError, match="labelled differently"):
        deduplicate(df)


def test_split_is_stratified_and_deterministic() -> None:
    df = load_dataset(SEED_DATA_PATH)
    train_a, test_a = split_dataset(df, test_size=0.25, seed=7)
    train_b, test_b = split_dataset(df, test_size=0.25, seed=7)
    assert test_a["id"].tolist() == test_b["id"].tolist()
    assert set(train_a["id"]).isdisjoint(test_a["id"])
    assert len(train_a) + len(test_a) == len(df)
    assert set(test_a["label"]) == set(LABELS)


def test_split_needs_two_examples_per_label() -> None:
    df = frame(row(id="a"), row(id="b", text="x"), row(id="c", label="legit", scam_type=""))
    with pytest.raises(DatasetError, match="at least 2 examples"):
        split_dataset(df)
