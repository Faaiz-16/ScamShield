"""Feedback export for retraining and end-to-end verdict evaluation."""

import json
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app.core.database import Base, build_engine, build_session_factory, utcnow
from app.detection.verdict import VerdictConfig
from app.models import Analysis, Feedback
from ml import evaluate_verdict, export_feedback, prepare_data, train_baseline
from ml.dataset import DatasetError, load_dataset
from ml.evaluate_verdict import binary_metrics, score_verdicts

# --- Feedback export ---------------------------------------------------------


@pytest.fixture
def database(tmp_path: Path) -> Iterator[tuple[str, Session]]:
    url = f"sqlite:///{tmp_path / 'feedback.db'}"
    engine = build_engine(url)
    Base.metadata.create_all(engine)
    with build_session_factory(engine)() as session:
        yield url, session
    engine.dispose()


def add_analysis(
    session: Session,
    verdict: str,
    votes: list[tuple[bool, str | None]],
    *,
    scam_type: str | None = None,
    text: str | None = None,
    age_days: int = 0,
) -> str:
    public_id = str(uuid.uuid4())
    analysis = Analysis(
        public_id=public_id,
        created_at=utcnow() - timedelta(days=age_days),
        channel="web",
        client_hash="c" * 64,
        anonymized_text=text or f"Message {public_id}",
        language="en",
        response_language="en",
        verdict=verdict,
        risk_score=0.5,
        confidence=0.7,
        scam_type=scam_type,
        rule_ids=[],
        link_domains=[],
    )
    session.add(analysis)
    session.flush()
    for i, (is_correct, label) in enumerate(votes):
        session.add(
            Feedback(
                analysis_id=analysis.id,
                client_hash=f"{i:064d}",
                is_correct=is_correct,
                correct_label=label,
            )
        )
    session.commit()
    return public_id


def labels_of(session: Session, **kwargs) -> dict[str, str]:
    rows, _ = export_feedback.collect_rows(session, since=utcnow() - timedelta(days=30), **kwargs)
    return {row["id"].removeprefix("fb-"): row["label"] for row in rows}


def test_votes_become_labels(database) -> None:
    _, session = database
    wrong_with_label = add_analysis(session, "likely_scam", [(False, "legit")])
    confirmed_scam = add_analysis(session, "likely_scam", [(True, None)], scam_type="kyc_scam")
    confirmed_safe = add_analysis(session, "safe", [(True, None)])
    confirmed_suspicious = add_analysis(session, "suspicious", [(True, None)])
    wrong_without_label = add_analysis(session, "safe", [(False, None)])
    add_analysis(session, "safe", [])  # never rated

    labels = labels_of(session)
    assert labels == {wrong_with_label: "legit", confirmed_scam: "scam"}
    assert confirmed_safe not in labels
    assert confirmed_suspicious not in labels and wrong_without_label not in labels

    with_safe = labels_of(session, confirmed_safe_as_legit=True)
    assert with_safe[confirmed_safe] == "legit"


def test_majority_wins_and_ties_are_skipped(database) -> None:
    _, session = database
    majority = add_analysis(session, "likely_scam", [(True, None), (True, None), (False, "legit")])
    tie = add_analysis(session, "likely_scam", [(True, None), (False, "legit")])
    _, summary = export_feedback.collect_rows(session, since=utcnow() - timedelta(days=30))
    labels = labels_of(session)
    assert labels[majority] == "scam"
    assert tie not in labels
    assert summary.skipped["users disagreed"] == 1


def test_min_votes_and_time_window(database) -> None:
    _, session = database
    single = add_analysis(session, "likely_scam", [(True, None)])
    double = add_analysis(session, "likely_scam", [(True, None), (True, None)])
    old = add_analysis(session, "likely_scam", [(True, None)], age_days=90)
    assert set(labels_of(session, min_votes=2)) == {double}
    assert old not in labels_of(session)
    assert single in labels_of(session)


def test_unknown_scam_type_becomes_other(database) -> None:
    _, session = database
    public_id = add_analysis(session, "likely_scam", [(True, None)], scam_type=None)
    rows, _ = export_feedback.collect_rows(session, since=utcnow() - timedelta(days=1))
    assert rows[0]["id"] == f"fb-{public_id}"
    assert rows[0]["scam_type"] == "other"
    assert rows[0]["source"] == "user_feedback"


def test_export_writes_a_valid_dataset(database, tmp_path: Path, capsys) -> None:
    url, session = database
    add_analysis(session, "likely_scam", [(True, None)], scam_type="upi_fraud")
    add_analysis(session, "safe", [(False, "scam")], text="Enter UPI PIN to receive <AMOUNT>")
    output = tmp_path / "out" / "feedback.csv"

    labels_dir = tmp_path / "labels"
    args = ["--database-url", url, "--output", str(output), "--labels-dir", str(labels_dir)]
    assert export_feedback.main(args) == 0
    df = load_dataset(output)
    assert len(df) == 2
    assert set(df["label"]) == {"scam"}
    assert "Wrote 2 rows" in capsys.readouterr().out


def test_export_with_nothing_to_write(database, tmp_path: Path, capsys) -> None:
    url, session = database
    add_analysis(session, "suspicious", [(True, None)])
    output = tmp_path / "feedback.csv"
    args = ["--database-url", url, "--output", str(output), "--labels-dir", str(tmp_path)]
    assert export_feedback.main(args) == 0
    assert not output.exists()
    assert "nothing written" in capsys.readouterr().out


def test_export_skips_reviewed_rows_and_marks_keep(database, tmp_path: Path) -> None:
    url, session = database
    first = add_analysis(session, "likely_scam", [(True, None)], scam_type="upi_fraud")
    labels_dir = tmp_path / "labels"
    output = tmp_path / "first.csv"
    base = ["--database-url", url, "--labels-dir", str(labels_dir)]
    assert export_feedback.main([*base, "--output", str(output)]) == 0
    assert set(pd.read_csv(output, dtype=str)["keep"]) == {"yes"}

    # The reviewer rejects the row and saves the file in the labels folder.
    labels_dir.mkdir()
    reviewed = pd.read_csv(output, dtype=str, keep_default_na=False)
    reviewed["keep"] = "no"
    reviewed.to_csv(labels_dir / "feedback-1.csv", index=False)

    second = add_analysis(session, "safe", [(False, "scam")], text="Enter UPI PIN to get <AMOUNT>")
    later = tmp_path / "second.csv"
    assert export_feedback.main([*base, "--output", str(later)]) == 0
    ids = set(pd.read_csv(later, dtype=str)["id"])
    assert ids == {f"fb-{second}"}  # rejected rows are not offered again
    assert f"fb-{first}" not in ids


# --- Reviewed label files ----------------------------------------------------


def write_reviewed(path: Path, keep: list[str]) -> None:
    rows = [
        {
            "id": f"r-{i}",
            "text": f"Message number {i} about your parcel",
            "language": "en",
            "label": "legit",
            "source": "user_feedback",
            "scam_type": "",
            "keep": value,
        }
        for i, value in enumerate(keep)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_keep_column_drops_rejected_rows(tmp_path: Path) -> None:
    path = tmp_path / "reviewed.csv"
    write_reviewed(path, ["yes", "No", "", "n"])
    assert load_dataset(path)["id"].tolist() == ["r-0", "r-2"]


def test_keep_column_rejects_unknown_values(tmp_path: Path) -> None:
    path = tmp_path / "reviewed.csv"
    write_reviewed(path, ["yes", "maybe"])
    with pytest.raises(DatasetError, match="keep column"):
        load_dataset(path)


def test_reviewed_files_are_found_and_used_by_default(tmp_path: Path, monkeypatch) -> None:
    from ml import dataset

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    write_reviewed(labels_dir / "b.csv", ["yes", "no"])
    (labels_dir / "notes.md").write_text("not a label file")
    monkeypatch.setattr(dataset, "LABELS_DIR", labels_dir)

    assert dataset.reviewed_label_files() == [labels_dir / "b.csv"]
    assert dataset.reviewed_ids() == {"r-0", "r-1"}
    assert dataset.reviewed_label_files(tmp_path / "missing") == []

    args = prepare_data.parse_args([])
    assert args.input == [dataset.SEED_DATA_PATH, labels_dir / "b.csv"]


# --- Verdict evaluation ------------------------------------------------------


def test_binary_metrics() -> None:
    is_scam = np.array([True, True, False, False, False])
    predicted = np.array([True, False, True, False, False])
    metrics = binary_metrics(is_scam, predicted)
    assert metrics == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_alarm_rate": pytest.approx(0.3333, abs=1e-4),
    }


def test_binary_metrics_without_positive_predictions() -> None:
    metrics = binary_metrics(np.array([True, False]), np.array([False, False]))
    assert metrics["precision"] is None
    assert metrics["f1"] == 0.0


def test_score_verdicts_distribution_and_sweep() -> None:
    labels = ["scam", "scam", "legit", "spam_not_scam"]
    levels = ["likely_scam", "suspicious", "safe", "safe"]
    risks = [0.9, 0.5, 0.1, 0.42]
    metrics = score_verdicts(labels, levels, risks, VerdictConfig())

    assert metrics["distribution"]["scam"] == {"safe": 0, "suspicious": 1, "likely_scam": 1}
    assert metrics["likely_scam"]["recall"] == 0.5
    assert metrics["flagged"]["recall"] == 1.0
    sweep = {point["threshold"]: point for point in metrics["threshold_sweep"]}
    assert sweep[0.4]["false_alarm_rate"] == 0.5  # the spam message at 0.42
    assert sweep[0.45]["recall"] == 1.0 and sweep[0.45]["false_alarm_rate"] == 0.0
    # 0.45 and 0.50 tie on F1; the tie goes to the one closer to the current 0.65.
    assert metrics["best_threshold"] == 0.5
    assert type(metrics["best_threshold"]) is float


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    output = tmp_path / "processed"
    assert prepare_data.main(["--output-dir", str(output)]) == 0
    return output


def test_evaluate_verdict_end_to_end(processed: Path, tmp_path: Path, monkeypatch) -> None:
    from app.core.config import get_settings

    model = tmp_path / "baseline.joblib"
    assert (
        train_baseline.main(
            [
                "--train",
                str(processed / "train.csv"),
                "--output",
                str(model),
                "--results-dir",
                str(tmp_path / "train-results"),
                "--c-values",
                "1",
            ]
        )
        == 0
    )
    monkeypatch.setenv("BASELINE_MODEL_PATH", str(model))
    get_settings.cache_clear()
    try:
        results = tmp_path / "results"
        code = evaluate_verdict.main(
            [
                "--test",
                str(processed / "test.csv"),
                "--classifiers",
                "none",
                "baseline",
                "not-a-model",
                "--results-dir",
                str(results),
            ]
        )
    finally:
        get_settings.cache_clear()

    assert code == 0
    data = json.loads((results / "verdict_evaluation.json").read_text())
    assert [row["classifier"] for row in data["results"]] == ["none", "baseline"]
    assert data["network_checks"] is False
    report = (results / "verdict_evaluation.md").read_text()
    assert "| none |" in report and "| baseline |" in report
    assert "0.65 (current)" in report


def test_evaluate_verdict_reports_missing_test_file(tmp_path: Path, capsys) -> None:
    assert evaluate_verdict.main(["--test", str(tmp_path / "missing.csv")]) == 1
    assert "not found" in capsys.readouterr().err
