"""End-to-end run of prepare -> train -> evaluate on the seed sample."""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import get_settings
from ml import evaluate, prepare_data, train_baseline


@pytest.fixture
def model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "models" / "baseline.joblib"
    monkeypatch.setenv("BASELINE_MODEL_PATH", str(path))
    # Keep any locally trained MuRIL model out of these tests.
    monkeypatch.setenv("MURIL_MODEL_DIR", str(tmp_path / "models" / "muril"))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


def test_full_pipeline(tmp_path: Path, model_path: Path) -> None:
    processed = tmp_path / "processed"
    results = tmp_path / "results"

    assert prepare_data.main(["--output-dir", str(processed)]) == 0
    train = pd.read_csv(processed / "train.csv", dtype=str, keep_default_na=False)
    assert not train["text"].str.contains(r"\b[6-9]\d{9}\b", regex=True).any()

    assert (
        train_baseline.main(
            [
                "--train",
                str(processed / "train.csv"),
                "--output",
                str(model_path),
                "--results-dir",
                str(results),
                "--c-values",
                "1,10",
            ]
        )
        == 0
    )
    assert model_path.exists()

    assert (
        evaluate.main(["--test", str(processed / "test.csv"), "--results-dir", str(results)]) == 0
    )
    for name in (
        "baseline_metrics.json",
        "baseline_confusion_matrix.csv",
        "baseline_errors.csv",
        "comparison.md",
        "comparison.json",
    ):
        assert (results / name).exists(), name

    metrics = json.loads((results / "baseline_metrics.json").read_text())["metrics"]
    assert set(metrics["per_class"]) == {"scam", "legit", "spam_not_scam"}
    assert "| baseline |" in (results / "comparison.md").read_text()


def test_prepare_reports_invalid_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\n1,hello\n", encoding="utf-8")
    assert prepare_data.main(["--input", str(bad), "--output-dir", str(tmp_path / "out")]) == 1
    assert "missing columns" in capsys.readouterr().err


def test_evaluate_fails_cleanly_without_a_model(
    tmp_path: Path, model_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processed = tmp_path / "processed"
    prepare_data.main(["--output-dir", str(processed)])
    code = evaluate.main(["--test", str(processed / "test.csv"), "--results-dir", str(tmp_path)])
    assert code == 1
    assert "no trained model" in capsys.readouterr().err
