import pytest

from ml.metrics import (
    bootstrap_macro_f1,
    comparison_table,
    compute_metrics,
    language_table,
    per_language_metrics,
)

Y_TRUE = ["scam", "scam", "legit", "legit", "spam_not_scam", "scam"]
Y_PRED = ["scam", "legit", "legit", "scam", "spam_not_scam", "scam"]
LANGS = ["en", "en", "hi", "hi", "hinglish", "hinglish"]


def test_per_language_metrics() -> None:
    result = per_language_metrics(Y_TRUE, Y_PRED, LANGS)
    # Only scam and legit occur in the English slice, so spam does not count as an F1 of 0.
    assert result["en"] == {
        "samples": 2,
        "accuracy": 0.5,
        "macro_f1": pytest.approx(0.3333, abs=1e-4),
        "scam_recall": 0.5,
    }
    assert result["hi"]["scam_recall"] is None  # no scams in this language
    assert result["hinglish"]["accuracy"] == 1.0


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible() -> None:
    y_true = Y_TRUE * 5
    y_pred = Y_PRED * 5
    first = bootstrap_macro_f1(y_true, y_pred, ["scam", "legit", "spam_not_scam"], 300, seed=1)
    second = bootstrap_macro_f1(y_true, y_pred, ["scam", "legit", "spam_not_scam"], 300, seed=1)
    point = compute_metrics(y_true, y_pred)["macro_avg"]["f1-score"]
    assert first == second
    assert first[0] <= point <= first[1]


def test_perfect_predictions_have_a_tight_interval() -> None:
    assert bootstrap_macro_f1(Y_TRUE, Y_TRUE, ["scam", "legit", "spam_not_scam"], 100) == [1.0, 1.0]


def test_optional_sections_are_off_by_default() -> None:
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert metrics["macro_f1_ci95"] is None
    assert metrics["by_language"] == {}


def test_tables_render_all_models() -> None:
    rows = [
        {
            "model": name,
            "ms_per_message": 1.0,
            "metrics": compute_metrics(Y_TRUE, Y_PRED, languages=LANGS, bootstrap_samples=50),
        }
        for name in ("baseline", "muril")
    ]
    overall = comparison_table(rows)
    assert "|---|" in overall
    assert overall.count("\n| ") == 2  # one row per model
    assert " to " in overall
    by_language = language_table(rows)
    assert "| muril |" in by_language
    assert "Test messages per language: en: 2, hi: 2, hinglish: 2." in by_language


def test_language_table_is_empty_without_language_data() -> None:
    rows = [{"model": "m", "ms_per_message": 1.0, "metrics": compute_metrics(Y_TRUE, Y_PRED)}]
    assert language_table(rows) == ""
