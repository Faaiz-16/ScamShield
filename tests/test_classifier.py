from pathlib import Path

import joblib
import pytest

from app.detection.classifier import (
    BaselineClassifier,
    ClassifierUnavailableError,
    load_classifier,
)
from ml.baseline import build_pipeline
from ml.dataset import LABELS, SEED_DATA_PATH, anonymize_dataset, load_dataset
from ml.metrics import compute_metrics


@pytest.fixture(scope="module")
def trained() -> BaselineClassifier:
    df = anonymize_dataset(load_dataset(SEED_DATA_PATH))
    pipeline = build_pipeline(c=10.0).fit(df["text"].tolist(), df["label"].tolist())
    return BaselineClassifier(pipeline=pipeline, metadata={"note": "test"})


def test_predictions_have_valid_probabilities(trained: BaselineClassifier) -> None:
    predictions = trained.predict(["Share your OTP now", "Lunch at 1?"])
    assert len(predictions) == 2
    for prediction in predictions:
        assert set(prediction.probabilities) == set(LABELS)
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
        assert prediction.label == max(prediction.probabilities, key=prediction.probabilities.get)
        assert prediction.model_name == "baseline"


def test_model_separates_obvious_examples(trained: BaselineClassifier) -> None:
    scam = trained.predict_one("Aapka account band ho jayega, turant KYC update karo")
    legit = trained.predict_one("482913 is your OTP. Do not share it with anyone.")
    assert scam.scam_probability > legit.scam_probability


def test_raw_text_is_anonymised_inside_the_pipeline(trained: BaselineClassifier) -> None:
    a = trained.predict_one("Call 9876543210 to update KYC")
    b = trained.predict_one("Call 9123456780 to update KYC")
    assert a.probabilities == pytest.approx(b.probabilities)


def test_empty_batch_returns_empty_list(trained: BaselineClassifier) -> None:
    assert trained.predict([]) == []


def test_save_and_load_round_trip(trained: BaselineClassifier, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "model.joblib"
    trained.save(path)
    loaded = BaselineClassifier.load(path)
    assert loaded.metadata == {"note": "test"}
    text = "Enter UPI PIN to receive cashback"
    assert loaded.predict_one(text).probabilities == pytest.approx(
        trained.predict_one(text).probabilities
    )


def test_missing_model_raises_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ClassifierUnavailableError, match="train_baseline"):
        BaselineClassifier.load(tmp_path / "missing.joblib")


def test_foreign_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.joblib"
    joblib.dump({"something": "else"}, path)
    with pytest.raises(ClassifierUnavailableError, match="not a ScamShield"):
        BaselineClassifier.load(path)


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ClassifierUnavailableError, match="unknown model"):
        load_classifier("does-not-exist")


def test_metrics_include_false_positive_rate_on_legit() -> None:
    y_true = ["scam", "scam", "legit", "legit", "legit", "legit", "spam_not_scam"]
    y_pred = ["scam", "legit", "legit", "scam", "legit", "legit", "spam_not_scam"]
    metrics = compute_metrics(y_true, y_pred, [0.9, 0.4, 0.1, 0.6, 0.2, 0.1, 0.3])
    assert metrics["legit_flagged_as_scam_rate"] == pytest.approx(0.25)
    assert metrics["per_class"]["scam"]["recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"]["matrix"][0] == [1, 1, 0]
    assert 0 <= metrics["scam_roc_auc"] <= 1
