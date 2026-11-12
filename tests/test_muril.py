"""MuRIL training and inference, exercised with a tiny randomly initialised BERT.

The real google/muril-base-cased is about 950 MB, so the tests build a model
with the same architecture family locally. Skipped when torch is not installed.
"""

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.detection.classifier import ClassifierUnavailableError, load_classifier  # noqa: E402
from app.detection.muril_classifier import MurilClassifier, resolve_device  # noqa: E402
from app.main import create_app  # noqa: E402
from ml import evaluate, prepare_data, train_baseline, train_muril  # noqa: E402
from ml.dataset import LABELS, SEED_DATA_PATH  # noqa: E402


def build_tiny_bert(directory: Path) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    texts = pd.read_csv(SEED_DATA_PATH)["text"]
    words = sorted({w for t in texts for w in re.findall(r"\w+", t.lower())})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    (directory / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
        num_labels=len(LABELS),
    )
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("muril")
    processed = root / "processed"
    assert prepare_data.main(["--output-dir", str(processed)]) == 0
    return {"root": root, "processed": processed, "base": build_tiny_bert(root / "tiny-bert")}


@pytest.fixture(scope="module")
def trained(workspace: dict[str, Path]) -> Path:
    output = workspace["root"] / "muril"
    code = train_muril.main(
        [
            "--train",
            str(workspace["processed"] / "train.csv"),
            "--model-name",
            str(workspace["base"]),
            "--output-dir",
            str(output),
            "--results-dir",
            str(workspace["root"] / "results"),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--learning-rate",
            "1e-3",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    return output


# --- Training ----------------------------------------------------------------


def test_training_writes_a_loadable_checkpoint(trained: Path, workspace: dict[str, Path]) -> None:
    for name in ("config.json", "model.safetensors", "tokenizer.json", "scamshield_metadata.json"):
        assert (trained / name).exists(), name
    assert not list(trained.glob("*.bin")), "weights must be saved as safetensors, not pickle"

    metadata = json.loads((trained / "scamshield_metadata.json").read_text())
    assert metadata["labels"] == list(LABELS)
    assert metadata["device"] == "cpu"
    assert metadata["validation_rows"] > 0
    assert 1 <= metadata["best_epoch"] <= 2
    assert len(metadata["history"]) >= 1
    assert "val_macro_f1" in metadata["history"][0]
    assert (workspace["root"] / "results" / "muril_training.json").exists()

    config = json.loads((trained / "config.json").read_text())
    assert config["id2label"] == {str(i): label for i, label in enumerate(LABELS)}


def test_training_without_validation_and_fp16_on_cpu(
    workspace: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    output = workspace["root"] / "no-val"
    with caplog.at_level(logging.WARNING):
        code = train_muril.main(
            [
                "--train",
                str(workspace["processed"] / "train.csv"),
                "--model-name",
                str(workspace["base"]),
                "--output-dir",
                str(output),
                "--results-dir",
                str(workspace["root"] / "results-no-val"),
                "--epochs",
                "1",
                "--val-size",
                "0",
                "--fp16",
                "--freeze-layers",
                "1",
                "--device",
                "cpu",
            ]
        )
    assert code == 0
    assert "fp16" in caplog.text
    metadata = json.loads((output / "scamshield_metadata.json").read_text())
    assert metadata["best_val_macro_f1"] is None
    assert metadata["frozen_layers"] == 1
    assert metadata["fp16"] is False


def test_training_reports_missing_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_muril.main(["--train", str(tmp_path / "missing.csv"), "--device", "cpu"]) == 1
    assert "not found" in capsys.readouterr().err


def test_class_weights_favour_rare_labels() -> None:
    weights = dict(
        zip(
            LABELS,
            train_muril.class_weights(["scam"] * 6 + ["legit"] * 3 + ["spam_not_scam"]),
            strict=True,
        )
    )
    assert weights["spam_not_scam"] > weights["legit"] > weights["scam"]


def test_freeze_lower_layers(workspace: dict[str, Path]) -> None:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(workspace["base"])
    assert train_muril.freeze_lower_layers(model, 1) == 1
    assert not any(p.requires_grad for p in model.bert.embeddings.parameters())
    assert not any(p.requires_grad for p in model.bert.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in model.bert.encoder.layer[1].parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert train_muril.freeze_lower_layers(model, 99) == 2
    assert train_muril.freeze_lower_layers(model, 0) == 0


def test_resolve_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")


# --- Inference ---------------------------------------------------------------


def test_classifier_predicts_probabilities(trained: Path) -> None:
    classifier = MurilClassifier.load(trained)
    predictions = classifier.predict(["Share your OTP now", "आपका ओटीपी 123456 है", "Lunch at 1?"])
    assert len(predictions) == 3
    for prediction in predictions:
        assert set(prediction.probabilities) == set(LABELS)
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert prediction.model_name == "muril"
    assert classifier.predict([]) == []


def test_classifier_batches_consistently(trained: Path) -> None:
    classifier = MurilClassifier.load(trained)
    texts = [f"Message number {i} about KYC" for i in range(5)]
    classifier.batch_size = 2
    batched = classifier.predict(texts)
    single = [classifier.predict([t])[0] for t in texts]
    for a, b in zip(batched, single, strict=True):
        assert a.probabilities == pytest.approx(b.probabilities, abs=1e-5)


def test_personal_details_do_not_change_predictions(trained: Path) -> None:
    classifier = MurilClassifier.load(trained)
    a, b = classifier.predict(["Call 9876543210 to update KYC", "Call 9123456780 to update KYC"])
    assert a.probabilities == pytest.approx(b.probabilities, abs=1e-6)


def test_registry_loads_muril(trained: Path) -> None:
    classifier = load_classifier("muril", Settings(_env_file=None, muril_model_dir=trained))
    assert classifier.name == "muril"


def test_missing_or_broken_model_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ClassifierUnavailableError, match="train_muril"):
        MurilClassifier.load(tmp_path / "missing")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ClassifierUnavailableError, match="could not load"):
        MurilClassifier.load(broken)


def test_api_can_serve_muril(trained: Path, tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'api.db'}",
        database_auto_create=True,
        url_network_checks=False,
        classifier="muril",
        muril_model_dir=trained,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["model"] == "muril"
        body = client.post("/api/analyze", json={"text": "Please share the OTP"}).json()
        assert body["model"]["name"] == "muril"
        assert 0 <= body["model"]["scam_probability"] <= 1


# --- Comparison --------------------------------------------------------------


@pytest.fixture
def both_models(
    trained: Path, workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    baseline = workspace["root"] / "baseline.joblib"
    assert (
        train_baseline.main(
            [
                "--train",
                str(workspace["processed"] / "train.csv"),
                "--output",
                str(baseline),
                "--results-dir",
                str(workspace["root"] / "results"),
                "--c-values",
                "1",
            ]
        )
        == 0
    )
    monkeypatch.setenv("BASELINE_MODEL_PATH", str(baseline))
    monkeypatch.setenv("MURIL_MODEL_DIR", str(trained))
    get_settings.cache_clear()
    yield workspace["root"] / "comparison"
    get_settings.cache_clear()


def test_evaluate_compares_both_models(both_models: Path, workspace: dict[str, Path]) -> None:
    code = evaluate.main(
        [
            "--test",
            str(workspace["processed"] / "test.csv"),
            "--results-dir",
            str(both_models),
            "--bootstrap",
            "200",
        ]
    )
    assert code == 0
    report = (both_models / "comparison.md").read_text()
    assert "| baseline |" in report and "| muril |" in report
    assert "## By language" in report and "hinglish macro F1" in report

    data = json.loads((both_models / "comparison.json").read_text())
    assert [m["model"] for m in data["models"]] == ["baseline", "muril"]
    for model in data["models"]:
        low, high = model["metrics"]["macro_f1_ci95"]
        assert 0 <= low <= high <= 1
        assert set(model["metrics"]["by_language"]) == {"en", "hi", "hinglish"}
    assert (both_models / "muril_confusion_matrix.csv").exists()
