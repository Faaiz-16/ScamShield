import pytest

from app.detection.language import detect_language
from ml.dataset import SEED_DATA_PATH, load_dataset


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Your account will be blocked today", "en"),
        ("आपका खाता बंद हो जाएगा", "hi"),
        ("प्रिय ग्राहक, आपका OTP 123456 है", "hi"),
        ("Aapka account band ho jayega", "hinglish"),
        ("Bhai kal cricket khelne chalega?", "hinglish"),
        ("Aapka OTP 552310 hai", "hinglish"),
        ("", "en"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_detector_mostly_agrees_with_seed_labels() -> None:
    df = load_dataset(SEED_DATA_PATH)
    agreement = (df["text"].map(detect_language) == df["language"]).mean()
    assert agreement >= 0.9
