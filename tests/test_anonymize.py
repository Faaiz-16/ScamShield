import pytest

from app.detection.anonymize import anonymize, anonymize_with_counts
from app.detection.preprocessing import prepare_for_model


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call 9876543210 now", "Call <PHONE> now"),
        ("Call +91 98765 43210 now", "Call <PHONE> now"),
        ("Call +91-9876543210 now", "Call <PHONE> now"),
        ("Mail help@sbi-care.com today", "Mail <EMAIL> today"),
        ("Pay to rahul.k@okicici", "Pay to <UPI_ID>"),
        ("PAN ABCDE1234F is linked", "PAN <PAN> is linked"),
        ("Card 4111 1111 1111 1111 used", "Card <CARD> used"),
        ("Card 4111111111111111 used", "Card <CARD> used"),
        ("Aadhaar 1234 5678 9012 verified", "Aadhaar <AADHAAR> verified"),
        ("debited from A/c XX1234 today", "debited from A/c <ACCOUNT> today"),
        ("card xxxx xxxx xxxx 4321 blocked", "card <ACCOUNT> blocked"),
        ("on 12-11-26 at", "on <DATE> at"),
        ("on 11-Nov-2026 at", "on <DATE> at"),
        ("Rs.1,250.00 debited", "<AMOUNT> debited"),
        ("Pay ₹1,499/- now", "Pay <AMOUNT> now"),
        ("INR 25,000.00 credited", "<AMOUNT> credited"),
        ("Earn 5000 rupees daily", "Earn <AMOUNT> daily"),
        ("आपके खाते से 1,200 रुपये कटे", "आपके खाते से <AMOUNT> कटे"),
        ("Your OTP is 482913", "Your OTP is <NUM>"),
        ("Account 50100234567890 credited", "Account <NUM> credited"),
        ("Dear Rahul Sharma, your bill", "Dear <NAME>, your bill"),
        ("Hi, I am Priya from HR", "Hi, I am <NAME> from HR"),
        ("प्रिय राहुल, आपका बिल", "प्रिय <NAME>, आपका बिल"),
    ],
)
def test_personal_data_is_replaced(text: str, expected: str) -> None:
    assert anonymize(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Dear Customer, your bill is ready",
        "Hello Sir, thank you",
        "प्रिय ग्राहक, धन्यवाद",
        "Valid for 24 hours, 50% off, 9:30 PM",
        "Call 1800-11-2211 for help",
        "Call 1800 425 3800 for help",
        "Call 18004253800 for help",
    ],
)
def test_non_personal_text_is_kept(text: str) -> None:
    assert anonymize(text) == text


def test_url_path_and_query_are_removed_but_host_kept() -> None:
    text = "Update at https://sbi-kyc.xyz/login?mobile=9876543210 now."
    assert anonymize(text) == "Update at https://sbi-kyc.xyz/<URL_PATH> now."


def test_bare_url_host_is_kept() -> None:
    assert anonymize("Visit www.Example.in/ today") == "Visit www.example.in/ today"


def test_amount_written_without_commas_is_not_mistaken_for_toll_free() -> None:
    assert anonymize("Rs 180000 credited") == "<AMOUNT> credited"


def test_counts_report_each_replacement() -> None:
    _, counts = anonymize_with_counts("Call 9876543210 or 9123456780, pay Rs 500")
    assert counts == {"<PHONE>": 2, "<AMOUNT>": 1}


@pytest.mark.parametrize(
    "text",
    [
        "Dear Rahul, Rs.500 debited from A/c XX1234 on 12-11-26. Call 9876543210.",
        "Visit https://bit.ly/abc?u=9876543210 and pay ₹1,499/-",
        "OTP 482913 for card 4111 1111 1111 1111, ref 431298765432",
    ],
)
def test_anonymize_is_idempotent(text: str) -> None:
    once = anonymize(text)
    assert anonymize(once) == once


def test_model_preprocessing_produces_lowercase_placeholders() -> None:
    assert prepare_for_model("Call  9876543210\nNOW") == "call <phone> now"
