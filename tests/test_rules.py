from pathlib import Path
from textwrap import dedent

import pytest

from app.detection.normalize import normalize_text
from app.detection.rules import (
    DEFAULT_RULES_PATH,
    RuleConfigError,
    RuleEngine,
    get_rule_engine,
    load_rule_set,
)


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return get_rule_engine()


def hit_ids(engine: RuleEngine, text: str) -> set[str]:
    return {hit.rule_id for hit in engine.evaluate(text)}


# --- Rule file integrity -----------------------------------------------------


def test_default_rules_file_is_valid_and_has_enough_rules() -> None:
    rule_set = load_rule_set(DEFAULT_RULES_PATH)
    assert len(rule_set.rules) >= 20


def test_every_rule_has_bilingual_reason(engine: RuleEngine) -> None:
    rule_set = load_rule_set(DEFAULT_RULES_PATH)
    for rule in rule_set.rules:
        assert rule.reason.en.strip(), rule.id
        assert rule.reason.hi.strip(), rule.id
        assert any("ऀ" <= ch <= "ॿ" for ch in rule.reason.hi), rule.id


def test_every_category_has_bilingual_label(engine: RuleEngine) -> None:
    for key in engine.categories:
        assert engine.category_label(key, "en") != key
        assert engine.category_label(key, "hi") != engine.category_label(key, "en")


# --- Scam messages must be caught --------------------------------------------

SCAM_CASES = [
    # OTP and credentials
    ("en", "Please share the OTP you just received to complete verification.", "OTP_SHARE_REQUEST"),
    ("hinglish", "Sir aapke number pe OTP aaya hoga, wo OTP bata do jaldi", "OTP_SHARE_REQUEST"),
    ("hi", "आपके मोबाइल पर आया ओटीपी तुरंत बताएं", "OTP_SHARE_REQUEST"),
    ("en", "To unblock your card, kindly tell us your CVV and ATM PIN", "PIN_CVV_PASSWORD_REQUEST"),
    ("hinglish", "apna atm pin bata do sir, card activate ho jayega", "PIN_CVV_PASSWORD_REQUEST"),
    ("hi", "कार्ड चालू करने के लिए अपना पिन बताइए", "PIN_CVV_PASSWORD_REQUEST"),
    (
        "en",
        "Reply with your account number and date of birth to verify",
        "PERSONAL_DETAILS_REQUEST",
    ),
    # Threats
    ("en", "Dear customer, your SBI account will be blocked today.", "ACCOUNT_BLOCK_THREAT"),
    ("hinglish", "Aapka account band ho jayega agar aaj update nahi kiya", "ACCOUNT_BLOCK_THREAT"),
    ("hi", "आपका खाता 24 घंटे में बंद हो जाएगा", "ACCOUNT_BLOCK_THREAT"),
    ("en", "Final warning: respond within 24 hours", "URGENCY_PRESSURE"),
    (
        "en",
        "This is CBI. A parcel with drugs was booked on your Aadhaar. You are under digital arrest.",
        "DIGITAL_ARREST_THREAT",
    ),
    ("hi", "आपके नाम पर गिरफ्तारी वारंट जारी हुआ है", "DIGITAL_ARREST_THREAT"),
    # KYC
    ("en", "Your KYC has expired. Update KYC now at http://bit.ly/x", "KYC_UPDATE"),
    ("hinglish", "Apna KYC update karo warna paytm wallet band", "KYC_UPDATE"),
    ("hi", "आपका केवाईसी अपडेट लंबित है", "KYC_UPDATE"),
    ("en", "Your PAN card will be blocked. Update PAN immediately.", "PAN_AADHAAR_UPDATE"),
    # UPI
    (
        "en",
        "You have received Rs 5000 cashback. Enter UPI PIN to receive the amount.",
        "UPI_PIN_TO_RECEIVE",
    ),
    ("hinglish", "Paise receive karne ke liye UPI PIN daalein", "UPI_PIN_TO_RECEIVE"),
    ("hi", "रिफंड प्राप्त करने के लिए अपना यूपीआई पिन डालें", "UPI_PIN_TO_RECEIVE"),
    ("en", "Scan this QR code to receive your refund of Rs 2000", "QR_SCAN_TO_RECEIVE"),
    (
        "hinglish",
        "Bhai galti se aapko 5000 transfer ho gaye, please wapas bhej do",
        "WRONG_TRANSFER_REFUND",
    ),
    # Utility and telecom
    (
        "en",
        "Dear consumer your electricity power will be disconnected tonight at 9.30 pm",
        "ELECTRICITY_DISCONNECTION",
    ),
    ("hinglish", "Aapka bijli connection aaj raat kaat diya jayega", "ELECTRICITY_DISCONNECTION"),
    ("hi", "आपकी बिजली आज रात 9:30 बजे काट दी जाएगी", "ELECTRICITY_DISCONNECTION"),
    ("en", "because your previous month bill was not updated", "ELECTRICITY_BILL_NOT_UPDATED"),
    ("en", "Your SIM card will be blocked in 2 hours as per TRAI. Press 9.", "SIM_BLOCK_THREAT"),
    # Jobs
    (
        "en",
        "Congratulations! You are selected for a work from home job. Pay registration fee of Rs 1500.",
        "JOB_REGISTRATION_FEE",
    ),
    ("hi", "नौकरी के लिए 2000 रुपये का पंजीकरण शुल्क जमा करें", "JOB_REGISTRATION_FEE"),
    ("en", "Like YouTube videos and earn Rs 3000 daily from home", "EASY_MONEY_TASK"),
    ("hinglish", "Ghar baithe kamaye 5000 roz, abhi join karo", "EASY_MONEY_TASK"),
    # Lottery
    ("en", "Congratulations! You have won Rs 25 lakh in the lucky draw", "LOTTERY_PRIZE_CLAIM"),
    ("hinglish", "Badhai ho aapne 25 lakh jeete hai", "LOTTERY_PRIZE_CLAIM"),
    ("hi", "बधाई हो! आपने 25 लाख की लॉटरी जीती है", "LOTTERY_PRIZE_CLAIM"),
    ("en", "KBC Kaun Banega Crorepati lottery winner 2026", "KBC_LOTTERY"),
    ("hi", "कौन बनेगा करोड़पति की ओर से आपको इनाम मिला है", "KBC_LOTTERY"),
    (
        "en",
        "You won a prize of 10 lakh. Pay processing fee of Rs 9500 to claim.",
        "PRIZE_PROCESSING_FEE",
    ),
    ("en", "Your reward points worth Rs 7,250 expire today. Redeem now.", "REWARD_POINTS_EXPIRY"),
    ("en", "Get free recharge of Rs 239 for 3 months under the new scheme", "FREE_GIFT_OR_SCHEME"),
    # Impersonation
    ("en", "For refund contact our customer care 98765 43210", "FAKE_CUSTOMER_CARE"),
    ("en", "Call helpline +91-9876543210 for help", "FAKE_CUSTOMER_CARE"),
    ("en", "Please download AnyDesk so our executive can help you", "REMOTE_ACCESS_APP"),
    ("en", "Install the SBI Rewards.apk from the link to claim", "APK_DOWNLOAD"),
    ("hinglish", "Yeh app download karein aur KYC complete karein", "APK_DOWNLOAD"),
    (
        "en",
        "India Post: your parcel is on hold due to incomplete address. Update here: http://indiapost-in.top",
        "PARCEL_DELIVERY_ISSUE",
    ),
    (
        "en",
        "Your income tax refund of Rs 15,490 has been approved. Verify your account here.",
        "INCOME_TAX_REFUND",
    ),
    (
        "en",
        "Your e-challan is pending. Pay now to avoid court: http://parivahan-pay.xyz",
        "TRAFFIC_CHALLAN",
    ),
    # Money offers
    (
        "en",
        "Join our VIP group for stock market tips with guaranteed returns",
        "INVESTMENT_GUARANTEED_RETURNS",
    ),
    ("hi", "सिर्फ 30 दिन में अपना पैसा दोगुना करें", "INVESTMENT_GUARANTEED_RETURNS"),
    ("en", "Instant personal loan without documents or CIBIL check", "INSTANT_LOAN_OFFER"),
    ("en", "Click the link below to continue", "LINK_CLICK_PROMPT"),
    ("hinglish", "is link par click karke apna reward le", "LINK_CLICK_PROMPT"),
]


@pytest.mark.parametrize(("language", "text", "expected_rule"), SCAM_CASES)
def test_scam_message_triggers_expected_rule(
    engine: RuleEngine, language: str, text: str, expected_rule: str
) -> None:
    assert expected_rule in hit_ids(engine, text), f"[{language}] {text}"


def test_every_rule_is_covered_by_a_scam_case(engine: RuleEngine) -> None:
    covered = {rule_id for _, _, rule_id in SCAM_CASES}
    assert set(engine.rule_ids) <= covered


# --- Genuine messages must not be flagged ------------------------------------

LEGIT_MESSAGES = [
    "482913 is your OTP for login to SBI YONO. Do not share it with anyone. SBI never asks for OTP.",
    "Your OTP for transaction of INR 1,250.00 at AMAZON is 739201. Valid for 5 mins. Never share your OTP with anyone. -HDFC Bank",
    "प्रिय ग्राहक, आपका ओटीपी 123456 है। इसे किसी के साथ साझा न करें। -एसबीआई",
    "Aapka OTP 552310 hai. Ise kisi ke saath share na karein.",
    "Rs.500.00 debited from A/c XX1234 on 12-11-26 to VPA abc@okicici. Not you? Call 18001234 to block. -SBI",
    "INR 25,000.00 credited to your A/c XX5678 on 11-Nov-26 by NEFT. Avl Bal INR 41,203.50 -ICICI Bank",
    "Your Amazon order #402-1234567 has been shipped and will be delivered by Thursday.",
    "Your Swiggy order is out for delivery. Delivery partner Ravi will reach in 10 minutes.",
    "Your debit card ending 4321 has been blocked as per your request. -Axis Bank",
    "Dear customer, scheduled maintenance: power supply will be cut from 10 AM to 2 PM tomorrow in your area.",
    "Your electricity bill of Rs 1,240 for October is generated. Due date 25-11-2026. Pay via the official app.",
    "Hi, are we still meeting for lunch tomorrow at 1?",
    "कल सुबह 10 बजे ऑफिस में मीटिंग है, समय पर आ जाना।",
    "Bhai kal cricket khelne chalega? 6 baje ground pe milte hai",
    "Re-KYC is due for your account. Please visit your nearest branch with valid documents. -Bank of Baroda",
    "Download the official IRCTC app from Play Store for faster booking.",
    "Your Jio recharge of Rs 299 is successful. Validity 28 days.",
]


@pytest.mark.parametrize("text", LEGIT_MESSAGES)
def test_legit_message_has_no_strong_hits(engine: RuleEngine, text: str) -> None:
    strong = [hit for hit in engine.evaluate(text) if hit.severity >= 0.5]
    assert strong == [], f"{text!r} -> {[h.rule_id for h in strong]}"


# --- Engine behaviour --------------------------------------------------------


def test_hits_are_sorted_by_severity(engine: RuleEngine) -> None:
    hits = engine.evaluate(
        "Your account will be blocked. Click the link and share OTP immediately."
    )
    severities = [hit.severity for hit in hits]
    assert len(hits) >= 3
    assert severities == sorted(severities, reverse=True)


def test_hit_exposes_reason_in_both_languages(engine: RuleEngine) -> None:
    hit = engine.evaluate("Enter UPI PIN to receive Rs 2000")[0]
    assert hit.rule_id == "UPI_PIN_TO_RECEIVE"
    assert hit.reason("en") == hit.reason_en
    assert hit.reason("hi") == hit.reason_hi
    assert hit.matched_text


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_input_returns_no_hits(engine: RuleEngine, text: str) -> None:
    assert engine.evaluate(text) == []


def test_obfuscated_text_is_still_detected(engine: RuleEngine) -> None:
    zero_width = "Please share the O​T​P received on your phone"
    fullwidth = "Please share the ＯＴＰ received on your phone"
    assert "OTP_SHARE_REQUEST" in hit_ids(engine, zero_width)
    assert "OTP_SHARE_REQUEST" in hit_ids(engine, fullwidth)


def test_nukta_spelling_variants_match(engine: RuleEngine) -> None:
    with_nukta = "कौन बनेगा करोड़पति"
    precomposed = "कौन बनेगा करोड़पति"
    assert "KBC_LOTTERY" in hit_ids(engine, with_nukta)
    assert "KBC_LOTTERY" in hit_ids(engine, precomposed)


def test_normalize_text_collapses_whitespace_and_case() -> None:
    assert normalize_text("  Your   KYC\n\nHAS expired ") == "your kyc has expired"


# --- Loading custom rule files -----------------------------------------------


def write_rules(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(dedent(body), encoding="utf-8")
    return path


CATEGORIES = """
categories:
  test: {en: "Test", hi: "परीक्षण"}
"""


def test_custom_rule_file_any_all_none_semantics(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        CATEGORIES
        + """
rules:
  - id: COMBO
    category: test
    severity: 0.5
    reason: {en: "combo", hi: "कॉम्बो"}
    match:
      any: ['alpha']
      all: [['beta', 'gamma'], ['delta']]
      none: ['omega']
""",
    )
    engine = RuleEngine.from_file(path)
    assert hit_ids(engine, "alpha gamma delta") == {"COMBO"}
    assert hit_ids(engine, "alpha beta") == set()  # missing the 'delta' group
    assert hit_ids(engine, "beta delta") == set()  # missing 'any'
    assert hit_ids(engine, "alpha beta delta omega") == set()  # suppressed


def test_disabled_rules_are_skipped(tmp_path: Path) -> None:
    path = write_rules(
        tmp_path,
        CATEGORIES
        + """
rules:
  - id: DISABLED_RULE
    category: test
    severity: 0.5
    enabled: false
    reason: {en: "off", hi: "बंद"}
    match: {any: ['alpha']}
""",
    )
    assert RuleEngine.from_file(path).evaluate("alpha") == []


@pytest.mark.parametrize(
    ("rules_body", "message"),
    [
        (
            """
  - id: BAD_REGEX
    category: test
    severity: 0.5
    reason: {en: "x", hi: "x"}
    match: {any: ['(unclosed']}
""",
            "invalid pattern",
        ),
        (
            """
  - id: DUP
    category: test
    severity: 0.5
    reason: {en: "x", hi: "x"}
    match: {any: ['a']}
  - id: DUP
    category: test
    severity: 0.5
    reason: {en: "x", hi: "x"}
    match: {any: ['b']}
""",
            "duplicate rule id",
        ),
        (
            """
  - id: UNKNOWN_CAT
    category: nope
    severity: 0.5
    reason: {en: "x", hi: "x"}
    match: {any: ['a']}
""",
            "unknown category",
        ),
        (
            """
  - id: NO_MATCH
    category: test
    severity: 0.5
    reason: {en: "x", hi: "x"}
    match: {none: ['a']}
""",
            "at least one of",
        ),
        (
            """
  - id: TOO_SEVERE
    category: test
    severity: 1.5
    reason: {en: "x", hi: "x"}
    match: {any: ['a']}
""",
            "severity",
        ),
    ],
)
def test_invalid_rule_files_are_rejected(tmp_path: Path, rules_body: str, message: str) -> None:
    path = write_rules(tmp_path, CATEGORIES + "\nrules:" + rules_body)
    with pytest.raises(RuleConfigError, match=message):
        RuleEngine.from_file(path)


def test_missing_rules_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuleConfigError, match="cannot read"):
        load_rule_set(tmp_path / "missing.yaml")
