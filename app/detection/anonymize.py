"""Remove personal data from message text before it is stored or used for training.

Replaced with placeholders: phone numbers, emails, UPI IDs, card, Aadhaar and
PAN numbers, masked account numbers, dates, amounts, names after a salutation,
and any other number of four or more digits (OTPs, account and reference
numbers).

Kept as-is: link hosts (their paths and query strings are removed, since those
often carry personal tokens), toll-free numbers (organisations, not people, and
a useful signal of genuine messages), and short numbers like "24 hours".

Name detection is heuristic. It only catches names after a salutation such as
"Dear Rahul" or "my name is Priya".

The function is idempotent: anonymising already-anonymised text changes nothing.
"""

import re
from collections import Counter
from urllib.parse import urlsplit

PLACEHOLDERS = (
    "<URL_PATH>",
    "<EMAIL>",
    "<UPI_ID>",
    "<PAN>",
    "<CARD>",
    "<ACCOUNT>",
    "<AADHAAR>",
    "<PHONE>",
    "<DATE>",
    "<AMOUNT>",
    "<NAME>",
    "<NUM>",
)

_URL = re.compile(r"(?:\bhttps?://|\bwww\.)[^\s<>\"']+", re.IGNORECASE)
# 1800/1860 numbers, either separated ("1800 11 2211") or as a full 10-11 digit
# run, so amounts such as "Rs 180000" are not mistaken for them.
_TOLL_FREE = re.compile(r"(?<!\d)1[89]00(?:[\s-]\d{2,4}(?:[\s-]?\d{3,4})?|\d{6,7})(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_UPI_ID = re.compile(r"\b[\w.-]{2,}@[a-z][a-z0-9]+\b", re.IGNORECASE)
_PAN = re.compile(r"\b[a-z]{5}\d{4}[a-z]\b", re.IGNORECASE)
_CARD = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")
_MASKED_ACCOUNT = re.compile(
    r"(?<![\w*])(?:[x*]{4}[ -]?){2,3}\d{4}\b|(?<![\w*])[x*]{2,}\d{2,6}\b", re.IGNORECASE
)
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)")
_PHONE = re.compile(r"(?<![\d+])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_DATE = re.compile(r"\b\d{1,2}[-/.](?:\d{1,2}|[a-z]{3})[-/.]\d{2,4}\b", re.IGNORECASE)
_AMOUNT = re.compile(
    r"(?:₹|\brs\.?|\binr\b|\brupees?\b)\s?\d[\d,]*(?:\.\d+)?(?:/-)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:/-|rupees?\b|rs\b|रुपये|रुपए|रु\.?)",
    re.IGNORECASE,
)
_GENERIC_ADDRESSEES = {
    "customer",
    "customers",
    "user",
    "users",
    "consumer",
    "sir",
    "madam",
    "member",
    "valued",
    "cardholder",
    "friend",
    "friends",
    "applicant",
    "candidate",
    "team",
    "all",
    "there",
    "everyone",
    "patron",
    "subscriber",
    "beneficiary",
    "student",
    "ग्राहक",
    "उपभोक्ता",
    "सदस्य",
    "मित्र",
    "कार्डधारक",
    "ग्राहकों",
    "आवेदक",
}
_SALUTATION_NAME = re.compile(
    r"(?P<prefix>(?i:\b(?:dear|hi|hello|hey|mr\.?|mrs\.?|ms\.?|shri|smt\.?|my name is|i am))\s+)"
    r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    r"|(?P<hprefix>(?:प्रिय|श्री|श्रीमती)\s+)(?P<hname>[^\s,।!?]+)"
)
_LONG_NUMBER = re.compile(r"(?<![\d<])\d{4,}(?![\d>])")
_PROTECTED = re.compile(r"\x00(\d+)\x00")


def _anonymize_url(url: str) -> str:
    has_scheme = bool(re.match(r"https?://", url, re.IGNORECASE))
    parts = urlsplit(url if has_scheme else f"http://{url}")
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    base = f"{parts.scheme.lower()}://{host}" if has_scheme else host
    rest = url.split(parts.netloc, 1)[-1] if parts.netloc else ""
    if rest.strip("/"):
        return f"{base}/<URL_PATH>"
    return base + rest


def anonymize_with_counts(text: str) -> tuple[str, Counter[str]]:
    """Return anonymised text and how many items of each kind were replaced."""
    counts: Counter[str] = Counter()
    protected: list[str] = []

    def protect(value: str) -> str:
        protected.append(value)
        return f"\x00{len(protected) - 1}\x00"

    def protect_url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;:!?)")
        trailing = match.group(0)[len(url) :]
        cleaned = _anonymize_url(url)
        if cleaned.endswith("<URL_PATH>"):
            counts["<URL_PATH>"] += 1
        return protect(cleaned) + trailing

    text = _URL.sub(protect_url, text)
    text = _TOLL_FREE.sub(lambda m: protect(m.group(0)), text)

    def replace(pattern: re.Pattern[str], placeholder: str, source: str) -> str:
        def substitute(match: re.Match[str]) -> str:
            counts[placeholder] += 1
            return placeholder

        return pattern.sub(substitute, source)

    text = replace(_EMAIL, "<EMAIL>", text)
    text = replace(_UPI_ID, "<UPI_ID>", text)
    text = replace(_PAN, "<PAN>", text)
    text = replace(_CARD, "<CARD>", text)
    text = replace(_MASKED_ACCOUNT, "<ACCOUNT>", text)
    text = replace(_AADHAAR, "<AADHAAR>", text)
    text = replace(_PHONE, "<PHONE>", text)
    text = replace(_DATE, "<DATE>", text)
    text = replace(_AMOUNT, "<AMOUNT>", text)

    def replace_name(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or match.group("hprefix")
        name = match.group("name") or match.group("hname")
        if name.split()[0].lower() in _GENERIC_ADDRESSEES:
            return match.group(0)
        counts["<NAME>"] += 1
        return f"{prefix}<NAME>"

    text = _SALUTATION_NAME.sub(replace_name, text)
    text = replace(_LONG_NUMBER, "<NUM>", text)

    text = _PROTECTED.sub(lambda m: protected[int(m.group(1))], text)
    return text, counts


def anonymize(text: str) -> str:
    return anonymize_with_counts(text)[0]
