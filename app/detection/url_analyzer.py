"""Find and assess links inside a message.

Offline checks (lookalike domains, suspicious endings, IP links, ...) always
run. Network checks run only when enabled in settings:

* Shortened links are expanded by requesting *only* hosts on the shortener
  allow-list, one hop at a time. The destination site itself is never
  contacted, so a scammer's server never sees our traffic and no page is
  ever rendered. Response bodies from these hosts are never read.
* Every host we contact must resolve exclusively to public IP addresses and
  use port 80 or 443 (SSRF protection).
* WHOIS and reputation results are cached to avoid repeated lookups.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

import httpx
import tldextract
import yaml
from rapidfuzz.distance import Levenshtein

from app.core.cache import TTLCache
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

Language = Literal["en", "hi"]
Resolver = Callable[[str, int], Awaitable[list[str]]]
WhoisLookup = Callable[[str], datetime | None]

URL_DATA_PATH = Path(__file__).with_name("url_data.yaml")
USER_AGENT = "ScamShield-LinkChecker/0.1"
SAFE_BROWSING_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
VIRUSTOTAL_ENDPOINT = "https://www.virustotal.com/api/v3/urls/"

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Brand:
    name: str
    domains: frozenset[str]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class OfficialLabel:
    brand: Brand
    domain: str
    label: str
    skeleton: str


@dataclass(frozen=True)
class UrlReferenceData:
    trusted_suffixes: tuple[str, ...]
    shorteners: frozenset[str]
    suspicious_tlds: frozenset[str]
    brand_check_exempt: frozenset[str]
    brands: tuple[Brand, ...]
    official_domains: dict[str, Brand]
    official_labels: tuple[OfficialLabel, ...]
    extractor: tldextract.TLDExtract


@lru_cache(maxsize=2)
def load_reference_data(path: Path = URL_DATA_PATH) -> UrlReferenceData:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    # suffix_list_urls=() forces the bundled public suffix snapshot, so domain
    # parsing never makes a network request.
    extractor = tldextract.TLDExtract(
        suffix_list_urls=(),
        cache_dir=None,
        extra_suffixes=tuple(raw.get("extra_public_suffixes", [])),
    )

    brands = tuple(
        Brand(
            name=entry["name"],
            domains=frozenset(d.lower() for d in entry["domains"]),
            keywords=tuple(k.lower() for k in entry.get("keywords", [])),
        )
        for entry in raw["brands"]
    )
    official_domains = {domain: brand for brand in brands for domain in brand.domains}
    official_labels = tuple(
        OfficialLabel(brand, domain, label, _skeleton(label))
        for domain, brand in official_domains.items()
        if (label := extractor(domain).domain)
    )

    return UrlReferenceData(
        trusted_suffixes=tuple(s.lower() for s in raw["trusted_suffixes"]),
        shorteners=frozenset(s.lower() for s in raw["url_shorteners"]),
        suspicious_tlds=frozenset(t.lower().lstrip(".") for t in raw["suspicious_tlds"]),
        brand_check_exempt=frozenset(d.lower() for d in raw.get("brand_check_exempt", [])),
        brands=brands,
        official_domains=official_domains,
        official_labels=official_labels,
        extractor=extractor,
    )


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UrlFinding:
    code: str
    severity: float
    reason_en: str
    reason_hi: str

    def reason(self, language: Language = "en") -> str:
        return self.reason_hi if language == "hi" else self.reason_en


_FINDINGS: dict[str, tuple[float, str, str]] = {
    "SHORTENED_URL": (
        0.3,
        "The link is shortened ({domain}), which hides the real website it leads to.",
        "यह लिंक छोटा किया गया है ({domain}), जिससे असली वेबसाइट छिप जाती है।",
    ),
    "LOOKALIKE_DOMAIN": (
        0.9,
        "The website {domain} imitates the official {brand} website ({official}).",
        "वेबसाइट {domain} आधिकारिक {brand} वेबसाइट ({official}) की नकल है।",
    ),
    "BRAND_IMPERSONATION": (
        0.8,
        "The link uses the name {brand} but is not an official {brand} website.",
        "लिंक में {brand} का नाम है, लेकिन यह {brand} की आधिकारिक वेबसाइट नहीं है।",
    ),
    "IDN_HOMOGRAPH": (
        0.8,
        "The website name uses look-alike letters from other alphabets to disguise itself.",
        "वेबसाइट के नाम में दूसरी लिपियों के मिलते-जुलते अक्षर हैं ताकि असली नाम छिप सके।",
    ),
    "YOUNG_DOMAIN": (
        0.75,
        "The website {domain} was registered only {days} days ago. Scam websites are usually brand new.",
        "वेबसाइट {domain} सिर्फ {days} दिन पहले बनाई गई है। धोखाधड़ी वाली वेबसाइटें अक्सर नई होती हैं।",
    ),
    "SUSPICIOUS_TLD": (
        0.5,
        "The link ends in .{tld}, a domain ending often used by scam websites.",
        "लिंक .{tld} पर खत्म होता है, जो अक्सर धोखाधड़ी वाली वेबसाइटों में इस्तेमाल होता है।",
    ),
    "IP_ADDRESS_URL": (
        0.7,
        "The link points to a bare IP address instead of a website name, which genuine services do not do.",
        "लिंक किसी वेबसाइट नाम की बजाय सीधे आईपी एड्रेस पर जाता है, जो असली सेवाएं नहीं करतीं।",
    ),
    "EXCESSIVE_SUBDOMAINS": (
        0.45,
        "The link has an unusually long chain of sub-domains, a trick to make a fake site look official.",
        "लिंक में असामान्य रूप से कई सब-डोमेन हैं, जो नकली साइट को असली दिखाने की चाल है।",
    ),
    "USERINFO_TRICK": (
        0.7,
        "The link hides its real destination after an '@' sign.",
        "लिंक '@' चिह्न के बाद अपना असली पता छिपाता है।",
    ),
    "NON_STANDARD_PORT": (
        0.4,
        "The link uses an unusual network port ({port}).",
        "लिंक असामान्य नेटवर्क पोर्ट ({port}) का उपयोग करता है।",
    ),
    "APK_LINK": (
        0.8,
        "The link downloads an Android app file (.apk) from outside the Play Store.",
        "यह लिंक प्ले स्टोर के बाहर से एंड्रॉइड ऐप फाइल (.apk) डाउनलोड करता है।",
    ),
    "BLOCKED_DESTINATION": (
        0.8,
        "The link leads to a hidden or internal network address.",
        "लिंक किसी छिपे हुए या आंतरिक नेटवर्क पते पर ले जाता है।",
    ),
    "TOO_MANY_REDIRECTS": (
        0.5,
        "The link bounces through many redirects to hide where it ends up.",
        "लिंक अपनी असली मंजिल छिपाने के लिए कई बार रीडायरेक्ट होता है।",
    ),
    "SAFE_BROWSING_MATCH": (
        1.0,
        "Google Safe Browsing lists this link as dangerous ({threat}).",
        "गूगल सेफ ब्राउजिंग ने इस लिंक को खतरनाक ({threat}) बताया है।",
    ),
    "VIRUSTOTAL_MALICIOUS": (
        0.95,
        "{count} security vendors on VirusTotal flag this link as malicious.",
        "वायरसटोटल पर {count} सुरक्षा कंपनियों ने इस लिंक को खतरनाक बताया है।",
    ),
    "VIRUSTOTAL_SUSPICIOUS": (
        0.6,
        "Security vendors on VirusTotal flag this link as suspicious.",
        "वायरसटोटल पर सुरक्षा कंपनियों ने इस लिंक को संदिग्ध बताया है।",
    ),
}


def make_finding(code: str, **params: object) -> UrlFinding:
    severity, en, hi = _FINDINGS[code]
    return UrlFinding(code, severity, en.format(**params), hi.format(**params))


# --------------------------------------------------------------------------
# Extraction and parsing
# --------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"""
    (?<![\w@.\-/:])                      # not in the middle of a word, email or path
    (?:
        https?://[^\s<>"'`]+
      |
        (?:[\w-]+\.)+[\w-]+              # bare host such as sbi-kyc.xyz
        (?::\d{1,5})?
        (?:[/?\#][^\s<>"'`]*)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"…।"
_NUMERIC_HOST = re.compile(r"^(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){0,3}$")


@dataclass(frozen=True, slots=True)
class ParsedUrl:
    url: str
    scheme: str
    host: str
    port: int | None
    path: str
    has_userinfo: bool
    subdomain: str
    domain: str
    suffix: str

    @property
    def registered_domain(self) -> str:
        return f"{self.domain}.{self.suffix}" if self.domain and self.suffix else ""

    @property
    def is_ip(self) -> bool:
        return _is_ip_host(self.host)


def _is_ip_host(host: str) -> bool:
    # Covers dotted IPv4, IPv6, and the decimal/hex forms browsers also accept
    # (http://3232235777 and http://0xc0a80001 both mean 192.168.0.1).
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(_NUMERIC_HOST.match(host))


def _decode_idna(host: str) -> str:
    if "xn--" not in host:
        return host
    try:
        return host.encode("ascii").decode("idna")
    except UnicodeError:
        return host


def parse_url(raw: str, data: UrlReferenceData | None = None) -> ParsedUrl | None:
    data = data or load_reference_data()
    candidate = raw if _SCHEME.match(raw) else f"http://{raw}"
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not host:
        return None

    host = _decode_idna(host.rstrip("."))
    if _is_ip_host(host):
        subdomain = domain = suffix = ""
    else:
        extracted = data.extractor(host)
        subdomain, domain, suffix = extracted.subdomain, extracted.domain, extracted.suffix

    return ParsedUrl(
        url=parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower()).geturl(),
        scheme=parts.scheme.lower(),
        host=host,
        port=port,
        path=parts.path,
        has_userinfo="@" in parts.netloc,
        subdomain=subdomain,
        domain=domain,
        suffix=suffix,
    )


def extract_urls(text: str, data: UrlReferenceData | None = None) -> list[str]:
    """Return unique URLs in order of appearance, normalised to include a scheme."""
    data = data or load_reference_data()
    found: list[str] = []
    seen: set[str] = set()

    for match in _URL_PATTERN.finditer(text):
        candidate = match.group(0).rstrip(_TRAILING_PUNCTUATION)
        has_scheme = bool(_SCHEME.match(candidate))
        parsed = parse_url(candidate, data)
        if parsed is None:
            continue
        if not has_scheme and not _looks_like_bare_host(parsed):
            continue
        if parsed.url not in seen:
            seen.add(parsed.url)
            found.append(parsed.url)
    return found


def _looks_like_bare_host(parsed: ParsedUrl) -> bool:
    # Without a scheme, require a real public suffix (rejects "Rs.500.00",
    # "e.g") or a strict dotted IPv4 address (rejects times like "9.30").
    if parsed.registered_domain:
        return True
    try:
        return isinstance(ipaddress.ip_address(parsed.host), ipaddress.IPv4Address)
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Offline checks
# --------------------------------------------------------------------------

_CONFUSABLE_CHARS = {
    # Cyrillic and Greek letters that render like Latin ones
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    "һ": "h",
    "ԁ": "d",
    "ɡ": "g",
    "ο": "o",
    "α": "a",
    "ν": "v",
    "τ": "t",
    "ρ": "p",
    "κ": "k",
    "ι": "i",
    "ı": "i",
    "ł": "l",
}
_DIGIT_SUBSTITUTES = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"}
_SKELETON_TABLE = str.maketrans({**_CONFUSABLE_CHARS, **_DIGIT_SUBSTITUTES, "-": "", "_": ""})


def _skeleton(label: str) -> str:
    """Reduce a domain label to how it looks, so 'hdfc-b4nk' and 'hdfcbank' compare equal."""
    return label.lower().translate(_SKELETON_TABLE).replace("rn", "m").replace("vv", "w")


def _official_brand(parsed: ParsedUrl, data: UrlReferenceData) -> str | None:
    """Return the brand name if the host is official, or a generic label for trusted suffixes."""
    brand = data.official_domains.get(parsed.registered_domain)
    if brand:
        return brand.name
    for suffix in data.trusted_suffixes:
        if parsed.host == suffix or parsed.host.endswith("." + suffix):
            return f"verified .{suffix} domain"
    return None


def _lookalike_finding(parsed: ParsedUrl, data: UrlReferenceData) -> UrlFinding | None:
    label = parsed.domain
    if not label:
        return None
    skeleton = _skeleton(label)

    for official in data.official_labels:
        if label == official.label or skeleton == official.skeleton:
            # Same name on another ending (sbi.co.com) or a disguised spelling (paytrn.com).
            is_lookalike = True
        elif len(official.label) >= 5:
            limit = 1 if len(official.label) < 8 else 2
            is_lookalike = 0 < Levenshtein.distance(skeleton, official.skeleton) <= limit
        else:
            is_lookalike = False

        if is_lookalike:
            return make_finding(
                "LOOKALIKE_DOMAIN",
                domain=parsed.registered_domain or parsed.host,
                brand=official.brand.name,
                official=official.domain,
            )
    return None


def _brand_keyword_finding(parsed: ParsedUrl, data: UrlReferenceData) -> UrlFinding | None:
    if parsed.registered_domain in data.brand_check_exempt:
        return None
    name_part = ".".join(p for p in (parsed.subdomain, parsed.domain) if p)
    tokens = set(re.split(r"[^a-z]+", name_part))
    squashed = _skeleton(name_part.replace(".", ""))

    for brand in data.brands:
        for keyword in brand.keywords:
            hit = keyword in tokens if len(keyword) < 5 else keyword in squashed
            if hit:
                return make_finding("BRAND_IMPERSONATION", brand=brand.name)
    return None


def static_findings(
    parsed: ParsedUrl, data: UrlReferenceData | None = None
) -> tuple[str | None, list[UrlFinding]]:
    """Run offline checks. Returns (official brand or None, findings)."""
    data = data or load_reference_data()
    findings: list[UrlFinding] = []

    if parsed.has_userinfo:
        findings.append(make_finding("USERINFO_TRICK"))
    if parsed.port is not None and parsed.port not in (80, 443):
        findings.append(make_finding("NON_STANDARD_PORT", port=parsed.port))
    if parsed.path.lower().endswith(".apk"):
        findings.append(make_finding("APK_LINK"))

    if parsed.is_ip:
        findings.append(make_finding("IP_ADDRESS_URL"))
        return None, findings

    official = _official_brand(parsed, data)
    if official:
        return official, findings

    if any(ch in _CONFUSABLE_CHARS for ch in parsed.host):
        findings.append(make_finding("IDN_HOMOGRAPH"))

    brand_finding = _lookalike_finding(parsed, data) or _brand_keyword_finding(parsed, data)
    if brand_finding:
        findings.append(brand_finding)

    tld = parsed.suffix.rsplit(".", 1)[-1]
    if tld in data.suspicious_tlds:
        findings.append(make_finding("SUSPICIOUS_TLD", tld=tld))

    sub_labels = [s for s in parsed.subdomain.split(".") if s]
    if sub_labels and sub_labels[0] == "www":
        sub_labels = sub_labels[1:]
    if len(sub_labels) >= 3:
        findings.append(make_finding("EXCESSIVE_SUBDOMAINS"))

    return None, findings


# --------------------------------------------------------------------------
# Network safety
# --------------------------------------------------------------------------


class BlockedDestinationError(Exception):
    """The URL points at a host we refuse to contact."""


def is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False  # includes scoped IPv6 such as fe80::1%eth0
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    # is_global excludes private (10/8, 172.16/12, 192.168/16), loopback,
    # link-local (169.254/16), carrier-grade NAT, reserved and unspecified.
    return address.is_global and not address.is_multicast


async def default_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, list):
        dates = [d for d in (_coerce_datetime(v) for v in value) if d is not None]
        return min(dates) if dates else None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return None


def default_whois_lookup(domain: str) -> datetime | None:
    import whois  # imported lazily: the package does work at import time

    record = whois.whois(domain)
    return _coerce_datetime(record.get("creation_date"))


# --------------------------------------------------------------------------
# Analyzer
# --------------------------------------------------------------------------


@dataclass
class UrlReport:
    url: str
    final_url: str
    host: str
    registered_domain: str
    is_shortened: bool = False
    redirect_chain: list[str] = field(default_factory=list)
    is_official: bool = False
    official_brand: str | None = None
    domain_age_days: int | None = None
    findings: list[UrlFinding] = field(default_factory=list)
    # check name -> "ok" | "flagged" | "skipped" | "error"
    checks: dict[str, str] = field(default_factory=dict)

    @property
    def max_severity(self) -> float:
        return max((f.severity for f in self.findings), default=0.0)


@dataclass
class _Expansion:
    final_url: str
    chain: list[str]
    too_many_redirects: bool = False
    blocked: bool = False
    error: bool = False


class UrlAnalyzer:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        resolver: Resolver = default_resolver,
        whois_lookup: WhoisLookup = default_whois_lookup,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        data: UrlReferenceData | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._data = data or load_reference_data()
        self._resolver = resolver
        self._whois_lookup = whois_lookup
        self._now = now
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.url_request_timeout),
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT},
        )
        self._whois_cache: TTLCache[datetime | None] = TTLCache(
            self._settings.whois_cache_ttl_seconds
        )
        self._reputation_cache: TTLCache[UrlFinding | None] = TTLCache(
            self._settings.reputation_cache_ttl_seconds
        )
        self._expansion_cache: TTLCache[_Expansion] = TTLCache(3600)

    async def __aenter__(self) -> UrlAnalyzer:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- public API ---------------------------------------------------------

    async def analyze_text(self, text: str) -> list[UrlReport]:
        urls = extract_urls(text, self._data)[: self._settings.url_max_per_message]
        return list(await asyncio.gather(*(self.analyze_url(url) for url in urls)))

    async def analyze_url(self, url: str) -> UrlReport:
        parsed = parse_url(url, self._data)
        if parsed is None:
            raise ValueError(f"not a valid URL: {url!r}")

        report = UrlReport(
            url=parsed.url,
            final_url=parsed.url,
            host=parsed.host,
            registered_domain=parsed.registered_domain,
            redirect_chain=[parsed.url],
        )
        network = self._settings.url_network_checks
        final = parsed

        if parsed.registered_domain in self._data.shorteners:
            report.is_shortened = True
            report.findings.append(make_finding("SHORTENED_URL", domain=parsed.registered_domain))
            if network:
                final = await self._apply_expansion(parsed, report)
            else:
                report.checks["expansion"] = "skipped"

        official, findings = static_findings(final, self._data)
        report.findings.extend(findings)
        report.is_official = official is not None
        report.official_brand = official

        lookup_domain = final.registered_domain
        if (
            network
            and not report.is_official
            and lookup_domain
            and lookup_domain not in self._data.shorteners
        ):
            await asyncio.gather(
                self._apply_domain_age(lookup_domain, report),
                self._apply_reputation(final.url, report),
            )

        report.findings = _dedupe_and_sort(report.findings)
        return report

    # -- expansion ----------------------------------------------------------

    async def _apply_expansion(self, parsed: ParsedUrl, report: UrlReport) -> ParsedUrl:
        expansion = self._expansion_cache.get(parsed.url)
        if expansion is None:
            expansion = await self._expand(parsed.url)
            if not expansion.error:
                self._expansion_cache.set(parsed.url, expansion)

        report.redirect_chain = expansion.chain
        report.final_url = expansion.final_url

        if expansion.blocked:
            report.findings.append(make_finding("BLOCKED_DESTINATION"))
            report.checks["expansion"] = "flagged"
        elif expansion.too_many_redirects:
            report.findings.append(make_finding("TOO_MANY_REDIRECTS"))
            report.checks["expansion"] = "flagged"
        else:
            report.checks["expansion"] = "error" if expansion.error else "ok"

        final = parse_url(expansion.final_url, self._data) or parsed
        report.host = final.host
        report.registered_domain = final.registered_domain
        if final.is_ip and not is_public_ip(final.host):
            report.findings.append(make_finding("BLOCKED_DESTINATION"))
        return final

    async def _expand(self, url: str) -> _Expansion:
        chain = [url]
        current = url
        redirects = 0

        while True:
            parsed = parse_url(current, self._data)
            # Only shortener hosts are contacted. Once the chain leaves them we
            # already know the destination from the Location header.
            if (
                parsed is None
                or parsed.scheme not in ("http", "https")
                or parsed.registered_domain not in self._data.shorteners
            ):
                return _Expansion(current, chain)

            try:
                location = await self._fetch_location(parsed)
            except BlockedDestinationError as exc:
                logger.warning("refusing to contact %s: %s", parsed.host, exc)
                return _Expansion(current, chain, blocked=True)
            except (httpx.HTTPError, TimeoutError, OSError) as exc:
                logger.info("could not expand %s: %s", current, exc)
                return _Expansion(current, chain, error=True)

            if not location:
                return _Expansion(current, chain)
            if redirects >= self._settings.url_max_redirects:
                return _Expansion(current, chain, too_many_redirects=True)

            redirects += 1
            current = urljoin(current, location.strip())
            chain.append(current)

    async def _fetch_location(self, parsed: ParsedUrl) -> str | None:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in (80, 443):
            raise BlockedDestinationError(f"port {port} not allowed")
        await self._ensure_public(parsed.host, port)

        timeout = self._settings.url_request_timeout
        response = await asyncio.wait_for(self._client.head(parsed.url), timeout)
        if response.is_redirect:
            return response.headers.get("location")
        if response.status_code < 400:
            return None

        # Some shorteners reject HEAD. Fall back to GET but close the stream
        # straight after the headers arrive, so no body is downloaded.
        async def get_location() -> str | None:
            async with self._client.stream("GET", parsed.url) as streamed:
                return streamed.headers.get("location") if streamed.is_redirect else None

        return await asyncio.wait_for(get_location(), timeout)

    async def _ensure_public(self, host: str, port: int) -> None:
        if _is_ip_host(host):
            if not is_public_ip(host):
                raise BlockedDestinationError(f"{host} is not a public address")
            return
        try:
            addresses = await asyncio.wait_for(
                self._resolver(host, port), self._settings.url_request_timeout
            )
        except (OSError, TimeoutError) as exc:
            raise BlockedDestinationError(f"cannot resolve {host}: {exc}") from exc
        if not addresses:
            raise BlockedDestinationError(f"{host} did not resolve")
        private = [a for a in addresses if not is_public_ip(a)]
        if private:
            raise BlockedDestinationError(f"{host} resolves to non-public {private}")

    # -- WHOIS --------------------------------------------------------------

    async def _apply_domain_age(self, domain: str, report: UrlReport) -> None:
        if domain in self._whois_cache:
            created = self._whois_cache.get(domain)
        else:
            try:
                created = await asyncio.wait_for(
                    asyncio.to_thread(self._whois_lookup, domain),
                    self._settings.url_request_timeout,
                )
            except Exception as exc:  # WHOIS libraries raise a wide range of errors
                logger.info("WHOIS lookup failed for %s: %s", domain, exc)
                self._whois_cache.set(domain, None, ttl=600)
                report.checks["whois"] = "error"
                return
            self._whois_cache.set(domain, created)

        if created is None:
            report.checks["whois"] = "error"
            return

        age_days = max(0, (self._now() - created).days)
        report.domain_age_days = age_days
        if age_days < self._settings.young_domain_days:
            report.findings.append(make_finding("YOUNG_DOMAIN", domain=domain, days=age_days))
            report.checks["whois"] = "flagged"
        else:
            report.checks["whois"] = "ok"

    # -- Reputation services ------------------------------------------------

    async def _apply_reputation(self, url: str, report: UrlReport) -> None:
        checks: list[tuple[str, Callable[[str], Awaitable[UrlFinding | None]]]] = []
        if self._settings.google_safe_browsing_api_key:
            checks.append(("safe_browsing", self._check_safe_browsing))
        if self._settings.virustotal_api_key:
            checks.append(("virustotal", self._check_virustotal))

        for name in ("safe_browsing", "virustotal"):
            report.checks.setdefault(name, "skipped")

        async def run(name: str, check: Callable[[str], Awaitable[UrlFinding | None]]) -> None:
            key = (name, url)
            if key in self._reputation_cache:
                finding = self._reputation_cache.get(key)
            else:
                try:
                    finding = await asyncio.wait_for(check(url), self._settings.url_request_timeout)
                except (httpx.HTTPError, TimeoutError, ValueError, KeyError) as exc:
                    logger.info("%s check failed for %s: %s", name, url, exc)
                    report.checks[name] = "error"
                    return
                self._reputation_cache.set(key, finding)
            report.checks[name] = "flagged" if finding else "ok"
            if finding:
                report.findings.append(finding)

        await asyncio.gather(*(run(name, check) for name, check in checks))

    async def _read_json(self, response: httpx.Response) -> Any:
        limit = self._settings.url_max_response_bytes
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError("response exceeded size limit")
        return json.loads(body or b"{}")

    async def _check_safe_browsing(self, url: str) -> UrlFinding | None:
        assert self._settings.google_safe_browsing_api_key is not None
        payload = {
            "client": {"clientId": "scamshield", "clientVersion": "0.1.0"},
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        async with self._client.stream(
            "POST",
            SAFE_BROWSING_ENDPOINT,
            params={"key": self._settings.google_safe_browsing_api_key.get_secret_value()},
            json=payload,
        ) as response:
            response.raise_for_status()
            data = await self._read_json(response)

        matches = data.get("matches") or []
        if not matches:
            return None
        threat = matches[0].get("threatType", "UNKNOWN").replace("_", " ").lower()
        return make_finding("SAFE_BROWSING_MATCH", threat=threat)

    async def _check_virustotal(self, url: str) -> UrlFinding | None:
        assert self._settings.virustotal_api_key is not None
        url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        # Look up existing reports only. Submitting a URL would make it (and any
        # personal tokens inside it) visible to every VirusTotal user.
        async with self._client.stream(
            "GET",
            VIRUSTOTAL_ENDPOINT + url_id,
            headers={"x-apikey": self._settings.virustotal_api_key.get_secret_value()},
        ) as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = await self._read_json(response)

        stats = data["data"]["attributes"]["last_analysis_stats"]
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        if malicious >= 2:
            return make_finding("VIRUSTOTAL_MALICIOUS", count=malicious)
        if malicious or suspicious:
            return make_finding("VIRUSTOTAL_SUSPICIOUS")
        return None


def _dedupe_and_sort(findings: list[UrlFinding]) -> list[UrlFinding]:
    unique: dict[str, UrlFinding] = {}
    for finding in findings:
        unique.setdefault(finding.code, finding)
    return sorted(unique.values(), key=lambda f: (-f.severity, f.code))
