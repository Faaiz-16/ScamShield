from collections.abc import AsyncIterator, Callable
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from app.core.config import Settings
from app.detection.url_analyzer import (
    UrlAnalyzer,
    _coerce_datetime,
    extract_urls,
    is_public_ip,
    parse_url,
    static_findings,
)

NOW = datetime(2026, 11, 12, tzinfo=UTC)
PUBLIC_IP = "93.184.216.34"

Handler = Callable[[httpx.Request], httpx.Response]


class FakeResolver:
    def __init__(self, mapping: dict[str, list[str]] | None = None) -> None:
        self.mapping = mapping or {}
        self.calls: list[str] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append(host)
        return self.mapping.get(host, [PUBLIC_IP])


class FakeWhois:
    def __init__(self, created: dict[str, datetime | Exception] | None = None) -> None:
        self.created = created or {}
        self.calls: list[str] = []

    def __call__(self, domain: str) -> datetime | None:
        self.calls.append(domain)
        value = self.created.get(domain, NOW - timedelta(days=3650))
        if isinstance(value, Exception):
            raise value
        return value


class Recorder:
    """Wraps a handler and remembers every request the analyzer sent."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]


def redirect(location: str, status: int = 301) -> httpx.Response:
    return httpx.Response(status, headers={"Location": location})


def not_found(_: httpx.Request) -> httpx.Response:
    return httpx.Response(404)


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "url_network_checks": True,
        "google_safe_browsing_api_key": None,
        "virustotal_api_key": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
async def build() -> AsyncIterator[Callable[..., tuple[UrlAnalyzer, Recorder]]]:
    clients: list[httpx.AsyncClient] = []

    def factory(
        handler: Handler = not_found,
        *,
        resolver: FakeResolver | None = None,
        whois: FakeWhois | None = None,
        **settings: object,
    ) -> tuple[UrlAnalyzer, Recorder]:
        recorder = Recorder(handler)
        client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
        clients.append(client)
        analyzer = UrlAnalyzer(
            make_settings(**settings),
            http_client=client,
            resolver=resolver or FakeResolver(),
            whois_lookup=whois or FakeWhois(),
            now=lambda: NOW,
        )
        return analyzer, recorder

    yield factory
    for client in clients:
        await client.aclose()


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


# --- Extraction --------------------------------------------------------------


def test_extracts_urls_with_and_without_scheme() -> None:
    text = (
        "Update KYC at https://sbi-kyc.xyz/login?id=1. Or visit www.HDFC-update.in, "
        "or sbi-verify.top/claim! Again: https://sbi-kyc.xyz/login?id=1"
    )
    assert extract_urls(text) == [
        "https://sbi-kyc.xyz/login?id=1",
        "http://www.hdfc-update.in",
        "http://sbi-verify.top/claim",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Rs.500.00 debited from A/c XX1234 on 12.11.26 at 9.30 pm",
        "Write to support@sbi.co.in for help",
        "Pay to merchant@okicici via UPI",
        "e.g. this is fine. Thanks.Bye",
        "आपका ओटीपी 123456 है।",
    ],
)
def test_ignores_text_that_only_looks_like_a_url(text: str) -> None:
    assert extract_urls(text) == []


def test_extracts_bare_ip_and_urls_inside_hindi_text() -> None:
    text = "लिंक पर क्लिक करें 45.33.12.9/kyc और फिर bit.ly/3xYz खोलें।"
    assert extract_urls(text) == ["http://45.33.12.9/kyc", "http://bit.ly/3xYz"]


def test_parse_url_splits_domain_parts() -> None:
    parsed = parse_url("https://netbanking.hdfcbank.com/login")
    assert parsed is not None
    assert (parsed.subdomain, parsed.registered_domain) == ("netbanking", "hdfcbank.com")
    rbi_bank = parse_url("https://sbi.bank.in")
    assert rbi_bank is not None and rbi_bank.registered_domain == "sbi.bank.in"


# --- Offline checks ----------------------------------------------------------

CYRILLIC_SBI = "http://" + "sbі.co.in".encode("idna").decode()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://hdfcbannk.com/login", {"LOOKALIKE_DOMAIN"}),
        ("http://paytrn.com", {"LOOKALIKE_DOMAIN"}),
        ("http://hdfc-bank.in", {"LOOKALIKE_DOMAIN"}),
        ("http://hdfcbank.online", {"LOOKALIKE_DOMAIN", "SUSPICIOUS_TLD"}),
        ("http://sbi.co.com", {"BRAND_IMPERSONATION"}),
        ("http://1cicibank.com", {"LOOKALIKE_DOMAIN"}),
        ("http://icici-kyc-update.in", {"BRAND_IMPERSONATION"}),
        (
            "http://sbi.co.in.verify-kyc.xyz/login",
            {"BRAND_IMPERSONATION", "SUSPICIOUS_TLD", "EXCESSIVE_SUBDOMAINS"},
        ),
        ("http://reward-claim.top", {"SUSPICIOUS_TLD"}),
        ("http://192.168.1.1/login", {"IP_ADDRESS_URL"}),
        ("http://3232235777/", {"IP_ADDRESS_URL"}),
        ("http://www.sbi.co.in@evil-site.com/", {"USERINFO_TRICK"}),
        ("http://rewards-portal.com:8080/", {"NON_STANDARD_PORT"}),
        ("http://files-share.com/SBI_Rewards.apk", {"APK_LINK"}),
        (CYRILLIC_SBI, {"IDN_HOMOGRAPH", "LOOKALIKE_DOMAIN"}),
    ],
)
def test_static_checks_flag_suspicious_urls(url: str, expected: set[str]) -> None:
    parsed = parse_url(url)
    assert parsed is not None
    official, findings = static_findings(parsed)
    assert official is None
    assert expected <= {f.code for f in findings}


@pytest.mark.parametrize(
    "url",
    [
        "https://www.onlinesbi.sbi/",
        "https://netbanking.hdfcbank.com/netbanking/",
        "https://www.incometax.gov.in/iec/foportal/",
        "https://sbi.bank.in",
        "https://www.amazon.in/gp/your-account",
        "https://paytm.com",
        "https://cybercrime.gov.in",
    ],
)
def test_official_domains_have_no_findings(url: str) -> None:
    parsed = parse_url(url)
    assert parsed is not None
    official, findings = static_findings(parsed)
    assert official is not None
    assert findings == []


@pytest.mark.parametrize(
    "url",
    ["https://sbirds.com", "https://example.com/about", "https://mybucket.s3.amazonaws.com/x"],
)
def test_unrelated_domains_are_not_brand_matches(url: str) -> None:
    parsed = parse_url(url)
    assert parsed is not None
    _, findings = static_findings(parsed)
    assert not {f.code for f in findings} & {"LOOKALIKE_DOMAIN", "BRAND_IMPERSONATION"}


def test_findings_have_bilingual_reasons() -> None:
    parsed = parse_url("http://hdfcbannk.com")
    assert parsed is not None
    finding = static_findings(parsed)[1][0]
    assert "HDFC Bank" in finding.reason("en")
    assert "HDFC Bank" in finding.reason("hi") and finding.reason("hi") != finding.reason("en")


# --- SSRF protection ---------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("8.8.8.8", True),
        (PUBLIC_IP, True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("172.31.255.255", False),
        ("192.168.0.10", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("0.0.0.0", False),
        ("::1", False),
        ("fe80::1", False),
        ("::ffff:127.0.0.1", False),
        ("fe80::1%eth0", False),
        ("not-an-ip", False),
    ],
)
def test_is_public_ip(address: str, expected: bool) -> None:
    assert is_public_ip(address) is expected


async def test_shortener_resolving_to_private_ip_is_never_contacted(build) -> None:
    resolver = FakeResolver({"bit.ly": ["10.0.0.5"]})
    analyzer, recorder = build(resolver=resolver)
    report = await analyzer.analyze_url("http://bit.ly/abc")
    assert recorder.requests == []
    assert "BLOCKED_DESTINATION" in codes(report)


async def test_redirect_to_internal_address_is_not_followed(build) -> None:
    analyzer, recorder = build(lambda r: redirect("http://169.254.169.254/latest/meta-data/"))
    report = await analyzer.analyze_url("https://bit.ly/meta")
    assert recorder.hosts == ["bit.ly"]
    assert report.final_url == "http://169.254.169.254/latest/meta-data/"
    assert {"BLOCKED_DESTINATION", "IP_ADDRESS_URL"} <= codes(report)


async def test_shortener_on_non_standard_port_is_refused(build) -> None:
    analyzer, recorder = build(lambda r: redirect("https://example.com"))
    report = await analyzer.analyze_url("http://bit.ly:8080/abc")
    assert recorder.requests == []
    assert "BLOCKED_DESTINATION" in codes(report)


# --- Shortener expansion -----------------------------------------------------


async def test_expands_chain_without_contacting_destination(build) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bit.ly":
            return redirect("https://tinyurl.com/xyz")
        if request.url.host == "tinyurl.com":
            return redirect("https://sbi-kyc-verify.xyz/login", status=302)
        raise AssertionError(f"destination contacted: {request.url}")

    whois = FakeWhois()
    analyzer, recorder = build(handler, whois=whois)
    report = await analyzer.analyze_url("https://bit.ly/abc")

    assert recorder.hosts == ["bit.ly", "tinyurl.com"]
    assert all(r.method == "HEAD" for r in recorder.requests)
    assert report.redirect_chain == [
        "https://bit.ly/abc",
        "https://tinyurl.com/xyz",
        "https://sbi-kyc-verify.xyz/login",
    ]
    assert report.final_url == "https://sbi-kyc-verify.xyz/login"
    assert report.registered_domain == "sbi-kyc-verify.xyz"
    assert {"SHORTENED_URL", "BRAND_IMPERSONATION", "SUSPICIOUS_TLD"} <= codes(report)
    assert whois.calls == ["sbi-kyc-verify.xyz"]


async def test_falls_back_to_get_when_head_is_rejected(build) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(405)
        return redirect("https://example.com/landing")

    analyzer, recorder = build(handler)
    report = await analyzer.analyze_url("https://cutt.ly/abc")
    assert [r.method for r in recorder.requests] == ["HEAD", "GET"]
    assert report.final_url == "https://example.com/landing"


async def test_redirect_loop_stops_at_limit(build) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        step = int(request.url.path.strip("/"))
        return redirect(f"https://bit.ly/{step + 1}")

    analyzer, recorder = build(handler, url_max_redirects=5)
    report = await analyzer.analyze_url("https://bit.ly/0")
    assert len(recorder.requests) == 6
    assert len(report.redirect_chain) == 6
    assert "TOO_MANY_REDIRECTS" in codes(report)


async def test_network_errors_are_handled_gracefully(build) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    analyzer, _ = build(handler)
    report = await analyzer.analyze_url("https://bit.ly/slow")
    assert report.checks["expansion"] == "error"
    assert report.final_url == "https://bit.ly/slow"
    assert codes(report) == {"SHORTENED_URL"}


async def test_official_shortener_resolves_to_official_site(build) -> None:
    whois = FakeWhois()
    analyzer, _ = build(lambda r: redirect("https://www.amazon.in/dp/B0TEST"), whois=whois)
    report = await analyzer.analyze_url("https://amzn.to/3abc")
    assert report.is_official and report.official_brand == "Amazon"
    assert codes(report) == {"SHORTENED_URL"}
    assert whois.calls == []


async def test_expansion_results_are_cached(build) -> None:
    analyzer, recorder = build(lambda r: redirect("https://example.com"))
    await analyzer.analyze_url("https://bit.ly/abc")
    await analyzer.analyze_url("https://bit.ly/abc")
    assert len(recorder.requests) == 1


# --- WHOIS -------------------------------------------------------------------


async def test_young_domain_is_flagged(build) -> None:
    whois = FakeWhois({"fresh-offer.com": NOW - timedelta(days=10)})
    analyzer, _ = build(whois=whois)
    report = await analyzer.analyze_url("https://fresh-offer.com")
    assert report.domain_age_days == 10
    assert "YOUNG_DOMAIN" in codes(report)
    assert "10 days" in next(f for f in report.findings if f.code == "YOUNG_DOMAIN").reason_en


async def test_old_domain_is_not_flagged(build) -> None:
    analyzer, _ = build()
    report = await analyzer.analyze_url("https://example.com")
    assert report.checks["whois"] == "ok"
    assert "YOUNG_DOMAIN" not in codes(report)


async def test_whois_failure_does_not_break_analysis(build) -> None:
    whois = FakeWhois({"broken.com": RuntimeError("whois server down")})
    analyzer, _ = build(whois=whois)
    report = await analyzer.analyze_url("https://broken.com")
    assert report.checks["whois"] == "error"
    assert report.domain_age_days is None


async def test_whois_results_are_cached(build) -> None:
    whois = FakeWhois()
    analyzer, _ = build(whois=whois)
    await analyzer.analyze_url("https://example.com/a")
    await analyzer.analyze_url("https://www.example.com/b")
    assert whois.calls == ["example.com"]


async def test_official_domains_skip_whois(build) -> None:
    whois = FakeWhois()
    analyzer, _ = build(whois=whois)
    report = await analyzer.analyze_url("https://www.onlinesbi.sbi")
    assert report.is_official
    assert whois.calls == []


def test_coerce_datetime_handles_whois_formats() -> None:
    naive = datetime(2020, 1, 1)
    assert _coerce_datetime(naive) == datetime(2020, 1, 1, tzinfo=UTC)
    assert _coerce_datetime([datetime(2021, 1, 1), naive]) == datetime(2020, 1, 1, tzinfo=UTC)
    assert _coerce_datetime(date(2019, 5, 2)) == datetime(2019, 5, 2, tzinfo=UTC)
    assert _coerce_datetime("2020-01-01") is None
    assert _coerce_datetime(None) is None


# --- Reputation services -----------------------------------------------------


async def test_reputation_checks_skipped_without_keys(build) -> None:
    analyzer, recorder = build()
    report = await analyzer.analyze_url("https://example.com")
    assert recorder.requests == []
    assert report.checks["safe_browsing"] == "skipped"
    assert report.checks["virustotal"] == "skipped"


async def test_safe_browsing_match_is_reported(build) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "safebrowsing.googleapis.com"
        assert request.url.params["key"] == "gsb-test-key"
        return httpx.Response(200, json={"matches": [{"threatType": "SOCIAL_ENGINEERING"}]})

    analyzer, recorder = build(handler, google_safe_browsing_api_key="gsb-test-key")
    report = await analyzer.analyze_url("https://phish.example.com")
    assert "SAFE_BROWSING_MATCH" in codes(report)
    assert report.checks["safe_browsing"] == "flagged"
    assert b"https://phish.example.com" in recorder.requests[0].content


@pytest.mark.parametrize(
    ("response", "expected_code", "expected_status"),
    [
        (
            httpx.Response(
                200, json={"data": {"attributes": {"last_analysis_stats": {"malicious": 7}}}}
            ),
            "VIRUSTOTAL_MALICIOUS",
            "flagged",
        ),
        (
            httpx.Response(
                200, json={"data": {"attributes": {"last_analysis_stats": {"suspicious": 1}}}}
            ),
            "VIRUSTOTAL_SUSPICIOUS",
            "flagged",
        ),
        (httpx.Response(404), None, "ok"),
        (httpx.Response(500), None, "error"),
    ],
)
async def test_virustotal_results(build, response, expected_code, expected_status) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.headers["x-apikey"] == "vt-test-key"
        return response

    analyzer, _ = build(handler, virustotal_api_key="vt-test-key")
    report = await analyzer.analyze_url("https://unknown.example.org")
    assert report.checks["virustotal"] == expected_status
    if expected_code:
        assert expected_code in codes(report)


async def test_reputation_results_are_cached(build) -> None:
    analyzer, recorder = build(
        lambda r: httpx.Response(200, json={}), google_safe_browsing_api_key="k"
    )
    await analyzer.analyze_url("https://example.com/x")
    await analyzer.analyze_url("https://example.com/x")
    assert len(recorder.requests) == 1


async def test_oversized_api_response_is_rejected(build) -> None:
    huge = httpx.Response(200, content=b"{" + b" " * 200_000 + b"}")
    analyzer, _ = build(lambda r: huge, google_safe_browsing_api_key="k")
    report = await analyzer.analyze_url("https://example.com")
    assert report.checks["safe_browsing"] == "error"


# --- Analyzer as a whole -----------------------------------------------------


async def test_network_checks_can_be_disabled(build) -> None:
    whois = FakeWhois()
    analyzer, recorder = build(whois=whois, url_network_checks=False)
    report = await analyzer.analyze_url("https://bit.ly/abc")
    assert recorder.requests == [] and whois.calls == []
    assert report.checks["expansion"] == "skipped"
    assert codes(report) == {"SHORTENED_URL"}


async def test_analyze_text_caps_number_of_urls(build) -> None:
    analyzer, _ = build(url_max_per_message=2)
    reports = await analyzer.analyze_text("a.com b.com c.com")
    assert [r.host for r in reports] == ["a.com", "b.com"]


async def test_findings_are_sorted_most_severe_first(build) -> None:
    analyzer, _ = build(whois=FakeWhois({"verify-kyc.xyz": NOW - timedelta(days=2)}))
    report = await analyzer.analyze_url("http://sbi.co.in.verify-kyc.xyz/login")
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, reverse=True)
    assert report.max_severity == severities[0]
