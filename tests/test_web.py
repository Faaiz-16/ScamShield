import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import BASE_DIR, Settings
from app.main import create_app

WEB_DIR = BASE_DIR / "web"
PAGES = ["index.html", "dashboard.html"]
SCRIPTS = ["app.js", "dashboard.js", "i18n.js"]


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'web.db'}",
        database_auto_create=True,
        url_network_checks=False,
        classifier="none",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/", "/dashboard"])
def test_pages_are_served_with_csp(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "frame-ancestors 'none'" in csp


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/js/app.js", "javascript"),
        ("/static/js/dashboard.js", "javascript"),
        ("/static/js/i18n.js", "javascript"),
        ("/static/css/styles.css", "text/css"),
        ("/static/img/favicon.svg", "image/svg+xml"),
    ],
)
def test_static_assets_are_served(client: TestClient, path: str, content_type: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_static_path_traversal_is_blocked(client: TestClient) -> None:
    assert client.get("/static/../app/core/config.py").status_code == 404
    assert client.get("/static/%2e%2e/.env.example").status_code == 404


@pytest.mark.parametrize("page", PAGES)
def test_page_has_no_inline_scripts_or_styles(page: str) -> None:
    html = (WEB_DIR / page).read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline <script> found"
    assert "<style" not in html
    assert not re.search(r"\sstyle=", html)
    assert not re.search(r"\son[a-z]+=", html), "inline event handler found"


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_scripts_never_use_inner_html(script_name: str) -> None:
    script = (WEB_DIR / "static" / "js" / script_name).read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script


def test_every_page_string_is_translated() -> None:
    html = "".join((WEB_DIR / page).read_text(encoding="utf-8") for page in PAGES)
    script = "".join(
        (WEB_DIR / "static" / "js" / name).read_text(encoding="utf-8")
        for name in ("app.js", "dashboard.js")
    )
    i18n = (WEB_DIR / "static" / "js" / "i18n.js").read_text(encoding="utf-8")

    used = set(re.findall(r'data-i18n(?:-placeholder|-aria-label)?="([\w.]+)"', html))
    used |= set(re.findall(r'\bt\("([\w.]+)"', script))
    used |= {f"verdict.{level}" for level in ("safe", "suspicious", "likely_scam")}
    en_block, hi_block = re.split(r"\n  hi: \{", i18n.split("export const EXAMPLES")[0])
    en_keys = set(re.findall(r'"([\w.]+)":', en_block))
    hi_keys = set(re.findall(r'"([\w.]+)":', hi_block))

    assert used <= en_keys, used - en_keys
    assert en_keys == hi_keys, en_keys ^ hi_keys


def test_analyze_response_includes_both_languages(client: TestClient) -> None:
    body = client.post(
        "/api/analyze", json={"text": "Update KYC now at http://sbi-kyc.xyz or account blocked"}
    ).json()
    for language in ("en", "hi"):
        texts = body["translations"][language]
        assert len(texts["reasons"]) == len(body["reasons"])
        assert len(texts["actions"]) == len(body["actions"])
        assert len(texts["link_findings"]) == len(body["links"])
        assert len(texts["link_findings"][0]) == len(body["links"][0]["findings"])
    assert body["translations"]["en"]["reasons"] == [r["text"] for r in body["reasons"]]
    assert body["translations"]["hi"]["verdict_label"] == "संभावित धोखाधड़ी"


def test_third_party_scripts_are_pinned_with_integrity() -> None:
    for page in PAGES:
        html = (WEB_DIR / page).read_text(encoding="utf-8")
        for tag in re.findall(r"<script[^>]*\bsrc=\"https?://[^>]*>", html):
            assert re.search(r'integrity="sha(256|384|512)-[A-Za-z0-9+/=]+"', tag), tag
            assert 'crossorigin="anonymous"' in tag, tag
            assert re.search(r"@\d+\.\d+\.\d+/", tag), f"version not pinned: {tag}"


def test_pages_link_to_each_other() -> None:
    index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    dashboard = (WEB_DIR / "dashboard.html").read_text(encoding="utf-8")
    assert 'href="/dashboard"' in index
    assert 'href="/"' in dashboard
