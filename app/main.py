"""Application entry point.

Run with:
    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import app.models  # noqa: F401  (registers tables on Base.metadata)
from app.api.errors import register_error_handlers
from app.api.routes import router as api_router
from app.core.config import BASE_DIR, Settings, get_settings
from app.core.database import Base, build_engine, build_session_factory
from app.core.rate_limit import RateLimiter
from app.detection.classifier import ClassifierUnavailableError, TextClassifier, load_classifier
from app.detection.rules import RuleEngine
from app.detection.url_analyzer import UrlAnalyzer
from app.detection.verdict import VerdictConfig, VerdictEngine
from app.services.analysis import AnalysisService

logger = logging.getLogger("scamshield")

WEB_DIR = BASE_DIR / "web"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

# Pages load scripts and styles only from this server, plus Chart.js from jsDelivr.
# No inline scripts or styles are used, so none are allowed.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


def build_analysis_service(settings: Settings) -> AnalysisService:
    rule_engine = RuleEngine.from_file(settings.rules_path)

    classifier: TextClassifier | None = None
    if settings.classifier.lower() != "none":
        try:
            classifier = load_classifier(settings.classifier, settings)
        except ClassifierUnavailableError as exc:
            logger.warning("running without a text classifier: %s", exc)

    return AnalysisService(
        rule_engine=rule_engine,
        url_analyzer=UrlAnalyzer(settings),
        verdict_engine=VerdictEngine(VerdictConfig.from_settings(settings), rule_engine.categories),
        classifier=classifier,
    )


def create_app(
    settings: Settings | None = None, *, analysis_service: AnalysisService | None = None
) -> FastAPI:
    settings = settings or get_settings()
    engine = build_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.database_auto_create:
            Base.metadata.create_all(engine)
        service = analysis_service or build_analysis_service(settings)
        app.state.analysis = service
        yield
        await service.url_analyzer.aclose()
        engine.dispose()

    app = FastAPI(
        title="ScamShield API",
        version="0.1.0",
        description="Detects scam SMS, WhatsApp and UPI messages in English, Hindi and Hinglish.",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.environment != "production" else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.environment != "production" else None,
    )
    app.state.settings = settings
    app.state.session_factory = build_session_factory(engine)
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        else:
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        return response

    register_error_handlers(app)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "dashboard.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
