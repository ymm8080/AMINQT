"""FastAPI app entrypoint + APScheduler (daily 14:50 auto-select).

Security:
  - API Key authentication via X-API-Key header (AMINQT_API_KEY env var).
  - When AMINQT_API_KEY is unset, auth is disabled (dev mode) with a warning.
  - CORS restricted to known Vite dev origins.
  - Security headers added to all responses.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.frontier_routes import router as frontier_router
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── API Key authentication ────────────────────────────────────
_API_KEY = os.getenv("AMINQT_API_KEY", "").strip()
_OPEN_ENDPOINTS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc"}


async def _api_key_middleware(request: Request, call_next):
    """Reject requests without valid X-API-Key header (except open endpoints)."""
    if _API_KEY and request.url.path not in _OPEN_ENDPOINTS:
        provided = request.headers.get("X-API-Key", "")
        if provided != _API_KEY:
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )
    response = await call_next(request)
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown lifecycle.

    Startup: start APScheduler (daily 14:50 Asia/Shanghai select).
    Shutdown: gracefully stop scheduler.
    """
    sched = None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        sched = BackgroundScheduler()
        # TODO(Phase 4): replace stub with real select() call.
        sched.add_job(
            lambda: logger.info("scheduled select tick"),
            "cron",
            hour=14,
            minute=50,
            timezone="Asia/Shanghai",
        )
        sched.start()
        logger.info("Scheduler started (daily 14:50 Asia/Shanghai)")
    except Exception as exc:  # noqa: BLE001
        logger.error("Scheduler init failed: %s", exc)

    if not _API_KEY:
        logger.warning(
            "AMINQT_API_KEY not set — API authentication DISABLED (dev mode). "
            "Set AMINQT_API_KEY in .env to enable."
        )

    yield

    if sched is not None:
        try:
            sched.shutdown(wait=False)
            logger.info("Scheduler stopped")
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(
    title="AMINQT A-Share Quant Platform",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],  # Vite dev
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.middleware("http")(_api_key_middleware)
app.include_router(router)
app.include_router(frontier_router)
