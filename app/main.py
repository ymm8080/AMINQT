# -*- coding: utf-8 -*-
"""FastAPI app entrypoint + APScheduler (daily 14:50 auto-select)."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.frontier_routes import router as frontier_router
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AMINQT A-Share Quant Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(frontier_router)


@app.on_event("startup")
def _startup() -> None:
    """Start background scheduler (daily 14:50 Asia/Shanghai select)."""
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
