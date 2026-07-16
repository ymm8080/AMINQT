# -*- coding: utf-8 -*-
"""FastAPI routes — stock selection + execution control (Phase 4)."""

import logging

from fastapi import APIRouter

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["quant"])


@router.get("/health")
def health() -> dict:
    """Liveness + current config snapshot."""
    return {
        "status": "ok",
        "data_source": settings.DATA_SOURCE,
        "exec_mode": settings.EXECUTION_MODE.value,
        "broker": settings.EXECUTION_BROKER,
    }


@router.post("/select")
def select() -> dict:
    """Run M1 pipeline: features → model → risk filter → Top 10.

    TODO(Phase 4): implement full select pipeline (<5s, in-memory pool).
    """
    return {"candidates": [], "note": "select pipeline — implement in Phase 4"}


@router.post("/execute")
def execute(order: dict) -> dict:
    """Execute or recommend an order per EXECUTION_MODE (auto/manual).

    TODO(M3): route to services.executor_base.get_executor().
    """
    return {"order": order, "mode": settings.EXECUTION_MODE.value}
