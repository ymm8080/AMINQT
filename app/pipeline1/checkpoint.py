"""
Training Checkpoint System (PIPELINE1 §Resilience)
====================================================
Saves partial training progress so interrupted runs can resume
rather than restart from scratch.

Two classes:
  TrainingCheckpoint — per-board, per-tag training progress
  PipelineState      — multi-step pipeline progress across Claude sessions

Design principles:
  - Atomic writes (temp file + os.replace) — never corrupt on crash
  - JSON manifest + partial pickle — human-readable state + ML objects
  - Idempotent — re-running a completed step is safe
  - Self-describing — checkpoint knows what it contains
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_KINDS = ("1d_reg", "1d_cls", "3d_reg", "5d_reg")
EXTRA_KINDS = ("quantile_models", "pain_model", "rank_model")


# ──────────────────────────────────────────────
# Training Checkpoint
# ──────────────────────────────────────────────


class TrainingCheckpoint:
    """Per-board training checkpoint: saves after each model kind completes.

    Checkpoint file: {model_dir}/.checkpoint_{board}_{tag}.json  (manifest)
    Partial bundle:  {model_dir}/.checkpoint_{board}_{tag}.pkl   (models)

    Usage:
        ck = TrainingCheckpoint("models/pipeline1", "main", "2026W31")
        ck.save_progress(trained_dict, completed_kinds=["1d_reg"])
        # ... crash ...
        ck = TrainingCheckpoint("models/pipeline1", "main", "2026W31")
        if ck.exists():
            partial = ck.load_partial()
            # partial has already-trained models; train remaining kinds
    """

    def __init__(self, model_dir: str, board: str, tag: str):
        self.model_dir = Path(model_dir)
        self.board = board
        self.tag = tag
        self.manifest_path = self.model_dir / f".checkpoint_{board}_{tag}.json"
        self.bundle_path = self.model_dir / f".checkpoint_{board}_{tag}.pkl"

    def exists(self) -> bool:
        """Check if a checkpoint exists for this board+tag."""
        return self.manifest_path.exists() and self.bundle_path.exists()

    @property
    def completed_kinds(self) -> list[str]:
        """List of already-completed model kinds."""
        if not self.manifest_path.exists():
            return []
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return list(manifest.get("completed_kinds", []))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Checkpoint manifest corrupted, ignoring")
            return []

    @property
    def completed_extras(self) -> list[str]:
        """List of already-completed extra model kinds (E1/E2/LambdaRank)."""
        if not self.manifest_path.exists():
            return []
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return list(manifest.get("completed_extras", []))
        except (json.JSONDecodeError, KeyError):
            return []

    def save_progress(
        self,
        trained: dict,
        completed_kinds: list[str] | None = None,
        completed_extras: list[str] | None = None,
    ) -> None:
        """Save partial training state atomically.

        Args:
            trained: The trained dict from train_window() — contains
                     board, feature_cols, models, segs, etc.
            completed_kinds: Which MODEL_KINDS have been fully trained.
            completed_extras: Which EXTRA_KINDS have been trained.
        """
        if completed_kinds is None:
            completed_kinds = list(trained.get("models", {}).keys())
        if completed_extras is None:
            extras = []
            for ek in EXTRA_KINDS:
                if ek in trained:
                    extras.append(ek)
            completed_extras = extras

        # Build manifest
        segment_dates = {}
        for seg_name, seg_df in trained.get("segs", {}).items():
            if "date" in seg_df.columns:
                segment_dates[seg_name] = sorted(seg_df["date"].unique())[
                    :2
                ]  # first 2 dates

        manifest = {
            "board": self.board,
            "tag": self.tag,
            "feature_cols": trained.get("feature_cols", []),
            "completed_kinds": completed_kinds,
            "completed_extras": completed_extras,
            "segment_date_ranges": segment_dates,
            "window_total": trained.get("_window_total", 770),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Build partial bundle (only what's been trained so far)
        partial_bundle = {
            "board": trained.get("board", self.board),
            "feature_cols": trained.get("feature_cols", []),
            "models": {
                k: v
                for k, v in trained.get("models", {}).items()
                if k in completed_kinds
            },
        }
        for ek in EXTRA_KINDS:
            if ek in completed_extras and ek in trained:
                partial_bundle[ek] = trained[ek]
        # calibrator is fitted after all models, save if present
        if "calibrator" in trained:
            partial_bundle["calibrator"] = trained["calibrator"]

        # Atomic write: manifest
        self._atomic_write_json(self.manifest_path, manifest)
        # Atomic write: bundle
        self._atomic_write_pickle(self.bundle_path, partial_bundle)

        logger.info(
            "[%s/%s] Checkpoint saved: kinds=%s extras=%s",
            self.board,
            self.tag,
            completed_kinds,
            completed_extras,
        )

    def load_partial(self) -> dict | None:
        """Load the partial training bundle. Returns None if no checkpoint."""
        if not self.exists():
            return None
        try:
            with open(self.bundle_path, "rb") as fh:
                bundle = pickle.load(fh)
            logger.info(
                "[%s/%s] Checkpoint loaded: %d model kinds, extras=%s",
                self.board,
                self.tag,
                len(bundle.get("models", {})),
                [k for k in EXTRA_KINDS if k in bundle],
            )
            return bundle
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.error("Failed to load checkpoint bundle: %s", e)
            return None

    def load_manifest(self) -> dict | None:
        """Load just the manifest (no pickle)."""
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load checkpoint manifest: %s", e)
            return None

    def clear(self) -> None:
        """Remove checkpoint files (called after successful full save)."""
        for p in (self.manifest_path, self.bundle_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    def remaining_kinds(self) -> list[str]:
        """MODEL_KINDS that still need to be trained."""
        done = set(self.completed_kinds)
        return [k for k in MODEL_KINDS if k not in done]

    def remaining_extras(self) -> list[str]:
        """EXTRA_KINDS that still need to be trained."""
        done = set(self.completed_extras)
        return [k for k in EXTRA_KINDS if k not in done]

    # ── internal helpers ──

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=".ckpt_", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_pickle(path: Path, obj: Any) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".pkl", prefix=".ckpt_", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                pickle.dump(obj, fh)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ──────────────────────────────────────────────
# Pipeline State (multi-step progress tracker)
# ──────────────────────────────────────────────


class PipelineState:
    """Track multi-step pipeline progress across Claude sessions.

    State file: data/pipeline_state/{run_name}_state.json

    Each step writes its status (pending → running → done/failed) and
    output path.  On resume, steps with status="done" are skipped.

    Usage:
        state = PipelineState("train_predict_main", tag="2026W31")
        if state.step_should_run("build_features"):
            output = build_features()
            state.mark_done("build_features", output=output)
        ...
    """

    STATE_DIR = "data/pipeline_state"

    def __init__(self, run_name: str, tag: str = "", board: str = "main"):
        self.run_name = run_name
        self.tag = tag
        self.board = board
        self.dir = Path(self.STATE_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        # State file: {run_name}_{board}_{tag}.json
        slug = f"{run_name}_{board}" + (f"_{tag}" if tag else "")
        self.path = self.dir / f"{slug}_state.json"
        self._state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Pipeline state corrupted, starting fresh")
        return {
            "run_name": self.run_name,
            "board": self.board,
            "tag": self.tag,
            "steps": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def _save(self) -> None:
        self._state["updated_at"] = datetime.now().isoformat()
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=".pstate_", dir=str(self.dir)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, str(self.path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def step_status(self, step_name: str) -> str:
        """Return 'pending', 'running', 'done', or 'failed'."""
        return self._state.get("steps", {}).get(step_name, {}).get("status", "pending")

    def step_should_run(self, step_name: str) -> bool:
        """True if this step hasn't completed successfully yet."""
        status = self.step_status(step_name)
        return status != "done"

    def mark_running(self, step_name: str) -> None:
        self._state.setdefault("steps", {})[step_name] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
        }
        self._save()

    def mark_done(
        self, step_name: str, output: str = "", meta: dict | None = None
    ) -> None:
        entry = self._state.setdefault("steps", {}).get(step_name, {})
        entry.update(
            {
                "status": "done",
                "completed_at": datetime.now().isoformat(),
                "output": output,
            }
        )
        if meta:
            entry.setdefault("meta", {}).update(meta)
        self._state.setdefault("steps", {})[step_name] = entry
        self._save()
        logger.info("[PipelineState] %s → done (output: %s)", step_name, output)

    def mark_failed(self, step_name: str, error: str) -> None:
        entry = self._state.setdefault("steps", {}).get(step_name, {})
        entry.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": str(error)[:500],
            }
        )
        self._state.setdefault("steps", {})[step_name] = entry
        self._save()

    def summary(self) -> str:
        lines = [f"Pipeline: {self.run_name} | board={self.board} tag={self.tag}"]
        for step, info in self._state.get("steps", {}).items():
            status = info.get("status", "?")
            icon = {"done": "✓", "running": "…", "failed": "✗", "pending": "○"}.get(
                status, "?"
            )
            lines.append(f"  {icon} {step}: {status}")
        return "\n".join(lines)

    def reset_step(self, step_name: str) -> None:
        """Force a step back to pending (for re-run)."""
        self._state.setdefault("steps", {}).pop(step_name, None)
        self._save()

    def clear_all(self) -> None:
        """Remove the state file entirely."""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
        self._state = self._load()
