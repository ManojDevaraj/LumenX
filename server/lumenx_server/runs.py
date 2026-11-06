"""Persistent run directories and completion state."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .publish import publish_json


_RUN_ID = re.compile(r"^[a-f0-9]{32}$")


class RunStore:
    def __init__(self, runs_dir: Path, retention: int = 30):
        self.root = runs_dir.resolve()
        self.retention = max(1, int(retention))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._states: dict[str, dict[str, Any]] = {}

    def create(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run identifier")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise ValueError("run path escaped storage root")
        path.mkdir(parents=True, exist_ok=False)
        state = {"run_id": run_id, "status": "running", "created_at": time.time()}
        with self._lock:
            self._states[run_id] = state
        publish_json(path / "status.json", state)
        return path

    def set_state(self, run_id: str, status: str, **extra: Any) -> None:
        with self._lock:
            state = {**self._states.get(run_id, {}), "run_id": run_id, "status": status, **extra}
            self._states[run_id] = state
        path = self.root / run_id
        if path.is_dir():
            publish_json(path / "status.json", state)

    def finish(self, run_id: str, response: dict[str, Any]) -> None:
        self.set_state(run_id, "complete", completed_at=time.time())
        path = self.root / run_id
        publish_json(path / "result.json", response)
        self.prune()

    def get(self, run_id: str) -> dict[str, Any] | None:
        if not _RUN_ID.fullmatch(run_id):
            return None
        path = (self.root / run_id).resolve()
        if path.parent != self.root or not path.is_dir():
            return None
        result_path = path / "result.json"
        if result_path.is_file():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        with self._lock:
            state = self._states.get(run_id)
        if state:
            return dict(state)
        state_path = path / "status.json"
        if state_path.is_file():
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"run_id": run_id, "status": "error"}
        return {"run_id": run_id, "status": "running"}

    def prune(self) -> None:
        directories = [path for path in self.root.iterdir() if path.is_dir()]
        directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        kept = 0
        for path in directories:
            try:
                state = json.loads((path / "status.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {"status": "unknown"}
            if state.get("status") == "running":
                continue
            kept += 1
            if kept > self.retention:
                for child in path.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
                path.rmdir()
                with self._lock:
                    self._states.pop(path.name, None)

