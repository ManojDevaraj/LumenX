"""Run orchestration, ordered event publication, and API response assembly."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from lumenx_core.contracts import EnhanceRequest
from lumenx_core.contracts import EnhanceResult, StageEvent

from .decode import DecodedImage
from .lane import EngineLane
from .publish import publish_input, publish_json, publish_visual
from .runs import RunStore
from .schemas import EnhanceResponse
from .settings import Settings


log = logging.getLogger("lumenx.server")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


class RunExecution:
    def __init__(
        self,
        *,
        engine: Any,
        lane: EngineLane,
        store: RunStore,
        settings: Settings,
        decoded: DecodedImage,
        mode: str,
        ablate: frozenset[str],
        mock: bool,
        request_started: float,
        decode_ms: float,
        seed: int = 0,
    ):
        self.engine = engine
        self.lane = lane
        self.store = store
        self.settings = settings
        self.decoded = decoded
        self.mode = mode
        self.ablate = ablate
        self.mock = mock
        self.request_started = request_started
        self.decode_ms = decode_ms
        self.seed = seed
        self.run_id = uuid.uuid4().hex
        self.run_dir = self.store.create(self.run_id)
        self.cancel = threading.Event()
        self.response: dict[str, Any] | None = None
        self._consumed = False
        self._sequence = 0
        self._assets: dict[str, str] = {}
        self._stage_events: list[dict[str, Any]] = []
        self._warnings: list[dict[str, Any]] = []
        self._items: list[Any] = []
        self._publish_ms = 0.0
        self._queue_ms: float | None = None
        self._queued_at = 0.0

    def _url(self, path: Path) -> str:
        return f"/files/runs/{self.run_id}/{path.name}"

    async def _publish_visuals(self, visuals: list[Any]) -> dict[str, str]:
        if not visuals:
            return {}
        started = time.perf_counter()
        paths = await asyncio.gather(
            *(asyncio.to_thread(publish_visual, self.run_dir, visual) for visual in visuals)
        )
        self._publish_ms += (time.perf_counter() - started) * 1000.0
        urls = {visual.key: self._url(path) for visual, path in zip(visuals, paths)}
        self._assets.update(urls)
        return urls

    def _event_payload(self, item: StageEvent) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "seq": self._sequence,
            "event": item.kind,
            "name": item.name,
            "title": item.title,
            "caption": item.caption,
            "tech_note": item.tech_note,
            "data": _jsonable(item.data),
            "visuals": [
                {
                    "key": visual.key,
                    "url": self._assets.get(visual.key),
                    "role": visual.role,
                    "max_side": visual.max_side,
                    "lossless": visual.lossless,
                }
                for visual in item.visuals
            ],
            "image_url": self._assets.get(item.primary_key) if item.primary_key else None,
            "t_ms": float(item.t_ms),
        }
        return event

    def _assemble(self, result: EnhanceResult, engine_health: dict[str, Any]) -> dict[str, Any]:
        now_ms = (time.perf_counter() - self.request_started) * 1000.0
        engine_timings = {key: float(value) for key, value in result.timings_ms.items()}
        core_total = float(engine_timings.get("core_total", 0.0))
        timings = {
            **engine_timings,
            "decode": self.decode_ms,
            "publish": self._publish_ms,
            "queue": float(self._queue_ms or 0.0),
            "server_total": max(0.0, now_ms - core_total),
            "total": now_ms,
        }
        final_url = self._assets.get("final")
        preview_url = self._assets.get("final_preview")
        if final_url is None or preview_url is None:
            raise RuntimeError("core result did not publish both final image assets")
        candidates = []
        for candidate in result.candidates:
            row = _jsonable(candidate)
            row["thumb_url"] = self._assets.get(candidate.thumb_key)
            candidates.append(row)
        stages = [row for row in self._stage_events if row["event"] == "stage"]
        story = []
        for step in result.story:
            row = _jsonable(step)
            row["image_url"] = self._assets.get(step.visual_key)
            row["highlight_url"] = self._assets.get(step.highlight_key) if step.highlight_key else None
            story.append(row)
        response = {
            "schema_version": "1.0",
            "status": "complete",
            "run_id": self.run_id,
            "mode": self.mode,
            "input": {
                "url": self._assets.get("input"),
                "width": self.decoded.processed_size[0],
                "height": self.decoded.processed_size[1],
                "original_width": self.decoded.original_size[0],
                "original_height": self.decoded.original_size[1],
                "format": self.decoded.format.lower(),
            },
            "final": {"url": final_url, "preview_url": preview_url},
            "headline": result.headline,
            "diagnosis": {
                "chips": [_jsonable(chip) for chip in result.chips],
                "raw": _jsonable(result.diagnosis_raw),
                "hotspots": [list(box) for box in result.hotspots],
            },
            "stages": stages,
            "maps": {
                "illumination_gray": self._assets.get("L_gray"),
                "snr": self._assets.get("snr"),
                "recoverability_overlay": self._assets.get("recoverability_overlay"),
                "gain": self._assets.get("gain"),
                "fusion": self._assets.get("fusion_map"),
                "denoise_strength": self._assets.get("denoise_strength"),
                "noise": self._assets.get("noise_map"),
                "heightmap": self._assets.get("heightmap"),
            },
            "relight": {
                **_jsonable(result.relight),
                "base_url": preview_url,
                "L_url": self._assets.get("L_gray"),
            },
            "recoverability": _jsonable(result.recoverability),
            "tuner": {
                "candidates": candidates,
                "search_ms": result.search_ms,
                "scoring_recipe": result.scoring_recipe,
            },
            "guardrails": [_jsonable(check) for check in result.guardrails],
            "decisions": list(result.decisions),
            "story": story,
            "metrics": _jsonable(result.metrics),
            "timings_ms": timings,
            "warnings": self._combined_warnings(result),
            "provenance": {
                **_jsonable(result.provenance),
                "original_size": list(self.decoded.original_size),
                "processed_size": list(self.decoded.processed_size),
                "downscaled": self.decoded.downscaled,
                "cache_hit": False,
            },
            "engine": {"fake": bool(engine_health.get("fake", False)), "health": engine_health},
        }
        return EnhanceResponse.model_validate(response).model_dump(mode="json")

    def _combined_warnings(self, result: EnhanceResult) -> list[dict[str, Any]]:
        warnings = [_jsonable(warning) for warning in result.warnings]
        seen = {(warning.get("code"), warning.get("message")) for warning in warnings}
        for warning in self._warnings:
            key = (warning.get("code"), warning.get("message"))
            if key not in seen:
                seen.add(key)
                warnings.append(warning)
        return warnings

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        if self._consumed:
            raise RuntimeError("a run stream can only be consumed once")
        self._consumed = True
        try:
            input_started = time.perf_counter()
            input_path = await asyncio.to_thread(
                publish_input, self.run_dir, self.decoded.image, self.decoded.alpha
            )
            self._publish_ms += (time.perf_counter() - input_started) * 1000.0
            self._assets["input"] = self._url(input_path)
            request = EnhanceRequest(
                image=self.decoded.image,
                original_size=self.decoded.original_size,
                alpha=self.decoded.alpha,
                mode=self.mode,
                ablate=self.ablate,
                seed=self.seed,
                cancel=self.cancel,
            )

            def on_started() -> None:
                self._queue_ms = (time.perf_counter() - self._queued_at) * 1000.0

            engine_health: dict[str, Any] = {}

            def start_engine() -> Any:
                items = self.engine.run(request)
                engine_health.update(self.engine.health())
                return items

            self._queued_at = time.perf_counter()
            async for item in self.lane.stream(
                start_engine, self.cancel, started_callback=on_started
            ):
                self._items.append(item)
                if isinstance(item, StageEvent):
                    await self._publish_visuals(item.visuals)
                    event = self._event_payload(item)
                    self._stage_events.append(event)
                    if item.kind == "warning":
                        self._warnings.append({"code": item.name, "message": item.tech_note})
                    yield event
                elif isinstance(item, EnhanceResult):
                    await self._publish_visuals(item.visuals)
                    if self.settings.debug_validate:
                        from lumenx_core.contract_check import validate_run

                        errors = validate_run(self._items, request)
                        if errors:
                            raise RuntimeError("Core stream contract violations: " + "; ".join(errors))
                    self.response = self._assemble(item, engine_health)
                    await asyncio.to_thread(
                        self.store.finish, self.run_id, self.response
                    )
                    self._sequence += 1
                    yield {"seq": self._sequence, "event": "done", "result": self.response}
                    return
                else:
                    raise RuntimeError(f"unexpected engine item: {type(item).__name__}")
            if self.response is None:
                raise RuntimeError("engine stream ended without an EnhanceResult")
        except asyncio.CancelledError:
            self.cancel.set()
            self.store.set_state(self.run_id, "cancelled", cancelled_at=time.time())
            raise
        except BaseException as exc:
            self.cancel.set()
            self.store.set_state(self.run_id, "error", message=str(exc), error_type=type(exc).__name__)
            log.exception("LumenX run %s failed", self.run_id)
            raise

    async def collect(self) -> dict[str, Any]:
        async for _event in self.events():
            pass
        if self.response is None:
            raise RuntimeError("run did not produce a completed response")
        return self.response

