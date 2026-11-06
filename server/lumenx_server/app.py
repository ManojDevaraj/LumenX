"""FastAPI application factory for the LumenX demo and processing API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from lumenx_core import create_engine, create_fake_engine
from lumenx_core.contracts import ABLATION_FLAGS, Cancelled, EngineUnavailable, InvalidRequest

from .decode import UploadDecodeError, decode_upload
from .lane import EngineLane, LaneBusy
from .runner import RunExecution
from .runs import RunStore
from .schemas import EnhanceResponse, ErrorResponse, GalleryItem
from .settings import Settings, get_settings


log = logging.getLogger("lumenx.server")


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(ErrorResponse(code=code, message=message).model_dump(), status_code=status_code)


def _parse_ablate(value: str) -> frozenset[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise UploadDecodeError("invalid_ablation", "ablate must be a JSON array of flag names.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise UploadDecodeError("invalid_ablation", "ablate must be a JSON array of flag names.")
    flags = frozenset(parsed)
    unknown = flags - ABLATION_FLAGS
    if unknown:
        raise UploadDecodeError("unknown_ablation", f"Unknown ablation flags: {', '.join(sorted(unknown))}.")
    if flags:
        if {"no_dl", "no_classical"}.issubset(flags):
            raise UploadDecodeError("contradictory_ablation", "At least one enhancement path must remain enabled.")
    return flags


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.gallery_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    frontend = Path(__file__).resolve().parent / "frontend"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.engine == "fake":
            engine = create_fake_engine(stage_delay_ms=config.mock_stage_delay_ms)
        else:
            engine = create_engine()
        lane = EngineLane(max_pending=config.max_pending)
        try:
            await lane.warmup(engine.warmup)
            app.state.engine = engine
            app.state.lane = lane
            app.state.store = RunStore(config.runs_dir, retention=config.run_retention)
            app.state.started_at = time.perf_counter()
            app.state.live_count = 0
            app.state.live_lock = asyncio.Lock()
            app.state.engine_health = await lane.call(engine.health)
            yield
        finally:
            lane.close()

    app = FastAPI(
        title="LumenX Low-Light Restoration",
        version="0.1.0",
        description="Adaptive low-light image restoration with progressive stage artifacts.",
        lifespan=lifespan,
    )
    app.state.settings = config
    app.mount("/files/runs", StaticFiles(directory=config.runs_dir), name="run-files")
    app.mount("/files/gallery", StaticFiles(directory=config.gallery_dir), name="gallery-files")
    app.mount("/files/results", StaticFiles(directory=config.results_dir), name="result-files")
    app.mount("/static", StaticFiles(directory=frontend), name="frontend-static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(frontend / "index.html")

    async def prepare_execution(
        request: Request,
        upload: UploadFile,
        mode: str,
        ablate_text: str,
        mock: bool,
        request_started: float,
    ) -> RunExecution:
        if mode not in ("hybrid", "classical", "dl"):
            raise UploadDecodeError("unknown_mode", "mode must be hybrid, classical, or dl.")
        ablate = _parse_ablate(ablate_text)
        if ablate and mode != "hybrid":
            raise UploadDecodeError("invalid_ablation", "Ablation flags are only valid in hybrid mode.")
        if mock and config.engine != "fake":
            raise HTTPException(
                status_code=503,
                detail={"code": "mock_engine_not_loaded", "message": "Start the server with LUMENX_MOCK=1 to use FakeEngine."},
                headers={"Retry-After": "1"},
            )
        content = await upload.read(config.max_upload_bytes + 1)
        decode_started = time.perf_counter()
        decoded = decode_upload(
            content,
            max_upload_bytes=config.max_upload_bytes,
            max_pixels=config.max_pixels,
            hard_max_pixels=config.hard_max_pixels,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        engine = request.app.state.engine
        return RunExecution(
            engine=engine,
            lane=request.app.state.lane,
            store=request.app.state.store,
            settings=config,
            decoded=decoded,
            mode=mode,
            ablate=ablate,
            mock=mock or config.engine == "fake",
            request_started=request_started,
            decode_ms=decode_ms,
        )

    @app.post("/enhance", response_model=EnhanceResponse)
    async def enhance(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Form("hybrid"),
        ablate: str = Form("[]"),
        mock: bool = Form(False),
    ) -> Any:
        request_started = time.perf_counter()
        try:
            execution = await prepare_execution(request, file, mode, ablate, mock, request_started)
            response = await execution.collect()
            request.app.state.engine_health = response["engine"]["health"]
            return response
        except UploadDecodeError as exc:
            return _error(exc.code, str(exc), exc.status_code)
        except HTTPException:
            raise
        except LaneBusy as exc:
            return JSONResponse(
                ErrorResponse(code="queue_full", message=str(exc)).model_dump(),
                status_code=503,
                headers={"Retry-After": "1"},
            )
        except InvalidRequest as exc:
            return _error(exc.code, str(exc), 422)
        except EngineUnavailable as exc:
            return JSONResponse(
                ErrorResponse(code="engine_unavailable", message=str(exc)).model_dump(),
                status_code=503,
                headers={"Retry-After": "1"},
            )
        except Cancelled as exc:
            return _error("cancelled", str(exc), 499)
        except Exception:
            log.exception("Enhancement request failed")
            return _error("internal_error", "The image could not be processed.", 500)

    @app.post("/enhance/stream")
    async def enhance_stream(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Form("hybrid"),
        ablate: str = Form("[]"),
        mock: bool = Form(False),
    ) -> Any:
        request_started = time.perf_counter()
        try:
            execution = await prepare_execution(request, file, mode, ablate, mock, request_started)
        except UploadDecodeError as exc:
            return _error(exc.code, str(exc), exc.status_code)
        except HTTPException:
            raise

        async def event_stream():
            completed = False
            last_seq = 0
            try:
                async for event in execution.events():
                    if await request.is_disconnected():
                        execution.cancel.set()
                        return
                    last_seq = int(event.get("seq", last_seq + 1))
                    yield (
                        f"id: {last_seq}\nevent: {event['event']}\n"
                        f"data: {json.dumps(event, ensure_ascii=False, allow_nan=False)}\n\n"
                    ).encode("utf-8")
                    if event["event"] == "done":
                        completed = True
            except LaneBusy as exc:
                yield _sse_error(last_seq + 1, "queue_full", str(exc))
            except InvalidRequest as exc:
                yield _sse_error(last_seq + 1, exc.code, str(exc))
            except EngineUnavailable as exc:
                yield _sse_error(last_seq + 1, "engine_unavailable", str(exc))
            except Cancelled as exc:
                yield _sse_error(last_seq + 1, "cancelled", str(exc))
            except Exception:
                log.exception("Streaming enhancement failed")
                yield _sse_error(last_seq + 1, "internal_error", "The image could not be processed.")
            finally:
                if not completed:
                    execution.cancel.set()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _sse_error(seq: int, code: str, message: str) -> bytes:
        event = {"seq": seq, "event": "error", "error": {"code": code, "message": message}}
        return f"id: {seq}\nevent: error\ndata: {json.dumps(event, allow_nan=False)}\n\n".encode("utf-8")

    @app.get("/runs/{run_id}")
    async def get_run(request: Request, run_id: str) -> Any:
        record = request.app.state.store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": "Run was not found."})
        return record

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        lane: EngineLane = request.app.state.lane
        health_data = request.app.state.engine_health
        if not lane.busy and lane.queue_depth == 0:
            try:
                fresh_health = await lane.call(request.app.state.engine.health, live=True)
                if fresh_health is not None:
                    health_data = fresh_health
                    request.app.state.engine_health = health_data
            except LaneBusy:
                pass
        return {
            "status": "ready" if health_data.get("ready") else "warming",
            "uptime_s": round(time.perf_counter() - request.app.state.started_at, 3),
            "lane": {
                "busy": lane.busy,
                "queue_depth": lane.queue_depth,
                "live_sessions": lane.live_sessions,
            },
            "engine": health_data,
        }

    @app.get("/config")
    async def public_config(request: Request) -> dict[str, Any]:
        try:
            engine_config = await request.app.state.lane.call(request.app.state.engine.public_config)
        except LaneBusy as exc:
            raise HTTPException(status_code=503, detail={"code": "queue_full", "message": str(exc)}, headers={"Retry-After": "1"}) from exc
        return {"schema_version": "1.0", "server": config.public(), "engine": engine_config}

    @app.get("/gallery", response_model=list[GalleryItem])
    async def gallery() -> list[dict[str, Any]]:
        metadata_path = config.gallery_dir / "metadata.json"
        if not metadata_path.is_file():
            return []
        try:
            items = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        output = []
        for item in items if isinstance(items, list) else []:
            filename = Path(str(item.get("image", ""))).name
            if not filename or not (config.gallery_dir / filename).is_file():
                continue
            output.append(
                {
                    "id": str(item.get("id", filename.rsplit(".", 1)[0])),
                    "title": str(item.get("title", "Untitled scene")),
                    "why_hard": str(item.get("why_hard", "Low-light image")),
                    "attribution": str(item.get("attribution", "")),
                    "image_url": f"/files/gallery/{filename}",
                    "cached_run_id": item.get("cached_run_id"),
                }
            )
        return output

    @app.get("/benchmarks")
    async def benchmarks() -> dict[str, Any]:
        summary_path = config.results_dir / "benchmark_summary.json"
        if summary_path.is_file():
            try:
                return json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        progress_path = config.results_dir / "progress.json"
        progress = None
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = None
        return {"status": "not_ready", "progress": progress, "tables": {}, "charts": []}

    @app.websocket("/live")
    async def live(websocket: WebSocket) -> None:
        await websocket.accept()
        state = websocket.app.state
        async with state.live_lock:
            if state.live_count >= config.max_live_sessions:
                await websocket.send_json({"status": "busy", "message": "The live session limit has been reached."})
                await websocket.close(code=1013)
                return
            state.live_count += 1
            state.lane.set_live_sessions(state.live_count)
        delivered = 0
        dropped = 0
        first_delivered: float | None = None
        fps = 0.0
        sequence = 0
        try:
            while True:
                packet = await websocket.receive()
                if packet.get("type") == "websocket.disconnect":
                    break
                raw = packet.get("bytes")
                if not raw:
                    continue
                sequence += 1
                if state.lane.busy or state.lane.queue_depth:
                    dropped += 1
                    await websocket.send_json({"seq": sequence, "fps": fps, "proc_ms": 0.0, "queue_ms": 0.0, "dropped": dropped, "status": "paused"})
                    continue
                try:
                    with Image.open(BytesIO(raw)) as frame_image:
                        if frame_image.width * frame_image.height > 12_000_000:
                            raise ValueError("frame exceeds the 12 megapixel processing limit")
                        frame_image.thumbnail((1280, 1280), Image.Resampling.BILINEAR)
                        frame = np.asarray(frame_image.convert("RGB"), dtype=np.uint8)
                except (UnidentifiedImageError, OSError, ValueError):
                    dropped += 1
                    await websocket.send_json({"seq": sequence, "fps": fps, "proc_ms": 0.0, "queue_ms": 0.0, "dropped": dropped, "status": "invalid_frame"})
                    continue
                started = time.perf_counter()
                try:
                    enhanced = await state.lane.call(
                        lambda frame=frame: state.engine.enhance_frame(frame), live=True
                    )
                except EngineUnavailable as exc:
                    await websocket.send_json({"seq": sequence, "fps": fps, "proc_ms": 0.0, "queue_ms": 0.0, "dropped": dropped, "status": "unavailable", "message": str(exc)})
                    continue
                if enhanced is None:
                    dropped += 1
                    await websocket.send_json({"seq": sequence, "fps": fps, "proc_ms": 0.0, "queue_ms": 0.0, "dropped": dropped, "status": "paused"})
                    continue
                proc_ms = (time.perf_counter() - started) * 1000.0
                now = time.perf_counter()
                delivered += 1
                if first_delivered is None:
                    first_delivered = now
                elif delivered > 1:
                    fps = (delivered - 1) / max(now - first_delivered, 1e-6)
                buffer = BytesIO()
                Image.fromarray(enhanced, mode="RGB").save(buffer, format="JPEG", quality=85)
                await websocket.send_json({"seq": sequence, "fps": fps, "proc_ms": proc_ms, "queue_ms": 0.0, "dropped": dropped, "status": "ok"})
                await websocket.send_bytes(buffer.getvalue())
        except WebSocketDisconnect:
            pass
        finally:
            async with state.live_lock:
                state.live_count = max(0, state.live_count - 1)
                state.lane.set_live_sessions(state.live_count)

    return app


app = create_app()

