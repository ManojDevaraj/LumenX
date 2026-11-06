# HTTP and live API

The service listens on port 8000 by default. All paths are relative to the origin. JSON responses use schema version `1.0`; errors have `{ "code": "...", "message": "..." }`.

## `POST /enhance`

Multipart fields:

| Field | Required | Values | Meaning |
| --- | --- | --- | --- |
| `file` | yes | PNG, JPEG, WebP, TIFF, BMP | Raster image upload |
| `mode` | no | `hybrid`, `classical`, `dl` | Processing path; defaults to `hybrid` |
| `ablate` | no | JSON string array | Hybrid ablation flags; defaults to `[]` |
| `mock` | no | `true`/`false` | Request FakeEngine; it must have been loaded with `LUMENX_MOCK=1` |

Success returns a complete run record with diagnosis, stage visual URLs, maps, candidates, report, metrics, timings, provenance, warnings, and final URL. Uploads over 32 MiB return 413 by default. Images over the configured hard decoded-pixel limit return 413; images above the process limit are downscaled with original and processed dimensions recorded.

Common errors: `415 unsupported_media`, `422 invalid_ablation`/`decode_failed`, `503 queue_full`/`engine_unavailable`, and `500 internal_error`. The queue-full response includes `Retry-After`.

## `POST /enhance/stream`

Uses the same multipart request. The response is Server-Sent Events. Each event has an increasing `id`, an `event` type, and a JSON `data` object. The server publishes its visual files before referencing their URLs.

- `stage`: one processing stage and its visuals/data.
- `candidate`: one of twelve tuned proxy results (hybrid mode with tuner enabled).
- `warning`: disclosed fallback or unresolved audit warning.
- `done`: contains the complete response under `result`.
- `error`: terminal stream error with an error object.

Clients should treat a stream without `done` as incomplete and may cancel by closing the request. Cancellation is observed at engine stage boundaries.

## Other routes

| Route | Result |
| --- | --- |
| `GET /health` | Service, model, lane, and uptime status |
| `GET /config` | Public-safe server and engine configuration |
| `GET /gallery` | Local gallery metadata; includes the bundled synthetic demo scenes |
| `GET /benchmarks` | Aggregated evaluation tables and chart URLs, or progress/not-ready state |
| `GET /runs/{run_id}` | Complete result or current run status |
| `/files/runs/{run_id}/{asset}` | Published run artifacts |
| `/files/gallery/{asset}` | Gallery image assets |
| `/files/results/{asset}` | Benchmark tables/charts |

## `WS /live`

The browser sends binary JPEG frames. For each accepted frame, the server sends one JSON text message containing `seq`, `fps`, `proc_ms`, `queue_ms`, `dropped`, and `status`, followed by one binary JPEG message for successful frames. Status may be `ok`, `paused`, `invalid_frame`, `unavailable`, or `busy`. At most one live session is accepted by default. New frames are dropped while the worker lane is occupied.
