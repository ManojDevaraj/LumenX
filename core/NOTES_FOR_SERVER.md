# Notes for server integration

The v1.0 public stream contract remains in `contracts.py`. The FastAPI service uses `Engine.run(EnhanceRequest)` and publishes each visual before sending the associated SSE event. All model, frame, and metric operations are serialized by the server's single worker lane.

Install the core with `pip install -e "../core[vision]"` and install `server/` separately. `create_engine()` requires OpenCV contrib with `cv2.ximgproc.guidedFilter`. The optional Zero-DCE++ model loads once from `core/weights/zerodce_pp.pth`, `LUMENX_WEIGHTS_DIR`, or `LUMENX_ZERODCE_CHECKPOINT`.

Missing model weights do not make the classical path unavailable. Health and provenance disclose which paths are loaded, which fallback ran, device, precision, and checkpoint hash. Learned IQA metrics remain null with an explanation unless pyiqa and its metric weights are available.

The server is a reference local application. Dataset evaluation scripts and release instructions are in `server/README.md`; no benchmark result or latency target is claimed until the relevant dataset and machine have been measured.
