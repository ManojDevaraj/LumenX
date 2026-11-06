FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LUMENX_ENGINE=real \
    LUMENX_WEIGHTS_DIR=/app/core/weights \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY core ./core
COPY server ./server

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" \
    && python -m pip install -e "./core[vision,models]" -e ./server \
    && python -m lumenx_core.download_weights --output /app/core/weights/zerodce_pp.pth

EXPOSE 7860

CMD ["uvicorn", "lumenx_server.main:app", "--app-dir", "/app/server", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
