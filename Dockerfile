# LumenX: CPU image for Hugging Face Spaces (Docker SDK) or any Docker host.
#   docker build -t lumenx .
#   docker run -p 7860:7860 lumenx        ->  http://localhost:7860
# The engine uses CUDA automatically when run on a GPU host with a CUDA torch build;
# this image installs the CPU build so it runs anywhere.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LUMENX_ENGINE=real \
    HOME=/home/user \
    TORCH_HOME=/home/user/.cache/torch \
    HF_HOME=/home/user/.cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user

WORKDIR /app

# Heavy, rarely changing layer first.
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user . /app

# pyiqa pulls opencv-python, which shadows opencv-contrib (needed for cv2.ximgproc); restore contrib afterwards.
RUN pip install -e "core[models,iqa]" -e server \
 && pip uninstall -y opencv-python opencv-python-headless \
 && pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86 \
 && python -c "import cv2, torch, pyiqa; assert hasattr(cv2, 'ximgproc'); print('stack ok', torch.__version__, cv2.__version__)"

USER user

# Model weights (CIDNet + Zero-DCE++) and image-quality model weights are fetched at build time,
# so the first request is fast. Set SKIP_WEIGHTS=1 only for offline test builds.
ARG SKIP_WEIGHTS=0
RUN if [ "$SKIP_WEIGHTS" != "1" ]; then \
      python core/scripts/download_weights.py && \
      python -c "import pyiqa; [pyiqa.create_metric(m, device='cpu') for m in ('clipiqa', 'musiq', 'lpips')]"; \
    fi
# Real night photos for the demo gallery (Unsplash / LIME; git-ignored, so fetched at build time).
RUN python server/scripts/fetch_demo_photos.py || echo "gallery photos unavailable; synthetic gallery only"

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "lumenx_server.main:app", "--app-dir", "server", "--host", "0.0.0.0", "--port", "7860"]
