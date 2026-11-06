"""ASGI entrypoint: ``uvicorn lumenx_server.main:app``."""

from .app import app

__all__ = ["app"]

