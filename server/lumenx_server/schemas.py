"""Transport schemas for the documented LumenX HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class ErrorResponse(StrictModel):
    code: str
    message: str


class EnhanceResponse(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["complete"] = "complete"
    run_id: str
    mode: Literal["hybrid", "classical", "dl"]
    input: dict[str, Any]
    final: dict[str, Any]
    headline: str
    diagnosis: dict[str, Any]
    stages: list[dict[str, Any]] = Field(default_factory=list)
    maps: dict[str, Any] = Field(default_factory=dict)
    relight: dict[str, Any] = Field(default_factory=dict)
    recoverability: dict[str, Any] = Field(default_factory=dict)
    tuner: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    story: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    engine: dict[str, Any] = Field(default_factory=dict)


class GalleryItem(StrictModel):
    id: str
    title: str
    why_hard: str
    attribution: str
    image_url: str
    cached_run_id: str | None = None

