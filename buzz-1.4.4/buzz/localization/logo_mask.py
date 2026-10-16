"""Validated, serializable configuration for source-logo masking.

This module deliberately has no Qt dependency: configurations are immutable
snapshots which can safely travel from the editor to a background render job.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


_MODES = {"legacy", "custom"}


@dataclass(frozen=True)
class LogoMaskRegion:
    id: str
    name: str
    enabled: bool
    x: float
    y: float
    w: float
    h: float
    blur_radius: int = 24

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("logo mask region id is required")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("logo mask region name is required")
        if not isinstance(self.enabled, bool):
            raise ValueError("logo mask region enabled must be a boolean")
        values = (self.x, self.y, self.w, self.h)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("logo mask coordinates must be finite numbers")
        if not 0 <= self.x <= 1 or not 0 <= self.y <= 1 or not 0 < self.w <= 1 or not 0 < self.h <= 1:
            raise ValueError("logo mask rectangle must be normalized within [0, 1]")
        if self.x + self.w > 1 or self.y + self.h > 1:
            raise ValueError("logo mask rectangle must fit within the frame")
        if isinstance(self.blur_radius, bool) or not isinstance(self.blur_radius, int) or not 1 <= self.blur_radius <= 40:
            raise ValueError("logo mask blur_radius must be an integer between 1 and 40")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "enabled": self.enabled,
                "x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "blur_radius": self.blur_radius}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LogoMaskRegion":
        if not isinstance(value, dict):
            raise ValueError("logo mask region must be an object")
        return cls(**{key: value[key] for key in ("id", "name", "enabled", "x", "y", "w", "h", "blur_radius")})


@dataclass(frozen=True)
class LogoMaskConfig:
    schema_version: int = 1
    mode: str = "legacy"
    regions: tuple[LogoMaskRegion, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported logo mask schema version")
        if self.mode not in _MODES:
            raise ValueError("logo mask mode must be 'legacy' or 'custom'")
        if not isinstance(self.regions, tuple):
            object.__setattr__(self, "regions", tuple(self.regions))
        if any(not isinstance(region, LogoMaskRegion) for region in self.regions):
            raise ValueError("logo mask regions must be LogoMaskRegion instances")
        if len({region.id for region in self.regions}) != len(self.regions):
            raise ValueError("logo mask region ids must be unique")
        if len([region for region in self.regions if region.enabled]) > 16:
            raise ValueError("at most 16 enabled logo mask regions are allowed")

    @property
    def enabled_regions(self) -> tuple[LogoMaskRegion, ...]:
        return tuple(region for region in self.regions if region.enabled)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "mode": self.mode,
                "regions": [region.to_dict() for region in self.regions]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LogoMaskConfig":
        if not isinstance(value, dict):
            raise ValueError("logo mask config must be an object")
        try:
            return cls(schema_version=value["schema_version"], mode=value["mode"],
                       regions=tuple(LogoMaskRegion.from_dict(region) for region in value["regions"]))
        except KeyError as exc:
            raise ValueError(f"missing logo mask field: {exc.args[0]}") from exc

    @classmethod
    def from_json(cls, value: str) -> "LogoMaskConfig":
        try:
            return cls.from_dict(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError("logo mask config is not valid JSON") from exc
