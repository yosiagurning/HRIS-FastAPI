from __future__ import annotations

import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "sasf": 0.70,
    "flrgb": 0.45,
    "icm2o": 0.564862,
    "iom2c": 0.218523,
    "cdcn": 0.53,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "sasf": 0.20,
    "flrgb": 0.20,
    "icm2o": 0.20,
    "iom2c": 0.20,
    "cdcn": 0.20,
}


class ApiConfig(BaseModel):
    thresholds: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    weights: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    enable_temporal: bool = True
    temporal_weight: float = 0.35


class ApiConfigPatch(BaseModel):
    thresholds: Optional[Dict[str, float]] = None
    weights: Optional[Dict[str, float]] = None
    enable_temporal: Optional[bool] = None
    temporal_weight: Optional[float] = None


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_config(path: str) -> ApiConfig:
    if not os.path.isfile(path):
        cfg = ApiConfig()
        _atomic_write_json(path, cfg.model_dump())
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ApiConfig.model_validate(raw)


def save_config(path: str, cfg: ApiConfig) -> None:
    _atomic_write_json(path, cfg.model_dump())


def apply_patch(cfg: ApiConfig, patch: ApiConfigPatch) -> ApiConfig:
    data = cfg.model_dump()
    p = patch.model_dump(exclude_unset=True)
    for k, v in p.items():
        if k in ("thresholds", "weights") and isinstance(v, dict):
            data[k].update(v)
        else:
            data[k] = v
    return ApiConfig.model_validate(data)
