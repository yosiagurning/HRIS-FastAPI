from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from api_config import ApiConfigPatch, apply_patch, load_config, save_config
from api_inference import HF_REPO_DIR, load_models, run_ensemble


_MODEL_KEYS = ("sasf", "flrgb", "icm2o", "iom2c", "cdcn")
_ENV_THR = {
    "sasf": "HF_THR_SASF",
    "flrgb": "HF_THR_FLRGB",
    "icm2o": "HF_THR_ICM2O",
    "iom2c": "HF_THR_IOM2C",
    "cdcn": "HF_THR_CDCN",
}
_ENV_W = {
    "sasf": "HF_W_SASF",
    "flrgb": "HF_W_FLRGB",
    "icm2o": "HF_W_ICM2O",
    "iom2c": "HF_W_IOM2C",
    "cdcn": "HF_W_CDCN",
}


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_settings.json")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _json_dict_from_env(name: str) -> Dict[str, float]:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).lower(): _safe_float(v, 0.0) for k, v in data.items()}


@dataclass
class HFAntiSpoofResult:
    ok: bool
    is_spoof: bool = False
    real_score: float = 0.0
    spoof_score: float = 1.0
    real_threshold: float = 0.0
    spoof_threshold: float = 1.0
    label: str = "Unknown"
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


class HFAntiSpoofEnsemble:
    def __init__(self, prefer_finetuned: bool = True, config_path: Optional[str] = None):
        self.prefer_finetuned = prefer_finetuned
        self.config_path = config_path or os.getenv("HF_ANTISPOOF_CONFIG_PATH", _default_config_path())
        self.cfg = load_config(self.config_path)
        self.models = None
        self.load_error: Optional[str] = None
        self.reload_models()

    def _thresholds(self) -> Dict[str, float]:
        thresholds = dict(self.cfg.thresholds)
        thresholds.update(_json_dict_from_env("HF_THRESHOLDS_JSON"))
        for key in _MODEL_KEYS:
            env_name = _ENV_THR[key]
            if os.getenv(env_name) is not None:
                thresholds[key] = _safe_float(os.getenv(env_name), thresholds.get(key, 0.5))
        return thresholds

    def _weights(self) -> Dict[str, float]:
        weights = dict(self.cfg.weights)
        weights.update(_json_dict_from_env("HF_WEIGHTS_JSON"))
        for key in _MODEL_KEYS:
            env_name = _ENV_W[key]
            if os.getenv(env_name) is not None:
                weights[key] = _safe_float(os.getenv(env_name), weights.get(key, 0.0))
        return weights

    def get_config(self) -> Dict[str, Any]:
        return {
            "hf_repo_dir": HF_REPO_DIR,
            "config_path": self.config_path,
            "thresholds": self._thresholds(),
            "weights": self._weights(),
            "models_ok": self.load_error is None,
            "load_error": self.load_error,
        }

    def update_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        patch_obj = ApiConfigPatch.model_validate(patch)
        self.cfg = apply_patch(self.cfg, patch_obj)
        save_config(self.config_path, self.cfg)
        return self.get_config()

    def reload_models(self) -> Dict[str, Any]:
        self.models = load_models(prefer_finetuned=self.prefer_finetuned)
        self.load_error = getattr(self.models, "load_error", None)
        return self.get_config()

    def predict_on_image(self, image_rgb: np.ndarray) -> HFAntiSpoofResult:
        if self.load_error:
            return HFAntiSpoofResult(ok=False, error=self.load_error)
        if image_rgb is None or not hasattr(image_rgb, "shape") or image_rgb.ndim != 3:
            return HFAntiSpoofResult(ok=False, error="Input image_rgb harus array RGB HxWx3")

        try:
            out = run_ensemble(
                self.models,
                image_rgb.astype(np.uint8),
                thresholds=self._thresholds(),
                weights=self._weights(),
            )
        except Exception as e:
            return HFAntiSpoofResult(ok=False, error=f"Anti-spoof gagal: {type(e).__name__}: {e}")

        if not out.get("ok"):
            return HFAntiSpoofResult(ok=False, error=out.get("error") or "Anti-spoof gagal", detail=out)

        ens = out.get("ensemble", {})
        real_score = float(ens.get("real_score", 0.0))
        spoof_score = float(ens.get("spoof_score", 1.0))
        real_threshold = float(ens.get("real_threshold", 0.0))
        spoof_threshold = float(ens.get("spoof_threshold", 1.0))
        is_spoof = bool(ens.get("is_spoof", real_score < real_threshold))

        out["config_used"] = {
            "thresholds": self._thresholds(),
            "weights": self._weights(),
        }
        return HFAntiSpoofResult(
            ok=True,
            is_spoof=is_spoof,
            real_score=real_score,
            spoof_score=spoof_score,
            real_threshold=real_threshold,
            spoof_threshold=spoof_threshold,
            label="Spoof" if is_spoof else "Live",
            detail=out,
        )

    def predict_on_face_crop(self, face_crop_rgb: np.ndarray) -> HFAntiSpoofResult:
        return self.predict_on_image(face_crop_rgb)
