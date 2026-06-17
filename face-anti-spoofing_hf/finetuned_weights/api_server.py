from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api_config import ApiConfig, ApiConfigPatch, apply_patch, load_config, save_config
from api_inference import HF_REPO_DIR, decode_image_to_rgb, load_models, run_ensemble


def _import_temporal_from_hf():
    if HF_REPO_DIR not in sys.path:
        sys.path.insert(0, HF_REPO_DIR)
    from liveness_temporal import TemporalLivenessChecker, fuse_with_spoof_score  # noqa: E402

    return TemporalLivenessChecker, fuse_with_spoof_score


TemporalLivenessChecker, fuse_with_spoof_score = _import_temporal_from_hf()


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_settings.json")

app = FastAPI(title="Face Anti-Spoofing API (No UI)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _State:
    cfg: ApiConfig
    models: Any


STATE = _State()


@app.on_event("startup")
def _startup() -> None:
    STATE.cfg = load_config(CONFIG_PATH)
    STATE.models = load_models(prefer_finetuned=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "hf_repo_dir": HF_REPO_DIR,
        "models_ok": bool(getattr(STATE.models, "load_error", None) is None),
        "error": getattr(STATE.models, "load_error", None),
    }


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return STATE.cfg.model_dump()


@app.put("/config")
def update_config(patch: ApiConfigPatch) -> Dict[str, Any]:
    STATE.cfg = apply_patch(STATE.cfg, patch)
    save_config(CONFIG_PATH, STATE.cfg)
    return STATE.cfg.model_dump()


@app.post("/models/reload")
def reload_models() -> Dict[str, Any]:
    STATE.models = load_models(prefer_finetuned=True)
    if STATE.models.load_error:
        raise HTTPException(status_code=500, detail=STATE.models.load_error)
    return {"ok": True}


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    thresholds: Optional[str] = None,
    weights: Optional[str] = None,
) -> Dict[str, Any]:
    raw = await file.read()
    try:
        img_rgb = decode_image_to_rgb(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    thr = dict(STATE.cfg.thresholds)
    w = dict(STATE.cfg.weights)

    import json

    if thresholds:
        try:
            thr.update(json.loads(thresholds))
        except Exception:
            raise HTTPException(status_code=400, detail="thresholds harus JSON string yang valid.")
    if weights:
        try:
            w.update(json.loads(weights))
        except Exception:
            raise HTTPException(status_code=400, detail="weights harus JSON string yang valid.")

    try:
        out = run_ensemble(STATE.models, img_rgb, thresholds=thr, weights=w)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out["config_used"] = {"thresholds": thr, "weights": w}
    return out


@app.post("/predict/frames")
async def predict_frames(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Kirim minimal 2 frame.")

    frames_rgb = []
    for f in files:
        raw = await f.read()
        try:
            frames_rgb.append(decode_image_to_rgb(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {e}")

    thr = dict(STATE.cfg.thresholds)
    w = dict(STATE.cfg.weights)

    try:
        ensemble_out = run_ensemble(STATE.models, frames_rgb[-1], thresholds=thr, weights=w)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not STATE.cfg.enable_temporal:
        ensemble_out["temporal"] = {"enabled": False}
        ensemble_out["config_used"] = {"thresholds": thr, "weights": w}
        return ensemble_out

    checker = TemporalLivenessChecker()
    no_face = 0
    multi_face = 0
    for fr in frames_rgb:
        bboxes, landmarks = STATE.models.detector(fr)
        if len(landmarks) == 0:
            no_face += 1
            continue
        if len(landmarks) > 1:
            multi_face += 1
            continue
        checker.add_frame(landmarks[0])

    temporal = checker.evaluate()
    spoof_score = float(ensemble_out.get("ensemble", {}).get("spoof_score", 1.0))

    spoof_weight = 1.0 - float(STATE.cfg.temporal_weight)
    is_live, fused_score, reason = fuse_with_spoof_score(
        spoof_score=spoof_score,
        temporal_result=temporal,
        spoof_weight=spoof_weight,
        temporal_weight=float(STATE.cfg.temporal_weight),
    )

    ensemble_out["temporal"] = {
        "enabled": True,
        "no_face_frames": no_face,
        "multi_face_frames": multi_face,
        "frame_count_used": temporal.frame_count,
        "motion_score": temporal.score,
        "motion_is_live": temporal.is_live,
        "fused_is_live": is_live,
        "fused_score": fused_score,
        "reason": reason,
    }
    ensemble_out["config_used"] = {"thresholds": thr, "weights": w}
    return ensemble_out

