from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms


def _default_hf_repo_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "face-anti-spoofing_hf"))


HF_REPO_DIR = os.environ.get("HF_REPO_DIR", _default_hf_repo_dir())
if HF_REPO_DIR not in sys.path:
    sys.path.insert(0, HF_REPO_DIR)


import IADG
import SASF
from infer_cdcnpp import load as load_cdcnpp_model


def _to_label_index(value: Any) -> int:
    try:
        idx = int(np.asarray(value).reshape(-1)[0])
    except Exception:
        idx = int(bool(value))
    return 1 if idx else 0


def _confidence(p: float, threshold: float) -> float:
    p = float(p)
    threshold = float(threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    if p < threshold:
        return (threshold - p) / threshold
    return (p - threshold) / (1 - threshold)


def _pick_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


class CDCNPPWrapper:
    def __init__(self, weights_path: str, threshold: float = 0.53, device: Optional[str] = None):
        self.threshold = float(threshold)
        self.model = load_cdcnpp_model(weights_path, device=device)
        self.tfm = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __call__(self, image: np.ndarray, bbox, landmark):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
        face = image if (x2 <= x1 or y2 <= y1) else image[y1:y2, x1:x2]

        t = self.tfm(face).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            out = self.model(t)
            spoof_prob = float(out["spoof_prob"].flatten()[0].detach().cpu())
        spoof_label = int(spoof_prob >= self.threshold)
        return spoof_label, spoof_prob, face


@dataclass
class LoadedModels:
    detector: Any
    sasf: Any
    flrgb: Any
    icm2o: Any
    iom2c: Any
    cdcn: Optional[Any]
    load_error: Optional[str] = None


def load_models(prefer_finetuned: bool = True) -> LoadedModels:
    try:
        finetuned_dir = os.path.join(HF_REPO_DIR, "finetuned_weights")
        weights_dir = os.path.join(HF_REPO_DIR, "weights")

        detector = IADG.aFaceDetect()
        sasf = SASF.aSASF(threshold=0.0094)
        flrgb = IADG.aSpoofONNX("modelrgb", threshold=0.0553)
        icm2o = IADG.aSpoof("ICM2O", threshold=0.564862)
        iom2c = IADG.aSpoof("IOM2C", threshold=0.218523)

        cdcn_weights = _pick_existing_file(
            [
                os.path.join(finetuned_dir, "cdcnpp.pth"),
                os.path.join(weights_dir, "cdcnpp.pth"),
            ]
        )
        cdcn = CDCNPPWrapper(cdcn_weights, threshold=0.53) if cdcn_weights else None

        if prefer_finetuned:
            for model, fname in ((icm2o, "ICM2O_finetuned.pth"), (iom2c, "IOM2C_finetuned.pth")):
                ft_path = os.path.join(finetuned_dir, fname)
                if os.path.isfile(ft_path):
                    try:
                        try:
                            state = torch.load(ft_path, map_location="cpu", weights_only=True)
                        except TypeError:
                            state = torch.load(ft_path, map_location="cpu")
                        model.model.load_state_dict(state, strict=False)
                        model.model.eval()
                    except Exception:
                        pass

        return LoadedModels(detector, sasf, flrgb, icm2o, iom2c, cdcn, None)
    except Exception as e:
        return LoadedModels(None, None, None, None, None, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _run_single_model(model, image_rgb, bbox, landmark):
    spoof, prob, crop = model(image_rgb, bbox, landmark)
    return _to_label_index(spoof), float(prob), crop


def run_ensemble(
    models: LoadedModels,
    image_rgb: np.ndarray,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    if models.load_error:
        raise RuntimeError(models.load_error)

    bboxes, landmarks = models.detector(image_rgb)
    if len(landmarks) < 1:
        return {"ok": False, "error": "No face detected."}

    bbox, landmark = bboxes[0], landmarks[0]

    models.sasf.threshold = float(thresholds.get("sasf", 0.70))
    models.flrgb.threshold = float(thresholds.get("flrgb", 0.45))
    models.icm2o.threshold = float(thresholds.get("icm2o", 0.564862))
    models.iom2c.threshold = float(thresholds.get("iom2c", 0.218523))
    if models.cdcn is not None:
        models.cdcn.threshold = float(thresholds.get("cdcn", 0.53))

    tasks = {
        "sasf": (models.sasf, image_rgb, bbox, landmark),
        "flrgb": (models.flrgb, image_rgb, bbox, landmark),
        "icm2o": (models.icm2o, image_rgb, bbox, landmark),
        "iom2c": (models.iom2c, image_rgb, bbox, landmark),
    }
    if models.cdcn is not None:
        tasks["cdcn"] = (models.cdcn, image_rgb, bbox, landmark)

    active_keys = list(tasks.keys())
    total_raw = float(sum(float(weights.get(k, 0.0)) for k in active_keys)) or 1.0
    norm_w = {k: float(weights.get(k, 0.0)) / total_raw for k in active_keys}

    results: Dict[str, Optional[Tuple[int, float, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as ex:
        futures = {ex.submit(_run_single_model, *args): k for k, args in tasks.items()}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                results[k] = fut.result()
            except Exception:
                results[k] = None

    per_model: Dict[str, Any] = {}
    usable = []
    for k in active_keys:
        thr = float(thresholds.get(k, 0.5))
        if results.get(k) is None:
            per_model[k] = {"ok": False, "error": "model_failed"}
            continue
        spoof_label, spoof_prob, _ = results[k]
        usable.append(k)
        per_model[k] = {
            "ok": True,
            "spoof_prob": float(spoof_prob),
            "spoof_label": int(spoof_label),
            "threshold": thr,
            "confidence": float(_confidence(spoof_prob, thr)),
            "weight": float(norm_w.get(k, 0.0)),
        }

    if not usable:
        return {"ok": False, "error": "All models failed."}

    total_w = float(sum(float(weights.get(k, 0.0)) for k in usable)) or 1.0
    norm_w_usable = {k: float(weights.get(k, 0.0)) / total_w for k in usable}

    ensemble_spoof_score = float(sum(norm_w_usable[k] * results[k][1] for k in usable))
    ensemble_spoof_thr = float(sum(norm_w_usable[k] * float(thresholds.get(k, 0.5)) for k in usable))
    ensemble_real_score = 1.0 - ensemble_spoof_score
    ensemble_real_thr = 1.0 - ensemble_spoof_thr
    is_spoof = bool(ensemble_real_score < ensemble_real_thr)

    return {
        "ok": True,
        "bbox": [float(x) for x in bbox[:4]],
        "landmarks": np.asarray(landmark).reshape(-1).astype(float).tolist(),
        "ensemble": {
            "real_score": float(ensemble_real_score),
            "real_threshold": float(ensemble_real_thr),
            "spoof_score": float(ensemble_spoof_score),
            "spoof_threshold": float(ensemble_spoof_thr),
            "is_spoof": bool(is_spoof),
            "label": "Spoof" if is_spoof else "Live",
            "confidence": float(_confidence(ensemble_real_score, ensemble_real_thr)),
        },
        "per_model": per_model,
    }


def decode_image_to_rgb(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("File bukan image yang valid / gagal dibaca oleh OpenCV.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
