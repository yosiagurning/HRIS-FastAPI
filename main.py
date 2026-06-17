import io
import os
import json
import time
import logging
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms, models
from contextlib import asynccontextmanager
from facenet_pytorch import InceptionResnetV1, MTCNN
from dotenv import load_dotenv

load_dotenv()

from api_config import ApiConfigPatch
from hf_antispoof_ensemble import HFAntiSpoofEnsemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("face-service")

MODEL_PATH = os.getenv("MODEL_PATH", "models/facenet_labersa_cpu.pt")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "160"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
FINAL_SCORE_THRESHOLD = float(os.getenv("FINAL_SCORE_THRESHOLD", "0.80"))
ANTI_SPOOFING_ENABLED = os.getenv("ANTI_SPOOFING_ENABLED", "1") == "1"
HF_REPO_DIR = os.getenv("HF_REPO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "face-anti-spoofing_hf"))
ANTI_SPOOF_MODEL_PATH = os.getenv("ANTI_SPOOF_MODEL_PATH", "HF_ENSEMBLE")
ANTI_SPOOF_THRESHOLD = float(os.getenv("ANTI_SPOOF_THRESHOLD", "0.0"))
ANTI_SPOOF_REAL_THRESHOLD = float(os.getenv("ANTI_SPOOF_REAL_THRESHOLD", "0.0"))
SPOOF_SCORE_THRESHOLD = float(os.getenv("SPOOF_SCORE_THRESHOLD", "0.65"))
FACE_DET_MIN_PROB = float(os.getenv("FACE_DET_MIN_PROB", "0.90"))
DEVICE = os.getenv("DEVICE", "cpu")
TORCH_DEVICE = torch.device("cuda" if str(DEVICE).startswith("cuda") and torch.cuda.is_available() else "cpu")
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.15"))
ANTI_SPOOF_IMGSZ = int(os.getenv("ANTI_SPOOF_IMGSZ", "224"))
LIVENESS_ENABLED = os.getenv("LIVENESS_ENABLED", "1") == "1"
LIVENESS_MIN_FRAMES = int(os.getenv("LIVENESS_MIN_FRAMES", "3"))
LIVENESS_STD_MEAN_THR = float(os.getenv("LIVENESS_STD_MEAN_THR", "0.008"))
LIVENESS_YAW_RANGE_THR = float(os.getenv("LIVENESS_YAW_RANGE_THR", "0.08"))
LIVENESS_MAX_FRAMES = int(os.getenv("LIVENESS_MAX_FRAMES", "6"))
SCREEN_SPOOF_ENABLED = os.getenv("SCREEN_SPOOF_ENABLED", "1") == "1"
SCREEN_RECT_MIN_AREA_RATIO = float(os.getenv("SCREEN_RECT_MIN_AREA_RATIO", "0.25"))
SCREEN_RECT_MAX_AREA_RATIO = float(os.getenv("SCREEN_RECT_MAX_AREA_RATIO", "0.95"))
SCREEN_RECT_ASPECT_MIN = float(os.getenv("SCREEN_RECT_ASPECT_MIN", "0.35"))
SCREEN_RECT_ASPECT_MAX = float(os.getenv("SCREEN_RECT_ASPECT_MAX", "0.85"))
SCREEN_BORDER_DARK_MAX = float(os.getenv("SCREEN_BORDER_DARK_MAX", "85"))
SCREEN_BORDER_DARK_DIFF = float(os.getenv("SCREEN_BORDER_DARK_DIFF", "25"))
API_KEY = os.getenv("FACE_API_KEY", "labersa-internal-api-key-2026")


class FaceNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2", classify=False)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        return nn.functional.normalize(self.backbone(x), p=2, dim=1)


class LightClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AntiSpoofNet(nn.Module):
    def __init__(self, dropout: float = 0.35, num_classes: int = 2):
        super().__init__()
        try:
            self.backbone = models.resnet18(weights=None)
        except TypeError:
            self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class FaceService:
    def __init__(self):
        self.extractor = None
        self.classifier = None
        self.class_names = []
        self.loaded = False
        self.face_det = None
        self.anti_spoof = None
        self.hf_antispoof = None
        self.anti_spoof_labels = []
        self.anti_spoof_real_index = 0
        self.anti_spoof_spoof_index = 1
        self.anti_spoof_threshold = ANTI_SPOOF_THRESHOLD
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.anti_spoof_transform = transforms.Compose([
            transforms.Resize((ANTI_SPOOF_IMGSZ, ANTI_SPOOF_IMGSZ)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("File gambar tidak valid atau rusak")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def load(self):
        try:
            self.face_det = MTCNN(
                keep_all=True,
                device=TORCH_DEVICE,
            )
            logger.info("MTCNN face detector loaded successfully")
            self.extractor = FaceNetExtractor().to(TORCH_DEVICE).eval()
            if os.path.exists(MODEL_PATH):
                ckpt = torch.load(MODEL_PATH, map_location="cpu")
                self.class_names = ckpt.get("class_names", [])
                if "classifier_state_dict" in ckpt and self.class_names:
                    self.classifier = LightClassifier(len(self.class_names)).to(TORCH_DEVICE).eval()
                    self.classifier.load_state_dict(ckpt["classifier_state_dict"])
                logger.info(f"Model loaded: {len(self.class_names)} classes")
            else:
                logger.warning(f"Model tidak ditemukan di {MODEL_PATH}, pakai pretrained VGGFace2")
            if ANTI_SPOOFING_ENABLED:
                self.hf_antispoof = HFAntiSpoofEnsemble(prefer_finetuned=True)
                if self.hf_antispoof.load_error:
                    raise RuntimeError(self.hf_antispoof.load_error)
                cfg = self.hf_antispoof.get_config()
                logger.info(
                    "HF anti-spoof ensemble loaded successfully. "
                    f"hf_repo_dir={cfg.get('hf_repo_dir')} thresholds={cfg.get('thresholds')} weights={cfg.get('weights')}"
                )
            self.loaded = True
        except Exception as e:
            logger.error(f"Gagal load model: {e}")
            raise

    def detect_faces(self, image_bytes: bytes) -> Tuple[bool, int, list]:
        try:
            if self.face_det is None:
                raise ValueError("Face detector belum diinisialisasi")
            img = self._decode_image(image_bytes)
            boxes, probs = self.face_det.detect(img)
            if boxes is None or probs is None:
                return False, 0, []
            valid_boxes = []
            for box, prob in zip(boxes, probs):
                if prob is None or float(prob) < FACE_DET_MIN_PROB:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                if (x2 - x1) >= 50 and (y2 - y1) >= 50:
                    valid_boxes.append([x1, y1, x2, y2])
            return len(valid_boxes) > 0, len(valid_boxes), valid_boxes
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return False, 0, []

    def detect_faces_with_scores(self, image_bytes: bytes) -> List[Tuple[np.ndarray, float]]:
        if self.face_det is None:
            raise ValueError("Face detector belum diinisialisasi")
        img = self._decode_image(image_bytes)
        boxes, probs = self.face_det.detect(img)
        out: List[Tuple[np.ndarray, float]] = []
        if boxes is None or probs is None:
            return out
        for box, prob in zip(boxes, probs):
            if prob is None or float(prob) < FACE_DET_MIN_PROB:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue
            out.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(prob)))
        return out

    def _crop_largest_face_rgb(self, image_bytes: bytes, boxes: list) -> Tuple[np.ndarray, List[float]]:
        img = self._decode_image(image_bytes)
        if not boxes:
            raise ValueError("Tidak ada wajah terdeteksi dalam foto")
        largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = [int(round(float(v))) for v in largest_box]
        height, width = img.shape[:2]
        margin_x = int(round((x2 - x1) * FACE_CROP_MARGIN))
        margin_y = int(round((y2 - y1) * FACE_CROP_MARGIN))
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        face = img[y1:y2, x1:x2]
        return face, [float(x1), float(y1), float(x2), float(y2)]

    @torch.no_grad()
    def detect_spoof(self, face_crop_rgb: np.ndarray, full_image_rgb: Optional[np.ndarray] = None) -> Tuple[bool, float, float]:
        if not ANTI_SPOOFING_ENABLED:
            return True, 1.0, 0.0
        if self.hf_antispoof is None:
            raise ValueError("HF anti-spoof ensemble belum diinisialisasi")
        image_for_spoof = full_image_rgb if full_image_rgb is not None else face_crop_rgb
        res = self.hf_antispoof.predict_on_image(image_for_spoof)
        if (not res.ok) and (full_image_rgb is not None):
            res = self.hf_antispoof.predict_on_face_crop(face_crop_rgb)
        if not res.ok:
            raise ValueError(res.error or "Anti-spoof gagal")
        self.anti_spoof_threshold = float(res.real_threshold)
        is_real = (not res.is_spoof) and (float(res.real_score) >= float(ANTI_SPOOF_REAL_THRESHOLD))
        return bool(is_real), float(res.real_score), float(res.spoof_score)

    @torch.no_grad()
    def extract_embedding_from_crop(self, face_crop_rgb: np.ndarray) -> list[float]:
        if self.extractor is None:
            raise ValueError("Extractor belum diinisialisasi")
        img = Image.fromarray(face_crop_rgb.astype(np.uint8))
        tensor = self.transform(img).unsqueeze(0).to(TORCH_DEVICE)
        emb = self.extractor(tensor)[0].detach().cpu()
        return emb.tolist()

    def motion_liveness(self, boxes_seq: List[np.ndarray]) -> Tuple[bool, Dict[str, Any]]:
        if not LIVENESS_ENABLED:
            return True, {"enabled": False}
        if len(boxes_seq) < LIVENESS_MIN_FRAMES:
            return False, {
                "enabled": True,
                "reason": "not_enough_frames",
                "min_frames": LIVENESS_MIN_FRAMES,
                "frames": len(boxes_seq),
            }
        feats = []
        cx_seq = []
        for b in boxes_seq:
            x1, y1, x2, y2 = [float(v) for v in b.tolist()]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            s = float(np.sqrt(bw * bh) + 1e-9)
            cx = float((x1 + x2) * 0.5) / s
            cy = float((y1 + y2) * 0.5) / s
            ar = float(bw / (bh + 1e-9))
            feats.append(np.array([cx, cy, ar], dtype=np.float32))
            cx_seq.append(cx)
        feats = np.stack(feats, axis=0)
        std = feats.std(axis=0)
        cx_range = float(np.max(cx_seq) - np.min(cx_seq))
        std_mean = float(std.mean())
        live = bool((std_mean >= LIVENESS_STD_MEAN_THR) and (cx_range >= LIVENESS_YAW_RANGE_THR))
        return live, {
            "enabled": True,
            "reason": "ok" if live else "liveness_fail",
            "feat_std_mean": std_mean,
            "cx_range": cx_range,
            "thr_feat_std_mean": LIVENESS_STD_MEAN_THR,
            "thr_cx_range": LIVENESS_YAW_RANGE_THR,
            "frames": len(boxes_seq),
        }

    def screen_spoof_check(self, img_rgb: np.ndarray, face_box_xyxy: List[float]) -> Tuple[bool, Dict[str, Any]]:
        if not SCREEN_SPOOF_ENABLED:
            return False, {"enabled": False}
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, {"enabled": True, "reason": "no_contours"}
        fx1, fy1, fx2, fy2 = [float(v) for v in face_box_xyxy]
        cx = float((fx1 + fx2) * 0.5)
        cy = float((fy1 + fy2) * 0.5)
        img_area = float(h * w + 1e-9)
        best = None
        best_area = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < img_area * 0.05:
                continue
            peri = float(cv2.arcLength(cnt, True))
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            rect = cv2.minAreaRect(approx)
            (rw, rh) = rect[1]
            rw = float(rw)
            rh = float(rh)
            if rw <= 1 or rh <= 1:
                continue
            aspect = min(rw, rh) / max(rw, rh)
            area_ratio = float((rw * rh) / img_area)
            if not (SCREEN_RECT_MIN_AREA_RATIO <= area_ratio <= SCREEN_RECT_MAX_AREA_RATIO):
                continue
            if not (SCREEN_RECT_ASPECT_MIN <= aspect <= SCREEN_RECT_ASPECT_MAX):
                continue
            box = cv2.boxPoints(rect)
            inside = cv2.pointPolygonTest(box.astype(np.float32), (cx, cy), False)
            if inside < 0:
                continue
            if area > best_area:
                best_area = area
                best = (box, area_ratio, aspect)
        if best is None:
            return False, {"enabled": True, "reason": "no_rect_candidate"}
        box, area_ratio, aspect = best
        poly = box.astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        band = max(2, int(round(0.012 * min(h, w))))
        k = 2 * band + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        inner = cv2.erode(mask, kernel, iterations=1)
        border = cv2.subtract(mask, inner)
        border_px = gray[border > 0]
        if border_px.size == 0:
            return False, {"enabled": True, "reason": "no_border_pixels", "area_ratio": round(area_ratio, 4), "aspect": round(aspect, 4)}
        border_mean = float(border_px.mean())
        overall_mean = float(gray.mean())
        dark = bool((border_mean <= SCREEN_BORDER_DARK_MAX) and ((overall_mean - border_mean) >= SCREEN_BORDER_DARK_DIFF))
        detail = {
            "enabled": True,
            "reason": "screen_like_rect" if dark else "rect_not_dark",
            "area_ratio": round(area_ratio, 4),
            "aspect": round(aspect, 4),
            "border_mean": round(border_mean, 2),
            "overall_mean": round(overall_mean, 2),
            "thr_border_max": SCREEN_BORDER_DARK_MAX,
            "thr_border_diff": SCREEN_BORDER_DARK_DIFF,
        }
        return dark, detail

    def _lbp_hist(self, gray_u8: np.ndarray) -> np.ndarray:
        g = gray_u8.astype(np.uint8)
        if g.shape[0] < 3 or g.shape[1] < 3:
            return np.zeros(256, dtype=np.float64)
        c = g[1:-1, 1:-1]
        code = np.zeros_like(c, dtype=np.uint8)
        code |= ((g[:-2, :-2] >= c) << 7).astype(np.uint8)
        code |= ((g[:-2, 1:-1] >= c) << 6).astype(np.uint8)
        code |= ((g[:-2, 2:] >= c) << 5).astype(np.uint8)
        code |= ((g[1:-1, 2:] >= c) << 4).astype(np.uint8)
        code |= ((g[2:, 2:] >= c) << 3).astype(np.uint8)
        code |= ((g[2:, 1:-1] >= c) << 2).astype(np.uint8)
        code |= ((g[2:, :-2] >= c) << 1).astype(np.uint8)
        code |= ((g[1:-1, :-2] >= c) << 0).astype(np.uint8)
        hist = np.bincount(code.reshape(-1), minlength=256).astype(np.float64)
        hist /= (hist.sum() + 1e-12)
        return hist

    def _entropy(self, p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def _laplacian_var(self, gray_u8: np.ndarray) -> float:
        g = gray_u8.astype(np.float32)
        if g.shape[0] < 3 or g.shape[1] < 3:
            return 0.0
        lap = (-4.0 * g[1:-1, 1:-1] +
               g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        return float(lap.var())

    def _fft_peak_ratio(self, gray_u8: np.ndarray) -> float:
        g = gray_u8.astype(np.float32) / 255.0
        g = g - g.mean()
        f = np.fft.fftshift(np.fft.fft2(g))
        mag = np.log1p(np.abs(f))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        r0 = max(6, int(0.06 * min(h, w)))
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 >= (r0 * r0)
        high = mag[mask]
        if high.size == 0:
            return 0.0
        return float(high.max() / (high.mean() + 1e-9))

    def anti_spoof_check(self, image_bytes: bytes, boxes: list) -> Tuple[bool, str, Dict[str, Any]]:
        if not ANTI_SPOOFING_ENABLED:
            return True, "ok", {"enabled": False}
        if not boxes:
            return False, "Tidak ada wajah terdeteksi (anti-spoofing)", {"enabled": True}
        img = self._decode_image(image_bytes)
        largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        screen_is_spoof, screen_detail = self.screen_spoof_check(img, [float(v) for v in largest_box])
        if screen_is_spoof:
            return False, "Terdeteksi tampilan layar/HP di depan kamera. Harap ambil selfie langsung dari kamera.", {
                "enabled": True,
                "reason": "screen_spoof",
                "screen": screen_detail,
            }
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in largest_box]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 40 or y2 - y1 < 40:
            return False, "Wajah terlalu kecil untuk verifikasi (anti-spoofing)", {"enabled": True}
        face = img[y1:y2, x1:x2]
        gray = (0.299 * face[..., 0] + 0.587 * face[..., 1] + 0.114 * face[..., 2]).astype(np.uint8)
        lap = self._laplacian_var(gray)
        hist = self._lbp_hist(gray)
        ent = self._entropy(hist)
        peak = self._fft_peak_ratio(gray)
        s_blur = np.clip((18.0 - lap) / 18.0, 0.0, 1.0)
        s_flat = np.clip((4.6 - ent) / 1.2, 0.0, 1.0)
        s_peak = np.clip((peak - 8.0) / 10.0, 0.0, 1.0)
        score = float(np.clip(0.45 * s_blur + 0.35 * s_flat + 0.20 * s_peak, 0.0, 1.0))
        detail = {
            "enabled": True,
            "spoof_score": round(score, 4),
            "lap_var": round(lap, 4),
            "lbp_entropy": round(ent, 4),
            "fft_peak_ratio": round(peak, 4),
            "threshold": SPOOF_SCORE_THRESHOLD,
            "screen": screen_detail,
        }
        if score >= SPOOF_SCORE_THRESHOLD:
            return False, "Foto terindikasi spoofing (foto cetak/layar). Harap ambil selfie langsung dari kamera.", detail
        return True, "ok", detail

    def detect_mask(self, face_region: np.ndarray) -> Tuple[bool, str]:
        try:
            height, width = face_region.shape[:2]
            mouth_region_y1 = int(height * 0.62)
            mouth_region_y2 = int(height * 0.95)
            mouth_region = face_region[mouth_region_y1:mouth_region_y2, :]
            if mouth_region.size == 0:
                return False, "Region mulut tidak valid"
            skin_region_y1 = int(height * 0.3)
            skin_region_y2 = int(height * 0.5)
            skin_region = face_region[skin_region_y1:skin_region_y2, int(width*0.3):int(width*0.7)]
            hsv_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2HSV)
            hsv_skin = cv2.cvtColor(skin_region, cv2.COLOR_RGB2HSV) if skin_region.size > 0 else None
            mouth_hue_mean = np.mean(hsv_mouth[:,:,0])
            mouth_sat_mean = np.mean(hsv_mouth[:,:,1])
            mouth_val_mean = np.mean(hsv_mouth[:,:,2])
            if hsv_skin is not None and hsv_skin.size > 0:
                skin_hue_mean = np.mean(hsv_skin[:,:,0])
                skin_sat_mean = np.mean(hsv_skin[:,:,1])
                skin_val_mean = np.mean(hsv_skin[:,:,2])
                hue_diff = abs(mouth_hue_mean - skin_hue_mean)
                sat_diff = abs(mouth_sat_mean - skin_sat_mean)
                val_diff = abs(mouth_val_mean - skin_val_mean)
            else:
                hue_diff, sat_diff, val_diff = 30, 50, 50
            total_pixels = mouth_region.shape[0] * mouth_region.shape[1]
            blue_mask = np.sum((hsv_mouth[:,:,0] > 85) & (hsv_mouth[:,:,0] < 135) & 
                            (hsv_mouth[:,:,1] > 60) & (hsv_mouth[:,:,2] > 50))
            blue_ratio = blue_mask / total_pixels
            green_mask = np.sum((hsv_mouth[:,:,0] > 35) & (hsv_mouth[:,:,0] < 90) & 
                            (hsv_mouth[:,:,1] > 50) & (hsv_mouth[:,:,2] > 50))
            green_ratio = green_mask / total_pixels
            black_mask = np.sum((hsv_mouth[:,:,2] < 55) & (hsv_mouth[:,:,1] < 60))
            black_ratio = black_mask / total_pixels
            white_mask = np.sum((hsv_mouth[:,:,2] > 190) & (hsv_mouth[:,:,1] < 35))
            white_ratio = white_mask / total_pixels
            grey_mask = np.sum((hsv_mouth[:,:,1] < 25) & (hsv_mouth[:,:,2] > 80) & (hsv_mouth[:,:,2] < 180))
            grey_ratio = grey_mask / total_pixels
            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray_mouth, cv2.CV_64F).var()
            edges = cv2.Canny(gray_mouth, 40, 120)
            edge_density = np.sum(edges > 0) / total_pixels if total_pixels > 0 else 0
            lab_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2LAB)
            a_channel = lab_mouth[:,:,1]
            lip_mask = a_channel > 145 
            lip_ratio = np.sum(lip_mask) / total_pixels if total_pixels > 0 else 0
            COLOR_THR = 0.22
            LAPLACIAN_THR = 80.0 
            LIP_THR = 0.08      
            if blue_ratio > COLOR_THR: return True, f"Terdeteksi masker"
            if green_ratio > COLOR_THR: return True, f"Terdeteksi masker"
            if black_ratio > 0.35: return True, f"Terdeteksi masker"
            if white_ratio > 0.30: return True, f"Terdeteksi masker"
            if grey_ratio > 0.30: return True, f"Terdeteksi masker"
            if laplacian_var < LAPLACIAN_THR and lip_ratio < LIP_THR:
                if sat_diff > 35 or val_diff > 40: 
                    return True, f"Area mulut tertutup material halus (var={laplacian_var:.1f})"
            if hue_diff > 25 and sat_diff > 45 and lip_ratio < 0.05:
                return True, "Warna area mulut tidak wajar (kemungkinan masker kain)"
            logger.info(f"[MASK] b={blue_ratio:.2f}, k={black_ratio:.2f}, w={white_ratio:.2f}, lip={lip_ratio:.2f}, lap={laplacian_var:.1f}")
            return False, "Tidak terdeteksi masker"
        except Exception as e:
            logger.error(f"Error detecting mask: {e}")
            return False, "Error deteksi masker"

    def detect_hat(self, face_region: np.ndarray, full_image: np.ndarray, box: list) -> Tuple[bool, str]:
        try:
            x1, y1, x2, y2 = [int(b) for b in box]
            head_h = int((y2 - y1) * 0.45)
            head_top = max(0, y1 - head_h)
            head_region = full_image[head_top:y1, x1:x2]
            if head_region.size == 0:
                return False, "Region kepala tidak valid"
            total_pixels = head_region.shape[0] * head_region.shape[1]
            if total_pixels < 50:
                return False, "Region terlalu kecil"
            gray_head = cv2.cvtColor(head_region, cv2.COLOR_RGB2GRAY)
            hsv_head  = cv2.cvtColor(head_region, cv2.COLOR_RGB2HSV)
            edges = cv2.Canny(gray_head, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                                    minLineLength=int(head_region.shape[1] * 0.4),
                                    maxLineGap=15)
            hue, sat, val = hsv_head[:,:,0], hsv_head[:,:,1], hsv_head[:,:,2]
            is_natural_hair = (
                (sat < 65) |                                  
                ((hue >= 5) & (hue <= 35) & (sat < 160)) |    
                (val < 45)                                    
            )
            non_hair_mask = ~is_natural_hair
            non_hair_ratio = np.sum(non_hair_mask) / total_pixels
            if np.sum(non_hair_mask) > 10:
                color_std = np.std(hue[non_hair_mask])
            else:
                color_std = 100 
            if edge_density > 0.15 and lines is not None and len(lines) >= 2:
                return True, f"Terdeteksi topi atau aksesoris kepala"
            if non_hair_ratio > 0.45 and color_std < 25:
                return True, f"Terdeteksi topi atau aksesoris"
            bright_color = np.sum(non_hair_mask & (sat > 150) & (val > 100)) / total_pixels
            if bright_color > 0.25:
                return True, f"Terdeteksi topi atau aksesoris "
            laplacian_head = cv2.Laplacian(gray_head, cv2.CV_64F).var()
            if laplacian_head < 40 and non_hair_ratio > 0.5:
                return True, "Terdeteksi topi atau aksesoris "
            logger.info(f"[HAT] edge={edge_density:.2f}, non_hair={non_hair_ratio:.2f}, std={color_std:.1f}, lap={laplacian_head:.1f}")
            return False, "Tidak terdeteksi topi"
        except Exception as e:
            logger.error(f"Error detecting hat: {e}")
            return False, "Error deteksi topi"

    def check_accessories(self, image_bytes: bytes, boxes: list) -> Tuple[bool, str]:
        try:
            img_array = self._decode_image(image_bytes)
            if len(boxes) == 0:
                return False, "Tidak ada wajah terdeteksi"
            largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            x1, y1, x2, y2 = [int(b) for b in largest_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_array.shape[1], x2), min(img_array.shape[0], y2)
            face_region = img_array[y1:y2, x1:x2]
            if face_region.size == 0:
                return False, "Region wajah tidak valid"
            has_mask, mask_msg = self.detect_mask(face_region)
            if has_mask:
                return False, f"{mask_msg}. Harap lepas masker."
            has_hat, hat_msg = self.detect_hat(face_region, img_array, largest_box)
            if has_hat:
                return False, f"{hat_msg}. Harap lepas topi/aksesoris kepala."
            return True, "Wajah valid, tidak ada aksesoris terdeteksi"
        except Exception as e:
            logger.error(f"Error checking accessories: {e}")
            return True, "Tidak dapat memeriksa aksesoris"

    @torch.no_grad()
    def extract_embedding(self, image_bytes: bytes) -> list[float]:
        has_face, face_count, boxes = self.detect_faces(image_bytes)
        if not has_face:
            raise ValueError("Tidak ada wajah terdeteksi dalam foto")
        if face_count > 1:
            raise ValueError(f"Terdeteksi {face_count} wajah. Hanya satu wajah yang diperbolehkan")
        is_valid, message = self.check_accessories(image_bytes, boxes)
        if not is_valid:
            raise ValueError(message)
        face_crop, _ = self._crop_largest_face_rgb(image_bytes, boxes)
        return self.extract_embedding_from_crop(face_crop)

    def cosine_similarity(self, emb1: list, emb2: list) -> float:
        a = np.array(emb1, dtype=np.float32)
        b = np.array(emb2, dtype=np.float32)
        a /= (np.linalg.norm(a) + 1e-8)
        b /= (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    @torch.no_grad()
    def verify(
        self,
        image_bytes: bytes,
        stored_embedding: list[float],
        threshold: float = None,
    ) -> dict:
        thr = threshold or SIMILARITY_THRESHOLD
        try:
            has_face, face_count, boxes = self.detect_faces(image_bytes)
            if not has_face:
                raise ValueError("Tidak ada wajah terdeteksi dalam foto")
            if face_count > 1:
                raise ValueError(f"Terdeteksi {face_count} wajah. Hanya satu wajah yang diperbolehkan")
            is_valid, message = self.check_accessories(image_bytes, boxes)
            if not is_valid:
                raise ValueError(message)
            face_crop, box_xyxy = self._crop_largest_face_rgb(image_bytes, boxes)
            img_rgb = self._decode_image(image_bytes)
            if SCREEN_SPOOF_ENABLED:
                screen_is_spoof, _ = self.screen_spoof_check(img_rgb, box_xyxy)
                if screen_is_spoof:
                    return {
                        "matched": False,
                        "similarity": 0.0,
                        "spoof_score": 0.0,
                        "final_score": 0.0,
                        "confidence": 0.0,
                        "threshold": thr,
                        "message": "Terdeteksi Kecurangan",
                    }
            is_real, real_score, spoof_score = self.detect_spoof(face_crop, img_rgb)
            if not is_real:
                return {
                    "matched": False,
                    "similarity": 0.0,
                    "real_score": round(float(real_score), 4),
                    "spoof_score": round(float(spoof_score), 4),
                    "anti_spoof_threshold": round(float(self.anti_spoof_threshold), 4),
                    "final_score": round(0.3 * float(real_score), 4),
                    "confidence": round(0.3 * float(real_score), 4),
                    "threshold": thr,
                    "message": "Terdeteksi Kecurangan",
                }
            live_emb = torch.tensor(self.extract_embedding_from_crop(face_crop), dtype=torch.float32)
            stored = torch.tensor(stored_embedding, dtype=torch.float32)
            similarity = float(F.cosine_similarity(live_emb, stored, dim=0).item())
            final_score = float((0.7 * similarity) + (0.3 * float(real_score)))
            matched = bool(
                (similarity >= float(thr))
                and is_real
                and (final_score >= float(FINAL_SCORE_THRESHOLD))
            )
            msg = "Wajah cocok" if matched else "Wajah tidak valid"
            return {
                "matched": matched,
                "similarity": round(similarity, 4),
                "real_score": round(float(real_score), 4),
                "spoof_score": round(float(spoof_score), 4),
                "anti_spoof_threshold": round(float(self.anti_spoof_threshold), 4),
                "final_score": round(final_score, 4),
                "confidence": round(final_score, 4),
                "threshold": float(thr),
                "message": msg,
            }
        except ValueError as e:
            return {
                "matched": False,
                "similarity": 0.0,
                "spoof_score": 0.0,
                "final_score": 0.0,
                "confidence": 0.0,
                "threshold": float(thr),
                "message": str(e),
            }


face_svc = FaceService()


class VerifyRequest(BaseModel):
    stored_embedding: list[float]
    employee_id: str
    threshold: Optional[float] = None


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "API Key tidak valid")
    return x_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Memuat model FaceNet...")
    face_svc.load()
    logger.info("✅ Face Recognition Service siap")
    yield
    logger.info("Service berhenti.")


app = FastAPI(
    title="Face Recognition — Hotel Labersa Toba",
    description="Internal face recognition service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", summary="Status service")
def health():
    return {
        "status": "ok",
        "model_loaded": face_svc.loaded,
        "num_classes": len(face_svc.class_names),
        "threshold": SIMILARITY_THRESHOLD,
        "anti_spoof_threshold": getattr(face_svc, "anti_spoof_threshold", ANTI_SPOOF_THRESHOLD),
        "anti_spoof_model_path": ANTI_SPOOF_MODEL_PATH,
        "hf_repo_dir": HF_REPO_DIR,
        "hf_antispoof": face_svc.hf_antispoof.get_config() if face_svc.hf_antispoof else None,
    }


@app.get("/face/antispoof/config", summary="Lihat threshold & weight anti-spoofing HF")
def get_antispoof_config(_=Depends(verify_api_key)):
    if not face_svc.loaded:
        face_svc.load()
    if face_svc.hf_antispoof is None:
        raise HTTPException(400, "Anti-spoofing tidak aktif")
    return face_svc.hf_antispoof.get_config()


@app.put("/face/antispoof/config", summary="Ubah threshold & weight anti-spoofing HF")
def update_antispoof_config(patch: ApiConfigPatch, _=Depends(verify_api_key)):
    if not face_svc.loaded:
        face_svc.load()
    if face_svc.hf_antispoof is None:
        raise HTTPException(400, "Anti-spoofing tidak aktif")
    return face_svc.hf_antispoof.update_config(patch.model_dump(exclude_unset=True))


@app.post("/face/antispoof/reload", summary="Reload model anti-spoofing HF")
def reload_antispoof_model(_=Depends(verify_api_key)):
    if not face_svc.loaded:
        face_svc.load()
    if face_svc.hf_antispoof is None:
        raise HTTPException(400, "Anti-spoofing tidak aktif")
    result = face_svc.hf_antispoof.reload_models()
    if face_svc.hf_antispoof.load_error:
        raise HTTPException(500, face_svc.hf_antispoof.load_error)
    return result


@app.post("/face/antispoof/predict", summary="Tes anti-spoofing HF tanpa verifikasi embedding")
async def predict_antispoof_only(
    photo: UploadFile = File(..., description="Foto wajah/selfie"),
    _=Depends(verify_api_key),
):
    _validate_image_file(photo)
    image_bytes = await photo.read()
    if not face_svc.loaded:
        face_svc.load()
    if face_svc.hf_antispoof is None:
        raise HTTPException(400, "Anti-spoofing tidak aktif")
    img_rgb = face_svc._decode_image(image_bytes)
    res = face_svc.hf_antispoof.predict_on_image(img_rgb)
    if not res.ok:
        return JSONResponse(status_code=400, content={"ok": False, "message": res.error, "detail": res.detail})
    return {
        "ok": True,
        "label": res.label,
        "is_spoof": res.is_spoof,
        "real_score": round(float(res.real_score), 4),
        "spoof_score": round(float(res.spoof_score), 4),
        "real_threshold": round(float(res.real_threshold), 4),
        "spoof_threshold": round(float(res.spoof_threshold), 4),
        "detail": res.detail,
    }


@app.post("/face/extract", summary="Ekstrak embedding dari foto wajah")
async def extract_embedding_endpoint(
    photo: UploadFile = File(..., description="Foto wajah (JPG/PNG/WEBP)"),
    employee_id: str = Form(..., description="ID pegawai untuk logging"),
    _=Depends(verify_api_key),
):
    _validate_image_file(photo)
    image_bytes = await photo.read()
    t0 = time.time()
    try:
        if not face_svc.loaded:
            face_svc.load()
        embedding = face_svc.extract_embedding(image_bytes)
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(f"[EXTRACT] employee={employee_id} elapsed={elapsed}ms | embedding dim={len(embedding)}")
        return {
            "success": True,
            "employee_id": employee_id,
            "embedding": embedding,
            "dimension": len(embedding),
            "elapsed_ms": elapsed,
            "message": "Embedding berhasil diekstrak",
        }
    except ValueError as e:
        logger.warning(f"[EXTRACT] Validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "employee_id": employee_id,
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "employee_id": employee_id,
                "message": f"Gagal mengekstrak embedding: {str(e)}"
            }
        )


@app.post("/face/verify", summary="Cocokkan foto vs embedding acuan")
async def verify_face(
    photo: UploadFile = File(..., description="Foto selfie saat absen"),
    data: str = Form(..., description='JSON: {"employee_id":"...","stored_embedding":[...],"threshold":0.75}'),
    liveness: str = Form(..., description='String "true" if liveness steps performed'),
    _=Depends(verify_api_key),
):
    if LIVENESS_ENABLED and liveness.lower() != "true":
        raise HTTPException(400, "Liveness verification belum terpenuhi. Arahkan wajah ke kiri dan kanan sebelum mengirim foto.")
    try:
        req = VerifyRequest(**json.loads(data))
    except Exception as e:
        raise HTTPException(400, f"Format 'data' tidak valid: {e}")
    if len(req.stored_embedding) != 512:
        # raise HTTPException(400, f"stored_embedding harus 512 dimensi, dapat {len(req.stored_embedding)}")
        raise HTTPException(400, f"Wajah kurang jelas, coba lagi")
    _validate_image_file(photo)
    image_bytes = await photo.read()
    t0 = time.time()
    try:
        if not face_svc.loaded:
            face_svc.load()
        result = face_svc.verify(image_bytes, req.stored_embedding, req.threshold)
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"[VERIFY] employee={req.employee_id} "
            f"matched={result['matched']} sim={result['similarity']:.3f} "
            f"elapsed={elapsed}ms"
        )
        return {
            **result,
            "employee_id": req.employee_id,
            "elapsed_ms": elapsed,
        }
    except Exception as e:
        logger.error(f"Error verifying face: {e}")
        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "matched": False,
            "similarity": 0.0,
            "confidence": 0.0,
            "threshold": req.threshold or SIMILARITY_THRESHOLD,
            "employee_id": req.employee_id,
            "elapsed_ms": elapsed,
            "message": f"Gagal verifikasi: {str(e)}"
        }


ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

def _validate_image_file(file: UploadFile):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Format tidak didukung. Gunakan: {ALLOWED_EXT}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
        workers=1,
    )
