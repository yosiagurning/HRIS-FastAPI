"""
prepare_dataset.py (CUSTOM)
==========================
Crop & split dataset untuk fine-tuning, tapi file kodenya berada di folder Anda.

Sumber code/model (detector & util) tetap memakai repo Hugging Face (asli):
  - folder HF ditentukan oleh env: HF_REPO_DIR
  - default: ..\\face-anti-spoofing_hf

Default dataset path disesuaikan dengan lokasi dataset Anda:
  - REAL_RAW_DIR = D:\\Dataset_Spoof\\samples
  - LCC_FASD_DIR = D:\\Dataset_Spoof\\LCC_FASD

Output:
  <HF_REPO_DIR>\\data\\train\\(real|spoof)
  <HF_REPO_DIR>\\data\\val\\(real|spoof)
"""

from __future__ import annotations

import os
import sys
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def _default_hf_repo_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "face-anti-spoofing_hf"))


HF_REPO_DIR = os.environ.get("HF_REPO_DIR", _default_hf_repo_dir())
if HF_REPO_DIR not in sys.path:
    sys.path.insert(0, HF_REPO_DIR)

from IADG import aFaceDetect  # noqa: E402


# ─── CONFIG PATHS (bisa override via ENV) ─────────────────────────────────────
REAL_RAW_DIR = os.environ.get("REAL_RAW_DIR", r"D:\Dataset_Spoof\samples")
LCC_FASD_DIR = os.environ.get("LCC_FASD_DIR", r"D:\Dataset_Spoof\LCC_FASD")

# optional
CASIA_DIR = os.environ.get("CASIA_DIR", "")
MSU_DIR = os.environ.get("MSU_DIR", "")

# output root di dalam repo HF agar trainer default bisa menemukannya
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HF_REPO_DIR, "data"))

VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.15"))
MAX_REAL = int(os.environ.get("MAX_REAL", "10000"))
MAX_SPOOF = int(os.environ.get("MAX_SPOOF", "10000"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".3gp", ".MOV"}
VIDEO_FRAME_STEP = int(os.environ.get("VIDEO_FRAME_STEP", "10"))

random.seed(42)


def get_detector():
    print("[detector] using YOLOv8-face from IADG.py (HF repo)")
    model = aFaceDetect()

    def detect(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bboxes, landmarks = model(img_rgb)
        if len(bboxes) == 0:
            return None
        b = bboxes[0]
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[0] + b[2])
        y2 = int(b[1] + b[3])
        return x1, y1, x2, y2

    return detect


def collect_media(root, max_n=None):
    entries = []
    if not root or not os.path.isdir(root):
        return entries
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMG_EXTS:
            entries.append(("image", str(p)))
        elif suffix in VID_EXTS:
            entries.append(("video", str(p)))
    random.shuffle(entries)
    return entries[:max_n] if max_n else entries


def collect_lcc_class_media(root, class_name, max_n=None):
    class_dirs = [str(p) for p in Path(root).rglob("*") if p.is_dir() and p.name.lower() == class_name.lower()]
    class_paths = []
    for d in class_dirs:
        class_paths += collect_media(d)
    random.shuffle(class_paths)
    return class_paths[:max_n] if max_n else class_paths


def crop_and_save(entries, dst_dir, detect_fn, label):
    os.makedirs(dst_dir, exist_ok=True)
    saved = 0

    def crop_face(img):
        box = detect_fn(img)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        return crop if crop.size else None

    for kind, src in tqdm(entries, desc=f"  {label} -> {dst_dir}"):
        if kind == "image":
            img = cv2.imread(src)
            if img is None:
                continue
            crop = crop_face(img)
            if crop is None:
                continue
            out_path = os.path.join(dst_dir, f"{saved:06d}.jpg")
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            continue

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            continue
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % VIDEO_FRAME_STEP == 0:
                crop = crop_face(frame)
                if crop is not None:
                    out_path = os.path.join(dst_dir, f"{saved:06d}.jpg")
                    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
            frame_idx += 1
        cap.release()

    print(f"  saved {saved} images to {dst_dir}")
    return saved


def split_into_train_val(src_dir, train_dir, val_dir, val_ratio=0.15):
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]
    random.shuffle(files)
    n_val = int(len(files) * val_ratio)
    val_f = files[:n_val]
    train_f = files[n_val:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # clear old
    for d in (train_dir, val_dir):
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                os.remove(os.path.join(d, f))

    for f in train_f:
        shutil.move(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in val_f:
        shutil.move(os.path.join(src_dir, f), os.path.join(val_dir, f))

    print(f"  split: {len(train_f)} train / {len(val_f)} val")


def rebalance_temp_dirs(real_dir, spoof_dir):
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(".jpg")]
    spoof_files = [f for f in os.listdir(spoof_dir) if f.lower().endswith(".jpg")]
    target = min(len(real_files), len(spoof_files))
    if target == 0:
        print("[warn] Cannot rebalance: one class has 0 images after cropping.")
        return
    random.shuffle(real_files)
    random.shuffle(spoof_files)
    for f in real_files[target:]:
        os.remove(os.path.join(real_dir, f))
    for f in spoof_files[target:]:
        os.remove(os.path.join(spoof_dir, f))
    print(f"[rebalance] using {target} per class")


def main():
    print(f"[HF_REPO_DIR] {HF_REPO_DIR}")
    print(f"[REAL_RAW_DIR] {REAL_RAW_DIR}")
    print(f"[LCC_FASD_DIR] {LCC_FASD_DIR}")
    print(f"[OUT_DIR] {OUT_DIR}")

    detect = get_detector()
    tmp_real = os.path.join(OUT_DIR, "_tmp_real")
    tmp_spoof = os.path.join(OUT_DIR, "_tmp_spoof")

    print("\n[1/3] Collecting REAL media …")
    real_pool = collect_media(REAL_RAW_DIR, max_n=MAX_REAL)
    crop_and_save(real_pool, tmp_real, detect, "real")

    print("\n[2/3] Collecting SPOOF media …")
    spoof_pool = collect_lcc_class_media(LCC_FASD_DIR, "spoof", max_n=MAX_SPOOF)
    crop_and_save(spoof_pool, tmp_spoof, detect, "spoof")
    rebalance_temp_dirs(tmp_real, tmp_spoof)

    print("\n[3/3] Splitting into train/val …")
    split_into_train_val(tmp_real, os.path.join(OUT_DIR, "train", "real"), os.path.join(OUT_DIR, "val", "real"), VAL_SPLIT)
    split_into_train_val(tmp_spoof, os.path.join(OUT_DIR, "train", "spoof"), os.path.join(OUT_DIR, "val", "spoof"), VAL_SPLIT)

    shutil.rmtree(tmp_real, ignore_errors=True)
    shutil.rmtree(tmp_spoof, ignore_errors=True)

    for split in ("train", "val"):
        for cls in ("real", "spoof"):
            d = os.path.join(OUT_DIR, split, cls)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    main()

