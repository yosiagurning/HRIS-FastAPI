"""
recalibrate_thresholds.py (CUSTOM)
=================================
Kalibrasi ulang threshold ICM2O/IOM2C setelah fine-tuning.

Input:
  - finetuned_weights ada di <HF_REPO_DIR>\\finetuned_weights
  - val set ada di <data_dir>\\val\\(real|spoof)
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm


def _default_hf_repo_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "face-anti-spoofing_hf"))


HF_REPO_DIR = os.environ.get("HF_REPO_DIR", _default_hf_repo_dir())
if HF_REPO_DIR not in sys.path:
    sys.path.insert(0, HF_REPO_DIR)


from IADG import aFaceDetect, crop_from_5landmarks, find_best_threshold, Framework, _load_checkpoint  # noqa: E402


FINETUNED_DIR = os.path.join(HF_REPO_DIR, "finetuned_weights")
WEIGHTS_DIR = os.path.join(HF_REPO_DIR, "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BPCER_BUDGETS = [10, 20, 30]  # percent


def _parse_args():
    p = argparse.ArgumentParser(description="Recalibrate thresholds for finetuned IADG model(s)")
    p.add_argument("--model", choices=["ICM2O", "IOM2C", "both"], default="both")
    p.add_argument("--data_dir", type=str, default=os.path.join(HF_REPO_DIR, "data"), help="Folder dataset berisi val/")
    return p.parse_args()


def load_finetuned_iadg(model_name):
    orig_ckpt = _load_checkpoint(os.path.join(WEIGHTS_DIR, f"{model_name}.pth.tar"), map_location="cpu")
    model_defs = orig_ckpt["args"].model
    transform_cfg = orig_ckpt["args"].transform

    ft_path = os.path.join(FINETUNED_DIR, f"{model_name}_finetuned.pth")
    state_dict = torch.load(ft_path, map_location="cpu")

    model = Framework(**model_defs["params"])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()

    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([transform_cfg["image_size"]] * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_cfg["mean"], std=transform_cfg["std"]),
        ]
    )
    return model, tfm, 0.7


def collect_val_images(data_dir: str):
    items = []
    val_root = os.path.join(data_dir, "val")
    for cls, lbl in [("real", 0), ("spoof", 1)]:
        d = os.path.join(val_root, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, f), lbl))
    return items


def run_iadg_model(model, tfm, crop_margin, img_rgb, landmarks):
    img = crop_from_5landmarks(img_rgb, landmarks, crop_margin)
    with torch.no_grad():
        t = tfm(img).unsqueeze(0).to(DEVICE)
        out = model(t)["out"]
        prob_spoof = float(torch.softmax(out, dim=1)[0, 1].cpu())
    return prob_spoof


def calibrate(model_name, data_dir: str, score_fn):
    print(f"\n--- Calibrating {model_name} ---")
    detector = aFaceDetect()
    val_imgs = collect_val_images(data_dir)

    spoof_probs = []
    skipped = 0
    for path, lbl in tqdm(val_imgs):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bboxes, landmarks = detector(img_rgb)
        if len(landmarks) != 1:
            skipped += 1
            continue
        prob = score_fn(img_rgb, bboxes[0], landmarks[0])
        spoof_probs.append([prob, lbl])

    if not spoof_probs:
        print("  No samples scored - check val path.")
        return

    print(f"  scored {len(spoof_probs)} images, skipped {skipped} (no/multi face)")
    print(f"\n  {'Budget':>8}  {'APCER':>8}  {'BPCER':>8}  {'Threshold':>12}")
    for budget in BPCER_BUDGETS:
        thre, err_real, err_spoof = find_best_threshold(spoof_probs, budget)
        print(f"  {budget:>7}%  {err_spoof*100:>7.2f}%  {err_real*100:>7.2f}%  {thre:>12.6f}")
    print()


def _calibrate_if_available(model_name, data_dir: str):
    ft_path = os.path.join(FINETUNED_DIR, f"{model_name}_finetuned.pth")
    if not os.path.exists(ft_path):
        print(f"{model_name}_finetuned.pth not found - run finetune_iadg.py --model {model_name} first")
        return
    model, tfm, crop = load_finetuned_iadg(model_name)
    calibrate(model_name, data_dir, lambda img, bbox, lm: run_iadg_model(model, tfm, crop, img, lm))


def main():
    args = _parse_args()
    print(f"[HF_REPO_DIR] {HF_REPO_DIR}")
    print(f"[data_dir] {os.path.abspath(args.data_dir)}")
    if args.model in ("ICM2O", "both"):
        _calibrate_if_available("ICM2O", args.data_dir)
    if args.model in ("IOM2C", "both"):
        _calibrate_if_available("IOM2C", args.data_dir)


if __name__ == "__main__":
    main()

