"""
finetune_sasf.py (CUSTOM)
========================
Fine-tune SASF (MiniFASNet) dengan code di folder Anda.

Catatan penting:
- Model definition & loader diambil dari repo HF (src/*).
- Output finetuned disimpan ke <HF_REPO_DIR>\\finetuned_weights agar otomatis terbaca oleh SASF.py.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm.auto import tqdm


def _default_hf_repo_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "face-anti-spoofing_hf"))


HF_REPO_DIR = os.environ.get("HF_REPO_DIR", _default_hf_repo_dir())
if HF_REPO_DIR not in sys.path:
    sys.path.insert(0, HF_REPO_DIR)


WEIGHTS_DIR = os.path.join(HF_REPO_DIR, "weights")
DEFAULT_DATA_DIR = os.path.join(HF_REPO_DIR, "data")
SAVE_DIR = os.path.join(HF_REPO_DIR, "finetuned_weights")
CKPT_DIR = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_TQDM = os.environ.get("FT_TQDM", "1") == "1"
BATCH_SIZE = 32
NUM_WORKERS = int(os.environ.get("FT_NUM_WORKERS", "2"))
LR_HEAD = 1e-4
LR_FULL = 5e-5
EPOCHS_HEAD = 5
EPOCHS_FULL = 10

# output SASF adalah 3 class: [spoof_low, real, spoof_high]
REAL_LOGIT_INDEX = int(os.environ.get("SASF_REAL_LOGIT_INDEX", "1"))

SASF_MODELS = {
    "2.7_80x80_MiniFASNetV2.pth": (80, 80),
    "4_0_0_80x80_MiniFASNetV1SE.pth": (80, 80),
}


def build_transforms(img_size=80, train=True):
    base = [transforms.Resize((img_size, img_size))]
    if train:
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.15),
            transforms.RandomGrayscale(p=0.05),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return transforms.Compose(base)


def build_loaders(img_size=80, data_dir: str = DEFAULT_DATA_DIR):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=build_transforms(img_size, True))
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=build_transforms(img_size, False))

    targets = train_ds.targets
    class_counts = [targets.count(c) for c in range(len(train_ds.classes))]
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    pin = DEVICE.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    print(f"  data_dir: {os.path.abspath(data_dir)}")
    print(f"  classes: {train_ds.class_to_idx}")
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")
    return train_dl, val_dl, train_ds.class_to_idx


def load_sasf_model(model_name):
    from src.anti_spoof_predict import AntiSpoofPredict

    predictor = AntiSpoofPredict(device_id=0 if torch.cuda.is_available() else -1)
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    model = predictor._load_model(model_path)
    return model


def sasf_loss(logits, labels, class_to_idx):
    real_idx = class_to_idx.get("real", 0)
    bin_labels = (labels == real_idx).float()

    probs = torch.softmax(logits, dim=1)
    if logits.size(1) <= REAL_LOGIT_INDEX:
        raise ValueError(f"REAL_LOGIT_INDEX={REAL_LOGIT_INDEX} out of range для logits {tuple(logits.shape)}")
    real_prob = probs[:, REAL_LOGIT_INDEX]
    loss = F.binary_cross_entropy(real_prob.clamp(1e-6, 1 - 1e-6), bin_labels)
    return loss, real_prob, bin_labels


def train_one_epoch(model, loader, optimizer, class_to_idx):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    batches = tqdm(loader, desc="train", leave=False) if USE_TQDM else loader
    for imgs, labels in batches:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss, real_prob, bin_labels = sasf_loss(logits, labels, class_to_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (real_prob > 0.5).long()
        correct += (preds == bin_labels.long()).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, class_to_idx):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    batches = tqdm(loader, desc="val", leave=False) if USE_TQDM else loader
    for imgs, labels in batches:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss, real_prob, bin_labels = sasf_loss(logits, labels, class_to_idx)
        total_loss += loss.item() * imgs.size(0)
        preds = (real_prob > 0.5).long()
        correct += (preds == bin_labels.long()).sum().item()
        total += imgs.size(0)
        tp += ((preds == 1) & (bin_labels == 1)).sum().item()
        tn += ((preds == 0) & (bin_labels == 0)).sum().item()
        fp += ((preds == 1) & (bin_labels == 0)).sum().item()
        fn += ((preds == 0) & (bin_labels == 1)).sum().item()
    acc = correct / total
    apcer = fp / max(fp + tn, 1)
    bpcer = fn / max(fn + tp, 1)
    acer = (apcer + bpcer) / 2
    return total_loss / total, acc, acer, apcer, bpcer


def _ckpt_path(model_name):
    safe = model_name.replace(".pth", "")
    return os.path.join(CKPT_DIR, f"{safe}_resume.pth")


def _current_resume_meta(model_name, img_size, data_dir: str):
    return {
        "model_name": model_name,
        "data_dir": os.path.abspath(data_dir),
        "weights_dir": os.path.abspath(WEIGHTS_DIR),
        "epochs_head": int(EPOCHS_HEAD),
        "epochs_full": int(EPOCHS_FULL),
        "img_size": int(img_size),
        "batch_size": int(BATCH_SIZE),
    }


def _save_resume_checkpoint(model_name, phase, epoch, model, optimizer, scheduler, best_acer, best_state, img_size, data_dir: str):
    torch.save(
        {
            "phase": phase,
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_acer": float(best_acer),
            "best_state": best_state,
            "epochs_head": int(EPOCHS_HEAD),
            "epochs_full": int(EPOCHS_FULL),
            "resume_meta": _current_resume_meta(model_name, img_size, data_dir),
        },
        _ckpt_path(model_name),
    )


def _set_phase1_trainable(model):
    for name, p in model.named_parameters():
        p.requires_grad = any(kw in name.lower() for kw in ["prob", "linear", "bn", "classifier", "fc", "last"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_params == 0:
        raise RuntimeError("Phase-1 freeze selected 0 trainable parameters; check head layer filters.")
    print(f"  trainable: {n_params:,}")


def finetune_one_model(model_name, img_size, resume=True, data_dir: str = DEFAULT_DATA_DIR):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning SASF model: {model_name}")
    print(f"  HF_REPO_DIR : {HF_REPO_DIR}")
    print(f"{'='*60}")

    train_dl, val_dl, class_to_idx = build_loaders(img_size, data_dir=data_dir)
    model = load_sasf_model(model_name).to(DEVICE)

    best_acer = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    resume_phase = "head"
    resume_epoch = 0
    resume_opt_state = None
    resume_sch_state = None

    if resume:
        ckpt_file = _ckpt_path(model_name)
        if os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=DEVICE)
            current_meta = _current_resume_meta(model_name, img_size, data_dir)
            saved_meta = ckpt.get("resume_meta")
            can_resume = True
            if saved_meta is not None and saved_meta != current_meta:
                can_resume = False
                print("  [resume] checkpoint config mismatch; starting fresh for this model.")
            if can_resume:
                resume_phase = ckpt.get("phase", "head")
                resume_epoch = int(ckpt.get("epoch", 0))
                model.load_state_dict(ckpt["model_state"])
                best_acer = float(ckpt.get("best_acer", best_acer))
                best_state = ckpt.get("best_state", best_state)
                resume_opt_state = ckpt.get("optimizer_state")
                resume_sch_state = ckpt.get("scheduler_state")
                print(f"  [resume] loaded: {ckpt_file}")
                print(f"  [resume] phase={resume_phase} epoch={resume_epoch}")
                if resume_phase == "done":
                    save_name = model_name.replace(".pth", "_finetuned.pth")
                    save_path = os.path.join(SAVE_DIR, save_name)
                    torch.save(best_state, save_path)
                    print(f"  [resume] already finished. Best ACER: {best_acer:.4f}")
                    print(f"  [resume] Saved best weights -> {save_path}")
                    return save_path

    print(f"\n--- Phase 1: head only ({EPOCHS_HEAD} epochs) ---")
    _set_phase1_trainable(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_HEAD)

    start_head = 1
    if resume_phase == "head":
        if resume_opt_state is not None:
            optimizer.load_state_dict(resume_opt_state)
        if resume_sch_state is not None:
            scheduler.load_state_dict(resume_sch_state)
        start_head = max(1, resume_epoch + 1)

    for epoch in range(start_head, EPOCHS_HEAD + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, class_to_idx)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, class_to_idx)
        scheduler.step()
        print(
            f"  ep {epoch:02d}/{EPOCHS_HEAD}  "
            f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
            f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
            f"ACER {acer:.4f} (APCER {apcer:.4f} BPCER {bpcer:.4f})  "
            f"{time.time()-t0:.1f}s"
        )
        if acer < best_acer:
            best_acer = acer
            best_state = copy.deepcopy(model.state_dict())
        _save_resume_checkpoint(model_name, "head", epoch, model, optimizer, scheduler, best_acer, best_state, img_size, data_dir)

    print(f"\n--- Phase 2: full model ({EPOCHS_FULL} epochs) ---")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_FULL)

    start_full = 1
    if resume_phase == "full":
        if resume_opt_state is not None:
            optimizer.load_state_dict(resume_opt_state)
        if resume_sch_state is not None:
            scheduler.load_state_dict(resume_sch_state)
        start_full = max(1, resume_epoch + 1)

    for epoch in range(start_full, EPOCHS_FULL + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, class_to_idx)
        vl_loss, vl_acc, acer, apcer, bpcer = validate(model, val_dl, class_to_idx)
        scheduler.step()
        print(
            f"  ep {epoch:02d}/{EPOCHS_FULL}  "
            f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
            f"acc {tr_acc:.3f}/{vl_acc:.3f}  "
            f"ACER {acer:.4f} (APCER {apcer:.4f} BPCER {bpcer:.4f})  "
            f"{time.time()-t0:.1f}s"
        )
        if acer < best_acer:
            best_acer = acer
            best_state = copy.deepcopy(model.state_dict())
        _save_resume_checkpoint(model_name, "full", epoch, model, optimizer, scheduler, best_acer, best_state, img_size, data_dir)

    save_name = model_name.replace(".pth", "_finetuned.pth")
    save_path = os.path.join(SAVE_DIR, save_name)
    torch.save(best_state, save_path)
    _save_resume_checkpoint(model_name, "done", EPOCHS_FULL, model, None, None, best_acer, best_state, img_size, data_dir)
    print(f"\n  [ok] Best ACER: {best_acer:.4f}")
    print(f"  [ok] Saved -> {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of SASF model names")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Folder dataset berisi train/ dan val/")
    parser.add_argument("--no_resume", action="store_true", help="Disable automatic resume from checkpoint")
    args = parser.parse_args()

    selected = args.models if args.models else list(SASF_MODELS.keys())
    for model_name in selected:
        if model_name not in SASF_MODELS:
            raise ValueError(f"Unknown SASF model: {model_name}")
        _h, w = SASF_MODELS[model_name]
        finetune_one_model(model_name, img_size=w, resume=not args.no_resume, data_dir=args.data_dir)


if __name__ == "__main__":
    main()

