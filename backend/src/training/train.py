"""
train.py â€” Training loop for Project Trinetra.

Hardware-aware optimizations for RTX 4060 (8 GB VRAM):
  â€¢ Mixed precision   (torch.amp)
  â€¢ Gradient accumulation  (simulate large batches)
  â€¢ Epoch checkpointing    (resume on crash)
  â€¢ ReduceLROnPlateau scheduler

Usage:
    python train.py
    python train.py --resume              # auto-resume from latest checkpoint
    python train.py --metadata path.csv   # custom metadata path
"""

import argparse
import os
import warnings
import shutil
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.core.config import (
    ACCUM_STEPS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    MODEL_DIR,
    DEVICE,
    EPOCHS,
    LR,
    METADATA_CSV,
    PROCESSED_DIR,
    RANDOM_SEED,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from src.data.dataset import DeepfakeDataset, build_dataloaders
from src.models.model import DeepfakeDetector


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Checkpoint helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  ðŸ’¾ Checkpoint saved â†’ {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
) -> tuple[int, float]:
    """Returns (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  âœ… Resumed from {path}  (epoch {ckpt['epoch']+1})")
    return ckpt["epoch"] + 1, ckpt["best_val_loss"]


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Sub-batch forward helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Label smoothing factor â€” softens hard 0/1 targets for better generalization
LABEL_SMOOTHING = 0.05

def forward_subbatch(
    model: nn.Module,
    frames: torch.Tensor | None,
    labels: torch.Tensor | None,
    criterion: nn.Module,
) -> tuple[torch.Tensor | None, int, int]:
    """
    Runs a 'mini-test' for a small group of images.
    
    This function takes a few images, asks the model 'Are these real?', and 
    then calculates how wrong the model was (loss). 
    
    We use 'Label Smoothing' (0.05) here. Instead of telling the AI 'This is 
    definitely 100% fake', we say 'This is 95% likely to be fake'. This keeps 
    the AI humble and prevents it from over-learning specific pixels.
    """
    if frames is None:
        return None, 0, 0

    frames = frames.to(DEVICE, non_blocking=True)
    labels = labels.to(DEVICE, non_blocking=True)

    # Label smoothing: 0 â†’ 0.05, 1 â†’ 0.95
    smooth_labels = labels * (1.0 - LABEL_SMOOTHING) + (1.0 - labels) * LABEL_SMOOTHING

    with autocast(device_type="cuda"):
        logits = model(frames)            # (B, 1)
        loss = criterion(logits.squeeze(1), smooth_labels)

    preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).float()
    correct = (preds == labels).sum().item()
    return loss, int(correct), len(labels)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Training epoch
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> tuple[float, float]:
    """
    The 'Study Session'.
    
    During one epoch, the AI looks at every single image in our dataset once. 
    
    Hardware Magic (RTX 4060 Optimization):
    We use 'Mixed Precision' (autocast). This allows the AI to do math with 
    half-sized numbers (float16) where possible, which is much faster and 
    uses less VRAM.
    
    We also use 'Gradient Accumulation'. Since our GPU is small, we can't fit 
    32 videos at once. Instead, we look at 4 videos, remember the 'lesson', 
    repeat 8 times, and *then* update the model's brain.
    """
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        # Process both sub-batches (sequences + statics) from the collate_fn
        loss_total = torch.tensor(0.0, device=DEVICE)

        for key_frames, key_labels in [
            ("seq_frames", "seq_labels"),
            ("static_frames", "static_labels"),
        ]:
            loss, correct, count = forward_subbatch(
                model, batch[key_frames], batch[key_labels], criterion,
            )
            if loss is not None:
                loss_total = loss_total + loss
                total_correct += correct
                total_samples += count

        # Scale loss for gradient accumulation
        scaled_loss = loss_total / ACCUM_STEPS
        scaler.scale(scaled_loss).backward()

        running_loss += loss_total.item()

        # Step optimizer every ACCUM_STEPS micro-batches
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Validation epoch
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
) -> tuple[float, float, dict]:
    """
    The 'Final Exam'.
    
    After studying, we test the AI on images it has NEVER seen before. 
    This tells us if the AI actually learned how to spot deepfakes, or if it 
    just memorized the training photos. 
    
    We calculate things like 'F1 Score' and 'Precision' to see how often 
    the AI is 'crying wolf' (false positives) versus how many fakes it missed.
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        for key_frames, key_labels in [
            ("seq_frames", "seq_labels"),
            ("static_frames", "static_labels"),
        ]:
            frames = batch[key_frames]
            labels = batch[key_labels]
            if frames is None:
                continue

            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(frames)
                loss = criterion(logits.squeeze(1), labels)

            running_loss += loss.item()
            preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)

    # Per-class metrics (label 1 = FAKE is positive class)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1_real": f1_score(all_labels, all_preds, pos_label=0, zero_division=0),
            "precision_real": precision_score(all_labels, all_preds, pos_label=0, zero_division=0),
            "recall_real": recall_score(all_labels, all_preds, pos_label=0, zero_division=0),
        }
    return avg_loss, accuracy, metrics


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Main
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main(args):
    print("=" * 60)
    print("  Project Trinetra â€” Training")
    print(f"  Device    : {DEVICE}")
    print(f"  Batch     : {BATCH_SIZE}  Ã—  {ACCUM_STEPS} accum  =  {BATCH_SIZE * ACCUM_STEPS} effective")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  LR        : {LR}")
    print("=" * 60)

    # â”€â”€ 1. Data split â”€â”€
    metadata_csv = args.metadata
    full_ds = DeepfakeDataset(metadata_csv, PROCESSED_DIR)
    labels = [s["label"] for s in full_ds.samples]

    MIN_SAMPLES = 20
    n_total = len(full_ds)
    if n_total < MIN_SAMPLES:
        raise ValueError(
            f"[ERROR] Dataset has only {n_total} sample(s) â€” need at least {MIN_SAMPLES}.\n"
            f"  Run preprocess.py on a proper dataset first."
        )
    groups = [s.get("source_file", str(i)) for i, s in enumerate(full_ds.samples)]
    if len(set(groups)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
        train_idx, val_idx = next(gss.split(range(n_total), labels, groups=groups))
        train_idx, val_idx = list(train_idx), list(val_idx)
    else:
        train_idx, val_idx = train_test_split(
            list(range(n_total)), test_size=VAL_SPLIT, stratify=labels, random_state=RANDOM_SEED
        )
    train_labels = [labels[i] for i in train_idx]
    n_real = train_labels.count(0)
    n_fake = train_labels.count(1)
    print(f"  Class dist : {n_real} real / {n_fake} fake")

    train_loader, val_loader = build_dataloaders(
        train_idx, val_idx, metadata_csv, PROCESSED_DIR,
    )

    # â”€â”€ 2. Model, optimizer, scheduler, scaler â”€â”€
    model = DeepfakeDetector().to(DEVICE)

    # Weighted loss to counter class imbalance (sampler balances batches,
    # but loss weighting further corrects gradient magnitude)
    pos_weight = torch.tensor([n_real / n_fake]).to(DEVICE)
    print(f"  pos_weight : {pos_weight.item():.4f} (penalises REAL misclass {n_fake/n_real:.1f}Ã— more)")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Only optimize parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()

    # â”€â”€ 3. Resume / Fine-tune from checkpoint â”€â”€
    start_epoch = 0
    best_val_loss = float("inf")

    if args.clean:
        # Delete old epoch checkpoints but preserve best_model.pt unless told
        finetune_abs = os.path.abspath(args.finetune) if args.finetune else None
        deleted = 0
        if os.path.isdir(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                fpath = os.path.join(CHECKPOINT_DIR, f)
                if finetune_abs and os.path.abspath(fpath) == finetune_abs:
                    continue
                if f.endswith(".pt"):
                    os.remove(fpath)
                    deleted += 1
        # Also skip best_model.pt in model/ if it's the finetune source
        best_path_abs = os.path.abspath(os.path.join(MODEL_DIR, "best_model.pt"))
        if finetune_abs and finetune_abs == best_path_abs:
            pass  # preserve it
        print(f"  \U0001f5d1  Deleted {deleted} old epoch checkpoint(s).")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if args.finetune:
        if not os.path.isfile(args.finetune):
            raise FileNotFoundError(f"Checkpoint not found: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location=DEVICE, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        # Use a lower LR for fine-tuning
        finetune_lr = LR / 10
        for pg in optimizer.param_groups:
            pg["lr"] = finetune_lr
        print(f"  âœ… Fine-tuning from {args.finetune}")
        print(f"     LR reduced to {finetune_lr:.2e} for fine-tuning")
        print(f"     Epoch counter reset to 1 (optimizer/scheduler fresh)")
    elif args.resume:
        latest = find_latest_checkpoint(CHECKPOINT_DIR)
        if latest:
            start_epoch, best_val_loss = load_checkpoint(
                latest, model, optimizer, scheduler, scaler,
            )
        else:
            print("  âš  No checkpoint found, starting from scratch.")

    # â”€â”€ 4. Training loop â”€â”€
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        print(f"\n{'â”€'*60}")
        print(f"  Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'â”€'*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
        )
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f}  |  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  |  Acc: {val_acc:.4f}")
        print(f"  FAKE â†’ F1: {val_metrics['f1']:.4f}  Prec: {val_metrics['precision']:.4f}  Rec: {val_metrics['recall']:.4f}")
        print(f"  REAL â†’ F1: {val_metrics['f1_real']:.4f}  Prec: {val_metrics['precision_real']:.4f}  Rec: {val_metrics['recall_real']:.4f}")
        print(f"  LR: {current_lr:.2e}  |  Time: {elapsed:.1f}s")

        # â”€â”€ Save epoch checkpoint â”€â”€
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1:03d}.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, ckpt_path)

        # â”€â”€ Track best model (saved to model/ not checkpoints/) â”€â”€
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(MODEL_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, best_path)
            print(f"  â­ New best val loss: {best_val_loss:.4f}")
            print(f"     Saved â†’ {best_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints â†’ {CHECKPOINT_DIR}")
    print(f"{'='*60}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trinetra Training Loop")
    parser.add_argument(
        "--metadata", type=str, default=METADATA_CSV,
        help="Path to metadata.csv from preprocessing",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from the latest checkpoint in CHECKPOINT_DIR",
    )
    parser.add_argument(
        "--finetune", type=str, default=None,
        help="Fine-tune from a specific checkpoint (loads model weights only, resets optimizer)",
    )
    parser.add_argument(
        "--clean", action="store_true", default=False,
        help="Delete old checkpoints before training",
    )
    args = parser.parse_args()
    main(args)

