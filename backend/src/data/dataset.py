"""
dataset.py â€” PyTorch Dataset and DataLoader for Project Trinetra.

Reads from the metadata.csv ledger produced by preprocess.py.
Handles two sample types:
  â€¢ Sequences (video):  (SEQ_LEN, C, H, W) tensors
  â€¢ Static images:      (1, C, H, W)  tensors

A custom collate_fn splits each batch into sequence and static sub-batches,
keeping shapes consistent and avoiding padding waste.

All augmentation runs on-CPU inside __getitem__ via albumentations,
preserving VRAM strictly for the forward pass.
"""

import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.core.config import (
    BATCH_SIZE,
    IMG_SIZE,
    METADATA_CSV,
    NUM_WORKERS,
    PROCESSED_DIR,
    SEQ_LEN,
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  ImageNet normalisation (applied after albumentations)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_tensor_and_normalize = transforms.Compose([
    transforms.ToTensor(),                                      # HWC uint8 â†’ CHW float [0,1]
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Albumentations pipelines
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_train_augmentation(img_size: int = IMG_SIZE) -> A.Compose:
    """
    The 'Digital Torture' pipeline.
    
    Why so aggressive? 
    In the real world, deepfakes are rarely high-quality original files. They 
    get compressed by WhatsApp, resized by Instagram, and blurred by poor 
    internet connections. 
    
    If we train our AI on 'clean' images, it will fail in the wild. So, we 
    intentionally 'torture' our training data with blur, noise, and 
    heavy JPEG compression to make the AI tough enough for social media.
    """
    return A.Compose([
        # â”€â”€ 0. Resize to model input â”€â”€
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),

        # â”€â”€ 1. Severe Compression Protocol â”€â”€
        #   Simulates brutal platform re-encoding.
        #   Quality floor at 30 strips high-frequency pixel anomalies.
        A.ImageCompression(
            quality_range=(30, 80),
            p=0.5,
        ),

        # â”€â”€ 2. Resolution Degradation â”€â”€
        #   Downscale â†’ re-upscale mimics bandwidth-saving algorithms.
        A.Downscale(
            scale_range=(0.5, 0.85),
            p=0.5,
        ),

        # â”€â”€ 3. High-Frequency Destruction â”€â”€
        #   Forces the backbone to rely on structural geometry, not
        #   superficial edge sharpness.
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.5),

        # â”€â”€ 4. Artificial Noise Injection â”€â”€
        #   Masks synthetic generation signatures; forces deeper
        #   feature-space reasoning.
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.1, 0.5), p=1.0),
        ], p=0.5),

        # â”€â”€ 5. Standard geometric / colour jitter â”€â”€
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
        ], p=0.4),
    ])


def _build_val_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """Clean resize-only transform for validation â€” no data corruption."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    ])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class DeepfakeDataset(Dataset):
    """
    The 'Librarian' of our training data.
    
    This class is responsible for looking at our metadata ledger (CSV) and 
    finding the correct image files on the hard drive. 
    
    It's smart: it knows that a video isn't just one image, but a sequence 
    of 20 frames. It packages them up neatly for the AI to process.
    """

    def __init__(
        self,
        metadata_csv: str = METADATA_CSV,
        processed_dir: str = PROCESSED_DIR,
        augmentation: A.Compose | None = None,
    ):
        self.processed_dir = processed_dir
        self.augmentation = augmentation

        df = pd.read_csv(metadata_csv)

        self.samples: list[dict] = []

        # â”€â”€ Group video sequences by sequence_id â”€â”€
        seq_df = df[df["is_sequence"] == True]  # noqa: E712
        if not seq_df.empty:
            for seq_id, group in seq_df.groupby("sequence_id"):
                group = group.sort_values("frame_index")
                paths = group["file_path"].tolist()
                label = int(group["label"].iloc[0])
                self.samples.append({
                    "paths": paths, "label": label, "is_sequence": True, "source_file": str(group["source_file"].iloc[0]),
                })

        # â”€â”€ Static images are individual samples â”€â”€
        static_df = df[df["is_sequence"] == False]  # noqa: E712
        for _, row in static_df.iterrows():
            self.samples.append({
                "paths": [row["file_path"]], "label": int(row["label"]), "is_sequence": False, "source_file": str(row.get("source_file", row["file_path"])),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_augment(self, path: str) -> torch.Tensor:
        """Load image â†’ albumentations augment â†’ ImageNet normalise â†’ CHW tensor."""
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)  # (H, W, 3) uint8 â€” albumentations native format

        if self.augmentation is not None:
            img_np = self.augmentation(image=img_np)["image"]
        else:
            # Fallback: just resize
            img_np = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE),
                                interpolation=cv2.INTER_LINEAR)

        # img_np is (H, W, 3) uint8 â†’ torchvision handles conversion
        return _to_tensor_and_normalize(img_np)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        frames = []
        for p in sample["paths"]:
            full_path = os.path.join(self.processed_dir, p)
            frames.append(self._load_and_augment(full_path))

        # Stack: (T, C, H, W)  where T = SEQ_LEN or 1
        tensor = torch.stack(frames, dim=0)

        # Truncate or pad sequences to exactly SEQ_LEN
        if sample["is_sequence"]:
            if tensor.shape[0] > SEQ_LEN:
                tensor = tensor[:SEQ_LEN]
            elif tensor.shape[0] < SEQ_LEN:
                pad = torch.zeros(
                    SEQ_LEN - tensor.shape[0], *tensor.shape[1:],
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, pad], dim=0)

        return {
            "frames": tensor,                       # (T, C, H, W)
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "is_sequence": sample["is_sequence"],
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Custom collate function
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def collate_fn(batch: list[dict]) -> dict:
    """
    The 'Sorter' for our batches.
    
    In our dataset, some items are single photos and some are 20-frame videos. 
    You can't easily put them in the same box (tensor) because their shapes 
    are different. 
    
    This function sorts them into two separate 'piles' (sub-batches) so the 
    AI can process the photos and videos efficiently without getting confused.
    """
    seq_frames, seq_labels = [], []
    static_frames, static_labels = [], []

    for item in batch:
        if item["is_sequence"]:
            seq_frames.append(item["frames"])
            seq_labels.append(item["label"])
        else:
            static_frames.append(item["frames"])
            static_labels.append(item["label"])

    result = {
        "seq_frames": torch.stack(seq_frames) if seq_frames else None,
        "seq_labels": torch.stack(seq_labels) if seq_labels else None,
        "static_frames": torch.stack(static_frames) if static_frames else None,
        "static_labels": torch.stack(static_labels) if static_labels else None,
    }
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Convenience builder
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_dataloaders(
    train_indices: list[int],
    val_indices: list[int],
    metadata_csv: str = METADATA_CSV,
    processed_dir: str = PROCESSED_DIR,
) -> tuple[DataLoader, DataLoader]:
    """Return train and val DataLoaders from pre-computed index splits."""

    # â”€â”€ Separate datasets with different augmentation policies â”€â”€
    train_dataset = DeepfakeDataset(
        metadata_csv, processed_dir,
        augmentation=_build_train_augmentation(),
    )
    val_dataset = DeepfakeDataset(
        metadata_csv, processed_dir,
        augmentation=_build_val_transform(),
    )

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # â”€â”€ Handle Class Imbalance via Oversampling â”€â”€
    # Problem: Deepfake datasets usually have way more 'Fake' than 'Real'.
    # If we just trained on that, the AI would become 'paranoid' and call 
    # everything fake just to be safe.
    #
    # Solution: The WeightedRandomSampler. It ensures the AI sees an equal 
    # number of Real and Fake samples during every training session.
    train_labels = [train_dataset.samples[i]["label"] for i in train_indices]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,               # Replaces shuffle=True
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader

