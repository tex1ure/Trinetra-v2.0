"""
preprocess.py â€” Adaptive Preprocessing Pipeline for Project Trinetra.

Supports the DFDC (Deepfake Detection Challenge) dataset format:
  - Flat directory of .mp4 videos
  - metadata.json with {"filename.mp4": {"label": "REAL"/"FAKE", ...}}

Also supports a generic real/fake subdirectory layout.

Usage:
    python preprocess.py                          # uses config defaults
    python preprocess.py --src path/to/dfdc_dir   # custom source
"""

import argparse
import csv
import json
import os
import uuid
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from src.core.config import (
    IMG_SIZE,
    SEQ_LEN,
    RAW_DATA_DIR,
    PROCESSED_DIR,
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  MTCNN face detector  (uses GPU if available)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import torch as _torch
_detect_device = "cuda" if _torch.cuda.is_available() else "cpu"
print(f"[INFO] MTCNN running on: {_detect_device.upper()}")
detector = MTCNN(
    image_size=IMG_SIZE,
    margin=40,
    keep_all=True,          # detect multiple faces per frame
    post_process=False,     # return PIL-compatible uint8 images
    device=_detect_device,
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Albumentations augmentation pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def sample_video_frames(video_path: str, n_frames: int = SEQ_LEN) -> list[np.ndarray]:
    """
    The 'Time Machine' function.
    
    We don't need every single frame of a video (that would be thousands!). 
    Instead, we jump through the video at equal intervals and grab 20 
    representative snapshots. This gives the AI enough information to see 
    the whole story without drowning in data.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def batch_extract_faces(
    frames_rgb: list[np.ndarray],
) -> list[list[np.ndarray]]:
    """
    The 'Face-Finder' (Optimized).
    
    Instead of finding a face in Frame 1, then Frame 2, then Frame 3, we 
    hand all 20 frames to the GPU at once. This 'Batch Extraction' is 
    way faster than doing them one by one. 
    
    It returns a list of faces it found in each frame.
    """
    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    boxes_batch, probs_batch = detector.detect(pil_frames)  # one batched GPU call

    result: list[list[np.ndarray]] = []
    for frame_rgb, boxes, probs in zip(frames_rgb, boxes_batch, probs_batch):
        if boxes is None or probs is None:
            result.append([])
            continue
        h, w = frame_rgb.shape[:2]
        faces = []
        for box, prob in zip(boxes, probs):
            if prob < 0.90:
                continue
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = frame_rgb[y1:y2, x1:x2]
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            faces.append(crop)
        result.append(faces)
    return result


def augment_and_save(face_rgb: np.ndarray, save_path: str) -> None:
    """Save clean face crop to disk. Augmentations are applied on-the-fly in dataset.py for train split only."""
    img = Image.fromarray(face_rgb)
    img.save(save_path)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Dataset discovery  (DFDC or generic)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def discover_files(src_dir: str) -> list[tuple[str, int, str]]:
    """
    The 'Detective' function.
    
    Deepfake datasets come in all shapes and sizes. Some have JSON metadata 
    (DFDC), some have weird folder names (FaceForensics++), and some are 
    just real/fake folders. 
    
    This function snoops around the source directory, figures out which 
    format it's looking at, and creates a master list of files to process.
    """
    meta_path = os.path.join(src_dir, "metadata.json")

    # â”€â”€â”€ Auto-detect nested DFDC layout e.g. dfdc_train_part_00/dfdc_train_part_0/ â”€â”€â”€
    if not os.path.isfile(meta_path):
        subdirs = [d for d in os.listdir(src_dir)
                   if os.path.isdir(os.path.join(src_dir, d))]
        if len(subdirs) == 1:
            candidate = os.path.join(src_dir, subdirs[0], "metadata.json")
            if os.path.isfile(candidate):
                print(f"[INFO] Detected nested DFDC layout â€” using: {os.path.join(src_dir, subdirs[0])}")
                src_dir = os.path.join(src_dir, subdirs[0])
                meta_path = candidate

    # â”€â”€â”€ Known FaceForensics++ fake-method folder names â”€â”€â”€
    FF_FAKE_FOLDERS = {
        "Deepfakes", "Face2Face", "FaceShifter",
        "FaceSwap", "NeuralTextures", "DeepFakeDetection",
    }

    if os.path.isfile(meta_path):
        # â”€â”€â”€ DFDC format â”€â”€â”€
        print(f"[INFO] Detected DFDC format (metadata.json found)")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        files = []
        for filename, info in metadata.items():
            filepath = os.path.join(src_dir, filename)
            if not os.path.isfile(filepath):
                continue
            label_str = info.get("label", "").upper()
            label = 0 if label_str == "REAL" else 1
            label_name = "real" if label == 0 else "fake"
            files.append((filepath, label, label_name))

        print(f"[INFO] Found {len(files)} entries in metadata.json "
              f"({sum(1 for _, l, _ in files if l == 0)} real, "
              f"{sum(1 for _, l, _ in files if l == 1)} fake)")
        return files

    elif os.path.isdir(os.path.join(src_dir, "original")):
        # â”€â”€â”€ FaceForensics++ format â”€â”€â”€
        print(f"[INFO] Detected FaceForensics++ layout (original/ folder found)")
        files = []

        # Real videos from original/
        real_dir = os.path.join(src_dir, "original")
        for fname in os.listdir(real_dir):
            ext = Path(fname).suffix.lower()
            if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                files.append((os.path.join(real_dir, fname), 0, "real"))

        # Fake videos from each deepfake method folder
        for folder_name in sorted(os.listdir(src_dir)):
            folder_path = os.path.join(src_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name not in FF_FAKE_FOLDERS:
                continue
            for fname in os.listdir(folder_path):
                ext = Path(fname).suffix.lower()
                if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                    files.append((os.path.join(folder_path, fname), 1, "fake"))

        n_real = sum(1 for _, l, _ in files if l == 0)
        n_fake = sum(1 for _, l, _ in files if l == 1)
        print(f"[INFO] Found {len(files)} files ({n_real} real, {n_fake} fake)")
        return files

    else:
        # â”€â”€â”€ Check for generic real/fake subdirs first â”€â”€â”€
        has_real = os.path.isdir(os.path.join(src_dir, "real"))
        has_fake = os.path.isdir(os.path.join(src_dir, "fake"))

        if has_real or has_fake:
            # â”€â”€â”€ Generic real/fake subdirectory format â”€â”€â”€
            print(f"[INFO] Using generic real/fake subdirectory layout")
            files = []
            for label_name in ("real", "fake"):
                label_dir = os.path.join(src_dir, label_name)
                if not os.path.isdir(label_dir):
                    print(f"[WARN] Directory not found, skipping: {label_dir}")
                    continue
                label = 0 if label_name == "real" else 1
                for fname in os.listdir(label_dir):
                    ext = Path(fname).suffix.lower()
                    if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                        files.append((os.path.join(label_dir, fname), label, label_name))
            print(f"[INFO] Found {len(files)} files")
            return files

        else:
            # â”€â”€â”€ Flat-fake format (e.g. DFD): all top-level videos â†’ FAKE â”€â”€â”€
            print(f"[INFO] Detected flat-fake layout (DFD-style) â€” all videos treated as FAKE")
            files = []
            for fname in os.listdir(src_dir):
                ext = Path(fname).suffix.lower()
                if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
                    files.append((os.path.join(src_dir, fname), 1, "fake"))
            print(f"[INFO] Found {len(files)} fake files")
            return files


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Main processing loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_dataset(src_dir: str, dst_dir: str, append: bool = False) -> None:
    """Walk the source directory, process each file, and write/append to metadata.csv."""

    os.makedirs(dst_dir, exist_ok=True)
    metadata_path = os.path.join(dst_dir, "metadata.csv")

    all_files = discover_files(src_dir)
    if not all_files:
        print("[ERROR] No files found to process. Check your --src path.")
        return

    write_mode = "a" if append and os.path.isfile(metadata_path) else "w"
    write_header = write_mode == "w"
    if append:
        print(f"[INFO] Append mode: {'adding to' if write_mode == 'a' else 'creating'} {metadata_path}")

    with open(metadata_path, write_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "file_path", "label", "source_file", "is_sequence",
                "sequence_id", "frame_index",
            ])

        saved_count = 0
        skipped_count = 0

        for filepath, label, label_name in tqdm(all_files, desc="Processing"):
            stem = Path(filepath).stem
            ext = Path(filepath).suffix.lower()
            out_subdir = os.path.join(dst_dir, label_name)
            os.makedirs(out_subdir, exist_ok=True)

            # â”€â”€â”€ Video path â”€â”€â”€
            if ext in VIDEO_EXTENSIONS:
                frames = sample_video_frames(filepath, SEQ_LEN)
                if not frames:
                    skipped_count += 1
                    continue

                seq_id = uuid.uuid4().hex[:12]
                seq_face_count = 0
                # One batched GPU call for all frames in this video
                faces_per_frame = batch_extract_faces(frames)
                for frame_idx, faces in enumerate(faces_per_frame):
                    for face_idx, face in enumerate(faces):
                        fname_out = f"{stem}_frame{frame_idx}_face{face_idx}.png"
                        save_path = os.path.join(out_subdir, fname_out)
                        augment_and_save(face, save_path)

                        rel_path = os.path.relpath(save_path, dst_dir)
                        writer.writerow([
                            rel_path, label, os.path.basename(filepath),
                            True, seq_id, frame_idx,
                        ])
                        seq_face_count += 1
                        saved_count += 1

                if seq_face_count == 0:
                    skipped_count += 1

            # â”€â”€â”€ Image path â”€â”€â”€
            elif ext in IMAGE_EXTENSIONS:
                img = cv2.imread(filepath)
                if img is None:
                    skipped_count += 1
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Use the batched extractor for consistency, even for one frame
                faces_per_frame = batch_extract_faces([img_rgb])
                faces = faces_per_frame[0] if faces_per_frame else []

                if not faces:
                    skipped_count += 1
                    continue

                for face_idx, face in enumerate(faces):
                    fname_out = f"{stem}_frame0_face{face_idx}.png"
                    save_path = os.path.join(out_subdir, fname_out)
                    augment_and_save(face, save_path)

                    rel_path = os.path.relpath(save_path, dst_dir)
                    writer.writerow([
                        rel_path, label, os.path.basename(filepath),
                        False, None, 0,
                    ])
                    saved_count += 1

    print(f"\n{'='*50}")
    print(f"  Preprocessing complete!")
    print(f"  Saved   : {saved_count} face crops")
    print(f"  Skipped : {skipped_count} files (no faces / unreadable)")
    print(f"  Output  : {dst_dir}")
    print(f"  Ledger  : {metadata_path}")
    print(f"{'='*50}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  CLI entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trinetra Preprocessing Pipeline")
    parser.add_argument("--src", type=str, default=RAW_DATA_DIR,
                        help="Path to raw dataset (DFDC dir with metadata.json, "
                             "or dir with real/ and fake/ subdirs)")
    parser.add_argument("--dst", type=str, default=PROCESSED_DIR,
                        help="Output directory for processed face crops")
    parser.add_argument("--append", action="store_true",
                        help="Append face crops to existing processed_data/ without "
                             "overwriting metadata.csv (use when adding a second dataset)")
    args = parser.parse_args()

    process_dataset(args.src, args.dst, append=args.append)

