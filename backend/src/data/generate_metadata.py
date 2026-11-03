"""
generate_metadata.py â€” Builds metadata.csv from processed_data/fake/ and real/

Supports both filename formats:
  Old format: {clip_id}_{frame_index}.png  (e.g., 0_137.png)
  New format: {stem}_frame{frame_index}_face{face_index}.png (e.g., 000_003_frame0_face0.png)

CSV columns produced (matching dataset.py expectations):
  file_path     â€” relative path like "fake/0_137.png"
  label         â€” 1 = fake, 0 = real
  source_file   â€” the original mp4 filename (reconstructed)
  is_sequence   â€” True / False
  sequence_id   â€” int (same for all frames of a clip) or NaN for statics
  frame_index   â€” int (parsed from filename)
"""

import os
import csv
import re
from collections import defaultdict
from pathlib import Path

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from src.core.config import PROCESSED_DIR, METADATA_CSV as OUTPUT_CSV, SEQ_LEN

# â”€â”€ Scan folders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
folders = {
    "fake": 1,   # label 1 = deepfake
    "real": 0,   # label 0 = real
}

# clip_data[label][clip_id] = sorted list of (frame_index, rel_path, stem)
clip_data: dict[int, dict[str, list]] = {0: defaultdict(list), 1: defaultdict(list)}

regex_new = re.compile(r"^(.*)_frame(\d+)_face\d+$")

# â”€â”€ Build rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rows = []
global_seq_id = 0
stats = {"seq_clips": 0, "seq_frames": 0, "static_images": 0}

for folder_name, label_val in folders.items():
    folder_path = os.path.join(PROCESSED_DIR, folder_name)
    if not os.path.isdir(folder_path):
        print(f"  âš   Folder not found: {folder_path}, skipping.")
        continue

    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            continue

        stem = Path(fname).stem
        
        rel_path = f"{folder_name}/{fname}"
        
        # Try new format: 000_003_frame0_face0
        m = regex_new.match(stem)
        if m:
            clip_id = m.group(1)      # "000_003"
            frame_index = int(m.group(2))
            clip_data[label_val][clip_id].append((frame_index, rel_path, clip_id))
        else:
            # Try old format: 0_137
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                clip_id = parts[0]
                frame_index = int(parts[1])
                clip_data[label_val][clip_id].append((frame_index, rel_path, clip_id))
            else:
                # Treat as a standalone static image
                # Using the filename as a unique clip_id so it doesn't group with others
                rows.append({
                    "file_path":   rel_path,
                    "label":       label_val,
                    "source_file": fname,
                    "is_sequence": False,
                    "sequence_id": "",
                    "frame_index": 0,
                })
                stats["static_images"] += 1

# â”€â”€ Build rows (continued: grouped sequences) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

for label_val, clips in clip_data.items():
    for clip_id, frame_list in clips.items():
        # Sort frames by index
        frame_list.sort(key=lambda x: x[0])

        # A clip can have multiple faces per frame sometimes, we only take up to SEQ_LEN distinct frames
        # For simplicity, if we have enough distinct frames, treat as sequence
        
        # Collect distinct frames (taking the first face of each frame if multiple are present)
        distinct_frames = {}
        for frame_index, rel_path, src_stem in frame_list:
            if frame_index not in distinct_frames:
                distinct_frames[frame_index] = rel_path
                
        sorted_frame_indices = sorted(list(distinct_frames.keys()))

        if len(sorted_frame_indices) >= SEQ_LEN:
            # Treat as a sequence â€” use first SEQ_LEN frames
            seq_frame_indices = sorted_frame_indices[:SEQ_LEN]
            for frame_index in seq_frame_indices:
                rel_path = distinct_frames[frame_index]
                rows.append({
                    "file_path":   rel_path,
                    "label":       label_val,
                    "source_file": f"{clip_id}.mp4",
                    "is_sequence": True,
                    "sequence_id": global_seq_id,
                    "frame_index": frame_index,
                })
            global_seq_id += 1
            stats["seq_clips"]  += 1
            stats["seq_frames"] += len(seq_frame_indices)

            # Extra frames become static
            for frame_index in sorted_frame_indices[SEQ_LEN:]:
                rel_path = distinct_frames[frame_index]
                rows.append({
                    "file_path":   rel_path,
                    "label":       label_val,
                    "source_file": f"{clip_id}.mp4",
                    "is_sequence": False,
                    "sequence_id": "",
                    "frame_index": frame_index,
                })
                stats["static_images"] += 1

        else:
            # Too few frames â†’ treat each as a standalone static image
            for frame_index in sorted_frame_indices:
                rel_path = distinct_frames[frame_index]
                rows.append({
                    "file_path":   rel_path,
                    "label":       label_val,
                    "source_file": f"{clip_id}.mp4",
                    "is_sequence": False,
                    "sequence_id": "",
                    "frame_index": frame_index,
                })
                stats["static_images"] += len(sorted_frame_indices)

# â”€â”€ Write CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fieldnames = ["file_path", "label", "source_file", "is_sequence", "sequence_id", "frame_index"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*55}")
print(f"  metadata.csv regenerated â†’ {OUTPUT_CSV}")
print(f"{'='*55}")
print(f"  Total rows        : {len(rows):,}")
print(f"  Sequence clips    : {stats['seq_clips']:,}   ({stats['seq_frames']:,} frames)")
print(f"  Static images     : {stats['static_images']:,}")
print(f"  Unique seq IDs    : {global_seq_id:,}")
print(f"{'='*55}\n")
print("  âœ… Ready to run:  python train.py --resume")

