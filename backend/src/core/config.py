"""
config.py — Centralized configuration for Project Trinetra.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", os.path.join(PROJECT_ROOT, "raw_data"))
METADATA_CSV  = os.path.join(PROCESSED_DIR, "metadata.csv")
MODEL_DIR     = os.path.join(PROJECT_ROOT, "model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Support both RD_API_KEYS (comma-separated) and RD_API_KEY (single key per README)
_rd_keys_raw = os.environ.get("RD_API_KEYS", "") or os.environ.get("RD_API_KEY", "")
RD_API_KEYS: list[str] = [k.strip() for k in _rd_keys_raw.split(",") if k.strip()]
RD_ENABLED = len(RD_API_KEYS) > 0
RD_TIMEOUT = int(os.environ.get("RD_TIMEOUT", "120"))
RD_RETRY_DELAY = int(os.environ.get("RD_RETRY_DELAY", "2"))

RD_MAX_IMAGE_SIZE_MB = 10
RD_MAX_VIDEO_SIZE_MB = 50
RD_MAX_AUDIO_SIZE_MB = 20
RD_MAX_DOC_SIZE_MB = 10

RD_SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".mp4", ".mov",
    ".mp3", ".wav", ".m4a", ".aac",
    ".pdf", ".docx"
}

_gemini_keys_raw = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEYS: list[str] = [k.strip() for k in _gemini_keys_raw.split(",") if k.strip()]
GEMINI_ENABLED = len(GEMINI_API_KEYS) > 0

IMG_SIZE = 384
SEQ_LEN = 20
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}

EFFICIENTNET_FEATURE_DIM = 1280
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_DIM_FF = 2048
FREEZE_BLOCKS = 5
DROPOUT = 0.3

BATCH_SIZE = 4
ACCUM_STEPS = 8
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_SPLIT = 0.2
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
