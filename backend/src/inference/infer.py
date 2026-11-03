"""
infer.py -- Explainable Inference Pipeline for Project Trinetra.

Provides:
  * GradCAM class           -- spatial artifact localisation via last EfficientNet block
  * run_inference()         -- master orchestrator (image / video routing)
  * InferenceResult         -- structured dataclass returned to callers (UI or CLI)

Usage (standalone CLI):
    python infer.py --input "path/to/file.mp4"
    python infer.py --input "path/to/face.jpg" --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import sys
import time
import threading

_INFERENCE_LOCK = threading.Lock()
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.core.config import (
    MODEL_DIR,
    DEVICE,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    SEQ_LEN,
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
)
from src.models.model import load_detector_from_checkpoint
from src.inference.xai import (
    geometric_jitter_mapping,
    noise_residual_extraction,
    temporal_attention_rollout,
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Constants
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Decision threshold
FAKE_THRESHOLD = 0.50

# Number of temporal windows to sample for video (more = better coverage)
N_SEGMENTS = 3

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Result dataclass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class InferenceResult:
    """Structured output returned by run_inference()."""
    fake_probability: float                     # 0.0 â€“ 1.0
    label: str                                  # "FAKE" or "REAL"
    face_crop: Image.Image | None = None        # Detected face (PIL RGB)
    gradcam_overlay: Image.Image | None = None  # Heatmap blended onto face
    per_frame_scores: list[float] | None = None # Per-frame fake% (video only)
    temporal_rollout: dict | None = None        # XAI: LSTM per-frame confidence
    noise_residual: Image.Image | None = None   # XAI: ELA noise map
    geometric_jitter: dict | None = None        # XAI: landmark jitter metrics
    rd_result: dict | None = None               # Cloud: Reality Defender API result
    forensic_log: dict = field(default_factory=dict)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Grad-CAM
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class GradCAM:
    """
    X-Ray vision for AI. 
    
    Grad-CAM (Gradient-weighted Class Activation Mapping) helps us see 'what' the 
    AI is looking at when it makes a decision. If the AI flags a face as fake, 
    this class generates a heatmap that highlights the specific pixels 
    (like a weird eye or a blurry jawline) that triggered the alarm.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Register hooks on the last conv block
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    @torch.enable_grad()
    def generate(self, frame_tensor: torch.Tensor) -> np.ndarray:
        with _INFERENCE_LOCK:
            return self._generate_locked(frame_tensor)

    @torch.enable_grad()
    def _generate_locked(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single frame.

        Parameters
        ----------
        frame_tensor : (1, C, H, W)  â€” preprocessed, on DEVICE

        Returns
        -------
        heatmap : np.ndarray of shape (H_feat, W_feat), values in [0, 1]
        """
        self.model.eval()
        frame_tensor = frame_tensor.requires_grad_(True)

        # Forward through spatial backbone only (single-frame path)
        features = self.model.features(frame_tensor)
        pooled = self.model.pool(features)
        pooled = torch.flatten(pooled, 1)
        logit = self.model.classifier(pooled)  # (1, 1)

        # Backward
        self.model.zero_grad()
        logit.backward(retain_graph=False)

        # Weighted combination
        grads = self._gradients          # (1, C, h, w)
        acts = self._activations         # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial dims
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Heatmap overlay
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def overlay_heatmap(face_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Paints the AI's 'suspicion heatmap' over the actual face.
    
    Think of this as a weather map, but for deepfakes. 'Red' areas mean the AI 
    is extremely suspicious of those pixels, while 'blue' means it's calm. 
    We blend it so you can still see the face underneath.
    """
    h, w = face_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended = np.uint8(alpha * colormap_rgb + (1 - alpha) * face_rgb)
    return Image.fromarray(blended)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Frame sampling
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def sample_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Uniformly sample n_frames RGB frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Video has no frames.")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_segments(
    video_path: str,
    n_segments: int = N_SEGMENTS,
    frames_per_segment: int = SEQ_LEN,
) -> list[list[np.ndarray]]:
    """
    Smart Video Sampling.
    
    Instead of just looking at the first 10 seconds, we chop the video into 
    sections (beginning, middle, end) and grab frames from each. 
    
    Why? Deepfakes often flicker or fail only in specific parts of a video. 
    This 'segment sampling' ensures we don't miss those hidden glitches.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= 0:
        raise ValueError("Video has no frames.")

    # Clamp n_segments so we don't request more segments than frames
    effective_segs = min(n_segments, max(1, total // frames_per_segment))

    # Divide video into equal-length windows
    boundaries = np.linspace(0, total, effective_segs + 1, dtype=int)
    segments = []
    for i in range(effective_segs):
        start = boundaries[i]
        end = max(boundaries[i + 1] - 1, start)
        segment_indices = np.linspace(start, end, frames_per_segment, dtype=int)

        cap = cv2.VideoCapture(video_path)
        seg_frames = []
        for idx in segment_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                seg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if seg_frames:
            segments.append(seg_frames)

    return segments if segments else [sample_frames(video_path, frames_per_segment)]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Face detection helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_mtcnn():
    try:
        from facenet_pytorch import MTCNN
        return MTCNN(
            keep_all=False,
            device=DEVICE,
            post_process=False,
            image_size=IMG_SIZE,
            margin=20,
        )
    except ImportError:
        warnings.warn(
            "facenet-pytorch not installed â€” using full-frame fallback. "
            "Install with: pip install facenet-pytorch"
        )
        return None


def _crop_face(
    frame_rgb: np.ndarray, mtcnn
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Finds the face in a photo and cuts it out.
    
    The AI model is trained specifically on faces, so we use MTCNN 
    (a face-finding specialist) to zoom in on the person's identity before 
    handing it over to our deepfake detector.
    """
    pil = Image.fromarray(frame_rgb)
    try:
        box, prob, landmarks = mtcnn.detect(pil, landmarks=True)
    except Exception:
        # MTCNN can raise ValueError / RuntimeError when no face proposals are
        # generated (e.g. torch.cat on empty list). Treat as no-face-detected.
        return None, None, 0.0
    if box is None or prob is None or len(prob) == 0 or prob[0] is None or prob[0] < 0.90:
        return None, None, 0.0
    conf = float(prob[0])
    x1, y1, x2, y2 = [int(v) for v in box[0]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None, None, 0.0
    lm = landmarks[0] if landmarks is not None else None  # (5, 2)
    return frame_rgb[y1:y2, x1:x2], lm, conf


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Master inference function
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_inference(
    file_path: str,
    checkpoint_path: str | None = None,
    *,
    allow_cloud: bool = False,
    _model_cache: dict = {},
) -> InferenceResult:
    """
    The 'Grand Central Station' of Trinetra's analysis.
    
    This function takes a file, figures out if it's an image or video, 
    loads the right AI brain (EfficientNet + LSTM), and runs a full suite 
    of forensic tests including:
    - Face detection
    - AI probability scoring
    - Heatmap generation (Grad-CAM)
    - Cloud verification (Reality Defender)
    """
    t_start = time.perf_counter()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")

    ext = os.path.splitext(file_path)[1].lower()
    is_image = ext in IMAGE_EXTENSIONS
    is_video = ext in VIDEO_EXTENSIONS
    is_audio = ext in AUDIO_EXTENSIONS

    if not is_image and not is_video and not is_audio:
        raise ValueError(f"Unsupported file extension: {ext}")

    if is_audio:
        # Skip local visual analysis for audio
        fake_prob = 0.0
        label = "AUDIO"
        rd_result_dict = None
        from src.core.config import RD_ENABLED
        if RD_ENABLED and allow_cloud:
            try:
                from src.inference.reality_defender import analyze_with_rd
                rd_res = analyze_with_rd(file_path)
                rd_result_dict = {
                    "status": rd_res.status,
                    "score": round(rd_res.overall_score, 4),
                    "models": rd_res.models,
                    "request_id": rd_res.request_id,
                    "latency_ms": round(rd_res.latency_ms, 2),
                    "error": rd_res.error
                }
            except Exception as e:
                rd_result_dict = {"status": "ERROR", "error": f"Integration error: {e}"}
        
        return InferenceResult(
            fake_probability=0.0,
            label="AUDIO",
            rd_result=rd_result_dict,
            forensic_log={"media_type": "audio", "reality_defender": rd_result_dict}
        )

    # â”€â”€ Load model (cached, auto-detects legacy vs current architecture) â”€â”€
    if "model" not in _model_cache or _model_cache.get("ckpt") != checkpoint_path:
        model = load_detector_from_checkpoint(checkpoint_path, DEVICE)
        _model_cache["model"] = model
        _model_cache["ckpt"] = checkpoint_path
        _model_cache["gradcam"] = GradCAM(model)
    model = _model_cache["model"]
    gradcam = _model_cache["gradcam"]

    # â”€â”€ Load MTCNN (cached) â”€â”€
    if "mtcnn" not in _model_cache:
        _model_cache["mtcnn"] = _load_mtcnn()
    mtcnn = _model_cache["mtcnn"]


    # â”€â”€ Face crop + landmark collection â”€â”€
    face_crop_rgb = None
    all_landmarks: list[np.ndarray] = []

    def _process_frames(
        frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[float], list]:
        """Crop faces from frames, return (crops, confidences, landmarks)."""
        crops, confs, lms = [], [], []
        nonlocal face_crop_rgb
        for frame in frames:
            crop, lm, conf = (None, None, 0.0)
            if mtcnn is not None:
                crop, lm, conf = _crop_face(frame, mtcnn)
            if crop is None:
                crop = frame  # fallback to full frame
                conf = 0.5    # neutral weight for fallback frames
            if face_crop_rgb is None:
                face_crop_rgb = crop.copy()
            crops.append(crop)
            confs.append(conf)
            if lm is not None:
                lms.append(lm)
        return crops, confs, lms

    def _score_segment(
        crops: list[np.ndarray],
        confs: list[float],
        is_seq: bool,
    ) -> tuple[float, list[float]]:
        """
        Score a segment of face crops.
        Returns (weighted_fake_prob, per_frame_scores).
        """
        # Per-frame scores (static path through model)
        frame_scores = []
        for crop in crops:
            t = _transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                inp = t.unsqueeze(1)   # (1, 1, C, H, W) â†’ static path
                logit = model(inp)
                prob = torch.sigmoid(logit).item()
            frame_scores.append(prob)

        # Also score the whole sequence (LSTM path for temporal coherence)
        if is_seq and len(crops) == SEQ_LEN:
            seq_tensors = [_transform(c).to(DEVICE) for c in crops]
            full_input = torch.stack(seq_tensors).unsqueeze(0)  # (1, T, C, H, W)
            with torch.no_grad():
                logit, _ = model.forward_with_hidden(full_input)
                seq_prob = torch.sigmoid(logit).item()
            # Blend: 0.5 frame-level avg + 0.5 full-sequence score
            frame_avg = float(np.average(frame_scores, weights=confs))
            blended = 0.5 * frame_avg + 0.5 * seq_prob
        else:
            blended = float(np.average(frame_scores, weights=confs))

        return blended, [round(s * 100, 2) for s in frame_scores]

    # â”€â”€ Run inference across all segments â”€â”€
    if is_video:
        segments = sample_segments(file_path)
        segment_probs = []
        all_frame_scores: list[float] = []
        for seg_frames in segments:
            crops, confs, lms = _process_frames(seg_frames)
            all_landmarks.extend(lms)
            seg_prob, fs = _score_segment(crops, confs, is_seq=True)
            segment_probs.append(seg_prob)
            all_frame_scores.extend(fs)
        per_frame_scores = all_frame_scores

        # Ensemble: 0.6 Ã— mean + 0.4 Ã— max  (max catches localised manipulation)
        fake_prob = 0.6 * float(np.mean(segment_probs)) + 0.4 * float(np.max(segment_probs))
        fake_prob = round(min(1.0, fake_prob), 6)

        # Consistency check: flag if segments strongly disagree
        temporal_consistent = float(np.max(segment_probs)) - float(np.min(segment_probs)) < 0.35
    else:
        raw_img = cv2.imread(file_path)
        if raw_img is None:
            raise IOError(f"Cannot read image file: {file_path}")
        raw_frames = [cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)]
        crops, confs, lms = _process_frames(raw_frames)
        all_landmarks.extend(lms)
        fake_prob, _ = _score_segment(crops, confs, is_seq=False)
        per_frame_scores = None
        temporal_consistent = True
        segment_probs = [fake_prob]

    label = "FAKE" if fake_prob >= FAKE_THRESHOLD else "REAL"

    # â”€â”€ Grad-CAM on first face â”€â”€
    if face_crop_rgb is not None:
        first_crop = face_crop_rgb
    elif is_image:
        fallback_img = cv2.imread(file_path)
        first_crop = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB) if fallback_img is not None else None
    else:
        first_crop = None
    gradcam_overlay = None
    face_pil = None
    if first_crop is not None:
        first_tensor = _transform(first_crop).unsqueeze(0).to(DEVICE)
        heatmap = gradcam.generate(first_tensor)
        gradcam_overlay = overlay_heatmap(first_crop, heatmap)
        face_pil = Image.fromarray(first_crop)

    # â”€â”€ XAI: Temporal Attention Rollout â”€â”€
    temporal_rollout = None
    if is_video and len(segments) > 0:
        # Re-score the first segment for LSTM hidden state
        first_seg_crops = []
        for f in segments[0][:SEQ_LEN]:
            cr, _, _ = (None, None, 0.0)
            if mtcnn is not None:
                cr, _, _ = _crop_face(f, mtcnn)
            first_seg_crops.append(cr if cr is not None else f)
        seq_tensors = [_transform(c).to(DEVICE) for c in first_seg_crops]
        while len(seq_tensors) < SEQ_LEN:
            seq_tensors.append(torch.zeros_like(seq_tensors[0]))
        full_input = torch.stack(seq_tensors[:SEQ_LEN]).unsqueeze(0)
        with torch.no_grad():
            _, transformer_hidden = model.forward_with_hidden(full_input)
        if transformer_hidden is not None:
            # Project each frame's transformer output through the classifier
            # to get per-frame confidence scores for temporal rollout
            frame_features = transformer_hidden.squeeze(0)  # (T, feat_dim)
            temporal_rollout = temporal_attention_rollout(
                frame_features, model.classifier,
            )

    # â”€â”€ XAI: Noise residual (ELA) â”€â”€
    noise_residual_pil = None
    if first_crop is not None:
        noise_map = noise_residual_extraction(first_crop)
        noise_residual_pil = Image.fromarray(noise_map)

    # â”€â”€ XAI: Geometric Jitter â”€â”€
    geo_jitter = None
    if is_video and len(all_landmarks) >= 2:
        geo_jitter = geometric_jitter_mapping(all_landmarks)

    # â”€â”€ Cloud: Reality Defender Analysis â”€â”€
    rd_result_dict = None
    from src.core.config import RD_ENABLED
    if RD_ENABLED and allow_cloud:
        try:
            from src.inference.reality_defender import analyze_with_rd
            rd_res = analyze_with_rd(file_path)
            rd_result_dict = {
                "status": rd_res.status,
                "score": round(rd_res.overall_score, 4),
                "models": rd_res.models,
                "request_id": rd_res.request_id,
                "key_used_idx": rd_res.key_used_index,
                "attempts": rd_res.attempts,
                "latency_ms": round(rd_res.latency_ms, 2),
                "error": rd_res.error
            }
        except Exception as e:
            rd_result_dict = {"status": "ERROR", "error": f"Integration error: {e}"}

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 2)

    face_tensor_dims = list(_transform(first_crop).shape) if first_crop is not None else []

    forensic_log = {
        "inference_latency_ms": latency_ms,
        "face_tensor_dims": face_tensor_dims,
        "classification_label": label,
        "fake_threshold": FAKE_THRESHOLD,
        "n_segments_analysed": len(segment_probs),
        "segment_scores_pct": [round(p * 100, 2) for p in segment_probs],
        "temporal_consistent": temporal_consistent if is_video else None,
        "reality_defender": rd_result_dict
    }
    if temporal_rollout:
        forensic_log["shatter_frame"] = temporal_rollout["shatter_index"]
        forensic_log["shatter_delta"] = temporal_rollout["shatter_delta"]
    if geo_jitter:
        forensic_log["jitter_score"] = geo_jitter["jitter_score"]

    return InferenceResult(
        fake_probability=round(fake_prob, 6),
        label=label,
        face_crop=face_pil,
        gradcam_overlay=gradcam_overlay,
        per_frame_scores=per_frame_scores,
        temporal_rollout=temporal_rollout,
        noise_residual=noise_residual_pil,
        geometric_jitter=geo_jitter,
        rd_result=rd_result_dict,
        forensic_log=forensic_log,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  CLI entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="Trinetra â€” Explainable Deepfake Inference")
    parser.add_argument("--input", required=True, help="Path to image or video file")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(MODEL_DIR, "best_model.pt"),
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] File not found: {args.input}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    result = run_inference(args.input, args.checkpoint)

    pct = result.fake_probability * 100
    print(f"\n{'='*60}")
    print(f"  VERDICT  : {'ðŸ”´ FAKE' if result.label == 'FAKE' else 'ðŸŸ¢ REAL'}")
    print(f"  Fake     : {pct:.2f}%  (threshold: {FAKE_THRESHOLD*100:.0f}%)")
    print(f"  Latency  : {result.forensic_log['inference_latency_ms']} ms")

    segs = result.forensic_log.get("segment_scores_pct")
    if segs:
        seg_str = "  ".join(f"Seg{i+1}={s:.1f}%" for i, s in enumerate(segs))
        print(f"  Segments : {seg_str}")
        consistent = result.forensic_log.get("temporal_consistent")
        if consistent is False:
            print("  âš   Temporal inconsistency â€” manipulation may be localised")

    if result.per_frame_scores:
        print(f"  Frames   : {result.per_frame_scores}")
    print(f"{'='*60}\n")

    import json
    print("Forensic Triage Log:")
    print(json.dumps(result.forensic_log, indent=2))


if __name__ == "__main__":
    main()


