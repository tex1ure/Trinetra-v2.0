"""
model.py — Deepfake Detector architectures for Project Trinetra.

Two architectures are supported, selected automatically based on checkpoint:

  • Legacy   : EfficientNet-B4 + LSTM    (best_model.pt from Trinetra v1/v2)
  • Current  : EfficientNet-V2-S + Transformer  (new training runs)

Use `load_detector_from_checkpoint(path, device)` to auto-detect and load.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models

from src.core.config import (
    DROPOUT,
    EFFICIENTNET_FEATURE_DIM,
    FREEZE_BLOCKS,
    SEQ_LEN,
    TRANSFORMER_DIM_FF,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  Legacy Architecture: EfficientNet-B4 + LSTM
# ══════════════════════════════════════════════

class LegacyDeepfakeDetector(nn.Module):
    """
    The 'Old Reliable' architecture.
    
    This is what we used in the early versions of Trinetra. It uses 
    EfficientNet-B4 for vision and an LSTM for memory. It's slower and 
    uses more VRAM, but it's very solid for basic face-swaps.
    """

    _FEATURE_DIM = 1792
    _LSTM_HIDDEN = 512

    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze lower blocks (same policy as new model)
        for i, block in enumerate(self.features):
            if i < FREEZE_BLOCKS:
                for param in block.parameters():
                    param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self._FEATURE_DIM,
            hidden_size=self._LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        self.fc_seq = nn.Linear(self._LSTM_HIDDEN, 1)
        self.fc_static = nn.Linear(self._FEATURE_DIM, 1)

    def _extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        features = self._extract_spatial(x_flat)

        if T > 1:
            features = features.reshape(B, T, -1)
            lstm_out, _ = self.lstm(features)
            last_hidden = lstm_out[:, -1, :]
            logits = self.fc_seq(last_hidden)
        else:
            features = features.reshape(B, -1)
            logits = self.fc_static(features)

        return logits

    def forward_with_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        features = self._extract_spatial(x_flat)

        if T > 1:
            features = features.reshape(B, T, -1)
            lstm_out, _ = self.lstm(features)
            last_hidden = lstm_out[:, -1, :]
            logits = self.fc_seq(last_hidden)
            return logits, lstm_out
        else:
            features = features.reshape(B, -1)
            logits = self.fc_static(features)
            return logits, None

    @property
    def classifier(self) -> nn.Module:
        """Compatibility: XAI modules access model.classifier for projections."""
        return self.fc_static


# ══════════════════════════════════════════════
#  Current Architecture: EfficientNet-V2-S + Transformer
# ══════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    """
    The 'State-of-the-Art' architecture.
    
    This is the current engine. It uses EfficientNet-V2-S (which is faster and 
    smarter) paired with a Transformer. 
    
    Why a Transformer?
    Unlike older LSTMs, Transformers can look at all 20 frames of a video 
    simultaneously. This makes them incredibly good at spotting 'glitches' that 
    don't just happen one after another, but across the entire clip.
    """

    def __init__(self):
        super().__init__()

        # ── 1. The 'Eyes' (EfficientNet-V2-S) ──
        # This part of the brain is trained to recognize faces, skin textures, 
        # and light reflections. We use a pre-trained version from ImageNet 
        # so it already knows what a 'real' object looks like.
        backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_dim = EFFICIENTNET_FEATURE_DIM

        # Freeze the lower blocks
        for i, block in enumerate(self.features):
            if i < FREEZE_BLOCKS:
                for param in block.parameters():
                    param.requires_grad = False

        # ── 2. The 'Memory' (Transformer Encoder) ──
        # This part looks at the sequence of frames. It's looking for things 
        # that don't belong in time—like an eye flickering for just one frame.
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
        )

        # ── 3. Classification head (shared for both paths) ──
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 1),
        )

    # ──────────────────────────────────────────

    def _extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, C, H, W)  →  (N, feature_dim)
        Passes through EfficientNet feature blocks + global avg pool.
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    # ──────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C, H, W)

        Adaptive routing:
          • T > 1  → spatial features → Transformer → classifier
          • T == 1 → spatial features →               classifier
        """
        B, T, C, H, W = x.shape

        # Flatten batch and time for the CNN: (B*T, C, H, W)
        x_flat = x.reshape(B * T, C, H, W)
        features = self._extract_spatial(x_flat)    # (B*T, feat_dim)

        if T > 1:
            # ── Sequence path ──
            features = features.reshape(B, T, -1)   # (B, T, feat_dim)
            features = features + self.pos_embedding[:, :T, :]
            transformer_out = self.transformer(features)  # (B, T, feat_dim)
            # Use mean pooling over temporal dimension
            pooled = transformer_out.mean(dim=1)     # (B, feat_dim)
            logits = self.classifier(pooled)          # (B, 1)
        else:
            # ── Static image path (bypass Transformer) ──
            features = features.reshape(B, -1)       # (B, feat_dim)
            logits = self.classifier(features)        # (B, 1)

        return logits

    # ──────────────────────────────────────────

    def forward_with_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Same as forward() but additionally returns per-frame transformer
        hidden states for XAI temporal analysis.

        Returns
        -------
        logits           : (B, 1)
        temporal_states  : (B, T, feat_dim) or None (if T == 1)
        """
        B, T, C, H, W = x.shape

        x_flat = x.reshape(B * T, C, H, W)
        features = self._extract_spatial(x_flat)

        if T > 1:
            features = features.reshape(B, T, -1)
            features = features + self.pos_embedding[:, :T, :]
            transformer_out = self.transformer(features)
            pooled = transformer_out.mean(dim=1)
            logits = self.classifier(pooled)
            return logits, transformer_out
        else:
            features = features.reshape(B, -1)
            logits = self.classifier(features)
            return logits, None


# ══════════════════════════════════════════════
#  Auto-detection: inspect checkpoint → pick model
# ══════════════════════════════════════════════

def load_detector_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    The 'Auto-Switcher'.
    
    You don't need to tell this function which model you're using. 
    It opens the 'best_model.pt' file, looks at the internal wiring (keys), 
    and automatically builds the correct architecture (Legacy or Current).
    """
    # Safe checkpoint loading without arbitrary code execution
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    # ── Detect architecture from checkpoint keys ──
    keys = set(state.keys())

    if "lstm.weight_ih_l0" in keys:
        log.info("Checkpoint detected: Legacy (EfficientNet-B4 + LSTM)")
        model = LegacyDeepfakeDetector()
    elif any(k.startswith("transformer.") for k in keys):
        log.info("Checkpoint detected: Current (EfficientNet-V2-S + Transformer)")
        model = DeepfakeDetector()
    else:
        log.warning("Could not detect architecture, defaulting to current (V2-S + Transformer)")
        model = DeepfakeDetector()

    model.to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ──────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.config import DEVICE, IMG_SIZE, SEQ_LEN

    model = DeepfakeDetector().to(DEVICE)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Sequence smoke test
    dummy_seq = torch.randn(2, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    out_seq = model(dummy_seq)
    print(f"Sequence  input: {dummy_seq.shape} -> output: {out_seq.shape}")

    # Static smoke test
    dummy_static = torch.randn(2, 1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    out_static = model(dummy_static)
    print(f"Static    input: {dummy_static.shape} -> output: {out_static.shape}")

    print("[OK] Model forward pass OK for both sequence and static inputs.")
