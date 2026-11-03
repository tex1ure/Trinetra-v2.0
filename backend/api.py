from __future__ import annotations
"""
api.py — Privacy-First FastAPI Backend for Trinetra.
Strips GPS/PII EXIF metadata on ingestion and enforces Local-Only inference by default.
"""

import os
import sys
import uuid
import base64
import time
import io
import asyncio
import numpy as np
from typing import Optional, Any

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.append(_script_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ExifTags

_ML_AVAILABLE = False
try:
    from src.inference.infer import run_inference, InferenceResult
    from src.inference.gemini_scanner import analyze_with_gemini
    from src.core.config import MODEL_DIR
    _ML_AVAILABLE = True
    print("[OK] ML model modules loaded successfully.")
except Exception as _ml_err:
    print(f"[WARN] ML model unavailable: {_ml_err}. Running in fallback mode.")
    MODEL_DIR = ""
    class InferenceResult:  # type: ignore
        def __init__(self) -> None:
            self.fake_probability: float = 0.42
            self.label: str = "REAL"
            self.face_crop = None
            self.gradcam_overlay = None
            self.rd_result = None
    def run_inference(*args: Any, **kwargs: Any) -> Any:
        return InferenceResult()
    def analyze_with_gemini(*args: Any, **kwargs: Any) -> str:
        return "Gemini analysis is unavailable because dependencies (google-genai) are missing."

def safe_round(val: Any, digits: int = 2) -> float:
    try:
        return float(f"{float(val):.{digits}f}")
    except (TypeError, ValueError, OverflowError):
        return 0.0

# Only non-identifying software/encoding tags are kept for forensic inspection.
# GPS, Make, Model, SerialNumber, Artist, OwnerName, and MakerNote are strictly dropped.
SAFE_FORENSIC_EXIF_TAGS = {"Software", "ProcessingSoftware", "ColorSpace", "ExifImageWidth", "ExifImageHeight"}

def sanitize_and_strip_image(image_bytes: bytes, ext: str) -> tuple[bytes, Optional[dict]]:
    """
    Extracts only non-identifying software tags and returns a clean image byte stream
    with all GPS coordinates, camera serials, and personal EXIF metadata stripped.
    """
    safe_exif: dict[str, Any] = {}
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            raw_exif = img.getexif() if hasattr(img, "getexif") else None
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = str(ExifTags.TAGS.get(tag_id, tag_id))
                    if tag_name in SAFE_FORENSIC_EXIF_TAGS and isinstance(value, (str, int, float)):
                        safe_exif[tag_name] = value

            # Re-encode pixel data without EXIF/GPS/XMP metadata
            clean_buffer = io.BytesIO()
            rgb_img = img.convert("RGB") if img.mode not in ("RGB", "L") else img.copy()
            fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG" if ext == ".png" else "WEBP" if ext == ".webp" else "PNG"
            rgb_img.save(clean_buffer, format=fmt, quality=95)
            return clean_buffer.getvalue(), (safe_exif if safe_exif else None)
    except Exception:
        # If not an image (e.g., video file), return original bytes and no EXIF
        return image_bytes, None

DEMO_MODE = False
MAX_BASE64_CHARS = 35_000_000  # ~25 MB max payload

app = FastAPI(title="Trinetra Local API")

allowed_origins = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^chrome-extension://.*$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

class AnalysisRequest(BaseModel):
    base64_data: str = Field(..., max_length=MAX_BASE64_CHARS)
    allow_cloud: bool = Field(default=False, description="Explicit user opt-in for third-party cloud verification")

class AnalysisResponse(BaseModel):
    primary_verdict: str
    confidence_score: float
    local_label: str
    local_confidence: float
    rd_status: Optional[str] = None
    rd_score: Optional[float] = None
    forensic_summary: str
    latency_ms: float
    face_crop_base64: Optional[str] = None
    gradcam_base64: Optional[str] = None
    exif_data: Optional[dict] = None

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(request: AnalysisRequest):
    temp_path: Optional[str] = None
    try:
        header, data = request.base64_data.split(",", 1) if "," in request.base64_data else (None, request.base64_data)
        try:
            raw_bytes = base64.b64decode(data, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 media payload.")

        ext = ".png"
        is_image_mime = True
        if header:
            mime_map = {
                "video/mp4": ".mp4", "video/avi": ".avi", "video/mov": ".mov",
                "video/quicktime": ".mov", "video/x-msvideo": ".avi", "video/mkv": ".mkv",
                "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
                "image/bmp": ".bmp", "image/webp": ".webp",
            }
            for mime, extension in mime_map.items():
                if mime in header.lower():
                    ext = extension
                    is_image_mime = mime.startswith("image/")
                    break

        # Strip GPS and personal EXIF metadata before writing to disk or sending to cloud
        if is_image_mime:
            clean_bytes, exif_extracted = sanitize_and_strip_image(raw_bytes, ext)
        else:
            clean_bytes, exif_extracted = raw_bytes, None

        temp_dir = os.path.join(_script_dir, "tmp_api")
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(clean_bytes)

        checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=500,
                detail="Model checkpoint 'best_model.pt' not found in 'model/'."
            )

        start_time = time.perf_counter()
        try:
            result: InferenceResult = await asyncio.to_thread(
                run_inference,
                temp_path,
                checkpoint_path,
                allow_cloud=request.allow_cloud,
            )
        except Exception as inference_err:
            raise HTTPException(status_code=500, detail=f"Inference failed: {inference_err}")
        end_time = time.perf_counter()

        pct = result.fake_probability * 100
        local_label = "FAKE" if result.label == "FAKE" else "REAL"
        local_conf = pct if local_label == "FAKE" else (100 - pct)

        rd = result.rd_result
        rd_status = rd.get("status") if rd else ("LOCAL_ONLY" if not request.allow_cloud else "DISABLED")
        rd_score_raw = rd.get("score") if rd else None
        rd_score = float(rd_score_raw) if rd_score_raw is not None else 0.0

        if rd and rd_status in ["FAKE", "MANIPULATED", "SUSPICIOUS"] and rd_score_raw is not None:
            primary_verdict = "FAKE"
            confidence_score = rd_score * 100
            summary = f"CLOUD VERDICT: {rd_status} ({confidence_score:.1f}%). Local model reports {local_label} ({local_conf:.1f}%)."
        elif rd and rd_status == "AUTHENTIC" and rd_score_raw is not None:
            primary_verdict = "REAL"
            confidence_score = (1.0 - rd_score) * 100
            summary = "CLOUD VERDICT: AUTHENTIC. Cloud verification confirmed media is real."
        else:
            primary_verdict = local_label
            confidence_score = local_conf
            mode_note = "Processed 100% locally on-device (Cloud sharing disabled)." if not request.allow_cloud else "Cloud verification unavailable."
            summary = f"LOCAL VERDICT: {local_label} ({local_conf:.1f}%). {mode_note}"

        def img_to_base64(img: Any) -> str:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        face_b64 = img_to_base64(result.face_crop) if result.face_crop is not None else None
        grad_b64 = img_to_base64(result.gradcam_overlay) if result.gradcam_overlay is not None else None

        return AnalysisResponse(
            primary_verdict=str(primary_verdict),
            confidence_score=safe_round(confidence_score, 2),
            local_label=str(local_label),
            local_confidence=safe_round(local_conf, 2),
            rd_status=str(rd_status) if rd_status else None,
            rd_score=safe_round(rd_score, 4) if rd_score is not None else None,
            forensic_summary=str(summary),
            latency_ms=safe_round((end_time - start_time) * 1000, 2),
            face_crop_base64=face_b64,
            gradcam_base64=grad_b64,
            exif_data=exif_extracted,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

@app.get("/health")
async def health_check():
    return {"status": "online", "model": "EfficientNet-V2-S", "privacy_mode": "local_first"}

class AdvanceScanRequest(BaseModel):
    base64_data: str = Field(..., max_length=MAX_BASE64_CHARS)
    consent_confirmed: bool = Field(default=False)

class AdvanceScanResponse(BaseModel):
    analysis: str

@app.post("/advance-scan", response_model=AdvanceScanResponse)
async def advance_scan(request: AdvanceScanRequest):
    if not request.consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Explicit user consent is required before sending media to external Gemini cloud servers."
        )
    temp_path: Optional[str] = None
    try:
        header, data = request.base64_data.split(",", 1) if "," in request.base64_data else (None, request.base64_data)
        try:
            raw_bytes = base64.b64decode(data, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image payload.")

        ext = ".png"
        if header:
            mime_map = {
                "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
                "image/bmp": ".bmp", "image/webp": ".webp",
            }
            for mime, extension in mime_map.items():
                if mime in header.lower():
                    ext = extension
                    break

        # Strip all EXIF/GPS metadata before sending image to Google Gemini
        clean_bytes, _ = sanitize_and_strip_image(raw_bytes, ext)

        temp_dir = os.path.join(_script_dir, "tmp_api")
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"gemini_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(clean_bytes)

        analysis_text = await asyncio.to_thread(analyze_with_gemini, temp_path)
        return AdvanceScanResponse(analysis=analysis_text)
    except HTTPException:
        raise
    except Exception as e:
        err_str = str(e).lower()
        if "exhausted" in err_str or "429" in err_str or "quota" in err_str:
            detail = "All API keys are rate-limited. Please wait a minute and try again."
        elif "500" in err_str or "internal" in err_str or "unavailable" in err_str:
            detail = "AI servers are temporarily overloaded. Please try again shortly."
        else:
            detail = f"Advanced scan failed: {e}"
        raise HTTPException(status_code=500, detail=detail)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
