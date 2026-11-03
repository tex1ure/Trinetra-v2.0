from __future__ import annotations
"""
api.py — FastAPI Backend for Trinetra Chrome Extension and Dashboard.
Hardened for ephemeral file safety, payload size limits, and CORS security.
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

DEMO_MODE = False
MAX_BASE64_CHARS = 35_000_000  # ~25 MB max payload to prevent DoS memory exhaustion

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
    if DEMO_MODE:
        import random
        await asyncio.sleep(random.uniform(0.8, 2.0))
        is_fake_demo = random.random() > 0.35
        fake_prob = random.uniform(72.0, 96.5) if is_fake_demo else random.uniform(8.0, 28.0)
        local_label = "FAKE" if is_fake_demo else "REAL"
        local_conf = fake_prob if is_fake_demo else (100.0 - fake_prob)
        return AnalysisResponse(
            primary_verdict=local_label,
            confidence_score=safe_round(local_conf, 2),
            local_label=local_label,
            local_confidence=safe_round(local_conf, 2),
            rd_status="SUSPICIOUS" if is_fake_demo else "AUTHENTIC",
            rd_score=safe_round(random.uniform(0.70, 0.95) if is_fake_demo else random.uniform(0.05, 0.20), 4),
            forensic_summary="DEMO MODE: Simulated deepfake forensic finding.",
            latency_ms=safe_round(random.uniform(800, 3200), 2),
        )

    temp_path: Optional[str] = None
    try:
        header, data = request.base64_data.split(",", 1) if "," in request.base64_data else (None, request.base64_data)
        try:
            image_bytes = base64.b64decode(data, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 media payload.")

        ext = ".png"
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
                    break

        temp_dir = os.path.join(_script_dir, "tmp_api")
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        exif_extracted = None
        try:
            with Image.open(temp_path) as img:
                exif_data_raw = img.getexif() if hasattr(img, "getexif") else None
                if exif_data_raw:
                    exif_extracted = {}
                    for tag_id, value in exif_data_raw.items():
                        tag_name = str(ExifTags.TAGS.get(tag_id, tag_id))
                        if tag_name in ("GPSInfo", "MakerNote", "UserComment"):
                            continue  # Strip sensitive location/device serial tags from response
                        if isinstance(value, (str, int, float)):
                            exif_extracted[tag_name] = value
        except Exception:
            pass

        checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=500,
                detail="Model checkpoint 'best_model.pt' not found in 'model/'."
            )

        start_time = time.perf_counter()
        try:
            result: InferenceResult = await asyncio.to_thread(run_inference, temp_path, checkpoint_path)
        except Exception as inference_err:
            raise HTTPException(status_code=500, detail=f"Inference failed: {inference_err}")
        end_time = time.perf_counter()

        pct = result.fake_probability * 100
        local_label = "FAKE" if result.label == "FAKE" else "REAL"
        local_conf = pct if local_label == "FAKE" else (100 - pct)

        rd = result.rd_result
        rd_status = rd.get("status") if rd else "DISABLED"
        rd_score_raw = rd.get("score") if rd else None
        rd_score = float(rd_score_raw) if rd_score_raw is not None else 0.0

        if rd and rd_status in ["FAKE", "MANIPULATED", "SUSPICIOUS"] and rd_score_raw is not None:
            primary_verdict = "FAKE"
            confidence_score = rd_score * 100
            summary = f"CLOUD VERDICT: {rd_status} ({confidence_score:.1f}%). Local model reports {local_label} ({local_conf:.1f}%)."
        elif rd and rd_status == "AUTHENTIC" and rd_score_raw is not None:
            primary_verdict = "REAL"
            confidence_score = (1.0 - rd_score) * 100
            summary = "CLOUD VERDICT: AUTHENTIC. High confidence cloud analysis confirmed media is real."
        else:
            primary_verdict = local_label
            confidence_score = local_conf
            rd_err = f": {rd.get('error')}" if rd and rd.get("error") else ""
            rd_reason = f" ({rd_status}{rd_err})" if rd_status and rd_status not in ["DISABLED", "None"] else ""
            summary = f"LOCAL VERDICT: {local_label} ({local_conf:.1f}%). Cloud verification unavailable{rd_reason}."

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

class SubscriptionRequest(BaseModel):
    email: str = Field(..., max_length=254)

@app.post("/subscribe")
async def subscribe(request: SubscriptionRequest):
    return {"status": "success", "message": f"[SIMULATION] Subscription recorded for {request.email}."}

@app.get("/health")
async def health_check():
    return {"status": "online", "model": "EfficientNet-V2-S"}

class AdvanceScanRequest(BaseModel):
    base64_data: str = Field(..., max_length=MAX_BASE64_CHARS)

class AdvanceScanResponse(BaseModel):
    analysis: str

@app.post("/advance-scan", response_model=AdvanceScanResponse)
async def advance_scan(request: AdvanceScanRequest):
    temp_path: Optional[str] = None
    try:
        header, data = request.base64_data.split(",", 1) if "," in request.base64_data else (None, request.base64_data)
        try:
            image_bytes = base64.b64decode(data, validate=True)
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

        temp_dir = os.path.join(_script_dir, "tmp_api")
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"gemini_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(image_bytes)

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
