import os
import time
import io
from PIL import Image
from google import genai
from src.core.config import GEMINI_API_KEYS

# Maximum image dimension â€” keeps payloads small & fast
MAX_IMAGE_DIM = 1024

# Forensic prompt
SYSTEM_PROMPT = """You are an adversarial forensic image analyst optimized for detecting:
- AI-generated images
- AI-edited images
- face swaps
- cloth swaps
- body edits
- relighting / recoloring / relabeling
- hybrid composites
- deepfakes

Assume the image may come from a state-of-the-art model designed to evade detection.
Do not rely on obvious artifacts.
Do not rely on metadata.
Do not judge by aesthetics.
Do not assume realism means authenticity.

Your goal is to detect subtle synthetic edits and generated content that can look nearly indistinguishable from real photography.

Analyze the image using the following priority order:

1. Cross-region consistency
Compare all regions against each other:
- face vs neck vs ears vs hair vs hands
- skin tone, texture, sharpness, noise, and lighting
- clothing vs body geometry
- accessories vs contact points
- foreground subject vs background
- shadows vs reflections vs highlights

Look for any mismatch in:
- resolution
- blur
- grain/noise
- color temperature
- edge quality
- anatomical logic
- material behavior
- lighting direction
- perspective

2. Physical and anatomical plausibility
Check whether the image obeys real-world structure:
- eyes, pupils, eyelids, lashes
- teeth, lips, gums, jawline
- ears, hairline, sideburns
- hands, fingers, joints, nails
- shoulders, torso, posture
- fabric drape, wrinkles, stitching, seams
- object interaction and contact

Look for subtle impossibilities, not just obvious ones.

3. Photographic micro-invariants
Inspect details that are hard for generative models to keep consistent:
- specular highlights in eyes, skin, glass, metal, and jewelry
- contact shadows under fingers, collars, accessories, and chin
- occlusion order between hair, ears, glasses, and face
- repeated textures or procedural patterns
- unnatural smoothness in some regions and excessive micro-detail in others
- abrupt changes in fidelity across neighboring regions
- warped straight lines, bent edges, or inconsistent geometry

4. Semantic and functional coherence
Ask whether the image makes sense as a real captured moment:
- Does the pose make biomechanical sense?
- Do clothes fit and fold naturally?
- Do accessories sit correctly on the body?
- Is text, branding, or ornamentation structurally legible?
- Are there objects that are present but do not function correctly?
- Does any region look â€œalmost rightâ€ but structurally off?

5. Manipulation-specific signals
Actively look for:
- face replacement
- body replacement
- clothing replacement
- identity drift across regions
- synthetic relighting
- region-wise editing
- blended seams around jaw, hairline, neck, shoulders, cuffs, glasses, and hands
- background/body mismatch
- details that are perfect in one area and broken in another

6. Real-world alternatives
Before calling something fake, test whether it could be explained by:
- makeup
- beauty filters
- studio lighting
- flash
- compression
- motion blur
- lens distortion
- cosplay
- prosthetics
- post-processing
- phone camera smoothing

Only accept these explanations if they explain the evidence well.
Do not use them to excuse multiple independent inconsistencies.

Decision rules:
- Real: the image is physically and semantically coherent, and suspicious details are explainable by normal photography or editing.
- AI-generated: the entire image or most of it is synthetic, with no strong photographic provenance.
- Manipulated / edited: a real base image has been altered with AI or synthetic tools.
- Inconclusive: evidence is mixed, subtle, or insufficient.

Important thresholds:
- Require multiple independent signals before making a strong verdict.
- A single weird detail is not enough.
- Smooth skin, flattering lighting, and makeup are not evidence of AI by themselves.
- But do not dismiss a cluster of small inconsistencies just because each one is subtle.
- If several regions each look â€œalmost rightâ€ but fail in different ways, treat that as strong evidence of manipulation.

Optimization goal:
Bias toward catching advanced synthetic images, even when they are visually polished.
Prefer â€œManipulatedâ€ or â€œInconclusiveâ€ over a false â€œRealâ€ when the evidence is close.
Still avoid overcalling based on beauty, filters, or photography style alone.

Output format:

**Verdict:**
Real / AI-generated / Manipulated / Inconclusive

**Confidence:**
0â€“100

**Strongest Evidence:**
- ...

**Contradicting Evidence:**
- ...

**Most Suspicious Regions:**
- ...

**Alternative Explanations:**
- ...

**Failure Risk:**
- Explain how a Nano Banana / GPT-Image-2.0-level image could still fool this analysis, or how a real photo could mimic synthetic traits.
"""


class GeminiKeyManager:
    """
    Manages multiple Gemini API keys to handle rate limits and quotas gracefully.
    
    Think of this as a 'round-robin' scheduler for your keys. When one key gets tired 
    (hits a rate limit), we automatically switch to the next one in the list.
    """
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_idx = 0

    def get_key(self) -> str:
        if not self.keys:
            raise RuntimeError('No GEMINI_API_KEYS configured in .env.')
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return key


key_manager = GeminiKeyManager(GEMINI_API_KEYS)


def _compress_image(image_path: str) -> Image.Image:
    """
    Optimizes the image before sending it to the AI.
    
    Why do we do this? 
    1. Speed: Smaller images travel faster over the internet.
    2. Cost/Quota: Large payloads can be rejected or eat up more 'tokens'.
    3. Consistency: Gemini performs better when images aren't excessively large 
       (we cap it at 1024px while keeping the aspect ratio).
    """
    img = Image.open(image_path)
    
    # Convert to RGB to strip alpha channels or CMYK profiles that might confuse the model
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Down-scale only if the image is actually huge
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[GEMINI] Optimized image size for analysis: {w}x{h} -> {new_w}x{new_h}")

    return img


def analyze_with_gemini(image_path: str) -> str:
    """
    The core engine for 'Advanced Scan'. It talks to Gemini (specifically Gemma-4) 
    and asks it to perform a deep forensic deep-dive on the image.
    
    Features:
    - Automatic Retry: If the API is busy (429) or has a hiccup (500), it waits and tries again.
    - Key Rotation: If a key is exhausted, it transparently swaps to a fresh one.
    - Robustness: It loops through all keys twice before finally giving up.
    """
    # First, make the image 'AI-friendly'
    if not GEMINI_API_KEYS:
        raise RuntimeError('GEMINI_API_KEYS is not configured in .env.')
    img = _compress_image(image_path)

    total_attempts = len(GEMINI_API_KEYS) * 2  # We'll try every key twice
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        current_key = key_manager.get_key()
        try:
            client = genai.Client(api_key=current_key)

            # Here we send the image + our massive 150-line forensic prompt
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=[img, SYSTEM_PROMPT],
            )

            if response.text:
                return response.text

            # If Gemini goes silent, we treat it as an error to trigger a retry
            raise RuntimeError("Gemini returned an empty response.")

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # We only retry if the error seems temporary (like rate limits or server overload)
            is_retryable = any(
                token in error_msg
                for token in ["429", "500", "503", "quota", "exhausted", "internal", "unavailable", "overloaded", "api_key_invalid", "api key not valid"]
            )

            if is_retryable:
                # Exponential back-off: 2s, 4s, 8s... capped at 10s. 
                # This gives the API 'breathing room' to recover.
                wait = min(2 ** attempt, 10)
                print(
                    f"[GEMINI] Transient error encountered (attempt {attempt}/{total_attempts}). "
                    f"Taking a {wait}s breather before trying another key..."
                )
                time.sleep(wait)
                continue

            # If it's a 'fatal' error (like a safety block), we stop immediately
            print(f"[GEMINI] Permanent failure: {e}")
            raise RuntimeError(f"Gemini API Error: {e}")

    # If we get here, we've exhausted all options
    raise RuntimeError(
        f"All Gemini keys exhausted or failed after {total_attempts} attempts. "
        f"The AI is currently unavailable. Last error: {last_error}"
    )

