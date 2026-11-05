"""
whatsapp_bot.py — Hardened WhatsApp Bot for Trinetra Deepfake Detection.
Includes Meta X-Hub-Signature-256 verification, bounded worker pool,
path-traversal prevention, and guaranteed ephemeral media deletion.
"""

import os
import sys
import hmac
import hashlib
import uuid
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import httpx
from flask import Flask, request, jsonify
from dotenv import load_dotenv

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from src.inference.infer import run_inference
from src.inference.ai_summary import generate_ai_summary

env_path = os.path.join(_backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_ID = os.getenv("WHATSAPP_PHONE_ID") or os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")
TEMP_DOWNLOAD_DIR = os.path.join(_backend_dir, "processed_data", "whatsapp_media")

os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trinetra.WhatsAppBot")

app = Flask(__name__)

_processed_messages: set[str] = set()
_msg_lock = threading.Lock()
_worker_pool = ThreadPoolExecutor(max_workers=2)

def cleanup_processed_messages():
    while True:
        time.sleep(3600)
        with _msg_lock:
            _processed_messages.clear()

threading.Thread(target=cleanup_processed_messages, daemon=True).start()

def verify_webhook_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """Verify Meta X-Hub-Signature-256 HMAC header."""
    if not APP_SECRET:
        logger.error("WHATSAPP_APP_SECRET is not configured. Webhook rejected (fail-closed).")
        return False
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    received = signature_header.split("sha256=", 1)[1]
    return hmac.compare_digest(expected, received)

def send_whatsapp_message(to: str, text: str) -> None:
    url = f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=15.0)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Error in send_whatsapp_message: {e}")

def mark_as_read(message_id: str) -> None:
    url = f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    try:
        httpx.post(url, json=payload, headers=headers, timeout=10.0)
    except Exception as e:
        logger.error(f"Failed to mark message as read: {e}")

def download_media(media_id: str) -> str | None:
    if not media_id or not str(media_id).isalnum():
        logger.error("Invalid media_id format rejected.")
        return None
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    try:
        resp = httpx.get(url, headers=headers, timeout=15.0)
        resp.raise_for_status()
        media_url = resp.json().get("url")
        mime_type = resp.json().get("mime_type", "")

        file_resp = httpx.get(media_url, headers=headers, timeout=30.0)
        file_resp.raise_for_status()

        ext = ".jpg"
        if "video" in mime_type:
            ext = ".mp4"
        elif "audio" in mime_type:
            ext = ".mp3"
        elif "png" in mime_type:
            ext = ".png"

        safe_name = f"wa_{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(TEMP_DOWNLOAD_DIR, safe_name)
        with open(file_path, "wb") as f:
            f.write(file_resp.content)

        return file_path
    except Exception as e:
        logger.error(f"Media download failed: {e}")
        return None

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token and VERIFY_TOKEN and hmac.compare_digest(token, VERIFY_TOKEN):
        return challenge or "", 200
    return "Forbidden", 403

def process_message_async(sender: str, msg_type: str, msg_data: dict, msg_id: str) -> None:
    file_path: str | None = None
    try:
        mark_as_read(msg_id)

        if msg_type == "text":
            text = msg_data.get("body", "").lower()
            if "hi" in text or "hello" in text:
                send_whatsapp_message(
                    sender,
                    "Welcome to Trinetra! Send an image or video to check for deepfakes.",
                )
            else:
                send_whatsapp_message(
                    sender,
                    "Please send a media file (Image/Video/Audio) for forensic analysis.",
                )
        elif msg_type in ["image", "video", "audio"]:
            media_id = msg_data.get("id")
            send_whatsapp_message(sender, f"Analyzing your {msg_type}... Please wait.")

            file_path = download_media(media_id)
            if not file_path:
                send_whatsapp_message(sender, "Sorry, could not download the file. Please try again.")
                return

            result = run_inference(file_path)
            rd = result.rd_result
            response_text = "🔍 *Trinetra Report*\n"

            if result.label == "AUDIO":
                response_text += "🤖 Local AI: ℹ️ Audio (Cloud only)\n"
            else:
                local_icon = "🟢" if result.label == "REAL" else "🔴"
                response_text += f"🤖 Local AI: {local_icon} {result.label} ({result.fake_probability * 100:.1f}%)\n"

            if rd:
                if rd.get("status") in ("ERROR", "SKIPPED"):
                    response_text += f"☁️ Cloud: ⚠️ {rd.get('error', 'unavailable')}\n"
                else:
                    rd_status = rd.get("status")
                    rd_icon = "🟢" if rd_status == "AUTHENTIC" else "🔴" if rd_status == "MANIPULATED" else "🟡"
                    response_text += f"☁️ Cloud: {rd_icon} {rd_status} ({rd.get('score', 0) * 100:.1f}%)\n"

            ai_summary = generate_ai_summary(result)
            if ai_summary:
                response_text += f"\n🧠 *Summary*: {ai_summary}"

            send_whatsapp_message(sender, response_text)
    except Exception as e:
        logger.error(f"Error in background process: {e}", exc_info=True)
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

@app.route("/webhook", methods=["POST"])
def handle_webhook():
    raw_body = request.get_data()
    sig_header = request.headers.get("X-Hub-Signature-256")
    if not verify_webhook_signature(raw_body, sig_header):
        logger.warning("Rejected webhook with invalid X-Hub-Signature-256")
        return jsonify({"error": "Invalid signature"}), 403

    data = request.get_json(silent=True, force=True) or {}
    try:
        if data.get("entry"):
            for entry in data["entry"]:
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    for msg in messages:
                        msg_id = msg.get("id")
                        sender = msg.get("from")
                        msg_type = msg.get("type")
                        if not msg_id or not sender or not msg_type:
                            continue
                        with _msg_lock:
                            if msg_id in _processed_messages:
                                continue
                            _processed_messages.add(msg_id)
                        msg_data = msg.get(msg_type, {}) if msg_type != "text" else msg.get("text", {})
                        _worker_pool.submit(process_message_async, sender, msg_type, msg_data, msg_id)
    except Exception as e:
        logger.error(f"Error handling webhook: {e}", exc_info=True)

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
