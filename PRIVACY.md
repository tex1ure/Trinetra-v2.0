# Trinetra Privacy Policy and User Privacy Guide

Last Updated: November 2026

## 1. Introduction and Core Philosophy

Trinetra is an open-source deepfake and synthetic media forensic analysis platform. The software was engineered under a strict Privacy-by-Design and Local-First philosophy. 

Unlike conventional cloud-only verification services that require uploading media to third-party servers, Trinetra provides powerful forensic capabilities directly on your local device. We believe digital forensics should never come at the expense of user privacy, intellectual property, or personal security.

---

## 2. Summary of Privacy Commitments

- Zero Personal Telemetry: Trinetra does not log, track, or transmit your identity, browsing habits, IP address, or device identifiers.
- On-Device First: The primary neural network (EfficientNet-V2-S + Transformer) executes entirely on your local machine.
- Automatic EXIF and Metadata Stripping: All GPS coordinates, device serial numbers, and camera metadata are wiped automatically upon ingestion before any analysis takes place.
- Explicit Opt-In for Cloud Features: No visual media or data is ever shared with external APIs (Reality Defender or Google Gemini) without your explicit toggle and confirmation.
- Ephemeral In-Memory Processing: Media submitted for analysis is processed statelessly in volatile buffers and immediately removed after inference.

---

## 3. Information Handled by Trinetra

### A. Media Files (Images and Video Frames)
- What is processed: Visual media you select through the web dashboard, highlight using the browser extension, or submit to the WhatsApp bot.
- How it is handled:
  - The file is decoded in volatile memory.
  - All non-essential metadata is stripped.
  - The frame is passed through the local face detector and feature extractor.
  - Once the verdict and Grad-CAM heatmap are returned, any temporary processing buffer is purged from disk.
- Storage: Zero permanent storage. Trinetra does not maintain an image database or repository of user submissions.

### B. Metadata and EXIF Information
Photographs often contain sensitive Exchangeable Image File Format (EXIF) data, including:
- Exact GPS coordinates of where a photo was taken.
- Device make, model, and hardware serial numbers.
- Timestamps, camera owner names, and proprietary vendor notes.

Trinetra enforces an automated sanitization function (`sanitize_and_strip_image`) before any model processing occurs. The following tags are permanently removed:
- GPSInfo (Latitude, Longitude, Altitude, Timestamp)
- Make and Model
- BodySerialNumber and LensSerialNumber
- Artist, Copyright, and OwnerName
- MakerNote and UserComment
- Software configuration history that could identify individuals

Only non-identifying technical tags essential for detecting software manipulation are retained (e.g., `ColorSpace`, `ExifImageWidth`, `ExifImageHeight`).

### C. Authentication Credentials
- Local API keys (`TRINETRA_API_KEY`) and third-party service tokens (`RD_API_KEY`, `GEMINI_API_KEY`, Meta tokens) are stored solely in your local `.env` configuration file on your machine.
- Keys are never transmitted to the project maintainers, remote servers, or client browsers.

---

## 4. Browser Extension Privacy Practices

The Trinetra Chrome Extension adheres strictly to Google Chrome Manifest V3 privacy standards:

### A. Scope of Permissions
- `activeTab`: Used exclusively to inspect media on the specific tab you are currently viewing when you explicitly click "Scan Region" or choose "Scan Image with Trinetra". It does not grant access to background tabs or unvisited domains.
- `scripting`: Used only to inject the on-screen region selector overlay and forensic modal onto the current page.
- `contextMenus`: Adds a right-click context menu option to trigger media inspection on demand.

### B. Browsing History and Page Content
- The extension does not record, inspect, or monitor your web browsing history.
- It does not read text, passwords, form fields, cookies, or user session tokens.
- It only captures the specific pixel region you manually click and drag across with the selection tool.

### C. Client-Side Scan History
- The extension includes an optional session history drawer.
- All scan history entries (verdicts and timestamps) are saved purely in local browser storage (`chrome.storage.local`).
- This data remains strictly within your browser sandbox and is never synced, uploaded, or exposed externally. You can clear this history at any time.

---

## 5. Local Processing vs. Third-Party Cloud Services

Trinetra provides a dual-layer architecture with full transparency on where processing occurs:

| Processing Tier | Engine | Where It Runs | Data Sent Externally | Consent Requirement |
| :--- | :--- | :--- | :--- | :--- |
| Layer 1: Default | Local CNN Hybrid (`EfficientNet-V2-S`) | Your Local CPU / GPU | None. 100% on-device. | Default enabled. |
| Layer 2: Cloud Fallback | Reality Defender API | Reality Defender Cloud | Sanitized image pixels only (EXIF stripped). | Requires explicit opt-in switch. |
| Layer 3: Advanced Multimodal | Google Gemini API | Google Cloud Platform | Sanitized image pixels only (EXIF stripped). | Requires explicit consent confirmation. |

If you do not enable cloud verification or provide external API keys, Trinetra functions entirely offline as a local-only tool.

---

## 6. WhatsApp Bot Integration Privacy

When operating the WhatsApp fact-checking chatbot (`whatsapp_bot.py`):
- Media submitted by users over WhatsApp is received via Meta's WhatsApp Cloud API webhook.
- The media is downloaded through temporary encrypted endpoints, processed through the Trinetra forensic engine, and the forensic summary is returned as a message.
- Temporary files generated during webhook processing are purged immediately following response dispatch.
- Trinetra does not build phone number databases, harvest user profiles, or sell contact data.

---

## 7. Data Security and Storage Lifecycle

- Data in Transit: Local communication between the browser extension and the local backend runs over local loopback (`http://127.0.0.1:8000`). In production deployments, HTTPS / TLS encryption is strongly mandated.
- Data at Rest: No visual payloads or user prompts are persisted to disk beyond temporary execution lifecycles.
- Temporary Buffers: Files saved in `backend/tmp_api/` for processing are generated with cryptographic UUID filenames and guaranteed to be deleted in execution `finally` blocks.

---

## 8. User Rights and Controls

As an open-source tool, Trinetra gives you complete control over your data:
- Inspect the Source Code: You can verify every line of code that handles media ingestion, EXIF stripping, and inference directly in this repository.
- Run Offline: You can disconnect from the internet and run the local model without any external network access.
- Wipe Local History: You can remove all browser extension history at any time by clearing extension storage or reinstalling the unpacked extension.

---

## 9. Contact and Inquiries

If you have questions, concerns, or feedback regarding Trinetra's privacy architecture:
- Open a discussion or issue on GitHub: https://github.com/
- Review our security guidelines in [SECURITY.md](SECURITY.md).
