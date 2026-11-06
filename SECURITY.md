# Trinetra Security Policy and Architecture

## 1. Overview and Security Principles

Trinetra is an open-source deepfake and synthetic media forensic analysis platform. Because digital forensics often deals with sensitive, contested, or evidentiary visual media, the project enforces a defense-in-depth security model across its backend, frontend, and browser extension.

Key security principles include:
- Least Privilege: Services and browser extensions request only the minimal permissions necessary to perform inference.
- Constant-Time Cryptographic Verification: All secret tokens are verified using timing-attack resistant comparisons.
- Strict Input Validation: All network payloads undergo schema validation, size limits, and sanitization before processing.
- Ephemeral Execution: Media payloads are kept in volatile memory and purged immediately after analysis to eliminate attack surfaces at rest.

---

## 2. Supported Versions

We actively maintain and provide security patches for the following versions:

| Version | Supported | Security Patch Policy |
| :--- | :--- | :--- |
| 2.0.x | Yes | Actively supported with high-priority security patches. |
| 1.x.x | No | Deprecated. Users are urged to upgrade to v2.0+. |

---

## 3. Reporting a Vulnerability (Coordinated Disclosure)

We take the security of Trinetra and its users seriously. If you discover a security vulnerability or potential threat, please report it responsibly before any public disclosure.

### How to Report
- Submit a confidential report via GitHub Private Security Advisories:
  Navigate to the repository's **Security** tab and click **Report a vulnerability**.
- Alternatively, email the security maintainers at: `security@trinetra-project.org` (or open an issue with the label `security-confidential`).

### What to Include
Please provide as much information as possible to help us reproduce and resolve the issue:
1. Type of vulnerability (e.g., Remote Code Execution, Denial of Service, Cross-Site Scripting, Privilege Escalation).
2. Affected component (Backend API, Next.js Frontend, Chrome Extension, or WhatsApp Bot).
3. Step-by-step instructions or a minimal Proof of Concept (PoC).
4. Potential impact or threat model assessment.

### Response Timeline
- Initial Acknowledgment: Within 48 hours of receipt.
- Triage and Validation: Within 5 business days.
- Remediation and Patch Release: Critical issues are addressed and patched within 14 days, followed by a public advisory crediting the reporter.

---

## 4. Application Security Controls

### A. Authentication and Timing-Attack Protection
The backend API (`backend/api.py`) guards forensic endpoints (`/analyze`, `/advance-scan`) with a bearer token authentication gate:
- Supported headers: `Authorization: Bearer <token>` and `X-API-Key: <token>`.
- Constant-Time Comparison: Secret comparison uses `hmac.compare_digest` to prevent side-channel timing attacks that could reveal the valid secret through response duration variances.
- Failure Mode: Requests without a valid token are rejected immediately with HTTP 401 Unauthorized before media decoding or computational inference occurs.

### B. Payload Validation and DoS Mitigation
- Max Payload Limit: Base64 strings sent to `/analyze` and `/advance-scan` are strictly capped at `MAX_BASE64_CHARS = 35_000_000` (approximately 25 MB). Payloads exceeding this threshold are rejected at the Pydantic schema validation layer.
- Memory Exhaustion Protection: Raw byte decoding is validated before passing to PIL or OpenCV, preventing memory overflow and decompression bomb attacks.
- Explicit Consent Enforcement: The `/advance-scan` endpoint strictly requires `consent_confirmed: true` in the JSON request body. Requests without explicit consent are blocked with HTTP 400.

### C. Metadata Sanitization (EXIF and GPS Scrubbing)
Before an uploaded image is written to disk or evaluated by any model:
- The `sanitize_and_strip_image` function strips all GPS coordinates, camera serial numbers, device model tags, user comments, and personal metadata.
- Images are re-encoded into pure RGB pixel buffers to ensure no malicious steganographic payloads or Malformed Header exploits execute against downstream libraries.

### D. Model Serialization and Checkpoint Security
PyTorch checkpoint loading is a known vector for arbitrary code execution if untrusted pickle files are loaded:
- Trinetra enforces safe checkpoint verification (`check_torch_format.py` and `src/models/`).
- Arbitrary pickle loading from untrusted external URLs is strictly prohibited.
- Production models use explicit state dictionary mapping (`torch.load(weights_path, map_location=device, weights_only=True)` when available).

---

## 5. Browser Extension Security (Manifest V3)

The Trinetra Chrome Extension adheres to Google Chrome Manifest V3 security requirements:

### A. Strict Origin Isolation and Content Security Policy (CSP)
- No Remote Code Execution: The extension does not use `eval()`, `new Function()`, or load external scripts from CDNs. All scripts (`background.js`, `content.js`, `popup.js`) are bundled locally within the extension directory.
- DOM Sanitization (XSS Prevention):
  - In `content.js`, text received from API responses (verdicts, summaries, confidence scores) is inserted into DOM elements strictly using `textContent` rather than `innerHTML`.
  - Image data URLs (`data:image/png;base64,...`) are validated against a strict base64 alphanumeric regex (`/^[A-Za-z0-9+/=]+$/`) before being bound to `src` attributes, preventing `javascript:` URI injection.

### B. Least-Privilege Permissions
- `activeTab`: Grants access to the current page only upon user initiation (clicking Scan Region or using the context menu).
- `storage`: Scoped to `chrome.storage.local` to store user settings and recent session scan logs on-device.
- `scripting`: Dynamically executes content scripts only on the targeted active tab when requested.

---

## 6. Network and CORS Policy

- Local API Binding: By default, the FastAPI server binds to loopback (`127.0.0.1:8000`).
- Cross-Origin Resource Sharing (CORS):
  - Allowed web origins are restricted to `http://localhost:3000` and `http://127.0.0.1:3000` by default.
  - Allowed browser extension origins are governed by a regex pattern (`r"^chrome-extension://.*$"`), preventing arbitrary unauthorized third-party websites from making cross-origin requests to your local analysis server.
  - Production deployments can customize allowed origins via the `ALLOWED_ORIGINS` environment variable.

---

## 7. Secrets and Environment Management

- Never Commit Secrets: The repository's `.gitignore` file explicitly ignores `.env`, `*.pt`, `*.pth`, `*.bin`, temporary media directories (`tmp_api/`, `processed_data/`), and virtual environments (`venv/`).
- Configuration Isolation: All API keys (`TRINETRA_API_KEY`, `RD_API_KEY`, `GEMINI_API_KEY`, `WHATSAPP_*`) are loaded from system environment variables or a local `.env` file via `python-dotenv`.
- Ephemeral Scratch Files: Any temporary files written to `backend/tmp_api/` are automatically deleted in `finally` execution blocks, ensuring no media artifacts remain on disk.

---

## 8. Deployment Recommendations for Production

When deploying Trinetra in a public or multi-user environment:
1. Enforce HTTPS: Always place the FastAPI backend behind an NGINX or Caddy reverse proxy with valid TLS certificates.
2. Rotate API Keys: Change the default `TRINETRA_API_KEY` from the development token to a strong, cryptographically generated string (e.g., `openssl rand -hex 32`).
3. Rate Limiting: Implement rate-limiting proxies (such as Redis-backed Cloudflare or NGINX `limit_req`) to protect against brute-force attacks and denial-of-service attempts.
4. Container Isolation: Run the backend and models inside an unprivileged Docker container with restricted filesystem write permissions.
