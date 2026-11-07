# Trinetra Chrome Extension — In-Browser Media Scanner

**A lightweight, privacy-first Manifest V3 browser extension for real-time deepfake detection and media authentication.**

---

## Overview & New White Theme

The **Trinetra Chrome Extension** brings the forensic power of the Trinetra AI engine directly into your daily web browsing experience. Operating via Chrome's modern **Manifest V3** standard, the extension lets you easily drag-and-select any region of any webpage (or right-click any image) to inspect it for deepfake and synthetic manipulation.

### What's New in v2.0:
- **Modern White Theme:** A clean, high-contrast, distraction-free aesthetic with crisp typography, soft elevations, and smooth animations.
- **On-Screen Region Selection:** Click & drag across faces, video frames, or suspicious images with a live laser scanline overlay.
- **Mandala Loading Animation:** An animated, Indian-heritage mandala spinner displaying live scanning states.
- **Forensic Modal:** Shows primary verdict (**REAL** or **FAKE**), confidence score gauge, source face extraction crop, and Explainable AI **Grad-CAM heatmap**.
- **Interactive Tools:** One-click **Copy Report** to clipboard and a sliding **Scan History Drawer** to review past inspections.
- **Local / Cloud Toggles:** Easily switch between on-device inference (Local CNN) and cloud verification (Reality Defender).

---

## Architecture & Files

The extension follows a clean, decoupled architecture:

- **Popup Interface (`popup.html`, `popup.css`, `popup.js`):**
  The toolbar popup for enabling detection engines, checking local backend health (port `8000`), and launching region scans.
- **In-Page Content Scripts (`content.js`, `content.css`):**
  Securely injected into the active webpage to render the crosshair region selector, scanline, mandala loader, forensic result modal, and history drawer.
- **Background Service Worker (`background.js`):**
  Coordinates tab capture, context-menu clicks ("Scan Image with Trinetra"), and API requests to the local Trinetra backend (`http://localhost:8000/analyze`).
- **Interactive UI Preview (`preview.html`):**
  A standalone developer playground to test and preview both the popup and in-page modal without having to trigger a live browser scan.

---

## Installation (Developer Mode)

1. Open **Google Chrome**, **Brave**, **Microsoft Edge**, or any Chromium-based browser.
2. In the URL bar, go to `chrome://extensions/`
3. Enable the **"Developer mode"** toggle in the top-right corner.
4. Click the **"Load unpacked"** button in the top-left.
5. Select this `extension/` folder.
6. The **Trinetra** icon will now appear in your browser's toolbar. Pin it for quick access!

---

## Quick UI Preview

Want to test or inspect the extension's design without running the backend?

Simply open [preview.html](preview.html) in your browser. You can:
- Toggle between **FAKE** and **REAL** verdict states.
- Inspect the confidence arc gauge animation.
- Test the one-click **Copy Report** button.
- Preview the extension popup side-by-side.

---

## Prerequisites

To perform real-time forensic scanning, the local Trinetra backend API must be running:

```bash
# In the Trinetra/backend directory:
python api.py
```

*The API runs at `http://127.0.0.1:8000`. By default, the extension connects to this endpoint using the developer authentication token.*

---

## Privacy & Permissions

Trinetra operates under strict privacy and least-privilege principles:
- **`activeTab`**: Temporary access to the currently visible tab only when you trigger a scan.
- **`scripting`**: Injects the selector and result card into the DOM on-demand.
- **`contextMenus`**: Adds the "Scan Image with Trinetra" right-click shortcut.
- **Zero Tracking**: No browsing history, telemetry, or personal data is collected or tracked.
