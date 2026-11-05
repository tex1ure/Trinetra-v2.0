(() => {
  if (window.__trinetraContentLoaded) return;
  window.__trinetraContentLoaded = true;

  let activeCleanup = null;

  function removeExistingUI() {
    document.querySelectorAll(
      '.trinetra-selection-overlay, .trinetra-selection-box, .trinetra-loader-overlay, .trinetra-modal, .trinetra-history-drawer'
    ).forEach((el) => el.remove());
    if (activeCleanup) {
      activeCleanup();
      activeCleanup = null;
    }
  }

  function showLoader() {
    removeExistingUI();
    const loader = document.createElement('div');
    loader.className = 'trinetra-loader-overlay';
    loader.innerHTML = `
      <div class="trinetra-mandala">
        <div class="trinetra-mandala-ring"></div>
        <div class="trinetra-mandala-ring"></div>
        <div class="trinetra-mandala-ring"></div>
        <div class="trinetra-mandala-ring"></div>
        <div class="trinetra-mandala-eye"></div>
      </div>
      <div class="trinetra-loader-text">ANALYZING</div>
      <div class="trinetra-loader-subtext">त्रिनेत्र — सत्य का अन्वेषण</div>
    `;
    document.body.appendChild(loader);
  }

  function sanitizeBase64(str) {
    if (!str || typeof str !== 'string') return null;
    return /^[A-Za-z0-9+/=]+$/.test(str) ? str : null;
  }

  function showErrorModal(errorMsg) {
    removeExistingUI();
    const modal = document.createElement('div');
    modal.className = 'trinetra-modal';
    modal.innerHTML = `
      <div class="trinetra-modal-content">
        <div class="trinetra-modal-inner">
          <div class="trinetra-header">
            <div class="trinetra-brand-group">
              <span class="trinetra-brand">TRINETRA <span>// V2</span></span>
            </div>
            <div class="trinetra-close" id="tri-close-btn">✕</div>
          </div>
          <div class="trinetra-error-panel">
            <div class="trinetra-error-icon">⚠️</div>
            <div class="trinetra-error-title">ANALYSIS FAILED</div>
            <div class="trinetra-error-msg"></div>
          </div>
        </div>
      </div>
    `;
    modal.querySelector('.trinetra-error-msg').textContent = String(errorMsg || 'Analysis failed.');
    document.body.appendChild(modal);
    modal.querySelector('#tri-close-btn').addEventListener('click', removeExistingUI);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) removeExistingUI();
    });
  }

  function showResultModal(result) {
    removeExistingUI();
    const isFake = result.primary_verdict === 'FAKE';
    const conf = Number(result.confidence_score || 0).toFixed(1);
    const arcLen = 251.33;
    const dashOffset = (arcLen * (1 - Math.min(100, Math.max(0, Number(conf))) / 100)).toFixed(2);
    const safeFace = sanitizeBase64(result.face_crop_base64);
    const safeGrad = sanitizeBase64(result.gradcam_base64);

    const modal = document.createElement('div');
    modal.className = 'trinetra-modal';
    modal.innerHTML = `
      <div class="trinetra-modal-content">
        <div class="trinetra-modal-inner">
          <div class="trinetra-header">
            <div class="trinetra-brand-group">
              <span class="trinetra-brand">TRINETRA <span>// V2</span></span>
              <span class="trinetra-hindi-sub">त्रिनेत्र — सत्य का तीसरा नेत्र</span>
            </div>
            <div class="trinetra-header-right">
              <span class="trinetra-latency"></span>
              <div class="trinetra-close" id="tri-close-btn">✕</div>
            </div>
          </div>

          <div class="trinetra-gauge-container">
            <svg class="trinetra-gauge-svg" viewBox="0 0 200 120">
              <path class="trinetra-gauge-track" d="M 20 100 A 80 80 0 0 1 180 100" />
              <path class="trinetra-gauge-fill"
                    d="M 20 100 A 80 80 0 0 1 180 100"
                    stroke="${isFake ? 'var(--fake-color)' : 'var(--real-color)'}"
                    stroke-dasharray="${arcLen}"
                    stroke-dashoffset="${dashOffset}" />
              <text class="trinetra-gauge-value" x="100" y="78">${conf}%</text>
              <text class="trinetra-gauge-unit" x="100" y="96">CONFIDENCE</text>
            </svg>
          </div>

          <div class="trinetra-verdict-main ${isFake ? 'status-fake' : 'status-real'}">
            <div class="trinetra-yantra-tl"></div>
            <div class="trinetra-yantra-tr"></div>
            <div class="trinetra-yantra-bl"></div>
            <div class="trinetra-yantra-br"></div>
            <div>
              <div class="trinetra-verdict-label"></div>
              <div class="trinetra-verdict-sub">CONSOLIDATED FORENSIC VERDICT</div>
            </div>
          </div>

          <div class="trinetra-visuals-row ${isFake ? 'status-fake-parent' : 'status-real-parent'}">
            <div class="trinetra-visual-card">
              <div class="trinetra-visual-card-label">Source Extraction</div>
              <div class="trinetra-face-crop-wrap">
                ${
                  safeFace
                    ? `<div class="trinetra-face-crop-circle"><img src="data:image/png;base64,${safeFace}" alt="Face" /></div>`
                    : `<div class="trinetra-no-face">No Face<br/>Detected</div>`
                }
              </div>
            </div>
            <div class="trinetra-visual-card">
              <div class="trinetra-visual-card-label">Grad-CAM Heatmap</div>
              <div class="trinetra-gradcam-wrap">
                ${
                  safeGrad
                    ? `<img class="trinetra-gradcam-img" src="data:image/png;base64,${safeGrad}" alt="Grad-CAM" />`
                    : `<div class="trinetra-no-gradcam">No heatmap<br/>available</div>`
                }
              </div>
            </div>
          </div>

          <div class="trinetra-results-grid">
            <div class="trinetra-result-cell">
              <span class="trinetra-label">LOCAL MODEL ⬥ HYBRID</span>
              <span class="trinetra-value trinetra-local-val ${result.local_label === 'FAKE' ? 'fake-text' : 'real-text'}"></span>
            </div>
            <div class="trinetra-result-cell">
              <span class="trinetra-label">RD CLOUD</span>
              <span class="trinetra-value trinetra-rd-val muted"></span>
            </div>
          </div>

          <div class="trinetra-summary"></div>

          <div class="trinetra-footer">
            <div class="trinetra-footer-source">
              <span class="trinetra-footer-source-dot online"></span>
              LOCAL ENGINE
            </div>
            <div class="trinetra-footer-esc">Esc to close</div>
          </div>
        </div>
      </div>
    `;

    // Safely assign dynamic textual content using textContent to prevent XSS
    modal.querySelector('.trinetra-latency').textContent = `${Math.round(result.latency_ms || 0)}ms`;
    modal.querySelector('.trinetra-verdict-label').textContent = String(result.primary_verdict || 'UNKNOWN');
    modal.querySelector('.trinetra-local-val').textContent = `${result.local_label || 'UNKNOWN'} (${Number(result.local_confidence || 0).toFixed(1)}%)`;
    modal.querySelector('.trinetra-rd-val').textContent = String(result.rd_status || 'LOCAL_ONLY');
    modal.querySelector('.trinetra-summary').textContent = String(result.forensic_summary || '');

    document.body.appendChild(modal);

    const onKeyDown = (e) => {
      if (e.key === 'Escape') removeExistingUI();
    };
    window.addEventListener('keydown', onKeyDown);
    activeCleanup = () => window.removeEventListener('keydown', onKeyDown);

    modal.querySelector('#tri-close-btn').addEventListener('click', removeExistingUI);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) removeExistingUI();
    });
  }

  function startRegionSelection(allowCloud) {
    removeExistingUI();
    const overlay = document.createElement('div');
    overlay.className = 'trinetra-selection-overlay';

    const box = document.createElement('div');
    box.className = 'trinetra-selection-box';
    box.style.display = 'none';
    box.innerHTML = '<div class="trinetra-scan-line"></div>';

    document.body.appendChild(overlay);
    document.body.appendChild(box);

    let startX = 0;
    let startY = 0;
    let dragging = false;

    const onMouseDown = (e) => {
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      box.style.left = `${startX}px`;
      box.style.top = `${startY}px`;
      box.style.width = '0px';
      box.style.height = '0px';
      box.style.display = 'block';
    };

    const onMouseMove = (e) => {
      if (!dragging) return;
      const x = Math.min(startX, e.clientX);
      const y = Math.min(startY, e.clientY);
      const w = Math.abs(e.clientX - startX);
      const h = Math.abs(e.clientY - startY);
      box.style.left = `${x}px`;
      box.style.top = `${y}px`;
      box.style.width = `${w}px`;
      box.style.height = `${h}px`;
    };

    const onMouseUp = (e) => {
      if (!dragging) return;
      dragging = false;
      const rect = box.getBoundingClientRect();
      removeExistingUI();

      if (rect.width < 24 || rect.height < 24) return;

      setTimeout(() => {
        chrome.runtime.sendMessage({ action: 'TRINETRA_CAPTURE_TAB' }, (resp) => {
          if (!resp || resp.error || !resp.dataUrl) {
            showErrorModal(resp?.error || 'Could not capture tab.');
            return;
          }
          showLoader();
          const img = new Image();
          img.onload = () => {
            const dpr = window.devicePixelRatio || 1;
            const canvas = document.createElement('canvas');
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(
              img,
              rect.left * dpr,
              rect.top * dpr,
              rect.width * dpr,
              rect.height * dpr,
              0,
              0,
              canvas.width,
              canvas.height
            );
            const croppedBase64 = canvas.toDataURL('image/png');
            chrome.runtime.sendMessage(
              { action: 'TRINETRA_ANALYZE_CROP', base64Data: croppedBase64, allowCloud },
              (analysisResp) => {
                if (!analysisResp || analysisResp.error) {
                  showErrorModal(analysisResp?.error || 'Analysis failed.');
                } else {
                  showResultModal(analysisResp.result);
                }
              }
            );
          };
          img.src = resp.dataUrl;
        });
      }, 60);
    };

    overlay.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    activeCleanup = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'TRINETRA_START_SELECTION') startRegionSelection(msg.allowCloud);
    if (msg.action === 'TRINETRA_SHOW_LOADER') showLoader();
    if (msg.action === 'TRINETRA_SHOW_RESULT') showResultModal(msg.result);
    if (msg.action === 'TRINETRA_SHOW_ERROR') showErrorModal(msg.error);
  });
})();
