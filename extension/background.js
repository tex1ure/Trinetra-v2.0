const API_URL = 'http://localhost:8000/analyze';

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'trinetra-scan-image',
    title: 'Scan Image with Trinetra',
    contexts: ['image'],
  });
});

async function fetchImageAsDataUrl(srcUrl) {
  const response = await fetch(srcUrl);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function runBackendAnalysis(base64Data, allowCloud) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      base64_data: base64Data,
      allow_cloud: Boolean(allowCloud),
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `Server error (${response.status})` }));
    throw new Error(err.detail || 'Analysis failed');
  }
  const result = await response.json();

  // Save to session history
  const { scanHistory = [] } = await chrome.storage.local.get({ scanHistory: [] });
  const updated = [
    {
      verdict: result.primary_verdict,
      confidence: result.confidence_score,
      timestamp: new Date().toLocaleTimeString(),
    },
    ...scanHistory.slice(0, 19),
  ];
  await chrome.storage.local.set({ scanHistory: updated });
  return result;
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== 'trinetra-scan-image' || !info.srcUrl || !tab?.id) return;

  try {
    chrome.tabs.sendMessage(tab.id, { action: 'TRINETRA_SHOW_LOADER' });
    const { useCloud = false } = await chrome.storage.local.get({ useCloud: false });
    const dataUrl = await fetchImageAsDataUrl(info.srcUrl);
    const result = await runBackendAnalysis(dataUrl, useCloud);
    chrome.tabs.sendMessage(tab.id, { action: 'TRINETRA_SHOW_RESULT', result });
  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'TRINETRA_SHOW_ERROR',
      error: err.message || 'Could not analyze image.',
    });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'TRINETRA_CAPTURE_TAB') {
    chrome.tabs.captureVisibleTab(null, { format: 'png' }, (dataUrl) => {
      if (chrome.runtime.lastError || !dataUrl) {
        sendResponse({ error: chrome.runtime.lastError?.message || 'Screen capture failed' });
      } else {
        sendResponse({ dataUrl });
      }
    });
    return true;
  }

  if (message.action === 'TRINETRA_ANALYZE_CROP') {
    runBackendAnalysis(message.base64Data, message.allowCloud)
      .then((result) => sendResponse({ result }))
      .catch((err) => sendResponse({ error: err.message || 'Analysis failed' }));
    return true;
  }
});
