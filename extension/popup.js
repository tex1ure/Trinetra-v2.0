document.addEventListener('DOMContentLoaded', () => {
  const switchLocal = document.getElementById('switch-local');
  const switchCloud = document.getElementById('switch-cloud');
  const warningEl = document.getElementById('tri-warning');
  const btnScan = document.getElementById('btn-scan');
  const statusContainer = document.getElementById('tri-status');
  const statusDot = statusContainer.querySelector('.tri-status-dot');
  const statusText = statusContainer.querySelector('.tri-status-text');

  // Privacy-first defaults: Local ON, Cloud OFF
  chrome.storage.local.get({ useLocal: true, useCloud: false }, (prefs) => {
    switchLocal.checked = prefs.useLocal;
    switchCloud.checked = prefs.useCloud;
  });

  function enforceAtLeastOne(changedSwitch, otherSwitch) {
    if (!switchLocal.checked && !switchCloud.checked) {
      changedSwitch.checked = true;
      warningEl.classList.add('visible');
      setTimeout(() => warningEl.classList.remove('visible'), 2500);
      return;
    }
    chrome.storage.local.set({
      useLocal: switchLocal.checked,
      useCloud: switchCloud.checked,
    });
  }

  switchLocal.addEventListener('change', () => enforceAtLeastOne(switchLocal, switchCloud));
  switchCloud.addEventListener('change', () => enforceAtLeastOne(switchCloud, switchLocal));

  // Check backend health
  fetch('http://localhost:8000/health')
    .then((res) => {
      if (!res.ok) throw new Error('Offline');
      return res.json();
    })
    .then((data) => {
      statusDot.classList.remove('offline');
      statusDot.classList.add('online');
      statusText.textContent = `Online · ${data.model || 'Local Engine'}`;
    })
    .catch(() => {
      statusDot.classList.remove('online');
      statusDot.classList.add('offline');
      statusText.textContent = 'Backend Offline (port 8000)';
    });

  // Trigger region selection on active tab
  btnScan.addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const activeTab = tabs[0];
      if (!activeTab || !activeTab.id) return;
      chrome.tabs.sendMessage(activeTab.id, {
        action: 'TRINETRA_START_SELECTION',
        allowCloud: switchCloud.checked,
      });
      window.close();
    });
  });
});
