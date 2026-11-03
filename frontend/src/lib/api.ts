const API_BASE = '/api/backend';

export interface AnalysisResponse {
  primary_verdict: 'FAKE' | 'REAL';
  confidence_score: number;
  local_label: string;
  local_confidence: number;
  rd_status: string | null;
  rd_score: number | null;
  forensic_summary: string;
  latency_ms: number;
  face_crop_base64?: string;
  gradcam_base64?: string;
  exif_data?: Record<string, string | number> | null;
}

export interface HealthResponse {
  status: string;
  model: string;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, {
    method: 'GET',
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function analyzeMedia(file: File, allowCloud: boolean = false): Promise<AnalysisResponse> {
  const base64Data = await fileToBase64(file);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);

  try {
    const res = await fetch('http://127.0.0.1:8000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base64_data: base64Data, allow_cloud: allowCloud }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const errorBody = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorBody.detail || `Analysis failed: ${res.status}`);
    }

    return res.json();
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new Error('Analysis timed out. Try a smaller file.');
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export interface AdvanceScanResponse {
  analysis: string;
}

export async function advanceScan(file: File, consentConfirmed: boolean = false): Promise<AdvanceScanResponse> {
  const base64Data = await fileToBase64(file);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120_000);

  try {
    const res = await fetch('http://127.0.0.1:8000/advance-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base64_data: base64Data, consent_confirmed: consentConfirmed }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const errorBody = await res.json().catch(() => ({ detail: 'Advance scan failed' }));
      throw new Error(errorBody.detail || `Advance scan failed: ${res.status}`);
    }

    return res.json();
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new Error('Analysis timed out. Try a smaller file.');
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}
