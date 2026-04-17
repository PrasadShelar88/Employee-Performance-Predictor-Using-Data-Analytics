const DEFAULT_BASE_URL = 'http://127.0.0.1:8000';

export function getBaseUrl() {
  return localStorage.getItem('epp_api_base_url') || DEFAULT_BASE_URL;
}

export function setBaseUrl(url) {
  localStorage.setItem('epp_api_base_url', url.trim() || DEFAULT_BASE_URL);
}

async function handleResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || data.message || 'Request failed');
  }
  return data;
}

export async function fetchRoot() {
  return handleResponse(await fetch(`${getBaseUrl()}/`));
}

export async function generateData(rows) {
  return handleResponse(await fetch(`${getBaseUrl()}/generate-data?rows=${rows}`, { method: 'POST' }));
}

export async function trainModel() {
  return handleResponse(await fetch(`${getBaseUrl()}/train`, { method: 'POST' }));
}

export async function fetchMetrics() {
  return handleResponse(await fetch(`${getBaseUrl()}/metrics`));
}

export async function predictEmployee(payload) {
  return handleResponse(await fetch(`${getBaseUrl()}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }));
}

export async function predictBatch(file) {
  const formData = new FormData();
  formData.append('file', file);
  return handleResponse(await fetch(`${getBaseUrl()}/predict-batch`, {
    method: 'POST',
    body: formData
  }));
}
