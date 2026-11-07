// Backend origin. Empty means same origin (the FastAPI server also serves this page). For separate static
// hosting, set window.LUMENX_CONFIG.apiBase in config.js to the backend URL, e.g. a tunnel or Space URL.
export const API_BASE = String(window.LUMENX_CONFIG?.apiBase || '').replace(/\/+$/, '');
const api = (path) => `${API_BASE}${path}`;

// Server responses carry root-relative asset paths (/files/...); point them at the backend origin.
function absolutize(value) {
  if (!API_BASE) return value;
  if (typeof value === 'string') return value.startsWith('/files/') ? API_BASE + value : value;
  if (Array.isArray(value)) return value.map(absolutize);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, absolutize(item)]));
  return value;
}

export function liveSocketUrl() {
  const base = API_BASE ? new URL(API_BASE) : window.location;
  return `${base.protocol === 'https:' ? 'wss:' : 'ws:'}//${base.host}/live`;
}

export async function streamEnhance(file, mode, onEvent, { ablate = [] } = {}) {
  const body = new FormData();
  body.append('file', file, file.name || 'image.png');
  body.append('mode', mode);
  body.append('ablate', JSON.stringify(ablate));
  const response = await fetch(api('/enhance/stream'), { method: 'POST', body });
  if (!response.ok) throw await responseError(response);
  if (!response.body) throw new Error('The browser did not provide a streaming response body.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = null;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let boundary;
    while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
      const block = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      if (!block.trim()) continue;
      const parsed = parseEventBlock(block);
      if (!parsed) continue;
      if (parsed.event === 'error') {
        const info = parsed.payload.error || parsed.payload;
        throw new Error(info.message || 'The enhancement run failed.');
      }
      const payload = absolutize(parsed.payload);
      onEvent?.(payload);
      if (parsed.event === 'done') completed = payload.result;
    }
    if (done) break;
  }
  if (!completed) throw new Error('The stream ended before a completion event arrived.');
  return completed;
}

export async function enhanceImage(file, mode) {
  const body = new FormData();
  body.append('file', file, file.name || 'image.png');
  body.append('mode', mode);
  body.append('ablate', '[]');
  const response = await fetch(api('/enhance'), { method: 'POST', body });
  if (!response.ok) throw await responseError(response);
  return absolutize(await response.json());
}

export async function getHealth() { return checkedJson('/health'); }
export async function getConfig() { return checkedJson('/config'); }
export async function getGallery() { return checkedJson('/gallery'); }
export async function getBenchmarks() { return checkedJson('/benchmarks'); }

async function checkedJson(url) {
  const response = await fetch(api(url), { headers: { Accept: 'application/json' } });
  if (!response.ok) throw await responseError(response);
  return absolutize(await response.json());
}

async function responseError(response) {
  let payload;
  try { payload = await response.json(); } catch { payload = { message: response.statusText }; }
  const detail = payload.detail || payload;
  const error = new Error(detail.message || detail.code || `Request failed (${response.status}).`);
  error.status = response.status;
  error.code = detail.code || null;
  error.retryAfter = Number(response.headers.get('Retry-After')) || null;
  return error;
}

function parseEventBlock(block) {
  const fields = { data: [] };
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const key = colon < 0 ? line : line.slice(0, colon);
    const value = colon < 0 ? '' : line.slice(colon + 1).replace(/^ /, '');
    if (key === 'data') fields.data.push(value);
    else fields[key] = value;
  }
  if (!fields.data.length) return null;
  let payload;
  try { payload = JSON.parse(fields.data.join('\n')); }
  catch { throw new Error('The server sent an unreadable progress event.'); }
  return { event: fields.event || payload.event || 'message', id: fields.id, payload };
}

export async function getText(url) {
  const response = await fetch(api(url));
  if (!response.ok) throw await responseError(response);
  return response.text();
}
