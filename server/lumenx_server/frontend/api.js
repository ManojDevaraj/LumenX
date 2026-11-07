export async function streamEnhance(file, mode, onEvent, { ablate = [] } = {}) {
  const body = new FormData();
  body.append('file', file, file.name || 'image.png');
  body.append('mode', mode);
  body.append('ablate', JSON.stringify(ablate));
  const response = await fetch('/enhance/stream', { method: 'POST', body });
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
      onEvent?.(parsed.payload);
      if (parsed.event === 'done') completed = parsed.payload.result;
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
  const response = await fetch('/enhance', { method: 'POST', body });
  if (!response.ok) throw await responseError(response);
  return response.json();
}

export async function getHealth() { return checkedJson('/health'); }
export async function getConfig() { return checkedJson('/config'); }
export async function getGallery() { return checkedJson('/gallery'); }
export async function getBenchmarks() { return checkedJson('/benchmarks'); }

async function checkedJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw await responseError(response);
  return response.json();
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
