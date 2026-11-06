import { enhanceImage, getBenchmarks, getGallery, getHealth, streamEnhance } from './api.js';
import { createRelight } from './relight.js';

const $ = (selector) => selector.startsWith('[') || selector.startsWith('.') || selector.startsWith('#')
  ? document.querySelector(selector)
  : document.querySelector(`[data-role="${selector}"]`);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  file: null,
  originalUrl: null,
  result: null,
  currentStageUrl: null,
  camera: null,
  socket: null,
  cameraTimer: null,
  liveMeta: null,
  stageAssets: new Map(),
  relightCleanup: null,
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function selectFile(file) {
  if (!file || !file.type.startsWith('image/')) return showError('Choose a supported image file.');
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
  state.file = file;
  state.originalUrl = URL.createObjectURL(file);
  state.result = null;
  state.stageAssets.clear();
  $('[data-role="before-image"]').src = state.originalUrl;
  $('[data-role="after-image"]').removeAttribute('src');
  const label = $('selected-file');
  label.textContent = `${file.name} · ${formatBytes(file.size)}`;
  label.hidden = false;
  $('run-button').disabled = false;
  $('compare-button').disabled = false;
  $('empty-state').hidden = false;
  $('run-view').hidden = true;
  clearError();
}

function setupUpload() {
  const input = $('image-file');
  input.addEventListener('change', () => selectFile(input.files?.[0]));
  const dropzone = $('dropzone');
  for (const eventName of ['dragenter', 'dragover']) {
    dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('is-dragging'); });
  }
  for (const eventName of ['dragleave', 'drop']) {
    dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('is-dragging'); });
  }
  dropzone.addEventListener('drop', (event) => selectFile(event.dataTransfer?.files?.[0]));
  $('run-button').addEventListener('click', runEnhancement);
  $('compare-button').addEventListener('click', runComparison);
  $('download-button').addEventListener('click', downloadFinal);
  $('compare-slider').addEventListener('input', updateComparison);
  $('overlay-select').addEventListener('change', updateOverlay);
}

function setupTabs() {
  $$('.nav-tab').forEach((button) => button.addEventListener('click', () => {
    const target = button.dataset.tab;
    $$('.nav-tab').forEach((tab) => tab.classList.toggle('is-active', tab === button));
    $$('[data-panel]').forEach((panel) => {
      const active = panel.dataset.panel === target;
      panel.classList.toggle('is-active', active);
      panel.hidden = !active;
    });
    if (target === 'benchmarks') loadBenchmarks();
  }));
}

async function runEnhancement() {
  if (!state.file) return;
  const button = $('run-button');
  const mode = $('mode-select').value;
  button.disabled = true;
  button.firstElementChild.textContent = 'Restoring…';
  $('empty-state').hidden = true;
  $('run-view').hidden = false;
  $('headline').textContent = 'Reading the light in your image…';
  $('run-mode').textContent = modeLabel(mode).toUpperCase();
  $('progress-list').replaceChildren();
  $('stage-strip').replaceChildren();
  $('candidate-grid').replaceChildren();
  $('warnings').replaceChildren();
  clearError();
  try {
    const result = await streamEnhance(state.file, mode, handleStreamEvent);
    state.result = result;
    renderResult(result);
  } catch (error) {
    showError(error.message);
    $('headline').textContent = 'This image could not be restored.';
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = 'Restore image';
  }
}

function handleStreamEvent(event) {
  if (event.event === 'done') return;
  if (event.event === 'warning') {
    appendProgress(event.name || 'warning', 'is-warning');
    addWarning(event.tech_note || event.caption || 'A processing note was reported.');
    return;
  }
  if (event.event === 'candidate') {
    appendProgress(event.name, 'is-done');
    renderCandidate(event.data, event.image_url);
    return;
  }
  appendProgress(event.title || event.name, 'is-done');
  for (const visual of event.visuals || []) {
    if (visual.url) state.stageAssets.set(visual.key, visual.url);
  }
  if (event.image_url) {
    state.stageAssets.set(event.name, event.image_url);
    addStageCard({ name: event.name, title: event.title, caption: event.caption, image_url: event.image_url });
  }
}

function appendProgress(label, modifier = '') {
  const row = node('div', `progress-item ${modifier}`.trim());
  row.append(node('span', 'progress-dot'), node('span', '', label.replaceAll('_', ' ')));
  $('progress-list').append(row);
  row.scrollIntoView({ block: 'nearest', inline: 'end', behavior: 'smooth' });
}

function renderResult(result) {
  state.result = result;
  state.currentStageUrl = result.final.preview_url;
  $('headline').textContent = result.headline;
  $('run-mode').textContent = modeLabel(result.mode).toUpperCase();
  $('image-size').textContent = `${result.input.width} × ${result.input.height}${result.provenance.downscaled ? ' · downscaled from original' : ''}`;
  $('[data-role="before-image"]').src = state.originalUrl || result.input.url;
  $('after-image').src = result.final.preview_url;
  $('download-button').disabled = false;
  $('empty-state').hidden = true;
  $('run-view').hidden = false;
  $('stage-strip').replaceChildren();
  for (const stage of result.stages) {
    if (stage.image_url) addStageCard(stage);
    if (stage.image_url) state.stageAssets.set(stage.name, stage.image_url);
  }
  state.stageAssets.set('final_preview', result.final.preview_url);
  for (const [name, url] of Object.entries(result.maps || {})) {
    if (url) state.stageAssets.set(name, url);
  }
  renderChips(result.diagnosis.chips || []);
  renderDecisions(result.decisions || []);
  renderMetrics(result.metrics || {});
  renderTimings(result.timings_ms || {});
  renderCandidates(result.tuner?.candidates || []);
  $('search-time').textContent = result.tuner?.search_ms == null ? 'Adaptive defaults' : `${Number(result.tuner.search_ms).toFixed(0)} ms search`;
  $('stage-count').textContent = `${result.stages.length} stages`;
  $('warnings').replaceChildren();
  (result.warnings || []).forEach((warning) => addWarning(warning.message || warning.code));
  $('overlay-select').value = 'none';
  updateOverlay();
  state.relightCleanup?.();
  state.relightCleanup = null;
  if (result.relight?.base_url && result.relight?.L_url) {
    state.relightCleanup = createRelight(
      $('relight-canvas'), result.relight.base_url, result.relight.L_url,
      $('gamma-slider'), $('exposure-slider'), $('relight-message'),
    );
  } else {
    $('relight-message').textContent = 'The illumination map is unavailable for this run.';
  }
}

function addStageCard(stage) {
  const button = node('button', 'stage-card');
  button.type = 'button';
  button.title = stage.caption || stage.name;
  const img = node('img');
  img.src = stage.image_url;
  img.alt = `${stage.title || stage.name} stage preview`;
  button.append(img, node('span', '', stage.title || stage.name));
  button.addEventListener('click', () => {
    $('after-image').src = stage.image_url;
    state.currentStageUrl = stage.image_url;
    $('overlay-select').value = 'none';
    updateOverlay();
  });
  $('stage-strip').append(button);
}

function renderChips(chips) {
  const container = $('diagnosis-chips');
  container.replaceChildren();
  for (const chip of chips) {
    const item = node('div', `metric-chip severity-${chip.severity || 'low'}`);
    const value = chip.value == null ? '—' : formatMetric(chip.label, chip.value);
    item.append(node('span', '', chip.label), node('strong', '', value));
    container.append(item);
  }
}

function renderDecisions(decisions) {
  const list = $('decisions');
  list.replaceChildren();
  for (const decision of decisions) list.append(node('li', '', decision));
}

function renderMetrics(metrics) {
  const container = $('metrics-table');
  container.replaceChildren();
  const directions = metrics.directions || {};
  const rows = [];
  for (const name of ['niqe', 'clipiqa', 'brisque', 'musiq']) {
    rows.push([name.toUpperCase(), metrics.before?.[name], metrics.after?.[name], directions[name]]);
  }
  rows.push(['Edge detail', null, metrics.edge_retention, 'higher']);
  rows.push(['Noise σ', metrics.noise_sigma?.before, metrics.noise_sigma?.after, 'lower']);
  rows.push(['Chroma shift', null, metrics.chroma_delta_e, 'lower']);
  rows.push(['Clipped highlights', metrics.clip_pct?.before, metrics.clip_pct?.after, 'lower']);
  for (const [label, before, after, direction] of rows) {
    const row = node('div', 'metric-row');
    row.append(node('span', '', label), node('span', '', valueText(before)), node('span', '', valueText(after)));
    const delta = compareValues(before, after, direction);
    const arrow = node('span', `metric-delta ${delta.state === 'worse' ? 'is-worse' : ''} ${delta.state === 'na' ? 'is-unavailable' : ''}`, delta.arrow);
    arrow.title = delta.label;
    row.append(arrow);
    container.append(row);
  }
  $('metrics-resolution').textContent = metrics.resolution ? `${metrics.resolution}px proxy` : '';
}

function renderTimings(timings) {
  const container = $('timings');
  container.replaceChildren();
  const keys = ['diagnose', 'search', 'classical', 'deep', 'fusion', 'denoise', 'guardrails', 'core_total', 'total'];
  for (const key of keys) {
    if (timings[key] == null) continue;
    const value = Number(timings[key]);
    const label = key.replaceAll('_', ' ');
    container.append(node('span', 'timing-pill', `${label} ${value.toFixed(0)} ms`));
  }
}

function renderCandidates(candidates) {
  $('candidate-grid').replaceChildren();
  for (const candidate of candidates) renderCandidate(candidate, candidate.thumb_url);
  $('candidates-section').hidden = candidates.length === 0;
}

function renderCandidate(candidate, imageUrl) {
  if (!candidate) return;
  const card = node('article', `candidate-card ${candidate.chosen ? 'is-winner' : ''}`);
  if (imageUrl) {
    const img = node('img');
    img.src = imageUrl;
    img.alt = `Candidate ${candidate.id + 1} preview`;
    card.append(img);
  }
  const info = node('div', 'candidate-info');
  const title = node('div', 'candidate-title');
  title.append(node('span', '', `Setting ${String(candidate.id + 1).padStart(2, '0')}`));
  title.append(node('span', 'winner-tag', candidate.chosen ? 'SELECTED' : (candidate.feasible ? '' : 'REVIEW')));
  info.append(title);
  info.append(node('p', '', `Score ${candidate.total == null ? '—' : Number(candidate.total).toFixed(3)} · ${candidate.reasons?.[0] || 'Measured preview'}`));
  card.append(info);
  $('candidate-grid').append(card);
}

function updateComparison() {
  const value = `${$('compare-slider').value}%`;
  $('[data-role="comparison"]').style.setProperty('--compare-position', value);
}

function updateOverlay() {
  const key = $('overlay-select').value;
  const overlay = $('comparison-overlay');
  const url = key === 'none' ? null : state.result?.maps?.[key];
  overlay.hidden = !url;
  if (url) overlay.src = url;
}

function downloadFinal() {
  if (!state.result?.final?.url) return;
  const anchor = node('a');
  anchor.href = state.result.final.url;
  anchor.download = `lumenx-${state.result.run_id}.png`;
  anchor.click();
}

async function loadGallery() {
  const grid = $('gallery-grid');
  try {
    const items = await getGallery();
    grid.replaceChildren();
    $('gallery-count').textContent = `${items.length} scenes`;
    for (const item of items) {
      const button = node('button', 'gallery-item');
      button.type = 'button';
      button.title = [item.why_hard, item.attribution].filter(Boolean).join('\n');
      const image = node('img');
      image.src = item.image_url;
      image.alt = item.title;
      button.append(image, node('span', '', item.title));
      button.addEventListener('click', async () => {
        try {
          const response = await fetch(item.image_url);
          if (!response.ok) throw new Error('Gallery image is unavailable.');
          const blob = await response.blob();
          selectFile(new File([blob], `${item.id}.${extensionFor(blob.type)}`, { type: blob.type }));
        } catch (error) { showError(error.message); }
      });
      grid.append(button);
    }
    if (!items.length) $('gallery-count').textContent = 'Add scenes';
  } catch { $('gallery-count').textContent = 'Unavailable'; }
}

async function loadHealth() {
  const status = $('health');
  try {
    const health = await getHealth();
    const fake = Boolean(health.engine?.fake);
    status.classList.toggle('is-ready', health.status === 'ready');
    status.classList.toggle('is-offline', false);
    $('health-label').textContent = health.status === 'ready' ? 'Ready' : 'Warming';
    $('mock-badge').hidden = !fake;
  } catch {
    status.classList.add('is-offline');
    $('health-label').textContent = 'Offline';
  }
}

async function runComparison() {
  if (!state.file) return;
  const button = $('compare-button');
  const container = $('compare-results');
  button.disabled = true;
  button.textContent = 'Comparing…';
  container.replaceChildren();
  for (const mode of ['classical', 'dl', 'hybrid']) {
    const card = node('article', 'method-card');
    card.append(node('div', '', ''));
    const body = card.lastElementChild;
    body.append(node('h3', '', modeLabel(mode)));
    body.append(node('p', '', 'Processing…'));
    container.append(card);
    try {
      const result = await enhanceImage(state.file, mode);
      const image = node('img');
      image.src = result.final.preview_url;
      image.alt = `${modeLabel(mode)} output`;
      card.insertBefore(image, body);
      body.replaceChildren(node('h3', '', modeLabel(mode)), node('p', '', result.headline));
      if (result.provenance.actual_method === 'classical_fallback') {
        body.append(node('p', 'warning-item', 'Learned weights were unavailable; this result used the disclosed classical fallback.'));
      }
      const score = result.metrics?.after?.niqe;
      body.append(node('p', '', `NIQE ${valueText(score)} · ${Number(result.timings_ms?.total || 0).toFixed(0)} ms`));
    } catch (error) {
      body.replaceChildren(node('h3', '', modeLabel(mode)), node('p', 'warning-item', error.message));
    }
  }
  button.disabled = false;
  button.textContent = 'Run three-way comparison';
}

async function loadBenchmarks() {
  const container = $('benchmark-content');
  try {
    const payload = await getBenchmarks();
    container.replaceChildren();
    if (payload.status === 'not_ready') {
      container.append(node('span', 'eyebrow', payload.progress ? 'EVALUATION IN PROGRESS' : 'NO RUNS YET'));
      container.append(node('p', '', payload.progress ? JSON.stringify(payload.progress) : 'Start the evaluation scripts from the server folder to populate benchmark results.'));
      return;
    }
    for (const [name, rows] of Object.entries(payload.tables || {})) {
      container.append(node('h3', '', name));
      const table = node('table', 'benchmark-table');
      const entries = Array.isArray(rows) ? rows : [];
      if (entries.length) {
        const columns = Object.keys(entries[0]);
        const head = node('tr');
        columns.forEach((column) => head.append(node('th', '', column)));
        table.append(head);
        entries.forEach((entry) => {
          const row = node('tr');
          columns.forEach((column) => row.append(node('td', '', valueText(entry[column]))));
          table.append(row);
        });
      }
      container.append(table);
    }
    for (const chart of payload.charts || []) {
      const image = node('img');
      image.src = chart.url || chart;
      image.alt = chart.title || 'Benchmark chart';
      container.append(image);
    }
  } catch (error) { container.textContent = error.message; }
}

async function startCamera() {
  if (state.socket) return stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) return showError('Camera access is unavailable in this browser.');
  try {
    state.camera = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    const video = $('camera-video');
    video.srcObject = state.camera;
    await video.play();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.socket = new WebSocket(`${protocol}//${location.host}/live`);
    state.socket.binaryType = 'blob';
    state.socket.onopen = () => {
      $('camera-button').textContent = 'Stop camera';
      $('live-stats').textContent = 'Connected · preparing frames';
      const capture = document.createElement('canvas');
      const context = capture.getContext('2d');
      state.cameraTimer = setInterval(() => {
        if (!state.socket || state.socket.readyState !== WebSocket.OPEN || state.socket.bufferedAmount > 300000 || video.readyState < 2) return;
        capture.width = Math.min(1280, video.videoWidth || 640);
        capture.height = Math.round(capture.width * (video.videoHeight || 480) / (video.videoWidth || 640));
        context.drawImage(video, 0, 0, capture.width, capture.height);
        capture.toBlob((blob) => { if (blob && state.socket?.readyState === WebSocket.OPEN) state.socket.send(blob); }, 'image/jpeg', 0.78);
      }, 100);
    };
    state.socket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          state.liveMeta = JSON.parse(event.data);
          $('live-stats').textContent = `${state.liveMeta.status} · ${Number(state.liveMeta.fps || 0).toFixed(1)} fps · ${Number(state.liveMeta.proc_ms || 0).toFixed(0)} ms · ${state.liveMeta.dropped || 0} dropped`;
        } catch { /* Ignore a malformed live status. */ }
      } else if (event.data instanceof Blob) {
        const output = $('live-output');
        const old = output.dataset.objectUrl;
        if (old) URL.revokeObjectURL(old);
        const url = URL.createObjectURL(event.data);
        output.dataset.objectUrl = url;
        output.src = url;
      }
    };
    state.socket.onclose = () => stopCamera(false);
  } catch (error) {
    stopCamera();
    showError(error.message || 'Camera permission was denied.');
  }
}

function stopCamera(closeSocket = true) {
  if (state.cameraTimer) clearInterval(state.cameraTimer);
  state.cameraTimer = null;
  if (closeSocket && state.socket) state.socket.close();
  state.socket = null;
  state.camera?.getTracks().forEach((track) => track.stop());
  state.camera = null;
  $('camera-video').srcObject = null;
  $('camera-button').textContent = 'Start camera';
  $('live-stats').textContent = 'Camera is off';
}

function setupCamera() { $('camera-button').addEventListener('click', () => startCamera()); }

function addWarning(message) {
  const warning = node('div', 'warning-item', message);
  $('warnings').append(warning);
}

function showError(message) {
  clearError();
  const element = node('div', 'warning-item', message);
  element.dataset.role = 'error-message';
  document.querySelector('main').prepend(element);
}

function clearError() { document.querySelector('[data-role="error-message"]')?.remove(); }
function modeLabel(mode) { return ({ hybrid: 'Adaptive hybrid', classical: 'Classical vision', dl: 'Zero-DCE++ model' })[mode] || mode; }
function extensionFor(type) { return ({ 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/tiff': 'tiff', 'image/bmp': 'bmp' })[type] || 'png'; }
function formatBytes(size) { return size < 1024 * 1024 ? `${(size / 1024).toFixed(0)} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`; }
function valueText(value) {
  if (value == null) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—';
}
function formatMetric(label, value) { return label.toLowerCase().includes('percent') || label.toLowerCase().includes('clip') ? `${Number(value).toFixed(1)}%` : Number(value).toFixed(3); }

function compareValues(before, after, direction) {
  if (before == null || after == null || !Number.isFinite(Number(before)) || !Number.isFinite(Number(after))) return { state: 'na', arrow: '·', label: 'Not available' };
  const delta = Number(after) - Number(before);
  const better = direction === 'higher' ? delta > 0 : direction === 'lower' ? delta < 0 : false;
  const worse = direction === 'higher' ? delta < 0 : direction === 'lower' ? delta > 0 : false;
  return { state: better ? 'better' : worse ? 'worse' : 'same', arrow: better ? '↑' : worse ? '↓' : '→', label: `Change ${delta.toFixed(3)}` };
}

async function boot() {
  setupTabs();
  setupUpload();
  setupCamera();
  await Promise.all([loadGallery(), loadHealth()]);
  setInterval(loadHealth, 15000);
}

boot();
