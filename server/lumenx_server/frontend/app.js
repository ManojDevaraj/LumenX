import { enhanceImage, getBenchmarks, getGallery, getHealth, streamEnhance } from './api.js';
import { createRelight } from './relight.js';
import { createFlashlight } from './flashlight.js';
import { createRelief } from './relief.js';

const $ = (selector) => selector.startsWith('[') || selector.startsWith('.') || selector.startsWith('#')
  ? document.querySelector(selector)
  : document.querySelector(`[data-role="${selector}"]`);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  file: null,
  originalUrl: null,
  result: null,
  mode: 'hybrid',
  overlayKey: 'none',
  stageCount: 0,
  candidateCount: 0,
  camera: null,
  socket: null,
  cameraTimer: null,
  liveMeta: null,
  stageAssets: new Map(),
  relightCleanup: null,
  ridgeCleanup: null,
  relief: null,
  viewerLight: null,
  heroFile: null,
  wipeFrame: null,
};

// Pipeline order as emitted by the core engine; modes that skip a stage leave it dimmed.
const PIPELINE = [
  ['diagnose', 'Diagnose'], ['recoverability', 'Signal'], ['tuner', 'Search'], ['classical', 'Classical'],
  ['deep', 'Neural'], ['fusion', 'Fusion'], ['denoise', 'Denoise'], ['guardrails', 'Guard'],
];
const LENSES = [
  ['recoverability_overlay', 'Signal confidence', 'where detail survived'],
  ['illumination_gray', 'Illumination', 'measured light field'],
  ['gain', 'Brightening gain', 'lift applied per pixel'],
  ['fusion', 'Path balance', 'classical vs neural'],
  ['noise', 'Noise estimate', 'grain in the input'],
  ['snr', 'Signal to noise', 'per-pixel SNR'],
  ['denoise_strength', 'Denoise strength', 'where grain was cleaned'],
];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const STAGE_DWELL_MS = 700;
const CANDIDATE_DWELL_MS = 90;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function setViewerState(value) { $('comparison').dataset.state = value; }

function setComparePosition(percent) {
  const value = Math.max(0, Math.min(100, percent));
  $('comparison').style.setProperty('--compare-position', `${value}%`);
  $('compare-slider').value = String(value);
}

function selectFile(file) {
  if (!file || !file.type.startsWith('image/')) return showError('Choose a supported image file.');
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
  state.file = file;
  state.originalUrl = URL.createObjectURL(file);
  state.result = null;
  state.stageAssets.clear();
  $('before-image').src = state.originalUrl;
  $('ambient-image').src = state.originalUrl;
  $('after-image').removeAttribute('src');
  setOverlay('none');
  const label = $('selected-file');
  label.textContent = `${file.name} · ${formatBytes(file.size)}`;
  label.hidden = false;
  $('run-button').disabled = false;
  $('compare-button').disabled = false;
  $('run-view').hidden = true;
  setViewerState('loaded');
  resetPipeline();
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
  $('before-image').addEventListener('load', (event) => {
    const { naturalWidth: width, naturalHeight: height } = event.target;
    if (!width || !height) return;
    $('comparison').style.setProperty('--ar', `${width} / ${height}`);
    $('comparison').style.setProperty('--ar-num', String(width / height));
  });
  $('run-button').addEventListener('click', runEnhancement);
  $('compare-button').addEventListener('click', runComparison);
  $('download-button').addEventListener('click', downloadFinal);
  $('compare-slider').addEventListener('input', () => setComparePosition(Number($('compare-slider').value)));
  $$('[data-mode]').forEach((button) => button.addEventListener('click', () => {
    state.mode = button.dataset.mode;
    $$('[data-mode]').forEach((item) => {
      item.classList.toggle('is-active', item === button);
      item.setAttribute('aria-checked', String(item === button));
    });
  }));
}

function setupViewerDrag() {
  const viewer = $('comparison');
  const move = (event) => {
    const rect = viewer.getBoundingClientRect();
    setComparePosition(((event.clientX - rect.left) / rect.width) * 100);
  };
  viewer.addEventListener('pointerdown', (event) => {
    if (viewer.dataset.state !== 'done' || viewer.dataset.view === 'flashlight' || event.target.closest('.v-modes')) return;
    cancelAnimationFrame(state.wipeFrame);
    viewer.setPointerCapture(event.pointerId);
    viewer.classList.add('is-dragging');
    move(event);
  });
  viewer.addEventListener('pointermove', (event) => { if (viewer.classList.contains('is-dragging')) move(event); });
  const end = () => viewer.classList.remove('is-dragging');
  viewer.addEventListener('pointerup', end);
  viewer.addEventListener('pointercancel', end);
}

function setupViewModes() {
  const viewer = $('comparison');
  state.viewerLight = createFlashlight(viewer, { radius: 0.17, wander: false, clickToFlood: false, reducedMotion });
  state.viewerLight.setEnabled(false);
  $$('[data-view]').forEach((button) => button.addEventListener('click', () => setViewMode(button.dataset.view)));
}

function setViewMode(mode) {
  $('comparison').dataset.view = mode;
  state.viewerLight?.setEnabled(mode === 'flashlight');
  $$('[data-view]').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.view === mode);
    button.setAttribute('aria-checked', String(button.dataset.view === mode));
  });
}

// Landing hero: the first real gallery photo is restored live on load, then revealed under the cursor.
async function setupHero(items) {
  const hero = $('hero');
  const hint = $('hero-hint');
  const setHint = (text) => hint.replaceChildren(node('span', 'rh-dot'), node('span', '', text));
  createFlashlight(hero, { radius: 0.16, reducedMotion });
  const updateProgress = () => {
    const progress = Math.min(1, Math.max(0, window.scrollY / Math.max(1, hero.offsetHeight)));
    hero.style.setProperty('--hero-progress', progress.toFixed(3));
  };
  window.addEventListener('scroll', () => requestAnimationFrame(updateProgress), { passive: true });
  $('hero-cta').addEventListener('click', (event) => {
    event.preventDefault();
    if (state.heroFile && !state.file) selectFile(state.heroFile);
    $('#console').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
  });

  const scene = items.find((item) => !String(item.id).startsWith('synthetic')) || items[0];
  if (!scene) { setHint('Add photos to the gallery to light the scene'); return; }
  $('hero-dark').src = scene.image_url;
  try {
    const response = await fetch(scene.image_url);
    const blob = await response.blob();
    state.heroFile = new File([blob], `${scene.id}.${extensionFor(blob.type)}`, { type: blob.type });
    const result = await enhanceImage(state.heroFile, 'dl');
    const lit = $('hero-lit');
    // The hero is full-bleed, so it needs the full-resolution output rather than the 1280 px preview.
    await new Promise((resolve, reject) => { lit.onload = resolve; lit.onerror = reject; lit.src = result.final.url; });
    hero.classList.remove('is-pending');
    const touch = window.matchMedia('(hover: none)').matches;
    setHint(touch ? 'Tap to flood the scene with light' : 'Move your light · click to flood the scene');
    const meta = $('hero-meta');
    meta.replaceChildren(node('strong', '', scene.title));
    meta.append(node('span', '', `${methodLabel(result.provenance.actual_method)} · restored in ${Number(result.timings_ms?.total || 0).toFixed(0)} ms`));
    if (result.provenance.fake) meta.append(node('span', '', ' · mock engine'));
    setupStory(state.heroFile, scene.title);
  } catch {
    setHint('The engine is offline; showing the original scene');
  }
}

// Wraps each character of a heading so CSS can stagger it in; inline elements such as <em> keep their styling.
function splitHeadline(heading) {
  let index = 0;
  const splitNode = (parent) => {
    for (const child of [...parent.childNodes]) {
      if (child.nodeType === Node.ELEMENT_NODE) { splitNode(child); continue; }
      if (child.nodeType !== Node.TEXT_NODE) continue;
      const fragment = document.createDocumentFragment();
      for (const part of child.textContent.split(/(\s+)/)) {
        if (!part) continue;
        if (/^\s+$/.test(part)) { fragment.append(document.createTextNode(' ')); continue; }
        const word = node('span', 'split-word');
        for (const character of part) {
          const span = node('span', 'split-char', character);
          span.style.setProperty('--i', String(index++));
          word.append(span);
        }
        fragment.append(word);
      }
      child.replaceWith(fragment);
    }
  };
  heading.setAttribute('aria-label', heading.textContent);
  splitNode(heading);
}

function setupCursor() {
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;
  const cursor = $('cursor');
  const dot = cursor.querySelector('.cursor-dot');
  const ring = cursor.querySelector('.cursor-ring');
  const target = { x: -100, y: -100 };
  const ringPosition = { x: -100, y: -100 };
  document.body.classList.add('has-cursor');
  window.addEventListener('pointermove', (event) => {
    target.x = event.clientX;
    target.y = event.clientY;
    dot.style.transform = `translate(${target.x}px, ${target.y}px) translate(-50%, -50%)`;
    const interactive = event.target.closest('a, button, label, .gallery-item, .lens, .stage-card, canvas, input');
    cursor.classList.toggle('is-hover', Boolean(interactive));
    // The hero and flashlight viewer draw their own light, so the ring steps aside there.
    cursor.classList.toggle('is-hidden', Boolean(event.target.closest('.reveal-hero:not(.is-flooded) .rh-media, .viewer[data-view="flashlight"][data-state="done"]')));
  }, { passive: true });
  document.addEventListener('pointerleave', () => cursor.classList.add('is-hidden'));
  const follow = () => {
    ringPosition.x += (target.x - ringPosition.x) * (reducedMotion ? 1 : 0.18);
    ringPosition.y += (target.y - ringPosition.y) * (reducedMotion ? 1 : 0.18);
    ring.style.transform = `translate(${ringPosition.x}px, ${ringPosition.y}px) translate(-50%, -50%)`;
    requestAnimationFrame(follow);
  };
  requestAnimationFrame(follow);
}

function setupMarquee(items) {
  const track = $('marquee-track');
  const real = items.filter((item) => !String(item.id).startsWith('synthetic')).slice(0, 4);
  const words = [...track.children];
  track.replaceChildren();
  words.forEach((word, index) => {
    track.append(word);
    const scene = real[index % Math.max(1, real.length)];
    if (!scene) return;
    const thumb = node('span', 'mq-thumb');
    const image = node('img');
    image.src = scene.image_url;
    image.alt = '';
    thumb.append(image);
    track.append(thumb);
  });
  // Two identical halves make the -50% loop seamless.
  [...track.children].forEach((child) => track.append(child.cloneNode(true)));
}

// Story: the hero photo is also run through the full hybrid path so its real stages can be scrolled through.
async function setupStory(file, sceneTitle) {
  const section = $('story');
  const track = $('story-track');
  let result;
  try { result = await enhanceImage(file, 'hybrid'); } catch { return; }
  const stages = result.stages.filter((stage) => stage.image_url);
  if (!stages.length) return;
  track.replaceChildren();
  const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
  $('story-count').textContent = words[stages.length + 1] || String(stages.length + 1);
  $('story-meta').textContent = `${sceneTitle} · adaptive hybrid · ${Number(result.timings_ms?.total || 0).toFixed(0)} ms on ${result.provenance.device.toUpperCase()}${result.provenance.fake ? ' · mock engine' : ''}`;
  const addPanel = (number, title, caption, url, isFinal = false) => {
    const panel = node('article', `story-panel${isFinal ? ' is-final' : ''}`);
    const frame = node('div', 'sp-img');
    const image = node('img');
    image.src = url;
    image.alt = title;
    image.loading = 'lazy';
    frame.append(image);
    const body = node('div', 'sp-body');
    body.append(node('h3', '', title), node('p', '', caption || ''));
    panel.append(node('span', 'sp-num', number), frame, body);
    track.append(panel);
    return body;
  };
  stages.forEach((stage, index) => addPanel(`STEP ${String(index + 1).padStart(2, '0')}`, stage.title || stage.name, stage.caption, stage.image_url));
  const finalBody = addPanel('RESULT', 'The restored night', 'Every step above was measured from this one photo. Now try it on yours.', result.final.preview_url, true);
  const cta = node('a', 'sp-cta', 'Restore your own →');
  cta.href = '#console';
  cta.addEventListener('click', (event) => {
    event.preventDefault();
    if (state.heroFile && !state.file) selectFile(state.heroFile);
    $('#console').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
  });
  finalBody.append(cta);
  section.hidden = false;

  const heightSource = result.maps?.heightmap || result.maps?.illumination_gray;
  if (heightSource) {
    try {
      createRelief($('landing-relief-canvas'), result.final.preview_url, heightSource, { reducedMotion });
      $('landing-relief').hidden = false;
    } catch (error) {
      console.warn('Landing relief unavailable:', error);
    }
  }

  const pinned = () => window.matchMedia('(min-width: 721px)').matches && !reducedMotion;
  const layout = () => {
    section.style.height = pinned() ? `${window.innerHeight + Math.max(0, track.scrollWidth - window.innerWidth)}px` : '';
    update();
  };
  const update = () => {
    if (!pinned()) return;
    const span = Math.max(1, section.offsetHeight - window.innerHeight);
    const progress = Math.min(1, Math.max(0, (window.scrollY - section.offsetTop) / span));
    track.style.transform = `translate3d(${-progress * Math.max(0, track.scrollWidth - window.innerWidth)}px, 0, 0)`;
    $('story-bar').style.transform = `scaleX(${progress})`;
  };
  window.addEventListener('scroll', () => requestAnimationFrame(update), { passive: true });
  window.addEventListener('resize', layout);
  layout();
}

function setupReveals() {
  const targets = $$('[data-reveal]');
  if (reducedMotion || !('IntersectionObserver' in window)) { targets.forEach((target) => target.classList.add('is-in')); return; }
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('is-in');
      observer.unobserve(entry.target);
    }
  }, { threshold: 0.12 });
  targets.forEach((target) => observer.observe(target));
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

function resetPipeline() {
  const list = $('progress-list');
  list.replaceChildren();
  for (const [key, label] of PIPELINE) {
    const item = node('li', 'pipe-node');
    item.dataset.stage = key;
    item.append(node('i'), node('b', '', label));
    list.append(item);
  }
}

function pipeNode(key) { return $(`.pipe-node[data-stage="${key}"]`); }

function markStage(key, status) {
  const item = pipeNode(key);
  if (!item) return;
  item.classList.remove('is-active');
  item.classList.add(status);
  const index = PIPELINE.findIndex(([name]) => name === key);
  $$('.pipe-node.is-active').forEach((other) => other.classList.remove('is-active'));
  const next = PIPELINE.slice(index + 1).map(([name]) => pipeNode(name)).find((candidate) => candidate && !candidate.classList.contains('is-done'));
  next?.classList.add('is-active');
}

async function runEnhancement() {
  if (!state.file) return;
  const button = $('run-button');
  button.disabled = true;
  button.classList.add('is-busy');
  button.firstElementChild.textContent = 'Restoring…';
  document.body.classList.add('has-run');
  $('run-view').hidden = true;
  $('stage-strip').replaceChildren();
  $('candidate-grid').replaceChildren();
  $('warnings').replaceChildren();
  state.stageCount = 0;
  state.candidateCount = 0;
  setOverlay('none');
  setViewMode('split');
  resetPipeline();
  pipeNode('diagnose').classList.add('is-active');
  $('after-image').src = state.originalUrl;
  setViewerState('running');
  $('comparison').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });
  setCaption('Reading the light', 'Measuring where the scene is dark and how much real signal is left.', 0);
  clearError();
  // Events can arrive faster than a viewer can follow; queue them so each stage holds the screen briefly.
  let playback = Promise.resolve();
  const enqueue = (event) => {
    playback = playback.then(() => {
      handleStreamEvent(event);
      if (reducedMotion) return null;
      return sleep(event.event === 'stage' ? STAGE_DWELL_MS : event.event === 'candidate' ? CANDIDATE_DWELL_MS : 0);
    });
  };
  try {
    const result = await streamEnhance(state.file, state.mode, enqueue);
    await playback;
    state.result = result;
    renderResult(result);
  } catch (error) {
    showError(error.message);
    setViewerState('loaded');
  } finally {
    button.disabled = false;
    button.classList.remove('is-busy');
    button.firstElementChild.textContent = 'Restore again';
  }
}

function setCaption(title, text, count) {
  const caption = $('viewer-caption');
  const body = node('div', 'cap-text');
  body.append(node('strong', '', title), node('span', '', text || ''));
  const counter = node('div', 'cap-count', String(count).padStart(2, '0'));
  counter.append(node('small', '', ` / ${String(PIPELINE.length).padStart(2, '0')}`));
  caption.replaceChildren(body, counter);
}

function handleStreamEvent(event) {
  if (event.event === 'done') return;
  if (event.event === 'warning') {
    addWarning(event.tech_note || event.caption || 'A processing note was reported.');
    return;
  }
  if (event.event === 'candidate') {
    state.candidateCount += 1;
    const tuner = pipeNode('tuner');
    if (tuner) {
      tuner.classList.add('is-active');
      let meta = tuner.querySelector('.pipe-meta');
      if (!meta) { meta = node('span', 'pipe-meta'); tuner.append(meta); }
      meta.textContent = `${state.candidateCount} tried`;
    }
    renderCandidate(event.data, event.image_url);
    return;
  }
  for (const visual of event.visuals || []) {
    if (visual.url) state.stageAssets.set(visual.key, visual.url);
  }
  state.stageCount += 1;
  markStage(event.name, 'is-done');
  setCaption(event.title || event.name, event.caption, state.stageCount);
  if (event.image_url) {
    state.stageAssets.set(event.name, event.image_url);
    showInViewer(event.image_url);
  }
}

function showInViewer(url) {
  const after = $('after-image');
  after.src = url;
  $('ambient-image').src = url;
  if (reducedMotion) return;
  after.classList.remove('is-flash');
  void after.offsetWidth;
  after.classList.add('is-flash');
}

function renderResult(result) {
  state.result = result;
  $('headline').textContent = result.headline;
  $('run-mode').textContent = modeLabel(result.mode);
  $('image-size').textContent = `${result.input.width} × ${result.input.height}${result.provenance.downscaled ? ' · downscaled' : ''}`;
  $('before-image').src = state.originalUrl || result.input.url;
  showInViewer(result.final.preview_url);
  $('download-button').disabled = false;
  for (const item of $$('.pipe-node')) {
    item.classList.remove('is-active');
    if (!item.classList.contains('is-done')) item.classList.add('is-skipped');
  }
  if (state.candidateCount === 0 && result.tuner?.candidates?.length) {
    const tuner = pipeNode('tuner');
    tuner?.append(node('span', 'pipe-meta', `${result.tuner.candidates.length} tried`));
  }
  setViewerState('done');
  wipeReveal();

  $('stage-strip').replaceChildren();
  result.stages.filter((stage) => stage.image_url).forEach((stage, index) => {
    state.stageAssets.set(stage.name, stage.image_url);
    addStageCard(stage, index);
  });
  addStageCard({ name: 'final', title: 'Final image', caption: 'The restored result, ready to download.', image_url: result.final.preview_url }, result.stages.length, true);
  $('stage-count').textContent = `${result.stages.length} stages`;
  state.stageAssets.set('final_preview', result.final.preview_url);
  for (const [name, url] of Object.entries(result.maps || {})) {
    if (url) state.stageAssets.set(name, url);
  }

  renderStats(result);
  renderLenses(result.maps || {});
  renderChips(result.diagnosis.chips || []);
  renderDecisions(result.decisions || []);
  renderMetrics(result.metrics || {});
  renderTimings(result.timings_ms || {});
  renderCandidates(result.tuner?.candidates || []);
  $('search-time').textContent = result.tuner?.search_ms == null ? 'Adaptive defaults' : `${Number(result.tuner.search_ms).toFixed(0)} ms search`;
  $('warnings').replaceChildren();
  const seen = new Set();
  for (const warning of result.warnings || []) {
    if (seen.has(warning.code)) continue;
    seen.add(warning.code);
    addWarning(warning.message || warning.code);
  }

  $('run-view').hidden = false;
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
  state.relief?.dispose();
  const heightSource = result.maps?.heightmap || result.maps?.illumination_gray;
  try {
    state.relief = heightSource ? createRelief($('relief-canvas'), result.final.preview_url, heightSource, { reducedMotion }) : null;
  } catch (error) {
    // The 3D view is a showcase; a GPU or shader failure must not take the rest of the report down with it.
    state.relief = null;
    console.warn('Relief unavailable:', error);
  }
  state.ridgeCleanup?.();
  const ridgeSource = result.maps?.illumination_gray || result.maps?.heightmap || result.relight?.L_url;
  state.ridgeCleanup = ridgeSource ? drawRidgelines($('ridge-canvas'), ridgeSource) : null;
}

// Opens the restored side from the right edge so the first view is a reveal, not a static split.
function wipeReveal() {
  if (reducedMotion) return setComparePosition(50);
  const start = performance.now();
  const duration = 1600;
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 4);
    setComparePosition(100 - eased * 50);
    if (t < 1) state.wipeFrame = requestAnimationFrame(step);
  };
  setComparePosition(100);
  state.wipeFrame = requestAnimationFrame(step);
}

function renderStats(result) {
  const stats = [];
  const meanY = result.diagnosis?.raw?.mean_y;
  if (meanY != null) stats.push([meanY * 100, 0, '%', 'Input light level']);
  const candidates = result.tuner?.candidates?.length || 0;
  if (candidates) stats.push([candidates, 0, '', 'Settings searched']);
  stats.push([result.stages.length, 0, '', 'Visible stages']);
  if (result.timings_ms?.total != null) stats.push([result.timings_ms.total / 1000, 2, 's', 'End to end']);
  const row = $('stat-row');
  row.replaceChildren();
  for (const [value, decimals, unit, label] of stats) {
    const cell = node('div', 'stat');
    const strong = node('strong');
    const number = node('span', '', (0).toFixed(decimals));
    strong.append(number);
    if (unit) strong.append(node('small', '', unit));
    cell.append(strong, node('span', '', label));
    row.append(cell);
    countUp(number, value, decimals);
  }
}

function countUp(element, target, decimals) {
  if (reducedMotion) { element.textContent = target.toFixed(decimals); return; }
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / 1400);
    element.textContent = (target * (1 - Math.pow(1 - t, 3))).toFixed(decimals);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderLenses(maps) {
  const row = $('overlay-select');
  row.replaceChildren();
  for (const [key, title, hint] of LENSES) {
    if (!maps[key]) continue;
    const button = node('button', 'lens');
    button.type = 'button';
    button.dataset.lens = key;
    const img = node('img');
    img.src = maps[key];
    img.alt = '';
    img.loading = 'lazy';
    const label = node('span', '', title);
    label.append(node('small', '', hint));
    button.append(img, label);
    button.addEventListener('click', () => setOverlay(state.overlayKey === key ? 'none' : key));
    row.append(button);
  }
}

function setOverlay(key) {
  state.overlayKey = key;
  const overlay = $('comparison-overlay');
  const url = key === 'none' ? null : state.result?.maps?.[key];
  overlay.hidden = !url;
  if (url) overlay.src = url;
  $$('.lens').forEach((lens) => lens.classList.toggle('is-active', lens.dataset.lens === key));
}

function addStageCard(stage, index, isFinal = false) {
  const button = node('button', `stage-card ${isFinal ? 'is-active' : ''}`.trim());
  button.type = 'button';
  button.style.animationDelay = `${index * 70}ms`;
  const img = node('img');
  img.src = stage.image_url;
  img.alt = `${stage.title || stage.name} stage preview`;
  img.loading = 'lazy';
  const body = node('div', 'stage-body');
  body.append(node('span', 'stage-num', isFinal ? 'RESULT' : `STEP ${String(index + 1).padStart(2, '0')}`));
  body.append(node('strong', '', stage.title || stage.name));
  if (stage.caption) body.append(node('p', '', stage.caption));
  button.append(img, body);
  button.addEventListener('click', () => {
    showInViewer(stage.image_url);
    setOverlay('none');
    $$('.stage-card').forEach((card) => card.classList.toggle('is-active', card === button));
    $('comparison').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });
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

// Only rows with a measured value are shown; unavailable scores are never filled in.
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
  const measured = rows.filter(([, before, after]) => before != null || after != null);
  if (!measured.length) {
    container.append(node('p', 'metrics-empty', 'Learned quality scores are not available in this build, so none are shown.'));
  } else {
    const head = node('div', 'metric-row is-head');
    head.append(node('span', '', 'Metric'), node('span', '', 'Before'), node('span', '', 'After'), node('span'));
    container.append(head);
  }
  for (const [label, before, after, direction] of measured) {
    const row = node('div', 'metric-row');
    row.append(node('span', '', label), node('span', '', valueText(before)), node('span', '', valueText(after)));
    const delta = compareValues(before, after, direction);
    const arrow = node('span', `metric-delta ${delta.state === 'worse' ? 'is-worse' : ''}`, delta.arrow);
    arrow.title = delta.label;
    row.append(arrow);
    container.append(row);
  }
  $('metrics-resolution').textContent = metrics.resolution ? `${metrics.resolution}px proxy` : '';
}

function renderTimings(timings) {
  const container = $('timings');
  container.replaceChildren();
  const keys = ['diagnose', 'recoverability', 'tuner', 'search', 'classical', 'deep', 'fusion', 'denoise', 'guardrails', 'metrics'];
  const present = keys.filter((key) => timings[key] != null);
  const max = Math.max(1, ...present.map((key) => Number(timings[key])));
  present.forEach((key, index) => {
    const value = Number(timings[key]);
    const row = node('div', 'timing-row');
    const bar = node('span', 'bar');
    const fill = node('i');
    fill.style.width = `${Math.max(2, (value / max) * 100)}%`;
    fill.style.animationDelay = `${index * 60}ms`;
    bar.append(fill);
    row.append(node('span', '', key), bar, node('b', '', `${value.toFixed(0)} ms`));
    container.append(row);
  });
  if (timings.total != null) {
    const row = node('div', 'timing-row is-total');
    row.append(node('span', '', 'total'), node('span'), node('b', '', `${Number(timings.total).toFixed(0)} ms`));
    container.append(row);
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
  title.append(node('span', '', `#${String(candidate.id + 1).padStart(2, '0')}`));
  title.append(node('span', 'winner-tag', candidate.chosen ? 'SELECTED' : (candidate.feasible ? '' : 'REVIEW')));
  info.append(title);
  info.append(node('p', '', `score ${candidate.total == null ? '—' : Number(candidate.total).toFixed(3)}`));
  card.title = candidate.reasons?.[0] || '';
  card.append(info);
  $('candidate-grid').append(card);
}

// Joy Division-style ridgelines: each line is one row of the illumination map, drawn back to front.
function drawRidgelines(canvas, url) {
  let disposed = false;
  let samples = null;
  let hoverRow = -1;
  let progress = reducedMotion ? 1 : 0;
  const image = new Image();
  const render = () => {
    if (!samples || disposed) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    const rows = samples.length;
    const padX = width * .06;
    const top = height * .2;
    const spacing = (height * .94 - top) / rows;
    const amplitude = spacing * 9;
    const visible = Math.ceil(rows * progress);
    for (let r = 0; r < visible; r += 1) {
      const line = samples[r];
      const base = top + r * spacing;
      context.beginPath();
      context.moveTo(padX, base);
      line.forEach((value, i) => context.lineTo(padX + (i / (line.length - 1)) * (width - padX * 2), base - value * amplitude));
      context.lineTo(width - padX, base);
      context.fillStyle = '#08090c';
      context.fill();
      const peak = Math.max(...line);
      const hot = r === hoverRow;
      context.strokeStyle = hot ? '#ffe9c7' : `rgba(244, 215, 168, ${.22 + peak * .78})`;
      context.lineWidth = hot ? 1.6 : 1;
      context.shadowColor = hot || peak > .5 ? 'rgba(244, 215, 168, .6)' : 'transparent';
      context.shadowBlur = hot ? 12 : peak * 8;
      context.stroke();
      context.shadowBlur = 0;
    }
  };
  image.onload = () => {
    if (disposed) return;
    const columns = 120;
    const rows = 44;
    const scratch = document.createElement('canvas');
    scratch.width = columns;
    scratch.height = rows;
    const context = scratch.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0, columns, rows);
    const data = context.getImageData(0, 0, columns, rows).data;
    let max = 0;
    samples = [];
    for (let r = 0; r < rows; r += 1) {
      const raw = [];
      for (let c = 0; c < columns; c += 1) raw.push(data[(r * columns + c) * 4] / 255);
      const smooth = raw.map((_, c) => (raw[Math.max(0, c - 1)] + raw[c] * 2 + raw[Math.min(columns - 1, c + 1)]) / 4);
      max = Math.max(max, ...smooth);
      samples.push(smooth);
    }
    if (max > 0) samples = samples.map((line) => line.map((value) => value / max));
    const start = performance.now();
    const step = (now) => {
      if (disposed) return;
      progress = reducedMotion ? 1 : Math.min(1, (now - start) / 1600);
      render();
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  image.src = url;
  const onMove = (event) => {
    if (!samples) return;
    const rect = canvas.getBoundingClientRect();
    const top = rect.height * .2;
    const spacing = (rect.height * .94 - top) / samples.length;
    const row = Math.round((event.clientY - rect.top - top) / spacing);
    hoverRow = row >= 0 && row < samples.length ? row : -1;
    render();
  };
  const onLeave = () => { hoverRow = -1; render(); };
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerleave', onLeave);
  window.addEventListener('resize', render);
  return () => {
    disposed = true;
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerleave', onLeave);
    window.removeEventListener('resize', render);
  };
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
    const all = await getGallery();
    // Synthetic scenes are only a fallback for when the real demo photos have not been fetched.
    const real = all.filter((item) => !String(item.id).startsWith('synthetic'));
    const items = real.length ? real : all;
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
    return items;
  } catch {
    $('gallery-count').textContent = 'Unavailable';
    return [];
  }
}

async function loadHealth() {
  const status = $('health');
  try {
    const health = await getHealth();
    const fake = Boolean(health.engine?.fake);
    status.classList.toggle('is-ready', health.status === 'ready');
    status.classList.toggle('is-offline', false);
    $('health-label').textContent = health.status === 'ready' ? 'Engine ready' : 'Warming up';
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
  button.firstElementChild.textContent = 'Comparing…';
  container.replaceChildren();
  const original = node('article', 'method-card is-original');
  const originalImage = node('img');
  originalImage.src = state.originalUrl;
  originalImage.alt = 'Original input';
  const originalBody = node('div');
  originalBody.append(node('span', 'method-kicker', 'Input'), node('h3', '', 'Original'), node('p', '', 'The photo as captured.'));
  original.append(originalImage, originalBody);
  container.append(original);
  const cards = ['classical', 'dl', 'hybrid'].map((mode, index) => {
    const card = node('article', 'method-card');
    card.style.animationDelay = `${(index + 1) * 90}ms`;
    const body = node('div');
    body.append(node('span', 'method-kicker', modeKicker(mode)), node('h3', '', modeLabel(mode)), node('p', '', 'Processing…'));
    card.append(node('div', 'method-ph'), body);
    container.append(card);
    return [mode, card, body];
  });
  for (const [mode, card, body] of cards) {
    try {
      const result = await enhanceImage(state.file, mode);
      const image = node('img');
      image.src = result.final.preview_url;
      image.alt = `${modeLabel(mode)} output`;
      card.firstElementChild.replaceWith(image);
      body.replaceChildren(node('span', 'method-kicker', modeKicker(mode)), node('h3', '', modeLabel(mode)), node('p', '', result.headline));
      if (result.provenance.actual_method === 'classical_fallback') {
        body.append(node('p', 'warning-item', 'Learned weights were unavailable; this result used the disclosed classical fallback.'));
      }
      const score = result.metrics?.after?.niqe;
      const parts = [`${Number(result.timings_ms?.total || 0).toFixed(0)} ms`];
      if (score != null) parts.unshift(`NIQE ${valueText(score)}`);
      body.append(node('span', 'method-time', parts.join(' · ')));
    } catch (error) {
      card.firstElementChild.remove();
      body.replaceChildren(node('h3', '', modeLabel(mode)), node('p', 'warning-item', error.message));
    }
  }
  button.disabled = false;
  button.firstElementChild.textContent = 'Run three-way comparison';
}

async function loadBenchmarks() {
  const container = $('benchmark-content');
  try {
    const payload = await getBenchmarks();
    container.replaceChildren();
    if (payload.status === 'not_ready') {
      container.append(node('span', 'label', payload.progress ? 'Evaluation in progress' : 'No runs yet'));
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
      $('camera-button').firstElementChild.textContent = 'Stop camera';
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
  $('camera-button').firstElementChild.textContent = 'Start camera';
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
function methodLabel(method) { return ({ zerodce: 'Zero-DCE++ neural model', hybrid: 'Adaptive hybrid', classical: 'Classical vision', classical_fallback: 'Classical fallback' })[method] || method; }
function modeKicker(mode) { return ({ hybrid: 'Per-pixel fusion', classical: 'Retinex + CLAHE', dl: 'Neural curve estimation' })[mode] || ''; }
function extensionFor(type) { return ({ 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/tiff': 'tiff', 'image/bmp': 'bmp' })[type] || 'png'; }
function formatBytes(size) { return size < 1024 * 1024 ? `${(size / 1024).toFixed(0)} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`; }
function valueText(value) {
  if (value == null) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—';
}
function formatMetric(label, value) { return label.toLowerCase().includes('percent') || label.toLowerCase().includes('clip') || label.toLowerCase().includes('brightness') ? `${(Number(value) * (label.toLowerCase().includes('brightness') ? 100 : 1)).toFixed(1)}%` : Number(value).toFixed(3); }

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
  setupViewerDrag();
  setupViewModes();
  setupCamera();
  setupReveals();
  setupCursor();
  $$('.rh-copy h1, .page-heading h1').forEach(splitHeadline);
  resetPipeline();
  const [items] = await Promise.all([loadGallery(), loadHealth()]);
  setInterval(loadHealth, 15000);
  setupMarquee(items);
  setupHero(items);
}

boot();
