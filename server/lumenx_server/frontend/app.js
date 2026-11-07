import { enhanceImage, getBenchmarks, getConfig, getGallery, getHealth, getText, liveSocketUrl, streamEnhance } from './api.js';
import { renderResults } from './results.js';
import { createRelight } from './relight.js';
import { createFlashlight } from './flashlight.js';
import { createRelief } from './relief.js';
import { createHeroScene } from './hero3d.js';

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
  maxUploadBytes: null,
  wipeFrame: null,
};

// Pipeline order as emitted by the core engine; modes that skip a stage leave it dimmed.
const PIPELINE = [
  ['diagnose', 'Diagnose'], ['recoverability', 'Signal'], ['tuner', 'Search'], ['classical', 'Classical'],
  ['deep', 'Neural'], ['fusion', 'Fusion'], ['denoise', 'Denoise'], ['guardrails', 'Guard'],
];
// Map legends follow the engine's renderer: viridis maps run purple (low) to yellow (high).
const LENSES = [
  ['recoverability_overlay', 'Recoverability', 'green · amber · red', 'Green: real detail was recorded. Amber: detail is estimated from weak signal. Red: too little light was recorded to recover.'],
  ['illumination_gray', 'Illumination', 'measured light field', 'How much light reached each part of the sensor. Lighter means more scene light.'],
  ['gain', 'Brightening gain', 'lift per pixel', 'How strongly each pixel was brightened. Purple is a gentle lift; yellow is the strongest.'],
  ['fusion', 'Path balance', 'per-pixel mix', 'The per-pixel mix of the deep-learning result, the classical result and highlight protection.'],
  ['denoise_strength', 'Denoise strength', 'grain cleaning', 'Where noise reduction was applied. Lighter means stronger smoothing.'],
  ['noise', 'Noise estimate', 'grain in the input', 'Estimated sensor grain in the input. Purple is clean; yellow is the noisiest.'],
  ['snr', 'Signal to noise', 'recording quality', 'Signal-to-noise per pixel. Yellow was recorded cleanly; purple is mostly noise.'],
];
const RECOVERY_CLASSES = [
  ['recovered_pct', 'Recovered', 'recorded detail', 'is-recovered'],
  ['estimated_pct', 'Estimated', 'weak signal', 'is-estimated'],
  ['unrecoverable_pct', 'Unrecoverable', 'too dark to recover', 'is-unrecoverable'],
];
const GUARD_LABELS = { overexposure: 'Blown highlights', color_shift: 'Colour shift', detail_loss: 'Detail kept', halo: 'Halo around lights' };
// How each check reads its value against the threshold, per core/config/default.yaml.
const GUARD_RULES = {
  overexposure: 'Added clipping vs limit; flags only if total clipping is also over its limit.',
  color_shift: 'Lower is better; flags above the limit.',
  detail_loss: 'Edge detail kept; flags below the limit.',
  halo: 'Lower is better; flags above the limit.',
};
const SUPPORTED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/tiff', 'image/bmp'];
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
  if (!file) return;
  if (!SUPPORTED_TYPES.includes(file.type)) return notify(`“${file.name}” isn't a supported image. Use PNG, JPEG, WebP, TIFF or BMP.`, 'error');
  if (state.maxUploadBytes && file.size > state.maxUploadBytes) return notify(`“${file.name}” is ${formatBytes(file.size)}; the limit is ${formatBytes(state.maxUploadBytes)}.`, 'error');
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

const SHOWCASE = '/static/showcase/';

// Landing hero: plays back a recorded engine run (scripts/build_showcase.py) so it opens instantly and works on
// a static host. With WebGL the photo becomes a 3D surface; otherwise a CSS flashlight reveals the restoration.
async function setupHero() {
  const hero = $('hero');
  const hint = $('hero-hint');
  const setHint = (text) => hint.replaceChildren(node('span', 'rh-dot'), node('span', '', text));
  const touch = window.matchMedia('(hover: none)').matches;
  if (touch) hero.querySelector('.rh-copy p').textContent = "A wandering light shows LumenX's restoration. Tap to raise the whole scene by the light it recovered.";
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

  let manifest;
  try {
    const response = await fetch(`${SHOWCASE}manifest.json`);
    if (!response.ok) throw new Error('No recorded showcase.');
    manifest = await response.json();
  } catch {
    setHint('Run scripts/build_showcase.py to record the landing scene');
    return;
  }
  const darkUrl = SHOWCASE + manifest.original;
  const litUrl = SHOWCASE + manifest.restored;
  const meta = $('hero-meta');
  meta.replaceChildren(node('strong', '', manifest.title));
  meta.append(node('span', '', `${methodLabel(manifest.method)} · recorded run, ${(manifest.total_ms / 1000).toFixed(1)} s on ${String(manifest.device || 'cpu').toUpperCase()}`));
  meta.append(node('br'), node('span', '', manifest.attribution));
  if (manifest.fake) meta.append(node('span', '', ' · mock engine'));
  fetch(darkUrl).then((response) => response.blob()).then((blob) => {
    state.heroFile = new File([blob], 'showcase.jpg', { type: blob.type || 'image/jpeg' });
  }).catch(() => {});

  const scene = createHeroScene($('hero-canvas'), {
    darkUrl, litUrl, heightUrl: SHOWCASE + manifest.heightmap, reducedMotion,
    onReady: () => {
      hero.classList.add('has-3d');
      hero.classList.remove('is-pending');
      setHint(touch ? 'Tap to raise the night into 3D' : 'Move your light · click to raise the night');
    },
  });
  if (scene) {
    let down = null;
    hero.addEventListener('pointerdown', (event) => { down = { x: event.clientX, y: event.clientY }; });
    hero.addEventListener('click', (event) => {
      if (event.target.closest('a, button')) return;
      // A drag to orbit should not also count as a click.
      if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) > 6) return;
      const raised = !scene.raised;
      scene.setRaised(raised);
      hero.classList.toggle('is-raised', raised);
      setHint(raised ? (touch ? 'Tap to lower' : 'Drag to orbit · click to lower') : (touch ? 'Tap to raise the night into 3D' : 'Move your light · click to raise the night'));
    });
  } else {
    $('hero-dark').src = darkUrl;
    $('hero-lit').src = litUrl;
    createFlashlight(hero, { radius: 0.16, reducedMotion });
    hero.classList.remove('is-pending');
    setHint(touch ? 'Tap to flood the scene with light' : 'Move your light · click to flood the scene');
  }
  setupStory(manifest);
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
    cursor.classList.toggle('is-hidden', Boolean(event.target.closest('.reveal-hero:not(.is-flooded):not(.is-raised) .rh-media, .reveal-hero:not(.is-raised) .rh-canvas, .viewer[data-view="flashlight"][data-state="done"]')));
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

// Story: the recorded showcase run's real stages, scrolled sideways while the section is pinned.
function setupStory(manifest) {
  const section = $('story');
  const track = $('story-track');
  const stages = manifest.stages || [];
  if (!stages.length) return;
  track.replaceChildren();
  const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
  $('story-count').textContent = words[stages.length + 1] || String(stages.length + 1);
  $('story-meta').textContent = `${manifest.title} · ${methodLabel(manifest.method)} · ${(manifest.total_ms / 1000).toFixed(1)} s on ${String(manifest.device || 'cpu').toUpperCase()}${manifest.fake ? ' · mock engine' : ''}`;
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
  stages.forEach((stage, index) => addPanel(`STEP ${String(index + 1).padStart(2, '0')}`, stage.title, stage.caption, SHOWCASE + stage.image));
  const finalBody = addPanel('RESULT', 'The restored scene', 'Every step above was measured from this one photo. Now try it on yours.', SHOWCASE + manifest.restored, true);
  const cta = node('a', 'sp-cta', 'Restore your own →');
  cta.href = '#console';
  cta.addEventListener('click', (event) => {
    event.preventDefault();
    if (state.heroFile && !state.file) selectFile(state.heroFile);
    $('#console').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
  });
  finalBody.append(cta);
  section.hidden = false;

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
    const result = await withRetry(() => streamEnhance(state.file, state.mode, enqueue));
    await playback;
    state.result = result;
    renderResult(result);
  } catch (error) {
    notify(friendlyError(error), 'error');
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
  renderRecoverability(result.recoverability);
  renderGuardrails(result.guardrails || []);
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
  renderLensLegend('none');
}

function renderLensLegend(key) {
  const legend = $('lens-legend');
  const entry = LENSES.find(([name]) => name === key);
  legend.replaceChildren();
  if (!entry) {
    legend.append(node('span', 'muted', 'Tap a map to overlay it on the result and read what it shows.'));
    return;
  }
  legend.append(node('strong', '', entry[1]), node('span', '', entry[3]));
  if (key === 'recoverability_overlay') {
    const key_ = node('span', 'legend-key');
    for (const [, label, , tone] of RECOVERY_CLASSES) key_.append(node('i', tone, label));
    legend.append(key_);
  }
}

function setOverlay(key) {
  state.overlayKey = key;
  const overlay = $('comparison-overlay');
  const url = key === 'none' ? null : state.result?.maps?.[key];
  overlay.hidden = !url;
  if (url) overlay.src = url;
  $$('.lens').forEach((lens) => lens.classList.toggle('is-active', lens.dataset.lens === key));
  if ($('lens-legend')) renderLensLegend(key);
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

function renderRecoverability(recovery) {
  const sentence = $('recover-sentence');
  const bar = $('recover-bar');
  const legend = $('recover-legend');
  bar.replaceChildren();
  legend.replaceChildren();
  if (!recovery) {
    sentence.textContent = 'Recoverability was not estimated for this run.';
    return;
  }
  sentence.textContent = recovery.sentence || '';
  for (const [field, label, hint, tone] of RECOVERY_CLASSES) {
    const value = Number(recovery[field]);
    if (!Number.isFinite(value)) continue;
    const segment = node('i', tone);
    segment.style.width = `${Math.max(0, value)}%`;
    segment.title = `${label}: ${value.toFixed(1)}%`;
    bar.append(segment);
    const item = node('div', `recover-item ${tone}`);
    item.append(node('b', '', `${value.toFixed(1)}%`), node('span', '', label), node('small', '', hint));
    legend.append(item);
  }
  if (recovery.estimator_confidence != null) {
    legend.append(node('p', 'recover-note', `Estimator confidence ${Number(recovery.estimator_confidence).toFixed(2)}. These are estimates from the input, not a certificate of recovered detail.`));
  }
}

function renderGuardrails(checks) {
  const grid = $('guardrails');
  grid.replaceChildren();
  if (!checks.length) {
    grid.append(node('p', 'muted', 'No safety checks were reported for this run.'));
    return;
  }
  for (const check of checks) {
    const passed = check.passed;
    const chip = node('div', `guard-chip ${passed === true ? 'is-pass' : passed === false ? 'is-flag' : 'is-na'}`);
    const top = node('div', 'guard-top');
    top.append(node('span', '', GUARD_LABELS[check.check] || check.check.replaceAll('_', ' ')), node('b', '', passed === true ? 'PASS' : passed === false ? 'FLAG' : '—'));
    const values = node('div', 'guard-values mono', `${check.value == null ? '–' : Number(check.value).toFixed(3)} / limit ${check.threshold == null ? '–' : Number(check.threshold).toFixed(2)}`);
    chip.append(top, values);
    if (GUARD_RULES[check.check]) chip.append(node('div', 'guard-rule', GUARD_RULES[check.check]));
    if (check.reason) chip.title = check.reason;
    grid.append(chip);
  }
}

// Every row is shown; a missing value is "–" with the engine's own reason, never a substituted number.
function renderMetrics(metrics) {
  const container = $('metrics-table');
  container.replaceChildren();
  const directions = metrics.directions || {};
  const unavailable = metrics.unavailable || {};
  const head = node('div', 'metric-row is-head');
  head.append(node('span', '', 'Measurement'), node('span', '', 'Before'), node('span', '', 'After'), node('span'));
  container.append(head);
  const rows = [
    ['CLIP-IQA', 'clipiqa', metrics.before?.clipiqa, metrics.after?.clipiqa, directions.clipiqa || 'higher', 3],
    ['MUSIQ', 'musiq', metrics.before?.musiq, metrics.after?.musiq, directions.musiq || 'higher', 2],
    ['NIQE', 'niqe', metrics.before?.niqe, metrics.after?.niqe, directions.niqe || 'lower', 3],
    ['Blown highlights %', 'clip_pct', metrics.clip_pct?.before, metrics.clip_pct?.after, 'lower', 2],
    ['Noise σ', 'noise_sigma', metrics.noise_sigma?.before, metrics.noise_sigma?.after, 'lower', 4],
    ['Edge retention', 'edge_retention', null, metrics.edge_retention, 'higher', 3],
    ['Colour shift ΔE', 'chroma_delta_e', null, metrics.chroma_delta_e, 'lower', 2],
  ];
  for (const [label, key, before, after, direction, digits] of rows) {
    const row = node('div', 'metric-row');
    const reason = unavailable[key] || 'Not measured for this run.';
    const cell = (value, onlyAfter) => {
      const span = node('span', value == null ? 'is-missing' : '', value == null ? '–' : Number(value).toFixed(digits));
      if (value == null) span.title = onlyAfter ? 'Measured on the result only.' : reason;
      return span;
    };
    row.append(node('span', '', label), cell(before, key === 'edge_retention' || key === 'chroma_delta_e'), cell(after, false));
    const delta = compareValues(before, after, direction);
    const arrow = node('span', `metric-delta ${delta.state === 'worse' ? 'is-worse' : ''} ${delta.state === 'na' ? 'is-unavailable' : ''}`.trim(), delta.arrow);
    arrow.title = delta.state === 'na' ? 'No before/after pair to compare.' : `${delta.state === 'better' ? 'Improved' : delta.state === 'worse' ? 'Worse' : 'Unchanged'} (${direction === 'higher' ? 'higher' : 'lower'} is better)`;
    row.append(arrow);
    container.append(row);
  }
  $('metrics-resolution').textContent = metrics.resolution ? `measured on a ${metrics.resolution}px proxy` : '';
}

function renderTimings(timings) {
  const container = $('timings');
  const stack = $('timing-stack');
  container.replaceChildren();
  stack.replaceChildren();
  const keys = ['diagnose', 'recoverability', 'tuner', 'search', 'classical', 'deep', 'fusion', 'denoise', 'guardrails', 'metrics'];
  const present = keys.filter((key) => timings[key] != null);
  const sum = present.reduce((total, key) => total + Number(timings[key]), 0) || 1;
  const max = Math.max(1, ...present.map((key) => Number(timings[key])));
  present.forEach((key, index) => {
    const value = Number(timings[key]);
    const segment = node('i', `seg-${index % 6}`);
    segment.style.width = `${(value / sum) * 100}%`;
    segment.title = `${key} · ${value.toFixed(0)} ms`;
    stack.append(segment);
    const row = node('div', 'timing-row');
    const bar = node('span', 'bar');
    const fill = node('i', `seg-${index % 6}`);
    fill.style.width = `${Math.max(2, (value / max) * 100)}%`;
    fill.style.animationDelay = `${index * 60}ms`;
    bar.append(fill);
    row.append(node('span', '', key), bar, node('b', '', `${value.toFixed(0)} ms`));
    container.append(row);
  });
  $('timing-total').textContent = timings.total != null ? `${Number(timings.total).toFixed(0)} ms total` : '';
}

function renderCandidates(candidates) {
  $('candidate-grid').replaceChildren();
  for (const candidate of candidates) renderCandidate(candidate, candidate.thumb_url);
  $('candidates-section').hidden = candidates.length === 0;
}

function renderCandidate(candidate, imageUrl) {
  if (!candidate) return;
  const card = node('article', `candidate-card ${candidate.chosen ? 'is-winner' : ''} ${candidate.feasible === false ? 'is-infeasible' : ''}`.trim());
  if (imageUrl) {
    const img = node('img');
    img.src = imageUrl;
    img.alt = `Candidate ${candidate.id + 1} preview`;
    img.loading = 'lazy';
    card.append(img);
  }
  const info = node('div', 'candidate-info');
  const title = node('div', 'candidate-title');
  title.append(node('span', '', `#${String(candidate.id + 1).padStart(2, '0')}`));
  title.append(node('span', 'winner-tag', candidate.chosen ? 'CHOSEN' : (candidate.feasible === false ? 'REJECTED' : '')));
  info.append(title);
  info.append(node('p', 'candidate-score', `score ${candidate.total == null ? '–' : Number(candidate.total).toFixed(3)}`));
  const reason = candidate.chosen ? 'Best score among the feasible settings.' : (candidate.reasons || []).join(' ');
  if (reason) info.append(node('p', 'candidate-reason', reason));
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
  image.crossOrigin = 'anonymous';
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

const COMPARE_MODES = ['hybrid', 'dl', 'classical'];
const COMPARE_ROWS = [
  ['Blown highlights %', (r) => r.metrics?.clip_pct?.after, 'lower', 2],
  ['Colour shift ΔE', (r) => r.metrics?.chroma_delta_e, 'lower', 2],
  ['Edge retention', (r) => r.metrics?.edge_retention, 'higher', 3],
  ['Noise σ (after)', (r) => r.metrics?.noise_sigma?.after, 'lower', 4],
  ['CLIP-IQA', (r) => r.metrics?.after?.clipiqa, 'higher', 3],
  ['MUSIQ', (r) => r.metrics?.after?.musiq, 'higher', 2],
  ['Latency ms', (r) => r.timings_ms?.total, 'lower', 0],
];

async function runComparison() {
  if (!state.file) return;
  const button = $('compare-button');
  const section = $('compare-section');
  const container = $('compare-results');
  button.disabled = true;
  button.firstElementChild.textContent = 'Comparing…';
  section.hidden = false;
  container.replaceChildren();
  $('compare-table').replaceChildren();
  section.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
  const original = node('article', 'method-card is-original');
  const originalImage = node('img');
  originalImage.src = state.originalUrl;
  originalImage.alt = 'Original input';
  const originalBody = node('div');
  originalBody.append(node('span', 'method-kicker', 'Input'), node('h3', '', 'Original'), node('p', '', 'The photo as captured.'));
  original.append(originalImage, originalBody);
  container.append(original);
  const cards = COMPARE_MODES.map((mode, index) => {
    const card = node('article', 'method-card');
    card.style.animationDelay = `${(index + 1) * 90}ms`;
    const body = node('div');
    body.append(node('span', 'method-kicker', modeKicker(mode)), node('h3', '', modeLabel(mode)), node('p', '', 'Waiting…'));
    card.append(node('div', 'method-ph'), body);
    container.append(card);
    return { mode, card, body, result: null };
  });
  for (const [index, entry] of cards.entries()) {
    $('compare-status').textContent = `running ${index + 1} of ${cards.length}`;
    entry.body.lastElementChild.textContent = 'Processing…';
    try {
      const result = await withRetry(() => enhanceImage(state.file, entry.mode));
      entry.result = result;
      const image = node('img');
      image.src = result.final.preview_url;
      image.alt = `${modeLabel(entry.mode)} output`;
      entry.card.firstElementChild.replaceWith(image);
      entry.body.replaceChildren(node('span', 'method-kicker', modeKicker(entry.mode)), node('h3', '', modeLabel(entry.mode)), node('p', '', result.headline));
      if (result.provenance.actual_method === 'classical_fallback') {
        entry.body.append(node('p', 'warning-item', 'Learned weights were unavailable; this result used the disclosed classical fallback.'));
      }
      const open = node('button', 'method-open', 'Open full report →');
      open.type = 'button';
      open.addEventListener('click', () => {
        renderResult(result);
        $('comparison').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });
      });
      entry.body.append(node('span', 'method-time', `${Number(result.timings_ms?.total || 0).toFixed(0)} ms`), open);
    } catch (error) {
      entry.card.firstElementChild.remove();
      entry.body.replaceChildren(node('h3', '', modeLabel(entry.mode)), node('p', 'warning-item', friendlyError(error)));
    }
    renderCompareTable(cards);
  }
  $('compare-status').textContent = '';
  button.disabled = false;
  button.firstElementChild.textContent = 'Compare all three';
}

// Highlights the best value per row among modes that reported it; missing values stay "–".
function renderCompareTable(cards) {
  const table = $('compare-table');
  table.replaceChildren();
  const head = node('tr');
  head.append(node('th', '', 'Measured on this image'));
  cards.forEach(({ mode }) => head.append(node('th', '', modeLabel(mode))));
  table.append(head);
  for (const [label, pick, direction, digits] of COMPARE_ROWS) {
    const values = cards.map(({ result }) => {
      const value = result ? pick(result) : null;
      return value == null || !Number.isFinite(Number(value)) ? null : Number(value);
    });
    const present = values.filter((value) => value != null);
    const best = present.length > 1 ? (direction === 'higher' ? Math.max(...present) : Math.min(...present)) : null;
    const row = node('tr');
    row.append(node('th', '', `${label} ${direction === 'higher' ? '↑' : '↓'}`));
    values.forEach((value, index) => {
      const cell = node('td', value != null && value === best ? 'is-best' : '', value == null ? '–' : value.toFixed(digits));
      if (value == null && cards[index].result) cell.title = 'Not measured for this run.';
      row.append(cell);
    });
    table.append(row);
  }
}

let benchmarksLoaded = false;
async function loadBenchmarks() {
  if (benchmarksLoaded) return;
  const container = $('benchmark-content');
  try {
    const payload = await getBenchmarks();
    if (payload.status === 'not_ready' || !payload.tables) {
      container.replaceChildren(node('span', 'label', payload.progress ? 'Evaluation in progress' : 'No results yet'));
      container.append(node('p', '', payload.progress ? 'The evaluation is still running; results appear here when it finishes.' : 'Run the evaluation scripts in the server folder to publish results.'));
      return;
    }
    // The failure review is optional; the page still renders without it.
    const failureCsv = await getText('/files/results/failure_review.csv').catch(() => null);
    renderResults($('panel-benchmarks'), payload, failureCsv);
    container.hidden = true;
    $('results').hidden = false;
    benchmarksLoaded = true;
  } catch (error) {
    container.replaceChildren(node('span', 'label', 'Results unavailable'), node('p', '', friendlyError(error)));
  }
}

async function startCamera() {
  if (state.socket) return stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) return showError('Camera access is unavailable in this browser.');
  try {
    state.camera = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    const video = $('camera-video');
    video.srcObject = state.camera;
    await video.play();
    state.socket = new WebSocket(liveSocketUrl());
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

function notify(message, tone = 'info') {
  const notice = $('console-notice');
  notice.className = `notice is-${tone}`;
  notice.textContent = message;
  notice.hidden = false;
}

function showError(message) { notify(message, 'error'); }
function clearError() { const notice = $('console-notice'); if (notice) notice.hidden = true; }

// Plain-language messages for the API's documented error codes (413/415/422/503).
function friendlyError(error) {
  const limit = state.maxUploadBytes ? formatBytes(state.maxUploadBytes) : 'the upload limit';
  if (error.status === 413) return `That image is too large. Keep files under ${limit} and within the server's pixel limit.`;
  if (error.status === 415) return 'That file type is not supported. Use PNG, JPEG, WebP, TIFF or BMP.';
  if (error.status === 422) return error.code === 'decode_failed' ? 'The image could not be read. It may be damaged or only partly downloaded.' : `The request was rejected: ${error.message}`;
  if (error.status === 503) return error.code === 'queue_full' ? 'The engine is still busy. Please try again in a moment.' : 'The engine is not available right now.';
  if (error instanceof TypeError) return 'Could not reach the LumenX server. Check that it is running.';
  return error.message || 'Something went wrong.';
}

// Retries only the "queue full" case, honouring the server's Retry-After header.
async function withRetry(task, attempts = 3) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await task();
    } catch (error) {
      if (!(error.status === 503 && error.code === 'queue_full') || attempt >= attempts) throw error;
      const wait = Math.max(1, error.retryAfter || 2);
      notify(`The engine is busy. Retrying in ${wait} s (attempt ${attempt + 1} of ${attempts})…`, 'info');
      await sleep(wait * 1000);
    }
  }
}
function modeLabel(mode) { return ({ hybrid: 'Adaptive hybrid', classical: 'Classical vision', dl: 'Deep learning (CIDNet)' })[mode] || mode; }
function methodLabel(method) { return ({ cidnet: 'Deep learning (CIDNet)', zerodce: 'Zero-DCE++', hybrid: 'Adaptive hybrid', classical: 'Classical vision', classical_fallback: 'Classical fallback' })[method] || method; }
function modeKicker(mode) { return ({ hybrid: 'CIDNet + highlight repair', classical: 'Retinex + CLAHE', dl: 'CIDNet' })[mode] || ''; }
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
  getConfig().then((config) => { state.maxUploadBytes = Number(config.server?.max_upload_bytes) || null; }).catch(() => {});
  const [items] = await Promise.all([loadGallery(), loadHealth()]);
  setInterval(loadHealth, 15000);
  setupMarquee(items);
  setupHero();
}

boot();
