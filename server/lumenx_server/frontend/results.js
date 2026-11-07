// Results & performance page. Everything is derived from GET /benchmarks and the failure-review CSV;
// the only fixed text is labelling (names, directions, plain-English descriptions).

const MODE_ORDER = ['classical', 'zerodce', 'dl', 'hybrid'];
const MODE_NAMES = { classical: 'Classical', zerodce: 'Zero-DCE++', dl: 'Deep learning (CIDNet)', hybrid: 'LumenX Hybrid' };
const HEADLINE_COLUMNS = [
  ['psnr', 'PSNR', 'higher', 2, ' dB'],
  ['ssim', 'SSIM', 'higher', 3, ''],
  ['lpips', 'LPIPS', 'lower', 3, ''],
  ['clip_pct_out', 'Blown-out px', 'lower', 2, '%'],
  ['chroma_delta_e', 'Colour shift ΔE', 'lower', 2, ''],
  ['edge_retention', 'Edge retention', 'higher', 3, ''],
  ['clipiqa', 'CLIP-IQA', 'higher', 3, ''],
  ['musiq', 'MUSIQ', 'higher', 1, ''],
  ['latency_ms', 'Latency', 'lower', 0, ' ms'],
];
const ABLATIONS = {
  no_dl: ['without the deep model', 'CIDNet does the core brightening; everything else refines its output.'],
  no_fusion: ['without highlight repair', 'Blends in a protected copy wherever the deep result would blow out lamps and windows.'],
  no_classical: ['without the classical path', 'Retinex + CLAHE offers a conventional estimate alongside the deep model.'],
  no_color_correction: ['without colour correction', 'Pulls any colour cast back toward the input’s own hues.'],
  no_gain_denoise: ['without gain-aware denoise', 'Removes grain only where brightening amplified it.'],
  no_guardrails: ['without the safety audit', 'Checks highlights, colour shift, detail and halos; it reports rather than edits by default.'],
  no_recoverability: ['without recoverability control', 'Limits smoothing to near-black areas where little signal was recorded.'],
  no_tuner: ['without adaptive search', 'Tries candidate settings per image and keeps the best-scoring safe one.'],
};
const ABLATION_METRICS = [
  ['psnr', 'PSNR', 'higher', 2, ' dB'],
  ['ssim', 'SSIM', 'higher', 3, ''],
  ['clip_pct_out', 'Blown-out pixels', 'lower', 2, '%'],
];
const FLAG_NAMES = { overexposure: 'Blown highlights', detail_loss: 'Detail loss', halo: 'Halo', color_shift: 'Colour shift' };

const node = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};
const num = (value) => (value == null || value === '' || !Number.isFinite(Number(value)) ? null : Number(value));
const fmt = (value, digits, unit = '') => (value == null ? '–' : `${value.toFixed(digits)}${unit}`);

export function renderResults(root, payload, failureCsv) {
  const $ = (role) => root.querySelector(`[data-role="${role}"]`);
  const rows = Object.values(payload.tables || {}).find(Array.isArray) || [];
  const datasets = [...new Set(rows.map((row) => row.dataset))];
  const headlineSet = datasets.find((name) => !/ablation|pretrained|finetun/i.test(name)) || datasets[0];
  const headline = MODE_ORDER.map((mode) => rows.find((row) => row.dataset === headlineSet && row.mode === mode && row.ablation === 'default')).filter(Boolean);
  const byMode = Object.fromEntries(headline.map((row) => [row.mode, row]));

  renderCallouts($('result-callouts'), byMode);
  renderHeadlineTable($('headline-table'), headline);
  $('headline-caption').textContent = headline.length
    ? `${headlineSet} · ${headline[0].images} paired images · default settings for every method.`
    : 'No default comparison rows were found in the results.';

  const ablationSet = datasets.find((name) => /ablation/i.test(name));
  const ablationRows = ablationSet ? rows.filter((row) => row.dataset === ablationSet && row.mode === 'hybrid') : [];
  renderAblation($('ablation-chart'), $('ablation-notes'), ablationRows);
  $('ablation-caption').textContent = ablationSet
    ? `${ablationSet} · LumenX Hybrid with one component removed at a time.`
    : 'No ablation results were found.';

  renderFinetune($('finetune-panel'), rows, datasets);
  renderLimitations($('limitations'), ablationRows);
  renderFailures($('failure-summary'), $('failure-table'), $('failure-count'), failureCsv);
  renderDatasets($('dataset-picker'), $('dataset-table'), rows, datasets);
  $('results-notice').textContent = payload.notice || '';
  setupTooltips(root);
}

function renderCallouts(container, byMode) {
  container.replaceChildren();
  const hybrid = byMode.hybrid;
  const deep = byMode.dl;
  const classical = byMode.classical;
  const tiles = [];
  if (hybrid && deep && num(hybrid.clip_pct_out) && num(deep.clip_pct_out) != null) {
    const ratio = num(deep.clip_pct_out) / num(hybrid.clip_pct_out);
    if (ratio > 1.5) tiles.push([`${ratio.toFixed(0)}×`, 'fewer blown-out pixels than the deep model alone', `${fmt(num(hybrid.clip_pct_out), 2, '%')} vs ${fmt(num(deep.clip_pct_out), 2, '%')}`]);
  }
  if (hybrid && deep && num(hybrid.psnr) != null && num(deep.psnr) != null) {
    const delta = num(hybrid.psnr) - num(deep.psnr);
    tiles.push([Math.abs(delta) < 0.05 ? '≈' : `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(2)}`, Math.abs(delta) < 0.05 ? 'PSNR, matching the deep model' : `dB PSNR vs the deep model`, `${fmt(num(hybrid.psnr), 2, ' dB')} vs ${fmt(num(deep.psnr), 2, ' dB')}`]);
  }
  if (hybrid && deep && num(hybrid.ssim) != null && num(deep.ssim) != null) {
    tiles.push([fmt(num(hybrid.ssim), 3), num(hybrid.ssim) >= num(deep.ssim) ? 'SSIM, at or above the deep model' : 'SSIM, just below the deep model', `deep model ${fmt(num(deep.ssim), 3)}`]);
  }
  if (hybrid && classical && num(hybrid.psnr) != null && num(classical.psnr) != null) {
    const delta = num(hybrid.psnr) - num(classical.psnr);
    tiles.push([`${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)}`, 'dB PSNR over classical enhancement', `${fmt(num(classical.psnr), 2, ' dB')} → ${fmt(num(hybrid.psnr), 2, ' dB')}`]);
  }
  for (const [value, label, detail] of tiles) {
    const tile = node('div', 'callout');
    tile.append(node('strong', '', value), node('span', '', label), node('small', 'mono', detail));
    container.append(tile);
  }
  container.hidden = tiles.length === 0;
}

function best(values, direction) {
  const present = values.filter((value) => value != null);
  if (present.length < 2) return null;
  return direction === 'higher' ? Math.max(...present) : Math.min(...present);
}

function renderHeadlineTable(table, rows) {
  table.replaceChildren();
  if (!rows.length) return;
  const head = node('tr');
  head.append(node('th', '', 'Method'));
  for (const [, label, direction] of HEADLINE_COLUMNS) head.append(node('th', '', `${label} ${direction === 'higher' ? '↑' : '↓'}`));
  table.append(head);
  const bests = HEADLINE_COLUMNS.map(([key, , direction]) => best(rows.map((row) => num(row[key])), direction));
  for (const row of rows) {
    const tr = node('tr', row.mode === 'hybrid' ? 'is-ours' : '');
    const name = node('th', '', MODE_NAMES[row.mode] || row.mode);
    if (row.failed_guardrail_runs) name.append(node('small', 'mono', ` · ${row.failed_guardrail_runs}/${row.images} audit flags`));
    tr.append(name);
    HEADLINE_COLUMNS.forEach(([key, , , digits, unit], index) => {
      const value = num(row[key]);
      const cell = node('td', value != null && value === bests[index] ? 'is-best' : '', fmt(value, digits, unit));
      if (value == null) cell.title = 'Not measured in this evaluation run.';
      tr.append(cell);
    });
    table.append(tr);
  }
}

function renderAblation(chart, notes, rows) {
  chart.replaceChildren();
  notes.replaceChildren();
  const full = rows.find((row) => row.ablation === 'default');
  if (!full) { chart.append(node('p', 'muted', 'No ablation results available.')); return; }
  const variants = [full, ...rows.filter((row) => row.ablation !== 'default').sort((a, b) => num(a.psnr) - num(b.psnr))];
  const label = (row) => (row.ablation === 'default' ? 'Full LumenX' : (ABLATIONS[row.ablation]?.[0] || row.ablation.replaceAll('_', ' ')));
  for (const [key, title, direction, digits, unit] of ABLATION_METRICS) {
    const panel = node('figure', 'bar-panel');
    panel.append(node('figcaption', '', `${title} ${direction === 'higher' ? '↑ higher is better' : '↓ lower is better'}`));
    const values = variants.map((row) => num(row[key]));
    const max = Math.max(...values.filter((value) => value != null), 1e-9);
    variants.forEach((row, index) => {
      const value = values[index];
      const line = node('div', `bar-row ${row.ablation === 'default' ? 'is-full' : ''}`);
      line.append(node('span', 'bar-label', label(row)));
      const track = node('span', 'bar-track');
      const fill = node('i');
      fill.style.width = `${value == null ? 0 : Math.max(0.6, (value / max) * 100)}%`;
      fill.style.animationDelay = `${index * 40}ms`;
      track.append(fill);
      const fullValue = num(full[key]);
      const delta = value != null && fullValue != null && row !== full ? value - fullValue : null;
      line.dataset.tip = `${label(row)} · ${title}: ${fmt(value, digits, unit)}${delta != null ? ` (${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(digits)} vs full)` : ''}`;
      line.append(track, node('span', 'bar-value mono', fmt(value, digits, unit)));
      panel.append(line);
    });
    chart.append(panel);
  }
  for (const row of variants.slice(1)) {
    const [name, description] = ABLATIONS[row.ablation] || [row.ablation, ''];
    const changes = ABLATION_METRICS.map(([key, title, direction, digits, unit]) => {
      const delta = num(row[key]) != null && num(full[key]) != null ? num(row[key]) - num(full[key]) : null;
      const threshold = key === 'ssim' ? 0.002 : key === 'psnr' ? 0.05 : 0.05;
      if (delta == null || Math.abs(delta) < threshold) return null;
      const worse = direction === 'higher' ? delta < 0 : delta > 0;
      return { text: `${title} ${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(digits)}${unit}`, worse };
    }).filter(Boolean);
    const item = node('article', `ablation-note ${changes.length ? '' : 'is-neutral'}`);
    const head = node('div', 'an-head');
    head.append(node('strong', '', name.charAt(0).toUpperCase() + name.slice(1)));
    const verdict = node('span', 'an-verdict mono', changes.length ? (changes.some((change) => change.worse) ? 'matters' : 'small gain') : 'no change');
    head.append(verdict);
    item.append(head, node('p', '', description));
    const effect = node('p', 'an-effect mono');
    effect.textContent = changes.length ? changes.map((change) => change.text).join(' · ') : 'Removing it left PSNR, SSIM and blown-out pixels unchanged on these images.';
    item.append(effect);
    notes.append(item);
  }
}

function renderFinetune(container, rows, datasets) {
  container.replaceChildren();
  const pretrainedSet = datasets.find((name) => /pretrained/i.test(name));
  const finetunedSet = datasets.find((name) => /finetun|ft-|fine-tun/i.test(name));
  const pretrained = pretrainedSet && rows.find((row) => row.dataset === pretrainedSet);
  const finetuned = finetunedSet && rows.find((row) => row.dataset === finetunedSet);
  const metric = (label, row) => {
    const block = node('div', 'ft-col');
    block.append(node('span', 'label', label));
    if (!row) { block.append(node('p', 'muted', 'Not in the published results.')); return block; }
    for (const [key, title, , digits, unit] of [['psnr', 'PSNR', '', 2, ' dB'], ['ssim', 'SSIM', '', 3, ''], ['lpips', 'LPIPS', '', 3, '']]) {
      const line = node('div', 'ft-row');
      line.append(node('span', '', title), node('b', 'mono', fmt(num(row[key]), digits, unit)));
      block.append(line);
    }
    return block;
  };
  const grid = node('div', 'ft-grid');
  grid.append(metric('Pretrained Zero-DCE++', pretrained), metric('Fine-tuned Zero-DCE++', finetuned));
  container.append(grid);
  let verdict;
  if (pretrained && finetuned) {
    const better = num(finetuned.psnr) > num(pretrained.psnr) && num(finetuned.ssim) >= num(pretrained.ssim);
    verdict = better ? 'Fine-tuning improved the scores on this set.' : 'Fine-tuning didn’t help, so the system keeps the pretrained weights.';
  } else if (pretrained) {
    verdict = 'Only the pretrained run is in the published results, and it is the checkpoint LumenX ships with. The fine-tuned comparison will appear here once its evaluation is added.';
  } else {
    verdict = 'No Zero-DCE++ comparison runs are in the published results yet.';
  }
  container.append(node('p', 'ft-verdict', verdict));
}

function renderLimitations(list, ablationRows) {
  list.replaceChildren();
  const full = ablationRows.find((row) => row.ablation === 'default');
  const noTuner = ablationRows.find((row) => row.ablation === 'no_tuner');
  const searchCost = full && noTuner && num(full.latency_ms) != null && num(noTuner.latency_ms) != null
    ? (num(full.latency_ms) - num(noTuner.latency_ms)) / 1000 : null;
  const items = [
    ['Halos near sharp lamp edges', 'Very bright point lights next to dark areas can keep a faint glow after brightening.'],
    ['Detail loss in very dark, noisy regions', 'Where almost no light was recorded, noise removal smooths texture that cannot be recovered.'],
    ['Adaptive search costs time', searchCost != null && searchCost > 0 ? `Searching for the best setting adds about ${searchCost.toFixed(1)} s per image on this benchmark.` : 'Searching for the best setting adds latency to each hybrid run.'],
    ['Large photos are processed in tiles', 'Big images are restored in overlapping tiles on CPU, which can leave faint brightness seams between tiles.'],
  ];
  for (const [title, body] of items) {
    const item = node('li');
    item.append(node('strong', '', title), node('span', '', body));
    list.append(item);
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i += 1; } else if (char === '"') quoted = false; else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(field); field = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [header, ...body] = rows.filter((line) => line.some((cell) => cell !== ''));
  return header ? body.map((line) => Object.fromEntries(header.map((key, index) => [key, line[index] ?? '']))) : [];
}

function renderFailures(summary, table, count, csvText) {
  summary.replaceChildren();
  table.replaceChildren();
  if (!csvText) { count.textContent = ''; summary.append(node('p', 'muted', 'The failure review file is not available.')); return; }
  const records = parseCsv(csvText);
  count.textContent = `${records.length} flagged runs`;
  const tally = {};
  for (const record of records) {
    for (const flag of String(record.guardrail_failures || '').split(',').map((value) => value.trim()).filter(Boolean)) tally[flag] = (tally[flag] || 0) + 1;
  }
  for (const [flag, total] of Object.entries(tally).sort((a, b) => b[1] - a[1])) {
    const chip = node('span', 'flag-chip');
    chip.append(node('b', 'mono', String(total)), node('span', '', FLAG_NAMES[flag] || flag.replaceAll('_', ' ')));
    summary.append(chip);
  }
  const head = node('tr');
  for (const title of ['Image', 'Dataset', 'Method', 'Variant', 'Flags', 'Latency']) head.append(node('th', '', title));
  table.append(head);
  for (const record of records.slice(0, 12)) {
    const tr = node('tr');
    tr.append(
      node('td', '', record.relative_path),
      node('td', '', record.dataset),
      node('td', '', MODE_NAMES[record.mode] || record.mode),
      node('td', '', record.ablation === 'default' ? 'full' : (ABLATIONS[record.ablation]?.[0] || record.ablation)),
    );
    const flags = node('td', 'flag-cell');
    for (const flag of String(record.guardrail_failures || '').split(',').map((value) => value.trim()).filter(Boolean)) flags.append(node('span', 'flag-chip is-small', FLAG_NAMES[flag] || flag));
    tr.append(flags, node('td', 'mono', record.latency_ms ? `${Number(record.latency_ms).toFixed(0)} ms` : '–'));
    table.append(tr);
  }
  if (records.length > 12) {
    const more = node('tr', 'more-row');
    const cell = node('td', '', `+ ${records.length - 12} more in failure_review.csv`);
    cell.colSpan = 6;
    more.append(cell);
    table.append(more);
  }
}

function renderDatasets(picker, table, rows, datasets) {
  picker.replaceChildren();
  const show = (name) => {
    [...picker.children].forEach((button) => {
      const active = button.dataset.dataset === name;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    table.replaceChildren();
    const subset = rows.filter((row) => row.dataset === name);
    const head = node('tr');
    for (const title of ['Method', 'Variant', 'Images', ...HEADLINE_COLUMNS.map(([, label, direction]) => `${label} ${direction === 'higher' ? '↑' : '↓'}`)]) head.append(node('th', '', title));
    table.append(head);
    const bests = HEADLINE_COLUMNS.map(([key, , direction]) => best(subset.map((row) => num(row[key])), direction));
    for (const row of subset) {
      const tr = node('tr', row.mode === 'hybrid' && row.ablation === 'default' ? 'is-ours' : '');
      tr.append(node('th', '', MODE_NAMES[row.mode] || row.mode), node('td', '', row.ablation === 'default' ? 'full' : (ABLATIONS[row.ablation]?.[0] || row.ablation)), node('td', 'mono', String(row.images ?? '–')));
      HEADLINE_COLUMNS.forEach(([key, , , digits, unit], index) => {
        const value = num(row[key]);
        tr.append(node('td', value != null && value === bests[index] ? 'is-best' : '', fmt(value, digits, unit)));
      });
      table.append(tr);
    }
  };
  for (const name of datasets) {
    const images = rows.find((row) => row.dataset === name)?.images;
    const button = node('button', 'dataset-chip');
    button.type = 'button';
    button.setAttribute('role', 'tab');
    button.dataset.dataset = name;
    button.append(node('span', '', name), node('small', 'mono', `${rows.filter((row) => row.dataset === name).length} runs · ${images ?? '–'} images`));
    button.addEventListener('click', () => show(name));
    picker.append(button);
  }
  if (datasets.length) show(datasets[0]);
}

let tooltipBound = false;
function setupTooltips(root) {
  if (tooltipBound) return;
  tooltipBound = true;
  const tip = document.querySelector('[data-role="chart-tip"]');
  if (!tip) return;
  root.addEventListener('pointermove', (event) => {
    const target = event.target.closest('[data-tip]');
    if (!target) { tip.hidden = true; return; }
    tip.textContent = target.dataset.tip;
    tip.hidden = false;
    tip.style.transform = `translate(${Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8)}px, ${event.clientY + 16}px)`;
  });
  root.addEventListener('pointerleave', () => { tip.hidden = true; });
}
