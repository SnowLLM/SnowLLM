// SPDX-License-Identifier: Apache-2.0
import { t } from './i18n.js';

const phases = {
  decode: [
    {key: 'decode_tok_s', label: 'decode tok/s', unit: 'tok/s', digits: 2, note: 'Output tokens per second.'},
    {key: 'step_hz', label: 'step/s', unit: 'step/s', digits: 2, note: 'Verification steps per second.'},
    {key: 'step_ms', label: 'ms/step', unit: 'ms', digits: 2, note: 'Milliseconds per verification step.'},
    {key: 'al', label: 'AL', unit: 'tok/step', digits: 3, note: 'Tokens emitted per step, including the correction token.'},
  ],
  prefill: [
    {key: 'prefill_tok_s', label: 'prefill tok/s', unit: 'tok/s', digits: 1, note: 'Prompt tokens divided by client-side TTFT.'},
    {key: 'ttft_s', label: 'TTFT s', unit: 's', digits: 3, note: 'Client-side time to first output.', log: true},
  ],
};
const metrics = Object.values(phases).flat();
const palettes = [
  ['#5572d4', '#1f6feb', '#12a594', '#7c5cd6', '#0f766e', '#3f8f4f', '#2563eb', '#0891b2', '#6d28d9', '#047857'],
  ['#e07a30', '#c2410c', '#b45309', '#9a3412', '#d97706'],
];
const positive = value => Number.isFinite(value) && value > 0;
const number = (value, metric) => Number.isFinite(value) ? value.toFixed(metric.digits) : '—';
const contextLabel = value => value >= 1024 ? `${Number((value / 1024).toPrecision(5))}K` : String(value);

export function resultMetrics(row) {
  const al = Number.isFinite(row.accept_length_mean) ? row.accept_length_mean
    : row.accept_unit === 'AL' && Number.isFinite(row.accept_mean) ? row.accept_mean : null;
  // Older records can derive step time from known AL and decode throughput.
  const seconds = positive(row.step_s) ? row.step_s : positive(al) && positive(row.decode_tok_s) ? al / row.decode_tok_s : null;
  return {decode_tok_s: row.decode_tok_s, step_hz: seconds ? 1 / seconds : null, step_ms: seconds ? seconds * 1000 : null,
    al, prefill_tok_s: row.prefill_tok_s, ttft_s: row.ttft_s};
}

function axisTicks(top, step) {
  return Array.from({length: Math.round(top / step) + 1}, (_, index) => index * step);
}

function valueScale(values, logarithmic) {
  const min = Math.min(...values), max = Math.max(...values);
  if (logarithmic && min > 0 && max / min > 20) {
    const low = Math.floor(Math.log10(min)), high = Math.ceil(Math.log10(max));
    return {ticks: Array.from({length: high - low + 1}, (_, index) => 10 ** (low + index)),
      fraction: value => (Math.log10(value) - low) / (high - low), log: true};
  }
  const raw = max / 4 || 1;
  const order = 10 ** Math.floor(Math.log10(raw));
  const ratio = raw / order;
  const step = (ratio < 1.5 ? 1 : ratio < 3 ? 2 : ratio < 7 ? 5 : 10) * order;
  const top = Math.max(step, Math.ceil(max / step) * step);
  return {ticks: axisTicks(top, step), fraction: value => value / top};
}

export function createBenchmarkResults() {
  const $ = id => document.getElementById(id);
  const node = (tag, text = '', className = '') => {
    const element = document.createElement(tag);
    element.textContent = text;
    element.className = className;
    return element;
  };
  const svg = $('bench-chart'), wrap = svg.parentElement, tooltip = $('bench-chart-tooltip');
  const phaseOptions = $('bench-phase'), metricOptions = $('bench-metric'), viewOptions = $('bench-view');
  const metricTable = $('bench-metric-table'), chartPanel = $('bench-chart-panel');
  const preferences = {decode: phases.decode[0].key, prefill: phases.prefill[0].key};
  let phase = 'decode', view = 'chart', signature, series = [], width = 0;
  let rungs = new Map(), guide = null, activeRung = null;
  const selectedMetric = () => phases[phase].find(metric => metric.key === preferences[phase]);
  const setPressed = (options, selected) => {
    for (const button of options.children) button.setAttribute('aria-pressed', String(button.dataset.value === selected));
  };
  const swatch = color => {
    const mark = node('i', '', 'bench-swatch');
    mark.style.backgroundColor = color;
    return mark;
  };
  function svgElement(tag, attributes, content) {
    const element = document.createElementNS(svg.namespaceURI, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (content !== undefined) element.textContent = content;
    svg.append(element);
    return element;
  }
  function makeMetricButtons() {
    for (const [section, options] of Object.entries(phases)) {
      for (const metric of options) {
        const button = node('button', t(metric.label));
        button.type = 'button';
        Object.assign(button.dataset, {value: metric.key, i18n: metric.label, phase: section});
        metricOptions.append(button);
      }
    }
  }
  function drawTable(drawable, contexts, metric) {
    const table = node('table', '', 'bench-table');
    const head = table.createTHead().insertRow();
    for (const text of [t('Series'), ...contexts.map(tokens => `${contextLabel(tokens)} ${metric.unit}`)]) {
      const cell = node('th', text);
      cell.scope = 'col';
      head.append(cell);
    }
    const ranges = new Map(contexts.map(tokens => {
      const values = rungs.get(tokens).map(point => point.value);
      return [tokens, [Math.min(...values), Math.max(...values)]];
    }));
    const body = table.createTBody();
    for (const line of drawable) {
      const row = body.insertRow();
      const header = node('th');
      header.scope = 'row';
      header.append(swatch(line.color), document.createTextNode(line.label));
      row.append(header);
      for (const tokens of contexts) {
        const value = line.values.get(tokens)?.value;
        const cell = row.insertCell();
        cell.textContent = number(value, metric);
        const [low, high] = ranges.get(tokens);
        if (Number.isFinite(value) && high > low) {
          const intensity = (value - low) / (high - low);
          cell.style.backgroundColor = `color-mix(in oklab, #3957c4 ${intensity * 72}%, transparent)`;
          if (intensity > 0.55) cell.style.color = '#172550';
        }
      }
    }
    metricTable.replaceChildren(table);
  }
  function showRung(hit) {
    if (hit === activeRung) return;
    activeRung = hit;
    tooltip.hidden = !hit;
    if (!hit) { guide?.setAttribute('visibility', 'hidden'); return; }
    const {tokens, x} = hit.dataset;
    const metric = selectedMetric();
    guide.setAttribute('x1', x);
    guide.setAttribute('x2', x);
    guide.setAttribute('visibility', 'visible');
    const rows = rungs.get(Number(tokens)).map(point => {
      const row = node('div', '', 'bench-tooltip-row');
      row.append(swatch(point.color), node('span', point.label), node('b', `${number(point.value, metric)} ${metric.unit}`));
      return row;
    });
    tooltip.replaceChildren(node('strong', `${t('Target input (tokens)')} ${tokens}`), ...rows);
    const ruler = guide.getBoundingClientRect().left - wrap.getBoundingClientRect().left;
    const right = ruler + 10;
    const left = right + tooltip.offsetWidth <= wrap.clientWidth ? right : ruler - 10 - tooltip.offsetWidth;
    tooltip.style.left = `${Math.max(0, left)}px`;
  }
  function drawChart(drawable, contexts, metric) {
    width = wrap.clientWidth || 720;
    const left = 50, right = width - 12, top = 12, bottom = 274;
    const step = Math.max(1, 2 ** Math.ceil(Math.log2(contexts.at(-1) / 5)));
    const maxTokens = Math.ceil(contexts.at(-1) / step) * step;
    const scale = valueScale([...rungs.values()].flatMap(points => points.map(point => point.value)), metric.log);
    const x = tokens => left + tokens / maxTokens * (right - left);
    const y = value => bottom - scale.fraction(value) * (bottom - top);
    svg.setAttribute('viewBox', `0 0 ${width} 300`);
    svg.setAttribute('aria-label', `${t(metric.label)} · ${t('Target input (tokens)')}`);
    if (scale.log) $('bench-metric-note').textContent += ' ' + t('Logarithmic Y scale.');
    for (const value of scale.ticks) {
      const position = y(value);
      svgElement('line', {x1: left, x2: right, y1: position, y2: position, class: 'bench-grid'});
      const label = value >= 1000 ? `${Number((value / 1000).toPrecision(4))}k` : String(Number(value.toPrecision(4)));
      svgElement('text', {x: left - 8, y: position + 4, 'text-anchor': 'end'}, label);
    }
    svgElement('line', {x1: left, x2: right, y1: bottom, y2: bottom, class: 'bench-axis'});
    for (const tokens of axisTicks(maxTokens, step)) {
      svgElement('text', {x: x(tokens), y: bottom + 18,
        'text-anchor': tokens === 0 ? 'start' : tokens === maxTokens ? 'end' : 'middle'}, contextLabel(tokens));
    }
    const legend = $('bench-legend');
    legend.replaceChildren();
    for (const line of drawable) {
      const key = node('span');
      key.append(swatch(line.color), document.createTextNode(line.label));
      legend.append(key);
      let drawing = false;
      let path = '';
      for (const tokens of contexts) {
        const point = line.values.get(tokens);
        if (!point) { drawing = false; continue; }
        path += `${drawing ? 'L' : 'M'}${x(tokens)},${y(point.value)} `;
        drawing = true;
      }
      svgElement('path', {d: path, class: 'bench-series', stroke: line.color});
      for (const [tokens, point] of line.values) {
        svgElement('circle', {cx: x(tokens), cy: y(point.value), r: 3, fill: line.color});
      }
    }
    const last = contexts.at(-1);
    const endpoints = [...rungs.get(last)].sort((a, b) => b.value - a.value);
    if (endpoints.length <= 4 && width >= 360) {
      let previous = top - 2;
      for (const [index, point] of endpoints.entries()) {
        const labelY = Math.min(bottom - 4 - (endpoints.length - 1 - index) * 14, Math.max(previous + 14, y(point.value) - 9));
        svgElement('text', {x: x(last) - 7, y: labelY, 'text-anchor': 'end', fill: point.color, class: 'bench-endpoint'},
          `${number(point.value, metric)} ${metric.unit} @ ${contextLabel(last)}`);
        previous = labelY;
      }
    }
    guide = svgElement('line', {y1: top, y2: bottom, class: 'bench-guide', visibility: 'hidden'});
    for (const [index, tokens] of contexts.entries()) {
      const center = x(tokens);
      const start = index === 0 ? left : (x(contexts[index - 1]) + center) / 2;
      const end = index === contexts.length - 1 ? right : (center + x(contexts[index + 1])) / 2;
      const labels = rungs.get(tokens).map(point => `${point.label} ${number(point.value, metric)} ${metric.unit}`);
      const hit = svgElement('rect', {x: start, y: top, width: end - start, height: bottom - top,
        class: 'bench-rung', tabindex: '0', role: 'img', 'aria-label': `${t('Target input (tokens)')} ${tokens} · ${labels.join(' · ')}`});
      Object.assign(hit.dataset, {tokens, x: center});
    }
  }
  function renderView() {
    const metric = selectedMetric();
    setPressed(phaseOptions, phase);
    setPressed(metricOptions, metric.key);
    setPressed(viewOptions, view);
    for (const button of metricOptions.children) button.hidden = button.dataset.phase !== phase;
    $('bench-metric-title').textContent = t(metric.label);
    $('bench-metric-note').textContent = t(metric.note);
    showRung(null);
    svg.replaceChildren();
    guide = null;
    rungs = new Map();
    const drawable = [];
    for (const line of series) {
      const values = new Map();
      for (const row of line.rows) {
        const value = row[metric.key];
        if (!positive(row.isl_target) || !Number.isFinite(value) || value < 0) continue;
        const point = {value, label: line.label, color: line.color};
        values.set(row.isl_target, point);
        if (!rungs.has(row.isl_target)) rungs.set(row.isl_target, []);
        rungs.get(row.isl_target).push(point);
      }
      if (values.size) drawable.push({...line, values});
    }
    const contexts = [...rungs.keys()].sort((a, b) => a - b);
    $('bench-empty').hidden = Boolean(contexts.length);
    chartPanel.hidden = !contexts.length || view !== 'chart';
    metricTable.hidden = !contexts.length || view !== 'table';
    if (!chartPanel.hidden) drawChart(drawable, contexts, metric);
    if (!metricTable.hidden) drawTable(drawable, contexts, metric);
  }
  function render(current, baseline) {
    const runs = [current, baseline];
    const rows = runs.map(run => (run?.rows || []).map(row => ({
      isl_target: row.isl_target, isl: row.isl, conc: row.conc, ...resultMetrics(row),
    })));
    const lang = document.documentElement.lang;
    const key = JSON.stringify([lang, runs.map(run => run?.id), rows]);
    if (key === signature) return;
    signature = key;
    const labels = [t('Selected run'), t('Comparison run')];
    const body = $('bench-results');
    body.replaceChildren();
    series = [];
    for (const [index, measured] of rows.entries()) {
      const label = labels[index];
      const concurrency = new Map();
      for (const row of measured) {
        if (!concurrency.has(row.conc)) concurrency.set(row.conc, []);
        concurrency.get(row.conc).push(row);
        const entry = body.insertRow();
        for (const value of [label, row.isl_target ?? '—', row.isl ?? '—', row.conc ?? '—',
          ...metrics.map(metric => number(row[metric.key], metric))]) entry.insertCell().textContent = value;
      }
      const palette = palettes[index];
      for (const [order, [conc, group]] of [...concurrency].sort(([a], [b]) => a - b).entries()) {
        series.push({label: `${label} · ${t('Concurrency')} ${conc ?? '—'}`, color: palette[order % palette.length], rows: group});
      }
    }
    renderView();
  }
  makeMetricButtons();
  for (const [options, update] of [[phaseOptions, value => { phase = value; }],
    [metricOptions, value => { preferences[phase] = value; }], [viewOptions, value => { view = value; }]]) {
    options.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button || button.getAttribute('aria-pressed') === 'true') return;
      update(button.dataset.value);
      renderView();
    });
  }
  const rung = element => element?.closest?.('.bench-rung') || null;
  svg.addEventListener('pointermove', event => {
    if (event.pointerType !== 'touch') showRung(rung(event.target) || rung(document.activeElement));
  });
  svg.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'touch') showRung(rung(document.activeElement));
  });
  svg.addEventListener('click', event => showRung(rung(event.target)));
  svg.addEventListener('focusin', event => showRung(rung(event.target)));
  svg.addEventListener('focusout', event => { if (!rung(event.relatedTarget)) showRung(null); });
  document.addEventListener('pointerdown', event => { if (!svg.contains(event.target)) showRung(null); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') showRung(null); });
  new ResizeObserver(() => {
    if (!chartPanel.hidden && wrap.clientWidth && wrap.clientWidth !== width) renderView();
  }).observe(wrap);
  return render;
}
