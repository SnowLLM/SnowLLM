// SPDX-License-Identifier: Apache-2.0
import { t } from './i18n.js';
import { benchmarkCommand } from './commands.js';

export function parseLengths(value) {
  return value.split(',').map(item => {
    const match = item.trim().match(/^(\d+)\s*([kKmM]?)$/);
    if (!match) throw new Error(t('Use comma-separated lengths such as 1k,8k,32k.'));
    const count = Number(match[1]) * ({'': 1, k: 1024, m: 1024 ** 2})[match[2].toLowerCase()];
    if (!Number.isSafeInteger(count) || count < 1) throw new Error(t('Lengths and concurrency must be positive integers.'));
    return count;
  });
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}

export function comparisonDifferences(left, right) {
  const differences = [];
  if ([left, right].some(run => run.state !== 'completed')) differences.push('Incomplete run');
  if (left.outdated || right.outdated) differences.push('Outdated result');
  for (const [key, label] of [['model_identity', 'Model identity'], ['draft_identity', 'Draft identity']]) {
    if (JSON.stringify(canonical(left[key])) !== JSON.stringify(canonical(right[key]))) differences.push(label);
  }
  if (!left.model_identity || !right.model_identity) differences.push('Missing model identity');
  for (const key of ['osl', 'reps', 'warm', 'thinking', 'conc', 'isl']) {
    if (JSON.stringify(left.config[key]) !== JSON.stringify(right.config[key])) differences.push(key);
  }
  if (JSON.stringify(canonical(left.model.runtime_config)) !== JSON.stringify(canonical(right.model.runtime_config))) differences.push('Inference settings');
  const corpus = run => run.events.find(event => event.type === 'validated')?.corpus_sha256;
  if (!corpus(left) || corpus(left) !== corpus(right)) differences.push('Corpus');
  const client = run => run.provenance?.client_sha256;
  if (!client(left) || client(left) !== client(right)) differences.push('Benchmark client');
  return differences;
}

export function createBenchmarkUI({ api, post, action, toast, getStatus, renderSettings, refreshStatus }) {
  const $ = id => document.getElementById(id);
  const node = (tag, text = '', className = '') => {
    const element = document.createElement(tag); element.textContent = text; element.className = className; return element;
  };
  let selected = '', current = null, baseline = null, refreshing = false, revision = 0;
  const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
  const stateLabel = state => t({queued: 'Queued', waiting: 'Waiting for active requests', running: 'Running', stopping: 'Stopping',
    cancelling: 'Cancelling', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted'}[state] || state);
  const display = value => Number.isFinite(value) ? value.toFixed(2) : '—';
  function request() {
    return {model_id: $('bench-model').value, isl: parseLengths($('bench-isl').value),
      osl: Number($('bench-osl').value), conc: parseLengths($('bench-conc').value),
      reps: Number($('bench-reps').value), warm: Number($('bench-warm').value), thinking: $('bench-thinking').value};
  }
  function update(status) {
    const models = (status?.models || []).filter(row => ['ready', 'draining', 'loading'].includes(row.state));
    for (const task of status?.tasks || []) {
      if (task.kind !== 'load' || terminal.has(task.state)) continue;
      for (const model of task.models || []) {
        if (model.id && !models.some(row => row.id === model.id)) models.push({id: model.id, state: 'queued'});
      }
    }
    const before = $('bench-model').value;
    const signature = JSON.stringify([models.map(row => [row.id, row.instance, row.state]), t('Waiting for load')]);
    if ($('bench-model').dataset.models !== signature) {
      $('bench-model').replaceChildren(...models.map(row => new Option(row.id + (row.state === 'ready' ? '' : ` · ${t('Waiting for load')}`), row.id)));
      if (models.some(row => row.id === before)) $('bench-model').value = before;
      $('bench-model').dataset.models = signature;
    }
    const model = models.find(row => row.id === $('bench-model').value);
    renderSettings($('bench-runtime'), model?.runtime_config);
    $('bench-fields').disabled = !status;
    $('bench-start').disabled = !model;
    $('bench-start').textContent = t((status?.tasks || []).some(task => task.kind !== 'inference' && !terminal.has(task.state)) ? 'Queue benchmark' : 'Start benchmark');
    $('bench-exclusive').hidden = !status?.benchmark;
    $('bench-no-model').hidden = Boolean(model);
    if (!selected && status?.benchmark) selected = status.benchmark.id;
  }
  function table() {
    const body = $('bench-results'); body.replaceChildren();
    for (const [run, label] of [[current, t('Selected run')], [baseline, t('Comparison run')]]) {
      for (const row of run?.rows || []) {
        const tr = node('tr');
        const al = row.accept_unit === 'AL' ? row.accept_mean : null;
        for (const value of [label, row.isl_target, row.conc, display(row.prefill_tok_s), display(row.decode_tok_s), display(row.ttft_s), display(al)]) {
          tr.append(node('td', String(value)));
        }
        body.append(tr);
      }
    }
    $('bench-comparison-warning').hidden = !baseline;
    if (baseline && current) {
      const differences = comparisonDifferences(current, baseline);
      $('bench-comparison-warning').textContent = differences.length
        ? t('Configurations differ: {0}. Compare the raw measurements with these differences in mind.', differences.map(key => t(key)).join(', '))
        : t('Recorded configurations and measurement methods match.');
    }
    chart();
  }
  function chart() {
    const svg = $('bench-chart'); svg.replaceChildren();
    const metric = $('bench-metric').value;
    const values = [];
    for (const [run, index] of [[current, 0], [baseline, 1]]) {
      for (const row of run?.rows || []) {
        if (Number.isFinite(row[metric])) values.push({x: row.isl_target, y: row[metric], index, conc: row.conc});
      }
    }
    if (!values.length) return;
    const maxX = Math.max(...values.map(point => point.x));
    const maxY = Math.max(1, ...values.map(point => point.y));
    const add = (tag, attrs, text) => {
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      if (text !== undefined) el.textContent = text;
      svg.append(el); return el;
    };
    add('path', {d: 'M65 15 V200 H700', class: 'bench-axis'});
    add('text', {x: 5, y: 25}, display(maxY));
    add('text', {x: 65, y: 223}, '0');
    add('text', {x: 580, y: 223}, `${maxX} tokens`);
    for (const key of new Set(values.map(point => `${point.index}:${point.conc}`))) {
      const points = values.filter(point => `${point.index}:${point.conc}` === key).sort((a, b) => a.x - b.x);
      const position = point => `${65 + point.x / maxX * 620},${200 - point.y / maxY * 180}`;
      const className = `bench-series bench-series-${points[0].index}`;
      add('polyline', {points: points.map(position).join(' '), class: className});
      for (const point of points) {
        const [cx, cy] = position(point).split(',');
        const dot = add('circle', {cx, cy, r: 4, class: className});
        const title = document.createElementNS(svg.namespaceURI, 'title');
        title.textContent = `${point.x} tokens · concurrency ${point.conc} · ${display(point.y)}`; dot.append(title);
      }
    }
  }
  function render() {
    $('bench-detail').hidden = !current;
    if (!current) return;
    $('bench-run-title').textContent = `${current.config.model_id} · ${stateLabel(current.state)}`;
    $('bench-run-id').textContent = `${current.id} · ${new Date(current.created_at).toLocaleString()}`;
    $('bench-cancel').disabled = terminal.has(current.state) || ['cancelling', 'stopping'].includes(current.state);
    $('bench-export').href = `/daemon/benchmarks/${current.id}/export`;
    $('bench-export-rows').href = `/daemon/benchmarks/${current.id}/export?results_only=true`;
    const progress = current.progress;
    $('bench-progress').max = progress?.total || 1;
    $('bench-progress').value = progress?.completed || 0;
    $('bench-progress-text').textContent = progress
      ? t('{0}/{1} requests · input {2} · concurrency {3} · {4} {5}', progress.completed, progress.total,
        progress.isl, progress.conc, progress.warm ? t('Warm-up') : t('Measurement'), progress.repetition || '')
      : stateLabel(current.state);
    const start = current.started_at || current.created_at;
    $('bench-elapsed').textContent = t('Elapsed: {0} s', Math.max(0, Math.floor((Date.parse(current.finished_at || new Date().toISOString()) - Date.parse(start)) / 1000)));
    $('bench-run-error').hidden = !current.error && !current.outdated;
    $('bench-run-error').textContent = current.error || (current.outdated ? t('Outdated result') : '');
    $('bench-output').textContent = current.log || '';
    $('bench-snapshot').textContent = JSON.stringify({config: current.config, model: current.model, provenance: current.provenance,
      model_identity: current.model_identity, draft_identity: current.draft_identity}, null, 2);
    table();
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    const version = revision;
    try {
      const {runs} = await api('/daemon/benchmarks');
      if (version !== revision) return;
      if (!selected && runs.length) selected = runs[0].id;
      const compareId = $('bench-baseline').value;
      const options = () => runs.map(run => new Option(`${run.config.model_id} · ${new Date(run.created_at).toLocaleString()} · ${stateLabel(run.state)}`, run.id));
      $('bench-history').replaceChildren(...options()); $('bench-history').value = selected;
      $('bench-baseline').replaceChildren(new Option(t('No comparison'), ''), ...options());
      if (runs.some(run => run.id === compareId)) $('bench-baseline').value = compareId;
      const [detail, comparison] = await Promise.all([selected ? api('/daemon/benchmarks/' + selected) : null,
        $('bench-baseline').value ? api('/daemon/benchmarks/' + $('bench-baseline').value) : null]);
      if (version !== revision) return;
      current = detail; baseline = comparison; render();
    } finally { refreshing = false; }
  }
  $('bench-model').addEventListener('change', () => update(getStatus()));
  $('bench-form').addEventListener('submit', event => {
    event.preventDefault(); action($('bench-start'), async () => {
      const result = await post('/daemon/benchmarks', request());
      selected = result.id; revision++; await refresh();
      await refreshStatus();
    });
  });
  $('bench-cancel').addEventListener('click', () => action($('bench-cancel'), async () => {
    await post(`/daemon/benchmarks/${selected}/cancel`); await refresh(); await refreshStatus();
  }));
  $('bench-copy').addEventListener('click', () => action($('bench-copy'), async () => {
    await navigator.clipboard.writeText(benchmarkCommand(request(), location.href)); toast(t('CLI command copied'));
  }));
  $('bench-history').addEventListener('change', () => { selected = $('bench-history').value; revision++; refresh().catch(error => toast(error.message)); });
  $('bench-baseline').addEventListener('change', () => { revision++; refresh().catch(error => toast(error.message)); });
  $('bench-metric').addEventListener('change', chart);
  $('bench-refresh').addEventListener('click', () => action($('bench-refresh'), refresh));
  return {update, refresh, select(id) { selected = id; revision++; refresh().catch(error => toast(error.message)); }};
}
