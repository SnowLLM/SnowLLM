// SPDX-License-Identifier: Apache-2.0
import { t } from './i18n.js';
import { benchmarkCommand } from './commands.js';
import { writeClipboard } from './clipboard.js';
import { createBenchmarkResults } from './benchmark-results.js?v=2';

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
  if ((left.config.workload || 'pg19') !== (right.config.workload || 'pg19')) differences.push('Workload');
  for (const key of ['osl', 'reps', 'warm', 'thinking', 'conc', 'isl']) {
    if (JSON.stringify(left.config[key]) !== JSON.stringify(right.config[key])) differences.push(key);
  }
  if (JSON.stringify(canonical(left.model?.runtime_config)) !== JSON.stringify(canonical(right.model?.runtime_config))) differences.push('Inference settings');
  const corpus = run => run.events.find(event => event.type === 'validated')?.corpus_sha256;
  if (!corpus(left) || corpus(left) !== corpus(right)) differences.push('Corpus');
  const client = run => run.provenance?.client_sha256;
  if (!client(left) || client(left) !== client(right)) differences.push('Benchmark client');
  return differences;
}

export function createBenchmarkUI({ api, post, action, toast, getStatus, renderSettings, refreshStatus }) {
  const $ = id => document.getElementById(id);
  let selected = '', current = null, baseline = null, refreshing = false, revision = 0;
  const renderResults = createBenchmarkResults();
  const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
  const workloadName = config => config.workload === 'file-edit' ? 'File Edit' : 'PG-19';
  const stateLabel = state => t({queued: 'Queued', waiting: 'Waiting for active requests', running: 'Running', stopping: 'Stopping',
    cancelling: 'Cancelling', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted'}[state] || state);
  function request() {
    return {model_id: $('bench-model').value, workload: $('bench-workload').value, isl: parseLengths($('bench-isl').value),
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
  function render() {
    $('bench-detail').hidden = !current;
    renderResults(current, current && baseline);
    if (!current) return;
    $('bench-run-title').textContent = `${current.config.model_id} · ${workloadName(current.config)} · ${stateLabel(current.state)}`;
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
    $('bench-comparison-warning').hidden = !baseline;
    if (baseline && current) {
      const differences = comparisonDifferences(current, baseline);
      $('bench-comparison-warning').textContent = differences.length
        ? t('Configurations differ: {0}. Compare the raw measurements with these differences in mind.', differences.map(key => t(key)).join(', '))
        : t('Recorded configurations and measurement methods match.');
    }
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
      const options = () => runs.map(run => new Option(`${run.config.model_id} · ${workloadName(run.config)} · ${new Date(run.created_at).toLocaleString()} · ${stateLabel(run.state)}`, run.id));
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
  const outputCaps = {pg19: '256', 'file-edit': '1024'};
  let workload = 'pg19';
  $('bench-workload').addEventListener('change', () => {
    outputCaps[workload] = $('bench-osl').value;
    workload = $('bench-workload').value;
    $('bench-osl').value = outputCaps[workload];
  });
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
    await writeClipboard(benchmarkCommand(request(), location.href)); toast(t('CLI command copied'));
  }));
  $('bench-history').addEventListener('change', () => { selected = $('bench-history').value; revision++; refresh().catch(error => toast(error.message)); });
  $('bench-baseline').addEventListener('change', () => { revision++; refresh().catch(error => toast(error.message)); });
  $('bench-refresh').addEventListener('click', () => action($('bench-refresh'), refresh));
  return {update, refresh, select(id) { selected = id; revision++; refresh().catch(error => toast(error.message)); }};
}
