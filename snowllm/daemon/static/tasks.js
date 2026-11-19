// SPDX-License-Identifier: Apache-2.0
import { t } from './i18n.js';

const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const labels = {load: 'Load model', unload: 'Unload model', benchmark: 'Benchmark', inference: 'Inference request'};
const states = {queued: 'Queued', waiting: 'Waiting', running: 'Running', cancelling: 'Cancelling',
  stopping: 'Stopping', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted'};

export function createTaskUI({ api, post, action, refreshStatus, onBenchmark }) {
  const $ = id => document.getElementById(id);
  const nodes = new Map();
  let taskById = new Map();
  let overviewTask = null, overviewOnline = false, benchmarkProgress = null;
  let selectedId = null, selectedGroup = null, focusSelected = false;
  const element = (tag, className, text = '') => {
    const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
  };
  const kindLabel = task => t(labels[task.kind] || task.kind);
  const modelNames = task => task.models?.length
    ? task.models.map(model => model.id || model.model) : [task.model || t('All models')];
  function summaryModel(task) {
    const models = modelNames(task);
    return models.length > 1 ? t('{0} + {1} more', models[0], models.length - 1) : models[0];
  }
  const taskHref = task => `#tasks/${task.id}`;
  function reasonFor(task) {
    if (task.wait_reason === 'previous_task') {
      const previous = taskById.get(task.waiting_for);
      return previous ? t('Waiting for {0}', `${kindLabel(previous)} · ${summaryModel(previous)}`) : t('Waiting to start');
    }
    if (task.wait_reason === 'active_requests') return t('Waiting for active requests to finish');
    if (task.state === 'cancelling') return t('Cancelling safely and releasing resources');
    if (task.state === 'stopping') return t('Releasing resources');
    if (task.state === 'running' && task.kind === 'load') return t('Reading weights and initializing');
    return '';
  }
  function elapsedFor(task) {
    const started = Date.parse(task.started_at || task.created_at);
    const finished = task.finished_at ? Date.parse(task.finished_at) : Date.now();
    const elapsed = Math.max(0, Math.floor((finished - started) / 1000));
    return Number.isFinite(elapsed) ? t(task.state === 'queued' ? 'Queued for {0} s' : 'Elapsed: {0} s', elapsed) : '';
  }
  const previewRows = Array.from({length: 3}, () => {
    const row = element('li', 'workspace-queued-task');
    const link = element('a', '');
    const model = element('span', '');
    link.append(element('strong', ''), model);
    row.append(link);
    $('overview-queued-tasks').append(row);
    return {row, link, kind: link.firstElementChild, model};
  });
  function renderCurrentDetail() {
    const detail = $('overview-current-detail');
    const meter = $('overview-task-progress');
    const progress = overviewTask?.state === 'running' && benchmarkProgress?.id === overviewTask.id
      ? benchmarkProgress.value : null;
    const measured = progress?.total > 0 && Number.isFinite(progress.completed);
    detail.textContent = measured ? t('{0}/{1} requests completed', progress.completed, progress.total)
      : overviewTask ? reasonFor(overviewTask) || elapsedFor(overviewTask) : '';
    detail.hidden = !detail.textContent;
    meter.hidden = !measured;
    if (measured) {
      meter.max = progress.total;
      meter.value = progress.completed;
    }
  }
  function renderOverview(active, queued, online) {
    overviewTask = active[0] || null;
    overviewOnline = online;
    $('overview-tasks-empty').hidden = Boolean(active.length || queued.length);
    $('overview-current-task').hidden = !overviewTask;
    $('overview-queue').hidden = !queued.length;
    if (overviewTask) {
      const link = $('overview-current-link');
      link.href = taskHref(overviewTask);
      link.textContent = `${kindLabel(overviewTask)} · ${summaryModel(overviewTask)}`;
      const state = $('overview-current-state');
      state.className = `pill ${overviewTask.state}`;
      state.textContent = t(states[overviewTask.state] || overviewTask.state);
    }
    renderCurrentDetail();
    previewRows.forEach(({row, link, kind, model}, index) => {
      const task = queued[index];
      row.hidden = !task;
      if (!task) return;
      row.value = task.position ?? index + 1;
      link.href = taskHref(task);
      kind.textContent = kindLabel(task);
      model.textContent = summaryModel(task);
    });
  }
  async function refreshProgress() {
    const task = overviewTask;
    if (!overviewOnline || task?.kind !== 'benchmark' || task.state !== 'running') return;
    try {
      const result = await api(`/daemon/benchmarks/${task.id}`);
      if (overviewTask?.id !== task.id || overviewTask.state !== 'running') return;
      benchmarkProgress = {id: task.id, value: result.progress};
      renderCurrentDetail();
    } catch { /* The task status remains available if its progress cannot be read. */ }
  }
  function taskNode(task, online) {
    let entry = nodes.get(task.id);
    if (!entry) {
      const node = element('article', 'task-row');
      const line = element('div', 'task-line');
      const position = element('span', 'task-position');
      const kind = element('strong', 'task-kind');
      const model = element('span', 'task-model');
      const state = element('span', 'pill');
      const detail = element('div', 'task-detail');
      detail.id = `task-details-${task.id}`;
      detail.hidden = true;
      const context = element('p', 'task-context');
      const metadata = element('p', 'task-metadata');
      detail.append(context, metadata);
      const tools = element('div', 'task-tools');
      const cancel = element('button', 'button secondary task-cancel');
      cancel.type = 'button';
      const link = element('button', 'button secondary');
      link.type = 'button';
      link.addEventListener('click', () => onBenchmark(task.id));
      const toggle = element('button', 'button secondary');
      toggle.type = 'button';
      toggle.setAttribute('aria-controls', detail.id);
      toggle.setAttribute('aria-expanded', 'false');
      toggle.addEventListener('click', () => {
        detail.hidden = !detail.hidden;
        toggle.setAttribute('aria-expanded', String(!detail.hidden));
      });
      cancel.addEventListener('click', () => {
        if (document.activeElement === cancel) toggle.focus({preventScroll: true});
        action(cancel, async () => {
          await post(`/daemon/tasks/${task.id}/cancel`);
          await refreshStatus();
        });
      });
      tools.append(link, cancel, toggle);
      line.append(position, kind, model, state, tools);
      node.append(line, detail);
      entry = {node, position, kind, state, model, detail, context, metadata, cancel, link, toggle};
      nodes.set(task.id, entry);
    }
    const position = task.state === 'queued' ? task.position : null;
    entry.position.textContent = position == null ? '' : `#${position}`;
    if (position == null) entry.position.removeAttribute('aria-label');
    else entry.position.setAttribute('aria-label', t('Queue position {0}', position));
    entry.kind.textContent = kindLabel(task);
    entry.state.textContent = t(states[task.state] || task.state);
    entry.state.className = `pill ${task.state}`;
    const target = task.targets?.length ? task.targets.map(target => target.toUpperCase()).join(' + ')
      : task.target === 'all' ? t('All devices') : task.target?.toUpperCase();
    const models = modelNames(task).join(', ');
    entry.model.textContent = `${models}${target ? ` · ${target}` : ''}`;
    entry.model.title = entry.model.textContent;
    entry.context.textContent = task.error || reasonFor(task);
    entry.context.hidden = !entry.context.textContent;
    entry.context.classList.toggle('task-error', task.state === 'failed');
    entry.metadata.textContent = [entry.model.textContent, elapsedFor(task), t('Task ID {0}', task.id)].filter(Boolean).join(' · ');
    entry.node.classList.toggle('task-selected', task.id === selectedId);
    entry.cancel.textContent = t('Cancel task');
    entry.cancel.setAttribute('aria-label', t('Cancel {0}', `${kindLabel(task)} ${models}`));
    entry.cancel.hidden = terminal.has(task.state) || !task.cancellable;
    entry.cancel.disabled = !online || !task.cancellable || task.cancel_requested;
    entry.link.textContent = t('View benchmark');
    entry.link.hidden = task.kind !== 'benchmark';
    entry.toggle.textContent = t('Details');
    entry.toggle.setAttribute('aria-label', t('Details for {0}', `${kindLabel(task)} ${models}`));
    return entry.node;
  }
  function reconcile(container, children) {
    const kept = new Set(children);
    [...container.children].forEach(node => { if (!kept.has(node)) node.remove(); });
    children.forEach((node, index) => { if (container.children[index] !== node) container.insertBefore(node, container.children[index] || null); });
  }
  function select(id) {
    if (id === selectedId) return false;
    selectedId = id;
    selectedGroup = null;
    focusSelected = Boolean(id);
    return true;
  }
  function update(status, online, page) {
    const tasks = status?.tasks || [];
    const current = tasks.filter(task => !terminal.has(task.state));
    const inference = current.filter(task => task.kind === 'inference');
    const management = current.filter(task => task.kind !== 'inference');
    const active = management.filter(task => task.state !== 'queued');
    const queued = management.filter(task => task.state === 'queued');
    taskById = new Map(tasks.map(task => [task.id, task]));
    const engines = (status?.models || []).map(model => model.engine).filter(Boolean);
    const running = engines.reduce((total, engine) => total + (engine.running || 0), 0);
    const waiting = engines.reduce((total, engine) => total + (engine.waiting || 0), 0);
    const count = t('{0} active · {1} queued', active.length, queued.length);
    $('nav-task-count').textContent = String(management.length);
    $('nav-task-count').hidden = !management.length;
    $('overview-task-count').textContent = count;
    $('overview-task-count').hidden = !management.length;
    $('active-count').textContent = engines.length || status?.active_requests === 0
      ? t('{0} running · {1} waiting', running, waiting) : status?.active_requests ?? '—';
    renderOverview(active, queued, online);
    if (page !== 'tasks') return;

    const recent = tasks.filter(task => terminal.has(task.state))
      .sort((left, right) => Date.parse(right.finished_at) - Date.parse(left.finished_at)).slice(0, 20);
    const selected = taskById.get(selectedId);
    if (selected && terminal.has(selected.state) && !recent.includes(selected)) recent.push(selected);
    $('task-missing').hidden = !online || !selectedId || Boolean(selected);
    $('tasks-count').textContent = count;
    $('tasks-count').hidden = !management.length;
    $('tasks-empty').hidden = management.length > 0;
    $('tasks-inference').hidden = !inference.length;
    $('tasks-inference-summary').textContent = t('{0} active requests · {1} running · {2} queued', inference.length, running, waiting);
    const selectedHadFocus = nodes.get(selectedId)?.node.contains(document.activeElement);
    reconcile($('tasks-active'), [...active, ...queued].map(task => taskNode(task, online)));
    reconcile($('tasks-requests'), inference.map(task => taskNode(task, online)));
    reconcile($('tasks-history'), recent.map(task => taskNode(task, online)));
    $('tasks-recent').hidden = !recent.length;
    $('tasks-history-count').textContent = String(recent.length);
    const visible = new Set([...current, ...recent].map(task => task.id));
    for (const key of nodes.keys()) if (!visible.has(key)) nodes.delete(key);
    const entry = nodes.get(selectedId);
    if (!entry) return;
    if (selectedGroup !== entry.node.parentElement) {
      selectedGroup = entry.node.parentElement;
      const group = entry.node.closest('.task-group');
      if (group) group.open = true;
    }
    if (focusSelected) {
      focusSelected = false;
      entry.detail.hidden = false;
      entry.toggle.setAttribute('aria-expanded', 'true');
      entry.node.scrollIntoView({block: 'nearest'});
      entry.toggle.focus({preventScroll: true});
    } else if (selectedHadFocus && !entry.node.contains(document.activeElement)) {
      entry.toggle.focus();
    }
  }
  return {select, update, refreshProgress};
}
