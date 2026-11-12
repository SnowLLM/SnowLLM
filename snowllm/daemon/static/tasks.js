// SPDX-License-Identifier: Apache-2.0
import { t } from './i18n.js';

const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted']);
const labels = {load: 'Load model', unload: 'Unload model', benchmark: 'Benchmark', inference: 'Inference request'};
const states = {queued: 'Queued', waiting: 'Waiting', running: 'Running', cancelling: 'Cancelling',
  stopping: 'Stopping', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted'};

export function createTaskUI({ post, action, refreshStatus, onBenchmark }) {
  const $ = id => document.getElementById(id);
  const nodes = new Map();
  const element = (tag, className, text = '') => {
    const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
  };
  function taskNode(task, tasks, online) {
    let entry = nodes.get(task.id);
    if (!entry) {
      const node = element('article', 'task-row');
      const content = element('div', 'task-content');
      const title = element('div', 'task-title');
      const kind = element('strong', '');
      const state = element('span', 'pill');
      const model = element('p', 'task-model');
      const detail = element('p', 'task-detail');
      const tools = element('div', 'task-tools');
      const cancel = element('button', 'button secondary');
      cancel.type = 'button';
      cancel.addEventListener('click', () => action(cancel, async () => {
        await post(`/daemon/tasks/${task.id}/cancel`); await refreshStatus();
      }));
      const link = element('button', 'button secondary');
      link.type = 'button';
      link.addEventListener('click', () => onBenchmark(task.id));
      title.append(kind, state); content.append(title, model, detail);
      tools.append(link, cancel); node.append(content, tools);
      entry = {node, kind, state, model, detail, cancel, link}; nodes.set(task.id, entry);
    }
    entry.kind.textContent = t(labels[task.kind] || task.kind);
    entry.state.textContent = t(states[task.state] || task.state);
    entry.state.className = `pill ${task.state}`;
    const target = task.targets?.length ? task.targets.map(target => target.toUpperCase()).join(' + ')
      : task.target === 'all' ? t('All devices') : task.target?.toUpperCase();
    const models = task.models?.map(model => model.id || model.model).join(', ') || task.model || t('All models');
    entry.model.textContent = `${models}${target ? ` · ${target}` : ''}`;
    let reason = '';
    if (task.wait_reason === 'previous_task') {
      const previous = tasks.find(row => row.id === task.waiting_for);
      reason = previous ? t('Waiting for {0}', `${t(labels[previous.kind] || previous.kind)} · ${previous.model || t('All models')}`) : t('Waiting to start');
    } else if (task.wait_reason === 'active_requests') reason = t('Waiting for active requests to finish');
    else if (task.state === 'cancelling') reason = t('Cancelling safely and releasing resources');
    else if (task.state === 'stopping') reason = t('Releasing resources');
    const elapsed = Math.max(0, Math.floor((Date.parse(task.finished_at || new Date().toISOString()) - Date.parse(task.created_at)) / 1000));
    entry.detail.textContent = [task.error || reason, t('Elapsed: {0} s', elapsed), task.id.slice(0, 8)].filter(Boolean).join(' · ');
    entry.detail.classList.toggle('task-error', task.state === 'failed');
    entry.cancel.textContent = t('Cancel task');
    entry.cancel.setAttribute('aria-label', t('Cancel {0}', `${t(labels[task.kind] || task.kind)} ${models}`));
    entry.cancel.hidden = terminal.has(task.state) || !task.cancellable;
    entry.cancel.disabled = !online || !task.cancellable || task.cancel_requested;
    entry.link.textContent = t('View benchmark');
    entry.link.hidden = task.kind !== 'benchmark';
    return entry.node;
  }
  function reconcile(container, children) {
    [...container.children].forEach(node => { if (!children.includes(node)) node.remove(); });
    children.forEach((node, index) => { if (container.children[index] !== node) container.insertBefore(node, container.children[index] || null); });
  }
  function update(status, online = true) {
    const tasks = status?.tasks || [];
    const current = tasks.filter(task => !terminal.has(task.state));
    const inference = current.filter(task => task.kind === 'inference');
    const management = current.filter(task => task.kind !== 'inference');
    const recent = tasks.filter(task => terminal.has(task.state))
      .sort((left, right) => Date.parse(right.finished_at) - Date.parse(left.finished_at)).slice(0, 20);
    $('tasks-nav-count').textContent = String(current.length);
    $('tasks-nav-count').hidden = !current.length;
    $('tasks-count').textContent = t('{0} active · {1} queued', current.filter(task => task.state !== 'queued').length, management.filter(task => task.state === 'queued').length);
    $('tasks-empty').hidden = current.length > 0;
    $('tasks-inference').hidden = !inference.length;
    const engines = (status?.models || []).map(model => model.engine).filter(Boolean);
    $('tasks-inference-summary').textContent = t('{0} active requests · {1} running · {2} queued', inference.length,
      engines.reduce((total, engine) => total + (engine.running || 0), 0), engines.reduce((total, engine) => total + (engine.waiting || 0), 0));
    reconcile($('tasks-active'), management.map(task => taskNode(task, tasks, online)));
    reconcile($('tasks-requests'), inference.map(task => taskNode(task, tasks, online)));
    reconcile($('tasks-history'), recent.map(task => taskNode(task, tasks, online)));
    $('tasks-recent').hidden = !recent.length;
    $('tasks-history-count').textContent = String(recent.length);
    const visible = new Set([...current, ...recent].map(task => task.id));
    for (const key of nodes.keys()) if (!visible.has(key)) nodes.delete(key);
  }
  return {update};
}
