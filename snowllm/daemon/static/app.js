// SPDX-License-Identifier: Apache-2.0
import { t, translatePage, language, setLanguage } from './i18n.js';
import { parseOptions, readEvents } from './stream.js?v=2';
import { thinkingProfile, thinkingParameters } from './thinking.js?v=3';
import { renderLogs } from './logs.js';
import { modelLoadCommand } from './commands.js';
import { writeClipboard } from './clipboard.js';
import { createBenchmarkUI } from './benchmark.js?v=2';
import { createTaskUI } from './tasks.js';
import { renderMarkdown } from './markdown.js';

const $ = (id) => document.getElementById(id);
const stateLabels = () => ({ queued: t("Queued"), skipped: t("Not attempted"), unloading: t("Unloading"), idle: t("Idle"), draining: t("Finishing requests"), loading: t("Loading"), ready: t("Ready"), failed: t("Failed") });
let savedRecipesUrl, savedLibraryPath;
let benchmarkUI = null;
let taskUI = null;
let status = null, recipes = [], recipeTypes = {}, localModels = [], page = 'workspace';
const pages = {workspace: 'Overview', models: 'Models', chat: 'Chat', benchmark: 'Benchmark', tasks: 'Tasks', settings: 'Settings'};
let bundleCatalog = false;
let toastTimer, directoryTarget = '', directoryData = null;
let polling = false, catalogLoading = false, modelRoot = null, lastDownloadExit = null;
let catalogRequest = 0;
let logRequest = 0;
let savedRoot;
let aliases = [], currentChatModel = null;
const activeChats = new Map();
let skipChatDeleteConfirmation = false;
try { skipChatDeleteConfirmation = sessionStorage.getItem('snowllm.skip-chat-delete-confirmation') === '1'; } catch {}
const chatHistoryMedia = matchMedia('(min-width: 720px) and (min-height: 620px)');
let chatDrafts = new Map();
let modelSignature = '', aliasRequest = 0;
let preferredThinkingLevel = '';
let conversation = [];
let sessions = [], sessionId = newSessionId(), loadSettings = {};
const specOptionFields = ['mtp-prompt-lookup', 'mtp-tool-draft'];
const loadFieldIds = ['load-id', 'load-target', 'context-length', 'gpu-budget', 'spec-depth', 'kv-type', ...specOptionFields, 'extra-options'];
const loadCards = [];
let loadCardSequence = 0, submittingLoad = false;
let seenLoadAdvice = null;
let savedLoadSelection = null;
try { seenLoadAdvice = sessionStorage.getItem('snowllm.load-advice'); } catch {}
try { loadSettings = JSON.parse(localStorage.getItem('snowllm.load-settings') || '{}'); if (!loadSettings || Array.isArray(loadSettings) || typeof loadSettings !== 'object') loadSettings = {}; } catch {}
try {
  const saved = JSON.parse(localStorage.getItem('snowllm.load-selection'));
  if (['recipe', 'path'].includes(saved?.source) && typeof saved.model === 'string' && saved.model.trim()) savedLoadSelection = saved;
} catch {}
try {
  const saved = JSON.parse(localStorage.getItem('snowllm.conversation') || '{}');
  if (Array.isArray(saved.messages)) conversation = saved.messages.filter(m => ['user', 'assistant'].includes(m.role) && typeof m.content === 'string').slice(-100);
  $('temperature').value = typeof saved.temperature === 'string' ? saved.temperature : '';
  $('max-tokens').value = typeof saved.maxTokens === 'string' ? saved.maxTokens : '2048';
  currentChatModel = typeof saved.model === 'string' ? saved.model : null;
  $('system-prompt').value = typeof saved.system === 'string' ? saved.system : '';
  preferredThinkingLevel = typeof saved.thinkingLevel === 'string' ? saved.thinkingLevel : '';
} catch { /* Storage may be unavailable or contain an old format. */ }

try {
  const stored = JSON.parse(localStorage.getItem('snowllm.sessions') || '{}');
  if (Array.isArray(stored.sessions)) sessions = stored.sessions.filter(item => item && typeof item.id === 'string' && Array.isArray(item.messages)).slice(-50).map(item => ({ ...item, messages: item.messages.filter(message => message && ['user', 'assistant'].includes(message.role) && typeof message.content === 'string').slice(-100) }));
  if (sessions.some(item => item.id === stored.current)) sessionId = stored.current;
} catch {}
try {
  chatDrafts = new Map(Object.entries(JSON.parse(localStorage.getItem('snowllm.chat-drafts') || '{}'))
    .filter(([, text]) => typeof text === 'string'));
} catch {}
function loadKey(card) { return card.source === 'recipe' ? card.fields['load-recipe'].value : card.fields['load-path'].value.trim(); }
function selectLoadModel(card) {
  card.restored = false;
  card.pendingRecipe = '';
  const model = loadKey(card);
  if (card.source === 'path') syncLocalModel(card);
  if (model) {
    try { localStorage.setItem('snowllm.load-selection', JSON.stringify({source: card.source, model})); } catch {}
  }
  restoreLoadSettings(card);
  renderLoadHints(card);
}
function saveLoadSettings(card, notify = true) {
  if (!loadKey(card)) { toast(t('Choose a recipe or enter a checkpoint directory.')); return; }
  loadSettings[loadKey(card)] = Object.fromEntries(loadFieldIds.map(id => [id, card.fields[id].value]));
  try { localStorage.setItem('snowllm.load-settings', JSON.stringify(loadSettings)); if (notify) toast(t('Load settings saved')); } catch { toast(t('Browser storage unavailable · this conversation is temporary')); }
}
function restoreLoadSettings(card) {
  const saved = loadSettings[loadKey(card)] || {};
  for (const id of loadFieldIds) card.fields[id].value = typeof saved[id] === 'string' ? saved[id] : '';
  if (!['gpu', 'npu'].includes(card.fields['load-target'].value)) card.fields['load-target'].value = 'gpu';
  updateTargetSettings(card);
}
function updateTargetSettings(card) {
  const npu = card.fields['load-target'].value === 'npu';
  card.fields['npu-load-hint'].hidden = !npu;
  for (const id of ['context-length', 'gpu-budget', 'spec-depth', 'kv-type', ...specOptionFields]) card.fields[id].disabled = npu;
}
function modelLoadRequest(card) {
  const fields = card.fields;
  const model = loadKey(card).trim();
  if (!model) throw new Error(t('Choose a recipe or enter a checkpoint directory.'));
  const options = [];
  for (const [id, flag] of [['context-length', '--max-model-len'], ['gpu-budget', '--max-gpu-mem'], ['spec-depth', '--num-spec'], ['kv-type', '--kv-cache-dtype']]) if (!fields[id].disabled && fields[id].value.trim()) options.push(flag, fields[id].value.trim());
  for (const id of specOptionFields) {
    const field = fields[id];
    if (!field.disabled && field.value) options.push(field.value === 'on' ? '--' + id : '--no-' + id);
  }
  options.push(...parseOptions(fields['extra-options'].value));
  return {model, options, target: fields['load-target'].value, id: fields['load-id'].value.trim() || null};
}
function renderLoadCards() {
  for (const [index, card] of loadCards.entries()) {
    card.node.querySelector('[data-role="load-title"]').textContent = t('Model {0}', index + 1);
    card.fields['remove-model'].hidden = loadCards.length === 1;
    card.fields['remove-model'].setAttribute('aria-label', t('Remove model {0}', index + 1));
  }
  $('load-model-count').textContent = loadCards.length === 1 ? t('1 model to load') : t('{0} models to load', loadCards.length);
  $('load-button').textContent = t(loadCards.length === 1 ? 'Load model' : 'Load all models');
}
function addLoadCard(selection = null) {
  const node = $('load-model-template').content.firstElementChild.cloneNode(true);
  const card = {node, source: selection?.source || 'recipe', fields: {}, restored: Boolean(selection),
    pendingRecipe: selection?.source === 'recipe' ? selection.model : ''};
  const prefix = `load-card-${++loadCardSequence}-`;
  node.querySelectorAll('[data-field]').forEach(field => {
    card.fields[field.dataset.field] = field;
    field.id = prefix + field.dataset.field;
  });
  node.querySelectorAll('[data-for]').forEach(label => label.htmlFor = prefix + label.dataset.for);
  loadCards.push(card); $('load-models').append(node);
  node.querySelectorAll('[data-source]').forEach(button => button.addEventListener('click', () => {
    if (card.source === button.dataset.source) return;
    chooseSource(card, button.dataset.source);
    selectLoadModel(card);
  }));
  const fields = card.fields;
  fields['load-recipe'].addEventListener('change', () => selectLoadModel(card));
  fields['load-path'].addEventListener('change', () => selectLoadModel(card));
  fields['load-target'].addEventListener('change', () => updateTargetSettings(card));
  fields['local-model'].addEventListener('change', () => {
    if (fields['local-model'].value) {
      fields['load-path'].value = fields['local-model'].value;
      selectLoadModel(card);
    }
  });
  fields['save-load-settings'].addEventListener('click', () => saveLoadSettings(card));
  fields['copy-load-command'].addEventListener('click', () => action(fields['copy-load-command'], () => copyText(modelLoadCommand(modelLoadRequest(card), location.href))));
  fields['reset-load-settings'].addEventListener('click', () => {
    for (const id of loadFieldIds) fields[id].value = '';
    fields['load-target'].value = 'gpu'; updateTargetSettings(card); saveLoadSettings(card);
  });
  fields['browse-model'].addEventListener('click', () => openDirectory(fields['load-path'].id));
  fields['remove-model'].addEventListener('click', () => {
    if (loadCards.length === 1 || submittingLoad) return;
    const index = loadCards.indexOf(card);
    loadCards.splice(index, 1); node.remove(); renderLoadCards();
    loadCards[Math.min(index, loadCards.length - 1)].node.querySelector('[data-role="load-title"]').focus();
  });
  node.querySelector('[data-role="load-title"]').tabIndex = -1;
  if (selection?.source === 'path') fields['load-path'].value = selection.model;
  renderRecipeOptions(); renderLocalOptions(); chooseSource(card, card.source);
  restoreLoadSettings(card); renderLoadHints(card);
  translatePage(); renderLoadCards();
  return card;
}
function configureModel(source, model) {
  const card = loadCards.find(item => item.restored || !loadKey(item)) || addLoadCard();
  chooseSource(card, source);
  card.fields[source === 'recipe' ? 'load-recipe' : 'load-path'].value = model;
  selectLoadModel(card);
  openModelLoader(card.fields[source === 'recipe' ? 'load-recipe' : 'load-path']);
}
function openModelLoader(field) {
  if (page !== 'workspace') {
    location.hash = 'workspace';
    navigate();
  }
  if (!$('load-dialog').open) {
    $('load-feedback').textContent = '';
    $('load-dialog').showModal();
  }
  field?.focus();
}
async function submitLoad(body) {
  const result = await post('/daemon/models/load', body);
  renderStatus(result);
  const link = $('load-task-link');
  link.hidden = !result.task_id;
  if (result.task_id) link.href = `#tasks/${result.task_id}`;
}
function renderLoadAdvice(notice) {
  if (!notice || seenLoadAdvice === notice.id) return;
  seenLoadAdvice = notice.id;
  try { sessionStorage.setItem('snowllm.load-advice', notice.id); } catch {}
  $('load-advice-link').href = notice.url;
  $('load-advice').hidden = false;
}
function renderSessions() {
  const history = $('chat-session');
  const focusedId = history.contains(document.activeElement) ? document.activeElement.dataset.sessionId : null;
  const scrollTop = history.scrollTop;
  const items = [...sessions].reverse();
  if (!items.some(item => item.id === sessionId)) items.unshift({id: sessionId});
  history.replaceChildren(...items.map(item => {
    const row = element('button', 'chat-session-row');
    row.type = 'button';
    row.dataset.sessionId = item.id;
    if (item.id === sessionId) row.setAttribute('aria-current', 'true');
    const title = item.title || chatDrafts.get(item.id)?.trim().slice(0, 48) || t('New conversation');
    const label = element('span', 'chat-session-title', title);
    label.title = title;
    row.append(label);
    const running = activeChats.has(item.id);
    const details = [running && t('Generating…'), item.model].filter(Boolean).join(' · ');
    if (details) row.append(element('span', running ? 'chat-session-running' : 'chat-session-model', details));
    return row;
  }));
  history.scrollTop = scrollTop;
  if (focusedId) {
    const focused = [...history.children].find(row => row.dataset.sessionId === focusedId);
    (focused || history.querySelector('[aria-current]'))?.focus({preventScroll: Boolean(focused)});
  }
}
function placeChatHistory() {
  const rail = $('chat-history-rail'), dialog = $('chat-history-dialog');
  const panel = $('chat-history-panel'), opener = $('open-chat-history');
  const inSidebar = page === 'chat' && chatHistoryMedia.matches;
  const host = inSidebar ? rail : dialog;
  const focused = panel.contains(document.activeElement) ? document.activeElement : null;
  const wasOpen = dialog.open;
  if (wasOpen && (inSidebar || page !== 'chat')) dialog.close();
  rail.hidden = !inSidebar;
  opener.hidden = inSidebar;
  if (panel.parentElement === host) return;
  host.append(panel);
  if (inSidebar) {
    const current = panel.querySelector('[aria-current]');
    current?.scrollIntoView({block: 'nearest'});
    if (focused || wasOpen) (focused || current)?.focus({preventScroll: true});
  } else if (focused && page === 'chat') opener.focus({preventScroll: true});
}
function newSessionId() {
  return crypto.randomUUID?.() ?? crypto.getRandomValues(new Uint32Array(4)).join('-');
}
function selectSession(id) {
  const item = sessions.find(item => item.id === id);
  sessionId = item?.id || newSessionId();
  conversation = activeChats.get(sessionId)?.conversation || item?.messages || [];
  currentChatModel = item?.model || null;
  $('chat-input').value = chatDrafts.get(sessionId) || '';
  $('system-prompt').value = item?.system || '';
  $('temperature').value = item?.temperature || '';
  $('max-tokens').value = item?.maxTokens || '2048';
  preferredThinkingLevel = item?.thinkingLevel || '';
  updateThinkingLevels(); renderChatModels(); renderConversation(); saveConversation(); controls();
}
function newSession() {
  saveConversation();
  sessionId = newSessionId(); conversation = []; currentChatModel = status?.default_model || status?.model || null;
  $('chat-input').value = '';
  renderChatModels(); updateThinkingLevels(); saveConversation(); renderConversation(); controls();
  $('chat-session').scrollTop = 0;
}
async function copyText(text) {
  try { await writeClipboard(text); toast(t('Copied')); } catch { toast(t('Copy failed')); }
}
function renderEffectiveSettings(container, config) {
  container.replaceChildren();
  if (!config) { container.append(element('div', 'muted', '—')); return; }
  const entries = [['Context per request', config.max_model_len], ['Shared KV capacity', config.kv_capacity_tokens], ['Maximum concurrent requests', config.max_num_seqs],
    ['GPU memory fraction', Number.isFinite(config.max_gpu_mem_util) ? config.max_gpu_mem_util.toFixed(4) : '—'], ['KV cache precision', config.kv_cache_dtype],
    ['Offload ratio', Number.isFinite(config.offload_ratio) ? `${(config.offload_ratio * 100).toFixed(2)}%` : '—'],
    ['MTP depth', config.num_spec], ['DFlash drafter', config.dflash_path || '—'], ['DFlash block', config.dflash_block ?? '—'],
    ['Execution mode', t(config.enforce_eager ? 'Eager' : 'CUDA graphs')], ['Prefill chunk', config.prefill_chunk]];
  for (const [key, label] of [['mtp_prompt_lookup', 'Prompt lookup'], ['mtp_tool_draft', 'Tool-call drafting']]) {
    if (typeof config[key] === 'boolean') entries.push([label, t(config[key] ? 'On' : 'Off')]);
  }
  if (Number.isFinite(config.max_gpu_mem_bytes)) entries.push(['GPU memory limit', bytes(config.max_gpu_mem_bytes)]);
  if (Number.isFinite(config.max_disk_mem_gib)) entries.push(['Disk offload limit', bytes(config.max_disk_mem_gib * 2 ** 30)]);
  if (Number.isFinite(config.max_host_mem_bytes)) entries.push(['Host memory limit', bytes(config.max_host_mem_bytes)]);
  for (const [label, value] of entries) { const row = element('div'); row.append(element('dt', '', t(label)), element('dd', '', String(value ?? '—'))); container.append(row); }
}
function element(tag, className = '', text = '') {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
}
function toast(message) {
  const node = $('toast');
  const parent = $('load-dialog').open ? $('load-dialog') : document.body;
  if (node.parentElement !== parent) parent.append(node);
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.hidden = true, 5500);
}
function bytes(value) { if (!Number.isFinite(value) || value <= 0) return '—'; return `${(value / 2 ** 30).toFixed(1)} GiB`; }
function transition() { return Boolean(status?.operation || status && ['loading', 'draining'].includes(status.state)); }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  if (!response.ok) {
    let detail = t("Request failed ({0})", response.status);
    try { const value = await response.json(); detail = typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail || value.error || value); } catch { /* Non-JSON upstream. */ }
    throw new Error(detail);
  }
  return response.json();
}
function post(path, body) { return api(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) }); }
async function action(button, task) {
  button.disabled = true;
  try { await task(); } catch (error) { toast(t(error.message)); }
  finally { button.disabled = false; if (status) controls(); }
}
function navigate() {
  const next = location.hash.slice(1);
  const previous = page;
  const taskId = /^tasks\/([\w-]+)$/.exec(next)?.[1] || null;
  page = taskId || next === 'tasks-panel' ? 'tasks' : Object.hasOwn(pages, next) ? next : 'workspace';
  for (const name of Object.keys(pages)) $('page-' + name).hidden = name !== page;
  placeChatHistory();
  document.querySelectorAll('nav a').forEach(link => { link.classList.toggle('active', link.dataset.page === page); if (link.dataset.page === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  $('page-name').textContent = t(pages[page]);
  const taskChanged = taskUI?.select(taskId);
  const changedTaskPage = page !== previous && (page === 'tasks' || previous === 'tasks');
  if (changedTaskPage || (page === 'tasks' && taskChanged && !taskId)) window.scrollTo(0, 0);
  taskUI?.update(status, Boolean(status) && $('connection-banner').hidden, page);
  if (page === 'workspace') refreshLogs().catch(error => toast(error.message));
  if (page === 'settings') storageSpace();
  if (page === 'benchmark') benchmarkUI?.refresh().catch(error => toast(error.message));
}
function controls() {
  const online = !$('connection-banner').hidden ? false : Boolean(status);
  $('load-button').disabled = !online || submittingLoad;
  $('add-load-model').disabled = submittingLoad;
  $('close-load-dialog').disabled = submittingLoad;
  for (const card of loadCards) card.node.inert = submittingLoad;
  syncChatControls();
  $('cancel-load').hidden = !transition() || status?.operation?.kind === 'unload';
  $('cancel-load').disabled = !online || status?.cancel_requested;
  $('retry-load').disabled = !online;
  $('retry-load').hidden = !status?.error || !status?.last_load;
  document.querySelectorAll('[data-download]').forEach(button => button.disabled = !online || Boolean(status?.benchmark) || status?.download?.exit_code === null);
  document.querySelectorAll('[data-use]').forEach(button => button.disabled = !online);
  benchmarkUI?.update(online ? status : null);
  taskUI?.update(status, online, page);
}
function renderLoadError(next) {
  const banner = $('model-error');
  banner.hidden = !next.error;
  const stopped = next.models?.some(row => row.state === 'skipped' && row.blocked_by);
  const title = stopped ? t('Batch loading stopped after a model failed')
    : next.error_kind === 'capacity' ? t('Not enough memory for these settings') : t('Model failed to load or stopped');
  const key = JSON.stringify([next.error, next.error_kind, title]);
  if (banner.dataset.failure === key) return;
  banner.dataset.failure = key;
  $('model-error-title').textContent = title;
  $('model-error-detail').textContent = next.error || '';
  $('model-error-hint').hidden = next.error_kind !== 'capacity';
  $('model-error-hint').textContent = t('Check Context length and memory limits in Inference settings. Reduce context, or set --max-num-seqs 1 in Additional options, then load the model again.');
}
function renderUpdate(update) {
  const releases = update?.releases?.length ? update.releases : update?.latest
    ? [{ version: update.latest, changelog: update.changelog,
      command: `python -m pip install --upgrade "snowllm==${update.latest}"` }] : [];
  $('update-notice').hidden = !releases.length;
  if (!releases.length) return;
  $('update-title').textContent = releases.length === 1
    ? t('SnowLLM {0} is available (installed: {1}).', releases[0].version, update.current)
    : t('SnowLLM updates are available (installed: {0}).', update.current);
  const key = JSON.stringify([releases, update.checking, t('Stable'), t('Dev')]);
  const list = $('update-releases');
  if (list.dataset.update === key) return;
  list.dataset.update = key;
  list.replaceChildren(...releases.map(release => {
    const row = element('section', 'update-release');
    const label = release.channel ? `${t(release.channel === 'dev' ? 'Dev' : 'Stable')} (${release.version})` : release.version;
    row.append(element('strong', '', label), element('pre', 'update-changelog', release.changelog
      || t(update.checking ? 'Loading release notes…' : 'Release notes are not available for this version.')),
      element('pre', 'update-command', release.command));
    return row;
  }));
}
function selectedModel() { return currentChatModel || status?.default_model || status?.model; }
function selectedAlias() { return aliases.find(item => item.id === selectedModel()); }
function chatModelReady(name = selectedModel()) {
  return (status?.models || []).some(row => row.state === 'ready' && (row.id === name || row.aliases?.includes(name)))
    || !status?.models?.length && status?.state === 'ready' && name === status.model;
}
function chatBlockReason() {
  if (activeChats.has(sessionId)) return 'Generating…';
  if (!status || !$('connection-banner').hidden) return 'Waiting for the local service…';
  if (status.benchmark) return 'Inference is paused during a benchmark.';
  if (status.state !== 'ready' || !chatModelReady()) return 'Select a loaded model to send messages.';
  return '';
}
function syncChatControls() {
  const running = activeChats.has(sessionId), blocked = chatBlockReason();
  $('stop-chat').hidden = !running;
  $('send-chat').hidden = running;
  $('send-chat').disabled = Boolean(blocked) || !$('chat-input').value.trim();
  $('chat-model').disabled = running;
  $('thinking-level').disabled = running || $('thinking-level').length < 2;
  $('delete-chat').disabled = running;
  $('export-chat').disabled = !conversation.length;
  const statusText = t(blocked || 'Enter to send · Shift+Enter for a new line');
  if ($('chat-status').textContent !== statusText) $('chat-status').textContent = statusText;
  $('chat-status').classList.toggle('chat-shortcut', !blocked);
  $('chat-empty-model').hidden = chatModelReady();
  $('chat-prompts').hidden = !chatModelReady();
  renderChatAction(blocked);
}
function renderChatModels() {
  const name = selectedModel();
  const names = [...new Set([name, ...aliases.map(item => item.id),
    ...(status?.models || []).filter(row => row.state === 'ready').map(row => row.id)].filter(Boolean))];
  const options = names.map(id => [id, chatModelReady(id) ? id : `${id} · ${t('Not loaded')}`]);
  if (!name) options.unshift(['', t(names.length ? 'Select a model' : 'None loaded')]);
  const select = $('chat-model');
  if (select.length !== options.length || options.some(([id, label], i) => select.options[i].value !== id || select.options[i].text !== label)) {
    select.replaceChildren(...options.map(([id, label]) => new Option(label, id)));
  }
  if (select.value !== (name || '')) select.value = name || '';
}
function renderResidentModels(next) {
  const operation = next.operation;
  $('model-operation').textContent = !operation ? '' : operation.kind === 'load'
    ? next.cancel_requested ? t('Cancelling after the current build finishes.') : t('Loading {0} on {1}. Inference on this target resumes after initialization.', operation.model, (operation.target || 'gpu').toUpperCase())
    : t('Unloading {0}.', operation.model);
  const queues = $('execution-queues'); queues.replaceChildren();
  for (const target of ['gpu', 'npu']) {
    const queue = next.queues?.[target];
    const unavailable = queue?.available === false || !queue && target === 'npu';
    const detail = queue?.error || (unavailable ? t('Backend not configured')
      : t('Running {0} · Queued {1}', queue?.running ? 1 : 0, queue?.waiting || 0));
    const row = element('div', `execution-queue${unavailable ? ' unavailable' : ''}`);
    row.append(element('strong', '', target.toUpperCase()), element('span', '', detail));
    queues.append(row);
  }
  const list = $('resident-models');
  const expanded = new Set([...list.querySelectorAll('.resident-model:has(details[open])')].map(node => node.dataset.modelId));
  list.replaceChildren();
  $('runtime-model-count').textContent = t('Ready {0} / Total {1}', (next.models || []).filter(row => row.state === 'ready').length, next.models?.length || 0);
  $('runtime-model-count').hidden = !next.models?.length;
  $('force-switch').closest('label').hidden = !next.models?.length;
  if (!next.models?.length) {
    const empty = element('div', 'runtime-empty');
    const icon = element('span', 'runtime-empty-icon', '◇');
    icon.setAttribute('aria-hidden', 'true');
    empty.append(icon, element('strong', '', t('None loaded')), element('p', '', t('Choose a model and load it to start.')));
    list.append(empty);
  }
  for (const row of next.models || []) {
    const card = element('section', 'resident-model');
    card.dataset.modelId = row.id;
    const title = element('div', 'resident-model-title');
    title.append(element('strong', '', row.id), element('span', `pill ${row.state}`, stateLabels()[row.state] || t(row.state)));
    card.append(title, element('p', 'model-device', `${(row.target || 'gpu').toUpperCase()}${next.default_model === row.id ? ` · ${t('Default model')}` : ''}`));
    const runtime = element('dl', 'runtime-list model-runtime');
    for (const [label, value] of [['Inference process', row.pid || next.pid || '—'],
      ['Running / waiting', `${row.engine?.running ?? '—'} / ${row.engine?.waiting ?? '—'}`],
      ['Tensor memory', bytes(row.memory?.total)],
      ['Scheduler wait', Number.isFinite(row.scheduling?.wait_seconds) ? `${row.scheduling.wait_seconds.toFixed(2)} s` : '—']]) {
      const detail = element('div'); detail.append(element('dt', '', t(label)), element('dd', '', String(value))); runtime.append(detail);
    }
    const settings = element('details', 'effective-settings');
    settings.open = expanded.has(row.id);
    const effective = element('dl', 'runtime-list');
    renderEffectiveSettings(effective, row.runtime_config);
    settings.append(element('summary', '', t('Model details')), element('p', 'field-hint model-source', row.source),
      runtime, element('h3', 'model-settings-title', t('Effective inference settings')), effective);
    card.append(settings);
    if (row.error) card.append(element('p', 'message-error', row.state === 'skipped' && row.blocked_by
      ? t('Not attempted: model {0} failed earlier in this batch. Fix the error, then retry loading.', row.blocked_by) : row.error));
    const buttons = element('div', 'runtime-actions');
    const use = element('button', 'button secondary', t('Set as default'));
    use.disabled = (next.tasks || []).some(task => task.kind !== 'inference'
      && !['completed', 'failed', 'cancelled', 'interrupted'].includes(task.state))
      || Boolean(operation) || Boolean(next.benchmark) || row.state !== 'ready' || next.default_model === row.id;
    use.addEventListener('click', () => action(use, async () => renderStatus(await post('/daemon/models/default', {id: row.id}))));
    const unload = element('button', 'button secondary', t(['failed', 'skipped'].includes(row.state) && !row.runtime_config ? 'Remove' : 'Unload model'));
    unload.disabled = row.state === 'unloading' || (next.tasks || []).some(task => task.kind === 'unload'
      && task.model === row.id && !['completed', 'failed', 'cancelled', 'interrupted'].includes(task.state));
    unload.addEventListener('click', () => action(unload, async () => renderStatus(await post('/daemon/models/unload', {id: row.id, force: $('force-switch').checked}))));
    buttons.append(use, unload); card.append(buttons); list.append(card);
  }
}
function renderStatus(next) {
  const oldModel = status?.model;
  const oldState = status?.state;
  status = next;
  benchmarkUI?.update(next);
  renderUpdate(next.update);
  renderLoadAdvice(next.load_advice);
  renderLoadCards();
  renderResidentModels(next);
  const inferenceUrl = `${next.inference_url}/v1`;
  $('api-address').textContent = inferenceUrl.replace(/^https?:\/\//, '');
  $('api-url').value = inferenceUrl;
  $('responses-api-url').value = inferenceUrl;
  $('anthropic-api-url').value = next.inference_url;
  $('metrics-link').href = `${next.inference_url}/metrics`;
  $('connection-banner').hidden = true;
  $('connection-dot').className = 'dot teal';
  $('connection-title').textContent = t("Daemon connected");
  $('connection-detail').textContent = location.host;
  $('engine-state').textContent = stateLabels()[next.state] || next.state;
  $('engine-state').dataset.state = next.state;
  $('current-model').textContent = next.default_model || next.model || t("None loaded");
  $('storage-preview').textContent = (next.models_dirs || [next.models_dir]).join('\n');
  $('config-source').textContent = { 'command-line': t("Command line"), environment: t("Environment"), config: t("Saved configuration"), default: t("Default") }[next.models_dir_source];
  renderLoadError(next);
  $('config-file').textContent = next.config_file || '';
  const modelOverride = ['command-line', 'environment'].includes(next.models_dir_source);
  const recipeOverride = next.recipes_url_source === 'environment';
  const libraryOverride = next.library_path_source === 'environment';
  const saved = Array.isArray(next.saved_models_dirs) ? next.saved_models_dirs.join('\n') :
    next.saved_models_dir ?? (modelOverride ? '' : next.models_dir);
  if (document.activeElement !== $('models-dir') && savedRoot !== saved) $('models-dir').value = saved;
  savedRoot = saved;
  const savedUrl = next.saved_recipes_url ?? (recipeOverride ? $('recipes-url').placeholder : next.recipes_url || '');
  const savedLib = next.saved_library_path || '';
  if (document.activeElement !== $('recipes-url') && savedRecipesUrl !== savedUrl) $('recipes-url').value = savedUrl;
  if (document.activeElement !== $('library-path') && savedLibraryPath !== savedLib) $('library-path').value = savedLib;
  $('library-path').placeholder = !libraryOverride && next.library_path_source === 'default' && next.library_path
    ? t("Bundled: {0}", next.library_path) : t("Bundled kernels (default)");
  savedRecipesUrl = savedUrl; savedLibraryPath = savedLib;
  for (const [id, overridden, source, effective] of [
    ['storage-override', modelOverride, next.models_dir_source === 'environment' ? 'SNOWLLM_MODELS / SNOWLLM_MODEL_DIRS' : '--models-dir / --models-dirs', (next.models_dirs || [next.models_dir]).join('; ')],
    ['recipes-override', recipeOverride, 'SNOWLLM_RECIPES_URL', next.recipes_url],
    ['library-override', libraryOverride, 'SNOWLLM_LIB', next.library_path],
  ]) {
    $(id).hidden = !overridden;
    $(id).textContent = overridden ? t("{0} overrides this setting. Effective value: {1}", source, effective) : '';
  }
  const download = next.download;
  $('download-card').hidden = !download;
  if (download) {
    $('download-name').textContent = download.recipe;
    const progress = download.progress;
    $('download-progress').hidden = !progress;
    $('download-progress-detail').hidden = !progress;
    if (progress) {
      const percent = progress.total_bytes > 0 ? Math.max(0, Math.min(100, 100 * progress.done_bytes / progress.total_bytes)) : 0;
      $('download-progress').value = percent;
      const amount = number => Number.isFinite(number) ? `${(number / 2 ** 20).toFixed(1)} MiB` : '—';
      const eta = Number.isFinite(progress.eta_seconds) ? `${Math.ceil(Math.max(0, progress.eta_seconds))} s` : '—';
      $('download-progress-detail').textContent = `${t(progress.phase === 'checking' ? 'Checking files' : 'Downloading files')} · ${percent.toFixed(1)}% · ${t('{0} / {1} · {2}/s · ETA {3}', amount(progress.done_bytes), amount(progress.total_bytes), amount(progress.bytes_per_second), eta)}`;
    }
    $('download-state').textContent = download.exit_code === null ? t("Downloading \u00b7 view output for transfer progress") : download.exit_code === 0 ? t("Download complete \u00b7 verified and ready") : t("Download stopped or failed (exit {0}). View output for details; a new download resumes partial files.", download.exit_code);
    $('cancel-download').hidden = download.exit_code !== null;
    if (download.exit_code !== null && lastDownloadExit === null) refreshCatalog(false);
    lastDownloadExit = download.exit_code;
  } else lastDownloadExit = null;
  const storageKey = JSON.stringify(next.models_dirs || [next.models_dir]);
  if (modelRoot !== storageKey) { modelRoot = storageKey; refreshCatalog(false); storageSpace(); }
  const signature = JSON.stringify((next.models || []).map(row => [row.id, row.instance, row.state]));
  const refresh = next.state === 'ready' && (signature !== modelSignature || !aliases.length);
  modelSignature = signature;
  if (refresh) refreshAliases();
  else if (next.state !== 'ready') {
    const hadAliases = aliases.length > 0;
    aliases = []; aliasRequest++;
    if (hadAliases) updateThinkingLevels();
  }
  const legacy = (next.models || []).filter(row => row.source === currentChatModel);
  if (legacy.length === 1 && legacy[0].id !== currentChatModel) { currentChatModel = legacy[0].id; saveConversation(); }
  renderChatModels();
  controls();
}
async function refreshStatus() {
  try { renderStatus(await api('/daemon/status')); }
  catch (error) { $('connection-banner').hidden = false; $('connection-dot').className = 'dot red'; $('connection-title').textContent = t("Disconnected"); $('connection-detail').textContent = t("Reconnecting\u2026"); controls(); throw error; }
}
async function refreshAliases() {
  const request = ++aliasRequest, signature = modelSignature;
  try {
    const result = await api('/daemon/inference/models');
    if (request !== aliasRequest || signature !== modelSignature) return;
    aliases = result.data || [];
    renderChatModels();
    updateThinkingLevels();
  } catch { if (request === aliasRequest) { aliases = []; updateThinkingLevels(); } }
}
function updateThinkingLevels() {
  const capabilities = selectedAlias()?.thinking;
  const {options} = thinkingProfile(capabilities);
  const select = $('thinking-level');
  select.replaceChildren(...options.map(([value, label]) =>
    new Option(t('Thinking: {0}', value.startsWith('effort:') ? label : t(label)), value)));
  select.value = options.some(([value]) => value === preferredThinkingLevel) ? preferredThinkingLevel : '';
  select.hidden = options.length < 2 && !capabilities?.required;
  select.disabled = activeChats.has(sessionId) || options.length < 2;
}
function renderChatStats() {
  const metrics = conversation.findLast(message => message.role === 'assistant' && message.metrics)?.metrics;
  $('chat-ttft').textContent = Number.isFinite(metrics?.ttft_seconds) ? `${metrics.ttft_seconds.toFixed(2)} s` : '—';
  $('chat-tokens').textContent = Number.isFinite(metrics?.completion_tokens) ? metrics.completion_tokens : '—';
  const rate = value => Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—';
  $('chat-prefill').textContent = rate(metrics?.prefill_tok_s);
  $('chat-decode').textContent = rate(metrics?.decode_tok_s);
}
async function refreshMetrics() {
  const ready = status?.state === 'ready';
  let data = null;
  if (ready) { try { data = await api('/daemon/metrics?format=json'); } catch {} }
  const selected = data?.models?.find(row => row.id === selectedModel() || row.aliases?.includes(selectedModel()));
  const engine = selected?.engine || data?.engine;
  $('metrics-requests').textContent = engine ? `${engine.running} / ${engine.waiting}` : '—';
  $('metrics-kv').textContent = engine?.free_kv_blocks ?? '—';
  $('metrics-cache').textContent = engine ? `${engine.cache_hits} / ${engine.cache_misses}` : '—';
  const rate = value => Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—';
  $('metrics-throughput').textContent = engine ? `${rate(engine.accounting?.prefill_tok_s)} / ${rate(engine.accounting?.decode_tok_s)}` : '—';
  const gpus = data?.gpus || [];
  $('gpu-metrics').replaceChildren();
  for (const gpu of gpus) {
    const row = element('div', 'gpu-metric-row');
    row.append(element('h3', '', `GPU · ${gpu.device}`));
    const list = element('dl', 'runtime-list');
    const value = (number, unit) => Number.isFinite(number) ? `${number.toFixed(1)} ${unit}` : '—';
    const memory = (used, total) => Number.isFinite(used) && Number.isFinite(total) ? `${(used / 2 ** 30).toFixed(2)} / ${(total / 2 ** 30).toFixed(2)} GiB` : '—';
    for (const [label, text] of [[t("Utilization"), value(gpu.utilization_percent, '%')], [t("Device power"), value(gpu.power_watts, 'W')], [t("Temperature"), value(gpu.temperature_celsius, '°C')], [t("VRAM"), memory(gpu.memory_used_bytes, gpu.memory_total_bytes)], [t("Shared memory"), memory(gpu.shared_memory_used_bytes, gpu.shared_memory_total_bytes)]]) {
      const entry = element('div'); entry.append(element('dt', '', label), element('dd', '', text)); list.append(entry);
    }
    row.append(list); $('gpu-metrics').append(row);
  }
  if (!gpus.length) $('gpu-metrics').append(element('p', 'muted', ready ? t("GPU sensors unavailable") : '—'));
}
async function refreshLogs() {
  const request = ++logRequest, kind = $('log-kind').value;
  const result = await api(`/daemon/logs?download=${kind === 'download'}`);
  if (request !== logRequest || kind !== $('log-kind').value) return;
  const logs = $('logs'), changed = logs.dataset.kind !== kind;
  renderLogs(logs, result.text, t("No activity."));
  if (changed) logs.scrollTop = logs.scrollHeight;
  logs.dataset.kind = kind;
}
const logsPanel = $('logs-panel'), logsParent = logsPanel.parentNode, logsNext = logsPanel.nextSibling;
function resizeLogs(expanded) {
  const logs = $('logs'), top = logs.scrollTop;
  const atBottom = top + logs.clientHeight >= logs.scrollHeight - 30;
  $('expand-logs').hidden = expanded;
  $('close-logs').hidden = !expanded;
  if (expanded) {
    $('logs-dialog').append(logsPanel);
    $('logs-dialog').showModal();
    $('close-logs').focus();
  } else {
    $('logs-dialog').close();
    logsParent.insertBefore(logsPanel, logsNext);
    $('expand-logs').focus({ preventScroll: true });
  }
  logs.scrollTop = atBottom ? logs.scrollHeight : top;
}
async function showLogs(kind) {
  $('log-kind').value = kind;
  location.hash = 'workspace';
  navigate();
  await refreshLogs();
  $('logs').scrollIntoView({ behavior: 'smooth', block: 'center' });
}
async function refreshCatalog(refresh, sourceChanged = false) {
  if (catalogLoading && !sourceChanged) return;
  const request = ++catalogRequest;
  catalogLoading = true; $('refresh-recipes').disabled = true;
  const root = JSON.stringify(status?.models_dirs || [status?.models_dir]);
  try {
    const [catalog, local] = await Promise.allSettled([api(`/daemon/recipes?refresh=${refresh}&bundle=${bundleCatalog}`), api('/daemon/local-models')]);
    if (request !== catalogRequest) return;
    if (root !== JSON.stringify(status?.models_dirs || [status?.models_dir])) return;
    if (catalog.status === 'fulfilled') {
      recipes = catalog.value.recipes;
      recipeTypes = catalog.value.recipe_types || {};
      $('catalog-error').hidden = true;
      renderRecipeOptions();
      renderRecipes(); loadCards.forEach(renderLoadHints);
    } else {
      $('catalog-error').hidden = false; $('catalog-error').textContent = t("Catalog unavailable: {0}. You can still load a local checkpoint from Overview.", catalog.reason.message);
      if (!recipes.length) {
        $('recipe-list').replaceChildren(element('p', 'empty-state', t("Could not load the catalog. Try refreshing, or use a local model.")));
        for (const card of loadCards) card.fields['load-recipe'].replaceChildren(new Option(t("Catalog unavailable \u00b7 use Local path"), ''));
      }
    }
    if (local.status === 'fulfilled') { localModels = local.value.models; renderLocal(); }
    else $('local-list').replaceChildren(element('p', 'muted', local.reason.message));
  } finally {
    if (request === catalogRequest) {
      catalogLoading = false; $('refresh-recipes').disabled = false;
      if (root !== JSON.stringify(status?.models_dirs || [status?.models_dir])) refreshCatalog(false);
    }
  }
}
function groupRecipes(rows) {
  const groups = new Map(Object.entries(recipeTypes).map(([type, label]) => [type, {type, label, rows: []}]));
  const other = {type: 'other', label: t('Other recipes'), rows: []};
  for (const recipe of rows) (groups.get(recipe.type) || other).rows.push(recipe);
  return [...groups.values(), other].filter(group => group.rows.length);
}
function renderRecipeOptions() {
  const groups = groupRecipes(recipes);
  for (const card of loadCards) {
    const select = card.fields['load-recipe'], previous = select.value;
    const selected = previous || card.pendingRecipe;
    select.replaceChildren(new Option(t("Choose a recipe\u2026"), ''));
    for (const {type, label, rows} of groups) {
      const group = document.createElement('optgroup');
      group.label = label;
      group.dataset.recipeType = type;
      for (const recipe of rows) {
        const option = new Option(`${recipe.id}${recipe.loadable ? t(" \u00b7 local") : ''}${recipe.status === 'coming soon' ? ` · ${t('Coming soon')}` : !recipe.servable ? t(" \u00b7 upgrade required") : ''}`, recipe.id);
        option.classList.toggle('local', Boolean(recipe.loadable));
        option.disabled = !recipe.servable;
        group.append(option);
      }
      select.append(group);
    }
    if (recipes.some(recipe => recipe.id === selected && recipe.servable)) {
      select.value = selected;
      card.pendingRecipe = '';
    }
    if (card.source === 'recipe' && select.value !== previous) restoreLoadSettings(card);
  }
}
function renderRecipes() {
  const search = $('model-search').value.toLowerCase();
  const visible = recipes.filter(recipe => (!search || [recipe.id, recipe.model, recipe.precision, recipe.type, recipeTypes[recipe.type]].join(' ').toLowerCase().includes(search)) && (!$('installed-only').checked || recipe.loadable));
  $('recipe-count').textContent = t("{0} recipes", visible.length);
  const list = $('recipe-list');
  list.replaceChildren();
  for (const {type, label, rows} of groupRecipes(visible)) {
    const section = element('section', 'recipe-group');
    section.dataset.recipeType = type;
    const title = element('h2', 'recipe-group-title', label);
    title.id = `recipe-group-${type}`;
    section.setAttribute('aria-labelledby', title.id);
    const grid = element('div', 'recipe-grid');
    grid.append(...rows.map(recipeCard));
    section.append(title, grid);
    list.append(section);
  }
  if (!visible.length) list.append(element('p', 'empty-state', t("No recipes match your filters.")));
  controls();
}
function recipeCard(recipe) {
  const card = element('article', `recipe-card${recipe.loadable ? ' local' : ''}`);
  const top = element('div', 'recipe-card-top');
  const availability = element('span', `availability-badge ${recipe.loadable ? 'local' : recipe.exists ? 'partial' : 'remote'}`);
  availability.append(element('span', 'availability-icon', recipe.loadable ? '✓' : recipe.exists ? '◐' : '↓'),
    element('span', '', t(recipe.loadable ? 'Available locally' : recipe.exists ? 'Partial copy' : 'Not downloaded')));
  top.append(element('span', 'pill', recipe.precision || t('Model')), availability);
  card.append(top, element('h3', '', recipe.model || recipe.id), element('div', 'recipe-id', recipe.id));
  if (!recipe.servable) card.append(element('span', 'pill', recipe.status === 'coming soon' ? t('Coming soon') : t('Requires {0}', recipe.requires)));
  const footer = element('div', 'recipe-footer');
  footer.append(element('span', 'recipe-size', t("{0} on disk", bytes(recipe.bytes))));
  const buttons = element('div', 'download-actions');
  if (recipe.loadable && recipe.servable) {
    const use = element('button', 'button secondary', t("Configure & load")); use.dataset.use = recipe.id;
    use.addEventListener('click', () => configureModel('recipe', recipe.id)); buttons.append(use);
  }
  const download = element('button', `button ${recipe.loadable ? 'secondary' : 'primary'}`, recipe.loadable ? t("Update files") : t("Download"));
  download.dataset.download = recipe.id;
  download.addEventListener('click', () => action(download, async () => { renderStatus(await post('/daemon/pull', { recipe: recipe.id })); toast(t("Downloading {0}", recipe.id)); }));
  buttons.append(download); footer.append(buttons); card.append(footer);
  return card;
}
function renderLocal() {
  $('nav-count').textContent = localModels.length;
  $('local-list').replaceChildren(); renderLocalOptions();
  for (const model of localModels) {
    const row = element('div', 'local-row'), text = element('div');
    text.append(element('strong', '', model.name), element('p', '', model.path), element('small', 'muted', model.bytes == null ? t('Storage size unavailable') : `${(model.bytes / 2 ** 30).toFixed(2)} GiB`));
    const use = element('button', 'button secondary', t("Use checkpoint")); use.dataset.use = model.path;
    use.disabled = model.loadable === false;
    if (model.loadable === false) { use.textContent = t('Partial download'); delete use.dataset.use; }
    use.addEventListener('click', () => configureModel('path', model.path));
    const buttons = element('div', 'local-actions'); buttons.append(use);
    if (model.deletable !== false) {
      const remove = element('button', 'button secondary danger', t('Delete model'));
      remove.addEventListener('click', () => { if (confirm(t('Delete {0}? This permanently removes its files.', model.name))) action(remove, async () => { await post('/daemon/models/delete', {path: model.path}); await refreshCatalog(false); storageSpace(); toast(t('Model deleted')); }); });
      buttons.append(remove);
    }
    row.append(text, buttons); $('local-list').append(row);
  }
  if (!localModels.length) $('local-list').append(element('p', 'muted', t("No checkpoints found here. Download a recipe or choose another storage directory.")));
  controls();
}
function renderLocalOptions() {
  for (const card of loadCards) {
    const select = card.fields['local-model'];
    select.replaceChildren(new Option(t("Choose a local model\u2026"), ''));
    for (const model of localModels) if (model.loadable !== false) select.append(new Option(model.name, model.path));
    syncLocalModel(card);
  }
}
function syncLocalModel(card) {
  const select = card.fields['local-model'], path = card.fields['load-path'].value.trim();
  select.value = Array.from(select.options).some(option => option.value === path) ? path : '';
}
function chooseSource(card, next) {
  card.source = next; card.fields['recipe-input'].hidden = next !== 'recipe'; card.fields['path-input'].hidden = next !== 'path';
  card.node.querySelectorAll('[data-source]').forEach(button => { button.classList.toggle('selected', button.dataset.source === next); button.setAttribute('aria-pressed', String(button.dataset.source === next)); });
}
function renderLoadHints(card) {
  const fields = card.fields;
  const recipe = recipes.find(row => row.id === fields['load-recipe'].value);
  fields['load-recipe'].classList.toggle('local-selected', Boolean(recipe?.loadable));
  fields['recipe-description'].textContent = recipe
    ? t(recipe.loadable ? 'Local copy available.' : 'Download this recipe from Models before loading.') : '';
  const context = card.source === 'recipe' ? recipe?.defaults?.max_model_len : null;
  fields['context-length'].placeholder = Number.isSafeInteger(context) && context > 0
    ? t('Recipe default · {0}', context % 1024 === 0 ? (context / 1024) + 'K' : context)
    : t('Default');
}
async function storageSpace() {
  try { const data = await api('/daemon/directories'); $('disk-free').textContent = bytes(data.free_bytes); }
  catch { $('disk-free').textContent = t("Unavailable"); }
}
async function browseDirectory(path) {
  $('directory-error').hidden = true; $('select-directory').disabled = true;
  try {
    directoryData = await api('/daemon/directories' + (path ? `?path=${encodeURIComponent(path)}` : ''));
    $('directory-path').value = directoryData.path; $('directory-space').textContent = t("{0} free", bytes(directoryData.free_bytes));
    $('directory-up').disabled = directoryData.parent === directoryData.path; $('directory-list').replaceChildren();
    for (const folder of directoryData.directories) { const button = element('button', '', `▱  ${folder.name}`); button.addEventListener('click', () => browseDirectory(folder.path)); $('directory-list').append(button); }
    if (!directoryData.directories.length) $('directory-list').append(element('p', 'empty-state', t("No subdirectories.")));
    $('select-directory').disabled = false;
  } catch (error) { $('directory-error').hidden = false; $('directory-error').textContent = error.message; }
}
function storagePaths(value) { return value.split(/\r?\n/).map(path => path.trim()).filter(Boolean); }
function openDirectory(target) {
  directoryTarget = target;
  const field = $(target === 'models-dir-add' ? 'models-dir' : target);
  const value = target.startsWith('models-dir') ? storagePaths(field.value)[0] : field.value;
  $('directory-dialog').showModal(); browseDirectory(value || undefined);
}
function storeChatDrafts() {
  try {
    localStorage.setItem('snowllm.chat-drafts', JSON.stringify(Object.fromEntries(chatDrafts)));
  } catch {}
}
function saveChatDraft(value) {
  if (value) chatDrafts.set(sessionId, value);
  else chatDrafts.delete(sessionId);
  storeChatDrafts();
}
function saveConversation() {
  const item = {id: sessionId, title: conversation.find(message => message.role === 'user')?.content.slice(0, 48) || '',
    messages: conversation.slice(-100), model: currentChatModel, system: $('system-prompt').value,
    thinkingLevel: preferredThinkingLevel, temperature: $('temperature').value, maxTokens: $('max-tokens').value};
  const index = sessions.findIndex(session => session.id === sessionId);
  if (index >= 0) sessions[index] = item; else sessions.push(item);
  sessions = sessions.filter((item, index) => activeChats.has(item.id) || index >= sessions.length - 50);
  const savedIds = new Set(sessions.map(session => session.id));
  let draftsChanged = false;
  for (const id of chatDrafts.keys()) {
    if (!savedIds.has(id)) {
      chatDrafts.delete(id);
      draftsChanged = true;
    }
  }
  if (draftsChanged) storeChatDrafts();
  try {
    localStorage.setItem('snowllm.conversation', JSON.stringify(item));
    localStorage.setItem('snowllm.sessions', JSON.stringify({current: sessionId, sessions}));
    $('chat-storage-warning').hidden = true;
  } catch { $('chat-storage-warning').hidden = false; }
  renderSessions();
}
const chatOutcomeLabels = {
  stopped: 'Generation stopped.',
  interrupted: 'The connection closed before the reply finished.',
  length: 'Output token limit reached.',
  failed: 'Generation failed.',
};
function chatResultText(message) {
  const label = Object.hasOwn(chatOutcomeLabels, message.outcome) ? t(chatOutcomeLabels[message.outcome]) : '';
  return [label, message.error && t(message.error)].filter(Boolean).join(' ');
}
function messageNode(message, attempt = false) {
  const root = element('article', `message ${message.role}`);
  const role = element('div', message.role === 'user' ? 'sr-only' : 'message-role', message.role === 'user' ? t('YOU') : 'SnowLLM');
  const reasoning = element('details');
  reasoning.append(element('summary', '', t('Thinking')), element('div', 'reasoning-content'));
  const content = element('div', 'message-content');
  const result = element('div', 'message-result');
  if (message.role === 'assistant') result.setAttribute('role', 'status');
  const actions = element('div', 'message-actions');
  root.append(role, reasoning, content, result, actions);
  let renderedContent, renderedReasoning;
  function update() {
    if (renderedContent !== message.content) {
      if (message.role === 'user') content.textContent = message.content;
      else renderMarkdown(content, message.content, code => {
        const copy = element('button', 'button secondary code-copy', t('Copy code'));
        copy.type = 'button';
        copy.addEventListener('click', () => copyText(code));
        return copy;
      });
      renderedContent = message.content;
    }
    if (renderedReasoning !== (message.reasoning || '')) {
      renderedReasoning = message.reasoning || '';
      reasoning.hidden = !renderedReasoning;
      reasoning.lastElementChild.textContent = renderedReasoning;
    }
  }
  function renderResult() {
    const status = chatResultText(message);
    if (status) result.append(element('p', message.outcome === 'length' ? 'message-notice' : 'message-error', status));
    if (message.stats) result.append(element('div', 'chat-token-line', message.stats));
    if (message.role === 'assistant' && message.content) {
      const copy = element('button', 'button secondary answer-copy', t('Copy answer'));
      copy.type = 'button';
      copy.addEventListener('click', () => copyText(message.content));
      actions.prepend(copy);
    }
  }
  function markAttempt() {
    root.classList.add('retry-attempt');
    role.textContent = t('Regeneration attempt');
  }
  update();
  if (attempt) markAttempt();
  return { root, update, renderResult, markAttempt };
}
function chatAtBottom(list) {
  // scrollTop can be fractional while the height measurements are rounded.
  return list.scrollHeight - list.clientHeight - list.scrollTop <= 1;
}
function renderConversation() {
  const list = $('messages');
  const active = activeChats.get(sessionId);
  const nodes = conversation.map(message => {
    const node = messageNode(message);
    node.renderResult();
    if (message.retry) {
      const retry = messageNode(message.retry, true);
      retry.renderResult();
      node.root.append(retry.root);
    }
    return node.root;
  });
  if (active) nodes.push(active.node.root);
  emptyChat.hidden = Boolean(nodes.length);
  list.replaceChildren(emptyChat, ...nodes);
  renderChatStats();
  renderChatAction(chatBlockReason());
  list.scrollTop = list.scrollHeight;
}
const emptyChat = $('chat-empty');
const chatAction = $('regenerate-chat');
function chatRetryIndex() {
  if (conversation.at(-1)?.role === 'user') return conversation.length;
  if (conversation.at(-1)?.role === 'assistant' && conversation.at(-2)?.role === 'user') return conversation.length - 1;
  return -1;
}
function renderChatAction(blocked) {
  const index = chatRetryIndex();
  chatAction.hidden = activeChats.has(sessionId) || index < 0;
  if (chatAction.hidden) return;
  const previous = conversation[index];
  const retry = !previous?.content || previous.retry || previous.error || (previous.outcome && previous.outcome !== 'length');
  const label = t(retry ? 'Retry' : 'Regenerate');
  if (chatAction.textContent !== label) chatAction.textContent = label;
  chatAction.disabled = Boolean(blocked);
  const last = $('messages').lastElementChild;
  const actions = last.querySelector('.retry-attempt > .message-actions') || last.querySelector('.message-actions');
  if (chatAction.parentElement !== actions) actions.append(chatAction);
}
async function sendMessage(retrying = false) {
  if (chatBlockReason()) return;
  const replyIndex = retrying ? chatRetryIndex() : conversation.length + 1;
  const text = $('chat-input').value.trim();
  if (replyIndex < 0 || (!retrying && !text)) return;
  currentChatModel = selectedModel();
  const runningSession = sessionId, runningConversation = conversation;
  const previous = conversation[replyIndex];
  const controller = new AbortController();
  const assistant = { role: 'assistant', content: '', reasoning: '' };
  const node = messageNode(assistant), list = $('messages');
  activeChats.set(runningSession, {controller, conversation: runningConversation, node});
  if (!retrying) {
    const user = { role: 'user', content: text };
    conversation.push(user);
    $('chat-input').value = '';
    saveChatDraft('');
    emptyChat.hidden = true;
    list.append(messageNode(user).root);
  }
  list.append(node.root);
  list.scrollTop = list.scrollHeight;
  saveConversation();
  syncChatControls();
  if (!retrying) $('chat-input').focus({preventScroll: true});
  const started = performance.now(); let firstToken = null, usage = null, finishReason = null;
  const messages = conversation.slice(0, replyIndex).filter(message => message.role === 'user' || message.content)
    .map(({ role, content }) => ({ role, content }));
  if ($('system-prompt').value.trim()) messages.unshift({ role: 'system', content: $('system-prompt').value.trim() });
  const payload = { model: currentChatModel, messages, stream: true, stream_options: { include_usage: true } };
  if ($('max-tokens').value) payload.max_tokens = Number($('max-tokens').value);
  if ($('temperature').value) payload.temperature = Number($('temperature').value);
  let frame = null;
  try {
    Object.assign(payload, thinkingParameters($('thinking-level').value, selectedAlias()?.thinking));
    const response = await fetch('/daemon/inference/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
    if (response.status === 499) { assistant.outcome = 'stopped'; return; }
    if (!response.ok) { let detail = t("Generation failed ({0})", response.status); try { const data = await response.json(); detail = data.detail || data.error?.message || detail; } catch {} throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)); }
    const completed = await readEvents(response.body, (chunk) => {
      if (chunk.error) throw new Error(chunk.error.message || t("Generation failed."));
      if (chunk.usage) usage = chunk.usage;
      const choice = chunk.choices?.[0], delta = choice?.delta || {};
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (!delta.content && !delta.reasoning_content) return;
      if (firstToken === null) firstToken = performance.now();
      assistant.content += delta.content || ''; assistant.reasoning += delta.reasoning_content || '';
      if (frame === null) {
        frame = requestAnimationFrame(() => {
          frame = null;
          const follow = node.root.isConnected && chatAtBottom(list);
          node.update();
          if (follow) list.scrollTop = list.scrollHeight;
        });
      }
    });
    if (!completed) assistant.outcome = 'interrupted';
    else if (finishReason === 'length') assistant.outcome = 'length';
    else if (!assistant.content.trim()) {
      assistant.outcome = 'failed';
      assistant.error = 'The model returned no answer.';
    }
    assistant.metrics = { ttft_seconds: firstToken === null ? null : (firstToken - started) / 1000 };
    if (usage) {
      const timing = usage.timings || {};
      Object.assign(assistant.metrics, { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens,
        prefill_tok_s: timing.prefill_tok_s, decode_tok_s: timing.decode_tok_s });
      const rate = value => Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—';
      assistant.stats = t("{0} output tokens \u00b7 {1} s \u00b7 Prefill {2} \u00b7 Decode {3}", usage.completion_tokens, ((performance.now() - started) / 1000).toFixed(1), rate(timing.prefill_tok_s), rate(timing.decode_tok_s));
    }
  } catch (error) {
    if (controller.signal.aborted) assistant.outcome = 'stopped';
    else if (error instanceof TypeError) assistant.outcome = 'interrupted';
    else {
      assistant.outcome = 'failed';
      if (error.message !== t('Generation failed.')) assistant.error = error.message;
    }
  } finally {
    if (frame !== null) cancelAnimationFrame(frame);
    const visible = sessionId === runningSession;
    const follow = visible && chatAtBottom(list);
    const stopFocused = visible && document.activeElement === $('stop-chat');
    const keepPrevious = (previous?.content || previous?.reasoning) && assistant.outcome
      && (assistant.outcome !== 'length' || !assistant.content.trim());
    if (keepPrevious) previous.retry = assistant;
    else runningConversation[replyIndex] = assistant;
    if (visible && previous) {
      const previousNode = node.root.previousElementSibling;
      if (keepPrevious) {
        previousNode.querySelector(':scope > .retry-attempt')?.remove();
        node.markAttempt();
        previousNode.append(node.root);
      } else previousNode.remove();
    }
    node.update();
    node.renderResult();
    activeChats.delete(runningSession);
    const saved = sessions.find(item => item.id === runningSession);
    if (saved) saved.messages = runningConversation.slice(-100);
    if (visible) { saveConversation(); renderChatStats(); }
    else {
      try { localStorage.setItem('snowllm.sessions', JSON.stringify({current: sessionId, sessions})); } catch {}
      renderSessions();
    }
    syncChatControls();
    if (follow) list.scrollTop = list.scrollHeight;
    if (stopFocused) $('chat-input').focus({preventScroll: true});
    refreshStatus().catch(() => {});
  }
}
async function copyApi(target = 'api-url') {
  if (!status?.inference_url) { toast(t("Wait for the local service connection before copying the API address.")); return; }
  const value = $(target).value;
  try { await writeClipboard(value); toast(t("API address copied")); }
  catch { toast(t("API address: {0}", value)); }
}

translatePage();
$('language').value = language();
$('language').addEventListener('change', () => {
  setLanguage($('language').value);
  translatePage(); navigate(); updateThinkingLevels();
  renderConversation(); renderSessions(); renderChatModels(); syncChatControls();
  if (status) renderStatus(status);
  renderRecipeOptions(); renderRecipes(); renderLocal(); loadCards.forEach(renderLoadHints); renderLoadCards();
  refreshMetrics(); refreshLogs().catch(() => {});
});
$('api-address').textContent = t("Connecting\u2026"); $('api-url').value = '';
$('copy-api').addEventListener('click', () => copyApi());
document.querySelectorAll('[data-copy-api]').forEach(button => button.addEventListener('click', () => copyApi(button.dataset.copyApi)));
window.addEventListener('hashchange', navigate);
$('open-load-dialog').addEventListener('click', () => openModelLoader());
$('open-load-from-chat').addEventListener('click', () => openModelLoader());
$('close-load-dialog').addEventListener('click', () => $('load-dialog').close());
$('load-dialog').addEventListener('cancel', event => { if (submittingLoad) event.preventDefault(); });
function dismissOnBackdrop(dialog, cancelButton) {
  function outside(event) {
    if (event.target !== dialog) return false;
    const box = dialog.getBoundingClientRect();
    return event.clientX < box.left || event.clientX >= box.right || event.clientY < box.top || event.clientY >= box.bottom;
  }
  let startedOutside = false;
  dialog.addEventListener('pointerdown', event => { startedOutside = event.button === 0 && outside(event); });
  dialog.addEventListener('click', event => {
    if (startedOutside && outside(event)) cancelButton.click();
    startedOutside = false;
  });
}
dismissOnBackdrop($('load-dialog'), $('close-load-dialog'));
dismissOnBackdrop($('delete-chat-dialog'), $('cancel-chat-delete'));
$('add-load-model').addEventListener('click', () => { const card = addLoadCard(); card.fields['load-target'].focus(); });
$('cancel-load').addEventListener('click', () => action($('cancel-load'), async () => { renderStatus(await post('/daemon/load/cancel')); toast(t('Load cancellation requested')); }));
$('retry-load').addEventListener('click', () => action($('retry-load'), async () => {
  const {model, options, id, target, models} = status.last_load;
  const remaining = models?.filter(item => !(status.models || []).some(row => row.state === 'ready'
    && (item.id ? row.id === item.id : row.source === item.model && (!item.target || row.target === item.target))));
  if (remaining && !remaining.length) throw new Error(t('All listed models are already loaded.'));
  await submitLoad(remaining ? {models: remaining} : {model, options, id, target});
  toast(t('Load task submitted'));
}));
$('dismiss-load-advice').addEventListener('click', () => { $('load-advice').hidden = true; });
$('model-search').addEventListener('input', renderRecipes); $('installed-only').addEventListener('change', renderRecipes);
$('refresh-state').addEventListener('click', () => action($('refresh-state'), refreshStatus));
$('refresh-recipes').addEventListener('click', () => refreshCatalog(true));
document.querySelectorAll('[data-catalog-source]').forEach(select => select.addEventListener('change', () => {
  bundleCatalog = select.value === 'bundle';
  document.querySelectorAll('[data-catalog-source]').forEach(control => { control.value = select.value; });
  recipes = []; renderRecipeOptions(); renderRecipes(); loadCards.forEach(renderLoadHints);
  refreshCatalog(false, true);
}));
$('refresh-logs').addEventListener('click', () => action($('refresh-logs'), refreshLogs)); $('log-kind').addEventListener('change', () => refreshLogs().catch(error => toast(error.message)));
$('expand-logs').addEventListener('click', () => resizeLogs(true));
$('close-logs').addEventListener('click', () => resizeLogs(false));
$('logs-dialog').addEventListener('cancel', event => { event.preventDefault(); resizeLogs(false); });
$('adjust-load-settings').addEventListener('click', () => {
  const card = loadCards.find(item => loadKey(item) === status?.last_load?.model) || loadCards[0];
  card.node.querySelector('.advanced').open = true;
  openModelLoader(card.fields['context-length'].disabled ? card.fields['load-target'] : card.fields['context-length']);
});
$('load-error-logs').addEventListener('click', () => action($('load-error-logs'), () => showLogs('inference')));
$('load-form').addEventListener('submit', async event => {
  event.preventDefault();
  if ($('load-button').disabled) return;
  $('load-feedback').textContent = '';
  try {
    const models = loadCards.map((card, index) => {
      try { return modelLoadRequest(card); }
      catch (error) { throw new Error(t('Model {0}: {1}', index + 1, error.message)); }
    });
    const ids = models.map(item => item.id).filter(Boolean);
    if (new Set(ids).size !== ids.length) throw new Error(t('Each model needs a unique model ID.'));
    loadCards.forEach(card => saveLoadSettings(card, false));
    submittingLoad = true; controls();
    await submitLoad({models});
    loadCards.forEach(card => { card.restored = false; });
    $('load-dialog').close();
    if (!$('load-task-link').hidden) $('load-task-link').focus();
    toast(t('Load task submitted'));
  } catch (error) {
    $('load-feedback').textContent = t(error.message);
  } finally {
    submittingLoad = false;
    controls();
  }
});
$('cancel-download').addEventListener('click', () => action($('cancel-download'), async () => { renderStatus(await post('/daemon/download/cancel')); toast(t("Download stopped. Partial files are kept for resuming.")); }));
$('download-logs-button').addEventListener('click', () => action($('download-logs-button'), () => showLogs('download')));
$('storage-form').addEventListener('submit', (event) => { event.preventDefault(); action(event.submitter, async () => { const paths = storagePaths($('models-dir').value); const settings = paths.length > 1 ? { models_dirs: paths } : { models_dir: paths[0] || '' }; renderStatus(await api('/daemon/config', { method: 'PUT', body: JSON.stringify(settings) })); await refreshCatalog(false); toast(['command-line', 'environment'].includes(status.models_dir_source) ? t("Directory saved to JSON. The startup override is still active.") : t("Model directory saved")); }); });
$('recipes-settings-form').addEventListener('submit', (event) => { event.preventDefault(); action(event.submitter, async () => { renderStatus(await api('/daemon/config', { method: 'PUT', body: JSON.stringify({ recipes_url: $('recipes-url').value.trim() || null }) })); await refreshCatalog(true); toast(status.recipes_url_source === 'environment' ? t("Catalogue saved. SNOWLLM_RECIPES_URL still overrides it.") : t("Catalogue saved")); }); });
$('library-settings-form').addEventListener('submit', (event) => { event.preventDefault(); action(event.submitter, async () => { renderStatus(await api('/daemon/config', { method: 'PUT', body: JSON.stringify({ library_path: $('library-path').value.trim() || null }) })); toast(status.library_path_source === 'environment' ? t("Library saved. SNOWLLM_LIB still overrides it.") : t("Library saved for the next model load")); }); });
$('browse-storage').addEventListener('click', () => openDirectory('models-dir'));
$('add-storage').addEventListener('click', () => openDirectory('models-dir-add'));
$('close-directory').addEventListener('click', () => $('directory-dialog').close());
$('directory-form').addEventListener('submit', (event) => { event.preventDefault(); browseDirectory($('directory-path').value); });
$('directory-up').addEventListener('click', () => browseDirectory(directoryData.parent));
$('select-directory').addEventListener('click', () => {
  const field = $(directoryTarget === 'models-dir-add' ? 'models-dir' : directoryTarget);
  $('directory-dialog').close();
  if (field) {
    if (directoryTarget.startsWith('models-dir')) {
      const paths = storagePaths(field.value);
      if (directoryTarget === 'models-dir-add') { if (!paths.includes(directoryData.path)) paths.push(directoryData.path); }
      else paths[0] = directoryData.path;
      field.value = paths.join('\n');
    } else field.value = directoryData.path;
    field.focus(); field.dispatchEvent(new Event('change'));
  }
});
$('chat-form').addEventListener('submit', (event) => { event.preventDefault(); sendMessage(); });
$('chat-input').addEventListener('input', () => { saveChatDraft($('chat-input').value); syncChatControls(); });
$('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$('send-chat').disabled) $('chat-form').requestSubmit(); } });
$('stop-chat').addEventListener('click', () => activeChats.get(sessionId)?.controller.abort());
$('new-chat').addEventListener('click', newSession);
chatHistoryMedia.addEventListener('change', placeChatHistory);
$('open-chat-history').addEventListener('click', () => {
  renderSessions();
  $('chat-history-dialog').showModal();
  $('chat-session').querySelector('[aria-current]')?.focus();
});
$('chat-session').addEventListener('click', (event) => {
  const next = event.target.closest('button[data-session-id]')?.dataset.sessionId;
  if (!next) return;
  if ($('chat-history-dialog').open) $('chat-history-dialog').close();
  if (next === sessionId) return;
  saveConversation();
  selectSession(next);
});
$('chat-model').addEventListener('change', () => { const next = $('chat-model').value; if (conversation.length) newSession(); currentChatModel = next; renderChatModels(); updateThinkingLevels(); saveConversation(); controls(); });
$('open-chat-settings').addEventListener('click', () => $('chat-settings-dialog').showModal());
function deleteCurrentChat() {
  saveChatDraft('');
  sessions = sessions.filter(item => item.id !== sessionId);
  selectSession(sessions.at(-1)?.id);
}
const deleteChatDialog = $('delete-chat-dialog');
$('delete-chat').addEventListener('click', () => {
  if (activeChats.has(sessionId)) return;
  if (skipChatDeleteConfirmation) return deleteCurrentChat();
  deleteChatDialog.dataset.sessionId = sessionId;
  $('skip-chat-delete').checked = false;
  deleteChatDialog.showModal();
});
$('confirm-chat-delete').addEventListener('click', () => {
  if (deleteChatDialog.dataset.sessionId !== sessionId || activeChats.has(sessionId)) return;
  if ($('skip-chat-delete').checked) {
    skipChatDeleteConfirmation = true;
    try { sessionStorage.setItem('snowllm.skip-chat-delete-confirmation', '1'); } catch {}
  }
  deleteCurrentChat();
});
$('export-chat').addEventListener('click', () => {
  const text = conversation.flatMap(message => message.retry ? [message, message.retry] : [message]).map(message => {
    const role = message.role === 'user' ? 'User' : 'SnowLLM';
    return [role, message.reasoning && `[Thinking]\n${message.reasoning}`, message.content, chatResultText(message)]
      .filter(Boolean).join('\n\n');
  }).join('\n\n---\n\n');
  const url = URL.createObjectURL(new Blob([text], {type: 'text/plain;charset=utf-8'}));
  const link = element('a'); link.href = url; link.download = `snowllm-chat-${new Date().toISOString().slice(0, 10)}.txt`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
chatAction.addEventListener('click', () => sendMessage(true));
$('temperature').addEventListener('change', saveConversation);
$('max-tokens').addEventListener('change', saveConversation);
$('system-prompt').addEventListener('change', saveConversation);
$('thinking-level').addEventListener('change', event => { preferredThinkingLevel = event.target.value; saveConversation(); });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
  const text = t(button.dataset.prompt);
  $('chat-input').value = text;
  saveChatDraft(text);
  syncChatControls();
  $('chat-input').focus();
}));
if (sessions.some(item => item.id === sessionId)) selectSession(sessionId);
else saveConversation();
benchmarkUI = createBenchmarkUI({api, post, action, toast, getStatus: () => status, renderSettings: renderEffectiveSettings, refreshStatus});
taskUI = createTaskUI({api, post, action, refreshStatus, onBenchmark: id => { benchmarkUI.select(id); location.hash = 'benchmark'; }});
if (savedLoadSelection?.source === 'recipe' && savedLoadSelection.model.startsWith('b/')) {
  bundleCatalog = true;
  document.querySelectorAll('[data-catalog-source]').forEach(select => { select.value = 'bundle'; });
}
addLoadCard(savedLoadSelection); renderConversation(); updateThinkingLevels(); navigate();
async function poll() {
  if (polling) return;
  polling = true;
  try {
    await refreshStatus();
    if (page === 'workspace') await Promise.allSettled([refreshLogs(), refreshMetrics(), taskUI.refreshProgress()]);
    if (page === 'benchmark') await benchmarkUI.refresh();
  }
  catch { /* The connection banner communicates daemon availability. */ }
  finally { polling = false; }
}
poll(); setInterval(poll, 2000);
