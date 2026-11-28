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
let savedOffloadDir;
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
let cliDefaults = {};
const inferenceOptions = {'context-length': 'max-model-len', 'gpu-budget': 'max-gpu-mem',
  'vision-placement': 'vision-placement', 'host-budget': 'max-host-mem', 'disk-budget': 'max-disk-offload',
  'weight-map': 'device-map', 'disk-expert-slots': 'disk-offload-gpu-cache-experts',
  'max-num-seqs': 'max-num-seqs', 'max-num-batched-tokens': 'max-num-batched-tokens', 'kv-type': 'kv-cache-dtype'};
const loadFieldIds = ['load-id', 'load-target', ...Object.keys(inferenceOptions), 'extra-options'];
const specLabels = {num_spec: 'MTP depth', mtp_adaptive: 'Adaptive MTP', dflash: 'Draft checkpoint',
  dflash_block: 'Draft block size', dflash_p_min: 'Draft confidence threshold', dflash_adaptive: 'Adaptive verification',
  dspark_adaptive: 'Adaptive DSpark', mtp_window: 'Draft attention window', mtp_sinks: 'Attention sink tokens',
  mtp_prompt_lookup: 'Prompt lookup', mtp_tool_draft: 'Tool-call drafting'};
const manualPreset = '@manual';
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
function loadKey(card) {
  if (card.source === 'path') return card.fields['load-path'].value.trim();
  return card.fields['vision-head'].selectedOptions[0]?.dataset.recipe || card.fields['spec-method'].selectedOptions[0]?.dataset.recipe || '';
}
function selectLoadModel(card, spec) {
  card.restored = false;
  card.pendingRecipe = '';
  const model = loadKey(card);
  if (card.source === 'path') syncLocalModel(card);
  const modelName = card.fields['load-model'].value;
  if (model || (card.source === 'recipe' && modelName)) {
    const selection = {source: card.source, model: model || modelName, modelName, bundle: bundleCatalog,
      preset: card.fields['load-recipe'].value, precision: card.fields['load-precision'].value,
      method: card.fields['spec-method'].value, vision: card.fields['vision-head'].value};
    try { localStorage.setItem('snowllm.load-selection', JSON.stringify(selection)); } catch {}
  }
  if (spec) renderLoadHints(card, spec);
  else restoreLoadSettings(card);
}
function loadSettingsKey(card) {
  const ref = loadKey(card);
  const preset = card.source === 'recipe' && card.fields['load-recipe'].value;
  return preset && preset !== ref ? JSON.stringify([preset, ref]) : ref;
}
function saveLoadSettings(card, notify = true) {
  if (!loadKey(card)) { toast(t('Choose a recipe or enter a checkpoint directory.')); return; }
  loadSettings[loadSettingsKey(card)] = {...Object.fromEntries(loadFieldIds.map(id => [id, card.fields[id].value])),
    ...Object.fromEntries(Object.entries(card.specInputs).map(([key, input]) => [key, input.value]))};
  try { localStorage.setItem('snowllm.load-settings', JSON.stringify(loadSettings)); if (notify) toast(t('Load settings saved')); } catch { toast(t('Browser storage unavailable · this conversation is temporary')); }
}
function restoreLoadSettings(card) {
  const saved = loadSettings[loadSettingsKey(card)] || {};
  for (const id of loadFieldIds) card.fields[id].value = typeof saved[id] === 'string' ? saved[id] : '';
  if (!['gpu', 'npu'].includes(card.fields['load-target'].value)) card.fields['load-target'].value = 'gpu';
  renderLoadHints(card, saved);
}
function updateTargetSettings(card) {
  const npu = card.fields['load-target'].value === 'npu';
  card.fields['npu-load-hint'].hidden = !npu;
  for (const id of Object.keys(inferenceOptions)) card.fields[id].disabled = npu;
  card.fields['vision-placement'].disabled = npu || card.fields['vision-placement-row'].hidden;
  for (const input of Object.values(card.specInputs)) input.disabled = npu;
  for (const button of card.fields['spec-fields'].querySelectorAll('.input-action button')) button.disabled = npu;
  card.fields['spec-method'].disabled = npu || !card.fields['spec-method'].options.length;
}
function modelLoadRequest(card) {
  const fields = card.fields;
  const model = loadKey(card).trim();
  if (!model && card.source === 'recipe' && fields['load-recipe'].value === manualPreset) throw new Error(t('Choose a weight precision…'));
  if (!model) throw new Error(t('Choose a recipe or enter a checkpoint directory.'));
  const options = [];
  const preset = card.source === 'recipe' && recipes.find(row => row.id === fields['load-recipe'].value);
  if (card.source === 'recipe' && fields['load-recipe'].value === manualPreset) options.push('--no-recipe-preset');
  else if (preset?.inference && preset.id !== model) options.push('--recipe-preset', preset.preset);
  for (const [id, flag] of Object.entries(inferenceOptions)) {
    if (!fields[id].disabled && fields[id].value.trim()) options.push('--' + flag, fields[id].value.trim());
  }
  for (const [key, input] of Object.entries(card.specInputs)) {
    if (input.disabled) continue;
    if (input.type === 'number' && !input.validity.valid) throw new Error(t(specLabels[key]) + ' · ' + input.validationMessage);
    if (!input.value.trim()) continue;
    const flag = key.replaceAll('_', '-');
    if (key !== 'dflash' && (input.value === 'on' || input.value === 'off')) options.push((input.value === 'on' ? '--' : '--no-') + flag);
    else options.push('--' + flag, input.type === 'number' ? String(input.valueAsNumber) : input.value.trim());
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
    specInputs: {}, pendingModel: selection?.modelName || '', pendingPrecision: selection?.precision || '', pendingSpec: selection?.method ?? (selection?.preset && selection.model !== selection.preset ? selection.model : ''),
    pendingRecipe: selection?.source === 'recipe' ? selection.preset || selection.model : '', pendingVision: selection?.vision || ''};
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
  fields['load-model'].addEventListener('change', () => {
    renderPresetOptions(card);
    selectLoadModel(card);
  });
  fields['load-recipe'].addEventListener('change', () => {
    const previous = recipes.find(row => row.id === loadKey(card));
    const manual = fields['load-recipe'].value === manualPreset;
    renderPrecisionOptions(card, manual ? weightPrecision(previous) : '', manual ? fields['spec-method'].value : '', manual ? fields['vision-head'].value : '');
    selectLoadModel(card);
    resetOverrides();
  });
  function resetOverrides() {
    for (const id of [...Object.keys(inferenceOptions), 'extra-options']) fields[id].value = '';
    for (const input of Object.values(card.specInputs)) input.value = '';
  }
  function applyOverrides() {
    const spec = Object.fromEntries(Object.entries(card.specInputs).map(([key, input]) => [key, input.value]));
    selectLoadModel(card, spec);
    const block = card.specInputs.dflash_block;
    if (block && (block.validity.rangeOverflow || block.validity.rangeUnderflow)) block.value = '';
  }
  fields['load-precision'].addEventListener('change', () => {
    renderMethodOptions(card, fields['spec-method'].value, fields['vision-head'].value);
    applyOverrides();
  });
  fields['spec-method'].addEventListener('change', () => {
    visionOptions(fields['vision-head'], fields['spec-method'], fields['vision-head'].value);
    applyOverrides();
  });
  fields['vision-head'].addEventListener('change', applyOverrides);
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
    if (card.source === 'recipe') {
      renderPrecisionOptions(card);
      selectLoadModel(card);
    }
    for (const id of loadFieldIds) fields[id].value = '';
    resetOverrides();
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
  restoreLoadSettings(card);
  translatePage(); renderLoadCards();
  return card;
}
function configureModel(source, model, preset = model, method = '', precision = '', vision = '') {
  const card = loadCards.find(item => item.restored || !loadKey(item)) || addLoadCard();
  chooseSource(card, source);
  if (source === 'recipe') {
    const recipe = recipes.find(row => row.id === model);
    card.fields['load-model'].value = recipe?.model || model;
    renderPresetOptions(card, preset, method, precision, vision);
  } else card.fields['load-path'].value = model;
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
  if (config.device_map) entries.push(['Weight placement', config.device_map]);
  if (config.disk_offload_gpu_cache_experts > 0) entries.push(['GPU-cached experts per disk-offloaded layer', config.disk_offload_gpu_cache_experts]);
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
  const offloadDir = next.disk_offload_dir || '';
  if (savedOffloadDir !== offloadDir) $('offload-dir').value = offloadDir;
  savedOffloadDir = offloadDir;
  $('offload-override').hidden = !next.moe_cache_dirs_override;
  $('offload-override').textContent = next.moe_cache_dirs_override
    ? t('SNOWLLM_MOE_CACHE_DIRS sets the expert cache location to {0}', next.moe_cache_dirs_override) : '';
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
      recipes = catalog.value.recipes.flatMap(row => [row, ...(row.vision_options || []).map(option =>
        ({...row, ...option, recommended: false, vision_base: row.id}))]);
      cliDefaults = catalog.value.load_defaults || {};
      recipeTypes = catalog.value.recipe_types || {};
      $('catalog-error').hidden = true;
      renderRecipeOptions();
      renderRecipes(); loadCards.forEach(card => renderLoadHints(card));
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
  const groups = new Map();
  for (const model of rows) {
    const family = model.family || (model.type?.startsWith('qwen') ? 'Qwen' : recipeTypes[model.type]) || t('Other recipes');
    const key = family.toLowerCase();
    if (!groups.has(key)) groups.set(key, {type: key, label: family, rows: []});
    groups.get(key).rows.push(model);
  }
  return [...groups.values()];
}
function modelChoices(rows) {
  const models = new Map();
  for (const row of rows) {
    if (row.vision_base) continue;
    const name = row.model || row.id;
    if (!models.has(name)) models.set(name, {name, family: row.family, type: row.type, choices: []});
    models.get(name).choices.push(row);
  }
  for (const model of models.values()) {
    if (model.choices.some(row => row.model_id)) {
      model.choices = model.choices.filter(row => row.preset && Number.isInteger(row.recommend_order))
        .sort((a, b) => a.recommend_order - b.recommend_order);
    }
  }
  return [...models.values()];
}
function presetSelect(select, choices, selected, forLoad = false) {
  const available = choices.filter(row => !forLoad || row.servable);
  const current = available.find(row => row.id === selected) || available.find(row => row.recommended)
    || available.find(row => !row.model_id);
  const manual = available.some(row => row.variant && row.base_defaults);
  select.replaceChildren();
  if (!manual && !current) select.append(new Option(t("Choose a preset…"), ''));
  for (const row of choices) {
    let label = row.preset || (row.model_id ? row.id.split(':').slice(1).join(' · ') : row.id);
    if (row.recommended) label += ' · ' + t('Recommended');
    if (row.loadable) label += t(" · local");
    const option = new Option(label, row.id);
    option.disabled = forLoad && !row.servable;
    select.append(option);
  }
  if (manual) select.append(new Option(t('No preset'), manualPreset));
  select.value = manual && (!current || selected === manualPreset) ? manualPreset : current?.id || '';
  select.disabled = !choices.length;
}
function precisionRows(preset) {
  if (!preset?.variant) return preset ? [preset] : [];
  const rows = recipes.filter(row => row.model_id === preset.model_id && row.variant && !row.vision_base);
  const named = rows.filter(row => row.preset);
  const variants = new Set(named.map(row => row.variant));
  const precisions = new Set(named.map(weightPrecision));
  return [preset, ...rows.filter(row => row !== preset && (variants.has(row.variant) || !precisions.has(weightPrecision(row))))];
}
function weightPrecision(row) {
  const version = row?.variant?.match(/-(pre-)?ud(\d+(?:\.\d+)+)$/i);
  return version ? `${version[1] || ''}UD${version[2]} · ${row.precision}` : row?.precision;
}
function precisionLabel(precision) {
  return precision?.replace(/^pre-UD\d+(?:\.\d+)+ · /i, '');
}
function selectionRow(selection, modelName) {
  return selection === manualPreset
    ? recipes.find(row => row.model === modelName && row.preset && row.variant)
    : recipes.find(row => row.id === selection);
}
function precisionOptions(select, preset, selected = '', manual = false) {
  select.replaceChildren();
  if (!preset?.variant) return;
  const rows = precisionRows(preset);
  const precisions = new Set(rows.map(weightPrecision));
  if (selected && !precisions.has(selected)) {
    // Older WebUI selections named only the precision and used the preset's weight revision.
    selected = weightPrecision(rows.find(row => row.preset && row.precision === selected));
  }
  if (!manual) select.append(new Option(t('Default ({0})', precisionLabel(weightPrecision(preset))), ''));
  else if (precisions.size > 1) select.append(new Option(t('Choose a weight precision…'), ''));
  for (const precision of precisions) {
    select.append(new Option(precisionLabel(precision), precision));
    if (selected === precision) select.value = selected;
  }
}
function methodOptions(select, preset, selected = '', precision = '', manual = false) {
  select.replaceChildren();
  if (!preset || (manual && !precision)) return;
  const labels = {mtp: 'MTP', dflash: 'DFlash', dspark: 'DSpark', none: 'Off'};
  const legacyMethod = spec => ({'mtp-adaptive': 'mtp', dflash2: 'dflash'}[spec] || spec);
  const method = row => row.spec_label || legacyMethod(row.spec);
  const choices = new Map();
  const rows = precisionRows(preset).filter(row => weightPrecision(row) === (precision || weightPrecision(preset)));
  for (const row of rows) {
    const key = method(row);
    const current = choices.get(key);
    if (!current || (current.variant !== preset.variant && row.variant === preset.variant)) choices.set(key, row);
  }
  const fallback = choices.get(method(preset)) || rows.find(row => row.recommended) || rows[0];
  if (!fallback) return;
  if (selected && !choices.has(selected)) {
    const previous = rows.find(row => row.id === selected || legacyMethod(row.spec) === selected);
    if (previous) selected = method(previous);
  }
  const label = row => row.spec_label || t(labels[method(row)] || row.spec || 'Default');
  if (!manual) {
    const inherited = new Option(t('Default ({0})', label(fallback)), '');
    inherited.dataset.recipe = fallback.id;
    select.append(inherited);
  } else if (!choices.has(selected)) {
    selected = choices.has('none') ? 'none' : method(fallback);
  }
  const rank = key => ({mtp: 0, dflash: 1, dflash2: 1, dspark: 2, none: 4}[key?.split('@')[0].toLowerCase()] ?? 3);
  for (const [key, row] of [...choices].sort(([a], [b]) => rank(a) - rank(b))) {
    const option = new Option(label(row), key);
    option.dataset.recipe = row.id;
    select.append(option);
    if (selected === key) select.value = key;
  }
}
function visionOptions(select, method, selected = '') {
  select.replaceChildren();
  const baseId = method.selectedOptions[0]?.dataset.recipe;
  const base = recipes.find(row => row.id === baseId);
  if (!base?.vision_heads) return;
  const choices = [base, ...recipes.filter(row => row.vision_base === baseId)];
  const label = key => key === 'none' ? t('Off') : key;
  const inherited = new Option(t('Default ({0})', label(base.vision_head)), '');
  inherited.dataset.recipe = base.id;
  select.append(inherited);
  for (const row of choices) {
    const key = row.vision_head;
    const option = new Option(label(key), key);
    option.dataset.recipe = row.id;
    select.append(option);
  }
  select.value = choices.some(row => row.vision_head === selected) ? selected : '';
}
function renderMethodOptions(card, selected, vision) {
  const fields = card.fields;
  const selection = fields['load-recipe'].value;
  methodOptions(fields['spec-method'], selectionRow(selection, fields['load-model'].value), selected,
    fields['load-precision'].value, selection === manualPreset);
  visionOptions(fields['vision-head'], fields['spec-method'], vision);
}
function renderPrecisionOptions(card, precision, method, vision) {
  const fields = card.fields;
  const selection = fields['load-recipe'].value;
  precisionOptions(fields['load-precision'], selectionRow(selection, fields['load-model'].value), precision, selection === manualPreset);
  renderMethodOptions(card, method, vision);
}
function renderPresetOptions(card, selected = '', method = '', precision = '', vision = '') {
  const name = card.fields['load-model'].value;
  const choices = modelChoices(recipes).find(model => model.name === name)?.choices || [];
  presetSelect(card.fields['load-recipe'], choices, selected, true);
  renderPrecisionOptions(card, precision, method, vision);
}
function renderRecipeOptions() {
  const models = modelChoices(recipes);
  for (const card of loadCards) {
    const select = card.fields['load-model'];
    const previous = card.fields['load-recipe'].value;
    const method = card.fields['spec-method'].value;
    const previousRef = loadKey(card);
    const selected = previous || card.pendingRecipe;
    const row = recipes.find(item => item.id === selected);
    const name = row?.model || row?.id || select.value || card.pendingModel;
    select.replaceChildren(new Option(t("Choose a model…"), ''));
    for (const group of groupRecipes(models)) {
      const options = document.createElement('optgroup');
      options.label = group.label;
      options.dataset.recipeType = group.type;
      for (const model of group.rows) options.append(new Option(model.name, model.name));
      select.append(options);
    }
    select.value = models.some(model => model.name === name) ? name : '';
    renderPresetOptions(card, selected, method || card.pendingSpec, card.fields['load-precision'].value || card.pendingPrecision,
      card.fields['vision-head'].value || card.pendingVision);
    if (select.value) { card.pendingModel = ''; card.pendingRecipe = ''; card.pendingSpec = ''; card.pendingPrecision = ''; card.pendingVision = ''; }
    if (card.source === 'recipe' && loadKey(card) !== previousRef) restoreLoadSettings(card);
  }
}
const libraryPresets = new Map();
const libraryOverrides = new Map();
function renderRecipes() {
  const search = $('model-search').value.trim().toLowerCase();
  const visible = [];
  for (const model of modelChoices(recipes)) {
    const variants = model.choices[0]?.variant;
    const rows = variants ? precisionRows(model.choices[0]) : model.choices;
    const matches = rows.filter(row => (!$('installed-only').checked || row.loadable || row.vision_options?.some(option => option.loadable))
      && [model.name, model.family, row.id, weightPrecision(row), row.type, recipeTypes[row.type]].join(' ').toLowerCase().includes(search));
    if (matches.length) visible.push(variants ? model : {...model, choices: matches});
  }
  $('recipe-count').textContent = t("{0} models", visible.length);
  const list = $('recipe-list');
  list.replaceChildren();
  for (const {type, label, rows} of groupRecipes(visible)) {
    const section = element('section', 'recipe-group');
    section.dataset.recipeType = type;
    const title = element('h2', 'recipe-group-title', label);
    title.id = `recipe-group-${encodeURIComponent(type)}`;
    section.setAttribute('aria-labelledby', title.id);
    const grid = element('div', 'recipe-grid');
    grid.append(...rows.map(modelCard));
    section.append(title, grid);
    list.append(section);
  }
  if (!visible.length) list.append(element('p', 'empty-state', t("No recipes match your filters.")));
  controls();
}
function modelCard(model) {
  const card = element('article', 'recipe-card');
  const select = document.createElement('select');
  select.setAttribute('aria-label', t('Preset for {0}', model.name));
  presetSelect(select, model.choices, libraryPresets.get(model.name));
  const presetLabel = element('label', '', t('Preset'));
  presetLabel.append(select);
  const overrides = element('fieldset', 'preset-overrides');
  const legend = element('legend');
  overrides.append(legend);
  const precision = document.createElement('select');
  const method = document.createElement('select');
  const vision = document.createElement('select');
  for (const [field, caption] of [[precision, 'Weight precision'], [vision, 'Vision head'], [method, 'Draft method']]) {
    if (field === method) {
      method.id = 'library-method-' + encodeURIComponent(model.name);
      overrides.append(specCaption(caption, method.id), method);
      continue;
    }
    const label = element('label', '', t(caption));
    label.append(field);
    overrides.append(label);
  }
  function render(saved = libraryOverrides.get(model.name)) {
    card.replaceChildren(element('h3', '', model.name), presetLabel);
    const manual = select.value === manualPreset;
    const preset = selectionRow(select.value, model.name);
    legend.textContent = t(manual ? 'Model settings' : 'Preset overrides');
    precisionOptions(precision, preset, saved?.precision, manual);
    methodOptions(method, preset, saved?.method, precision.value, manual);
    visionOptions(vision, method, saved?.vision);
    vision.parentElement.hidden = !vision.options.length;
    if (preset?.variant) card.append(overrides);
    const recipe = recipes.find(row => row.id === (vision.selectedOptions[0]?.dataset.recipe || method.selectedOptions[0]?.dataset.recipe));
    card.classList.toggle('local', Boolean(recipe?.loadable));
    if (recipe) renderRecipeDetails(card, recipe, select.value, method.value, precision.value, vision.value);
  }
  for (const field of [precision, method, vision]) field.addEventListener('change', () => {
    render({precision: precision.value, method: method.value, vision: vision.value});
    libraryOverrides.set(model.name, {precision: precision.value, method: method.value, vision: vision.value});
    controls();
  });
  select.addEventListener('change', () => {
    libraryPresets.set(model.name, select.value);
    const previous = recipes.find(row => row.id === method.selectedOptions[0]?.dataset.recipe);
    if (select.value === manualPreset) libraryOverrides.set(model.name, {precision: weightPrecision(previous), method: method.value, vision: vision.value});
    else libraryOverrides.delete(model.name);
    render();
    controls();
  });
  render();
  return card;
}
function effectiveDefaults(recipe, selection) {
  if (selection === manualPreset) return {...cliDefaults, ...recipe?.base_defaults};
  const preset = recipes.find(row => row.id === selection);
  if (preset?.inference && recipe?.base_defaults && recipe.id !== preset.id) {
    const draft = recipe.spec === preset.spec ? preset.draft_params : {};
    return {...cliDefaults, ...recipe.base_defaults, ...draft, ...preset.inference};
  }
  return {...cliDefaults, ...recipe?.defaults};
}
function renderRecipeDetails(card, recipe, preset, method, precision, vision) {
  const top = element('div', 'recipe-card-top');
  const partial = recipe.partial ?? recipe.exists;
  const availability = element('span', `availability-badge ${recipe.loadable ? 'local' : partial ? 'partial' : 'remote'}`);
  availability.append(element('span', 'availability-icon', recipe.loadable ? '✓' : partial ? '◐' : '↓'),
    element('span', '', t(recipe.loadable ? 'Available locally' : partial ? 'Partial copy' : 'Not downloaded')));
  top.append(element('span', 'pill', precisionLabel(weightPrecision(recipe)) || t('Model')), availability);
  card.append(top);
  if (recipe.model_id) {
    const settings = element('details', 'effective-settings');
    settings.append(element('summary', '', t('Effective inference settings')));
    const defaults = effectiveDefaults(recipe, preset);
    const labels = {max_model_len: 'Context length', max_num_seqs: 'Max concurrent requests',
      max_num_batched_tokens: 'Prefill chunk size', kv_cache_dtype: 'KV cache precision'};
    for (const key of Object.keys(recipe.parameters || {})) labels[key] = specLabels[key] || key;
    const list = element('dl', 'runtime-list model-runtime');
    for (const [key, label] of Object.entries(labels)) {
      list.append(element('dt', '', t(label)), element('dd', '', settingValue(key, defaults[key], recipe.spec_label || recipe.spec)));
    }
    settings.append(list);
    card.append(settings);
  }
  if (!recipe.servable) card.append(element('span', 'pill', recipe.status === 'coming soon' ? t('Coming soon') : t('Requires {0}', recipe.requires)));
  const footer = element('div', 'recipe-footer');
  footer.append(element('span', 'recipe-size', t("{0} on disk", bytes(recipe.bytes))));
  const buttons = element('div', 'download-actions');
  if (recipe.loadable && recipe.servable) {
    const use = element('button', 'button secondary', t("Configure & load")); use.dataset.use = recipe.id;
    use.addEventListener('click', () => configureModel('recipe', recipe.id, preset, method, precision, vision)); buttons.append(use);
  }
  if (recipe.repo === 'TBD') {
    footer.append(element('span', 'field-hint', t('Local weights only')), buttons);
    card.append(footer);
    return;
  }
  const download = element('button', `button ${recipe.loadable ? 'secondary' : 'primary'}`, recipe.loadable ? t("Update files") : t("Download"));
  download.dataset.download = recipe.id;
  download.addEventListener('click', () => action(download, async () => { renderStatus(await post('/daemon/pull', { recipe: recipe.id })); toast(t("Downloading {0}", recipe.id)); }));
  buttons.append(download); footer.append(buttons); card.append(footer);
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
function settingValue(key, value, spec) {
  if (key === 'dflash' && value == null) return t('No draft checkpoint');
  if (key === 'dflash_block' && value === 0) return t(['dflash', 'dflash2'].includes(spec?.split('@')[0].toLowerCase())
    ? '0 (engine default: 8)' : '0 (checkpoint default)');
  if (value == null) return t('Automatic');
  if (typeof value === 'boolean') return t(value ? 'On' : 'Off');
  return String(value);
}
let visibleSpecTip;
function hideSpecTip() {
  if (visibleSpecTip) visibleSpecTip.hidden = true;
  visibleSpecTip = null;
}
document.addEventListener('pointerdown', event => {
  if (!visibleSpecTip?.parentElement.contains(event.target)) hideSpecTip();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && visibleSpecTip?.isConnected) {
    hideSpecTip();
    event.preventDefault();
    event.stopPropagation();
  }
}, true);
function specCaption(caption, fieldId, helpKey = caption) {
  const group = element('span', 'spec-caption');
  const label = element('label', '', t(caption));
  label.htmlFor = fieldId;
  const button = element('button', 'spec-help', '?');
  button.type = 'button';
  button.setAttribute('aria-label', t('About {0}', t(caption)));
  const tip = element('span', 'spec-tooltip', t(helpKey + ' help'));
  tip.id = fieldId + '-help';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  button.setAttribute('aria-describedby', tip.id);
  function show() {
    hideSpecTip();
    visibleSpecTip = tip;
    tip.hidden = false;
  }
  for (const event of ['pointerenter', 'focus', 'click']) button.addEventListener(event, show);
  group.addEventListener('pointerleave', () => {
    if (visibleSpecTip === tip && document.activeElement !== button) hideSpecTip();
  });
  button.addEventListener('blur', () => { if (visibleSpecTip === tip) hideSpecTip(); });
  group.append(label, button, tip);
  return group;
}
function renderSpecFields(card, recipe, defaults, saved) {
  const container = card.fields['spec-fields'];
  const previous = saved || Object.fromEntries(Object.entries(card.specInputs).map(([key, field]) => [key, field.value]));
  container.replaceChildren();
  card.specInputs = {};
  if (!recipe && card.source === 'recipe') return;
  const rules = recipe?.parameters;
  for (const [key, caption] of Object.entries(specLabels)) {
    if (rules && !Object.hasOwn(rules, key)) continue;
    const rule = rules?.[key] || {};
    const value = defaults[key];
    const boolean = key.endsWith('_adaptive') || ['mtp_prompt_lookup', 'mtp_tool_draft'].includes(key);
    const hint = settingValue(key, value, recipe?.spec_label || recipe?.spec);
    const input = document.createElement(boolean || rule.values ? 'select' : 'input');
    if (input.tagName === 'SELECT') {
      input.append(new Option(t('Default ({0})', hint), ''));
      for (const v of rule.values || [true, false]) input.append(new Option(settingValue(key, v), boolean ? (v ? 'on' : 'off') : String(v)));
    } else {
      input.type = key === 'dflash' ? 'text' : 'number';
      if (input.type === 'number') {
        input.min = key === 'num_spec' ? 0 : rule.min ?? 0;
        if (rule.max != null) input.max = rule.max;
        if (key === 'dflash_p_min') { input.max = 1; input.step = 'any'; }
      }
      input.placeholder = hint;
    }
    const id = key.replaceAll('_', '-');
    input.id = card.fields['spec-method'].id + '-' + id;
    const wrapper = element('div');
    const inkling = key === 'dflash_adaptive' && recipe?.spec?.startsWith('dspark');
    wrapper.append(specCaption(caption, input.id, inkling ? 'DSpark verification' : caption));
    if (key === 'dflash_block' && rule.suggestions?.length && input.type === 'number') {
      const controls = element('div', 'input-action');
      controls.append(input);
      for (const size of rule.suggestions) {
        const button = element('button', 'button secondary', String(size));
        button.type = 'button';
        button.setAttribute('aria-label', t('Use draft block size {0}', size));
        button.addEventListener('click', () => { input.value = String(size); input.focus(); });
        controls.append(button);
      }
      const help = element('p', 'field-hint', t('Enter a custom integer from {0} to {1}. Use 0 for the engine default (8).', rule.min, rule.max));
      help.id = input.id + '-hint';
      input.setAttribute('aria-describedby', help.id);
      wrapper.append(controls, help);
    } else wrapper.append(input);
    container.append(wrapper);
    input.value = previous[key] ?? previous[key === 'num_spec' ? 'spec-depth' : id] ?? '';
    card.specInputs[key] = input;
  }
}
function renderLoadHints(card, saved) {
  const fields = card.fields;
  const recipe = card.source === 'recipe' ? recipes.find(row => row.id === loadKey(card)) : null;
  const selection = card.source === 'recipe' ? fields['load-recipe'].value : '';
  const manual = selection === manualPreset;
  const defaults = effectiveDefaults(recipe, selection);
  fields['precision-row'].hidden = !manual && !recipe?.variant;
  fields['vision-row'].hidden = card.source !== 'recipe' || !fields['vision-head'].options.length;
  fields['vision-placement-row'].hidden = card.source === 'recipe' && !recipe?.modalities?.includes('vision') &&
    !recipe?.modalities?.includes('image') && (!recipe?.vision_head || recipe.vision_head === 'none');
  fields['override-title'].hidden = fields['precision-row'].hidden;
  fields['override-title'].dataset.i18n = manual ? 'Model settings' : 'Preset overrides';
  fields['override-title'].textContent = t(fields['override-title'].dataset.i18n);
  fields['method-row'].hidden = !recipe;
  fields['method-caption'].replaceChildren(specCaption('Draft method', fields['spec-method'].id));
  const checkpoint = recipe && defaults.dflash;
  fields['draft-checkpoint'].hidden = !checkpoint;
  fields['draft-checkpoint'].textContent = checkpoint ? t('Draft checkpoint · {0}',
    checkpoint.replace(/^@model(?=\/|$)/, () => recipe.path || '@model')) : '';
  fields['load-recipe'].classList.toggle('local-selected', Boolean(recipe?.loadable));
  fields['recipe-description'].textContent = recipe ? t(recipe.loadable ? 'Local copy available.'
    : recipe.repo === 'TBD' ? 'Local weights only' : 'Download this recipe from Models before loading.') : '';
  fields['context-length'].placeholder = defaults.max_model_len ?? '';
  fields['max-num-seqs'].placeholder = defaults.max_num_seqs ?? '';
  fields['max-num-batched-tokens'].placeholder = defaults.max_num_batched_tokens ?? '';
  fields['gpu-budget'].placeholder = defaults.max_gpu_mem_gib != null ? defaults.max_gpu_mem_gib + ' GiB' : defaults.max_gpu_mem_util ?? '';
  fields['host-budget'].placeholder = defaults.max_host_mem_gib != null ? defaults.max_host_mem_gib + ' GiB' : defaults.max_host_mem_util ?? '';
  fields['weight-map'].placeholder = defaults.device_map || 'auto';
  fields['disk-expert-slots'].placeholder = defaults.moe_cache_experts || '';
  fields['disk-budget'].placeholder = defaults.max_disk_mem_gib != null ? defaults.max_disk_mem_gib + ' GiB' : t('e.g. 16GiB');
  fields['kv-type'].options[0].textContent = t('Default ({0})', (defaults.kv_cache_dtype || 'auto').toUpperCase());
  renderSpecFields(card, recipe, defaults, saved);
  updateTargetSettings(card);
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
  renderRecipeOptions(); renderRecipes(); renderLocal(); loadCards.forEach(card => renderLoadHints(card)); renderLoadCards();
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
  recipes = []; renderRecipeOptions(); renderRecipes(); loadCards.forEach(card => renderLoadHints(card));
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
$('offload-dir-form').addEventListener('submit', (event) => {
  event.preventDefault();
  action(event.submitter, async () => {
    const next = await api('/daemon/config', { method: 'PUT', body: JSON.stringify({ disk_offload_dir: $('offload-dir').value.trim() || null }) });
    renderStatus(next);
    $('offload-dir').value = next.disk_offload_dir || '';
    toast(t('Offload directory saved'));
  });
});
$('browse-offload').addEventListener('click', () => openDirectory('offload-dir'));
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
if (savedLoadSelection?.source === 'recipe' && (savedLoadSelection.bundle || savedLoadSelection.model.startsWith('b/'))) {
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
