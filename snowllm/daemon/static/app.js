// SPDX-License-Identifier: Apache-2.0
import { t, translatePage, language, setLanguage } from './i18n.js';
import { parseOptions, readEvents } from './stream.js';
import { thinkingProfile, thinkingParameters } from './thinking.js';

const $ = (id) => document.getElementById(id);
const stateLabels = () => ({ idle: t("Idle"), draining: t("Finishing requests"), loading: t("Loading"), ready: t("Ready"), failed: t("Failed") });
let savedRecipesUrl, savedLibraryPath;
let status = null, recipes = [], localModels = [], source = 'recipe', page = 'workspace';
let bundleCatalog = false;
let toastTimer, chatController = null, directoryTarget = '', directoryData = null;
let polling = false, catalogLoading = false, modelRoot = null, lastDownloadExit = null;
let catalogRequest = 0;
let savedRoot;
let aliases = [], currentChatModel = null;
let preferredThinkingLevel = '';
let conversation = [];
let sessions = [], sessionId = crypto.randomUUID(), loadSettings = {};
const loadFieldIds = ['context-length', 'gpu-budget', 'spec-depth', 'kv-type', 'extra-options'];
try { loadSettings = JSON.parse(localStorage.getItem('snowllm.load-settings') || '{}'); if (!loadSettings || Array.isArray(loadSettings) || typeof loadSettings !== 'object') loadSettings = {}; } catch {}
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
function loadKey() { return source === 'recipe' ? $('load-recipe').value : $('load-path').value.trim(); }
function saveLoadSettings(notify = true) {
  if (!loadKey()) { toast(t('Choose a recipe or enter a checkpoint directory.')); return; }
  loadSettings[loadKey()] = Object.fromEntries(loadFieldIds.map(id => [id, $(id).value]));
  try { localStorage.setItem('snowllm.load-settings', JSON.stringify(loadSettings)); if (notify) toast(t('Load settings saved')); } catch { toast(t('Browser storage unavailable · this conversation is temporary')); }
}
function restoreLoadSettings() {
  const saved = loadSettings[loadKey()] || {};
  for (const id of loadFieldIds) $(id).value = typeof saved[id] === 'string' ? saved[id] : '';
}
function renderSessions() {
  $('chat-session').replaceChildren(...sessions.map(item => new Option(item.title || t('New conversation'), item.id)));
  if (!sessions.some(item => item.id === sessionId)) $('chat-session').append(new Option(t('New conversation'), sessionId));
  $('chat-session').value = sessionId;
  $('chat-session').disabled = Boolean(chatController);
  $('delete-chat').disabled = Boolean(chatController);
  $('new-chat').disabled = Boolean(chatController);
  $('regenerate-chat').hidden = Boolean(chatController) || conversation.at(-1)?.role !== 'assistant';
  $('regenerate-chat').disabled = status?.state !== 'ready' || Boolean(currentChatModel && currentChatModel !== status?.model);
}
function selectSession(id) {
  const item = sessions.find(item => item.id === id);
  sessionId = item?.id || crypto.randomUUID();
  conversation = item?.messages || [];
  currentChatModel = item?.model || null;
  $('system-prompt').value = item?.system || '';
  $('temperature').value = item?.temperature || '';
  $('max-tokens').value = item?.maxTokens || '2048';
  preferredThinkingLevel = item?.thinkingLevel || '';
  updateThinkingLevels(); renderConversation(); saveConversation();
}
function newSession() {
  saveConversation();
  sessionId = crypto.randomUUID(); conversation = []; currentChatModel = status?.model || null;
  saveConversation(); renderConversation();
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast(t('Copied')); } catch { toast(t('Copy failed')); }
}
function renderEffectiveSettings(next) {
  $('effective-settings').replaceChildren();
  const config = next.runtime_config;
  if (!config) { $('effective-settings').append(element('div', 'muted', '—')); return; }
  const entries = [['Context per request', config.max_model_len], ['Shared KV capacity', config.kv_capacity_tokens], ['Maximum concurrent requests', config.max_num_seqs],
    ['GPU memory fraction', Number.isFinite(config.max_gpu_mem_util) ? config.max_gpu_mem_util.toFixed(4) : '—'], ['KV cache precision', config.kv_cache_dtype],
    ['Offload ratio', Number.isFinite(config.offload_ratio) ? `${(config.offload_ratio * 100).toFixed(2)}%` : '—'],
    ['MTP depth', config.num_spec], ['DFlash drafter', config.dflash_path || '—'], ['DFlash block', config.dflash_block ?? '—'],
    ['Execution mode', t(config.enforce_eager ? 'Eager' : 'CUDA graphs')], ['Prefill chunk', config.prefill_chunk]];
  if (Number.isFinite(config.max_gpu_mem_bytes)) entries.push(['GPU memory limit', bytes(config.max_gpu_mem_bytes)]);
  if (Number.isFinite(config.max_disk_mem_gib)) entries.push(['Disk offload limit', bytes(config.max_disk_mem_gib * 2 ** 30)]);
  if (Number.isFinite(config.max_host_mem_bytes)) entries.push(['Host memory limit', bytes(config.max_host_mem_bytes)]);
  for (const [label, value] of entries) { const row = element('div'); row.append(element('dt', '', t(label)), element('dd', '', String(value ?? '—'))); $('effective-settings').append(row); }
}
function element(tag, className = '', text = '') {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
}
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5500); }
function bytes(value) { if (!Number.isFinite(value) || value <= 0) return '—'; return `${(value / 2 ** 30).toFixed(1)} GiB`; }
function transition() { return status && ['loading', 'draining'].includes(status.state); }
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
  page = ['workspace', 'models', 'chat', 'settings'].includes(next) ? next : 'workspace';
  for (const name of ['workspace', 'models', 'chat', 'settings']) $('page-' + name).hidden = name !== page;
  document.querySelectorAll('nav a').forEach(link => { link.classList.toggle('active', link.dataset.page === page); if (link.dataset.page === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  $('page-name').textContent = { workspace: t("Overview"), models: t("Models"), chat: t("Chat"), settings: t("Settings") }[page];
  if (page === 'workspace') refreshLogs().catch(error => toast(error.message));
  if (page === 'settings') storageSpace();
}
function controls() {
  const online = !$('connection-banner').hidden ? false : Boolean(status);
  $('load-button').disabled = !online || transition();
  $('unload-button').disabled = !online || transition() || !status?.model;
  $('send-chat').disabled = !online || status?.state !== 'ready' || Boolean(chatController);
  $('cancel-load').hidden = !transition();
  $('cancel-load').disabled = !online;
  $('retry-load').disabled = !online;
  $('retry-load').hidden = status?.state !== 'failed' || !status?.last_load;
  renderSessions();
  document.querySelectorAll('[data-download]').forEach(button => button.disabled = !online || status?.download?.exit_code === null);
  document.querySelectorAll('[data-use]').forEach(button => button.disabled = !online || transition() || Boolean(chatController));
}
function renderLoadError(next) {
  const banner = $('model-error');
  banner.hidden = !next.error;
  const title = next.error_kind === 'capacity' ? t('Not enough memory for these settings') : t('Model failed to load or stopped');
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
function renderStatus(next) {
  const oldModel = status?.model;
  const oldState = status?.state;
  status = next;
  renderUpdate(next.update);
  renderEffectiveSettings(next);
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
  $('engine-detail').textContent = { idle: t("Ready to load a model"), draining: t("Waiting for active requests"), loading: t("Reading weights and initializing"), ready: t("Accepting inference requests"), failed: t("See the error above") }[next.state] || '';
  $('current-model').textContent = next.model || t("None loaded");
  $('active-count').textContent = next.active_requests;
  $('runtime-state').textContent = stateLabels()[next.state] || next.state;
  $('runtime-pid').textContent = next.pid || '—';
  $('storage-preview').textContent = next.models_dir;
  $('config-source').textContent = { 'command-line': t("Command line"), environment: t("Environment"), config: t("Saved configuration"), default: t("Default") }[next.models_dir_source];
  renderLoadError(next);
  $('config-file').textContent = next.config_file || '';
  const modelOverride = ['command-line', 'environment'].includes(next.models_dir_source);
  const recipeOverride = next.recipes_url_source === 'environment';
  const libraryOverride = next.library_path_source === 'environment';
  const saved = next.saved_models_dir ?? (modelOverride ? '' : next.models_dir);
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
    ['storage-override', modelOverride, next.models_dir_source === 'environment' ? 'SNOWLLM_MODELS' : '--models-dir', next.models_dir],
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
  if (modelRoot !== next.models_dir) { modelRoot = next.models_dir; refreshCatalog(false); storageSpace(); }
  if (next.state === 'ready' && (oldState !== 'ready' || oldModel !== next.model || !aliases.length)) refreshAliases();
  if (oldModel !== next.model) { aliases = []; updateThinkingLevels(); }
  if (next.state !== 'ready') $('chat-model-label').textContent = next.model ? `${next.model} · ${stateLabels()[next.state]}` : t("Load a model to start a conversation.");
  if (oldModel && oldModel !== next.model && chatController) chatController.abort();
  controls();
}
async function refreshStatus() {
  try { renderStatus(await api('/daemon/status')); }
  catch (error) { $('connection-banner').hidden = false; $('connection-dot').className = 'dot red'; $('connection-title').textContent = t("Disconnected"); $('connection-detail').textContent = t("Reconnecting\u2026"); controls(); throw error; }
}
async function refreshAliases() {
  try {
    const result = await api('/daemon/inference/models'); aliases = result.data || [];
    $('chat-model-label').textContent = aliases[0]?.id || status.model;
    updateThinkingLevels();
  } catch { aliases = []; updateThinkingLevels(); }
}
function updateThinkingLevels() {
  const profile = thinkingProfile(aliases[0]?.thinking);
  $('thinking-level').replaceChildren(...profile.options.map(([value, label]) => new Option(value.startsWith('effort:') ? label : t(label), value)));
  $('thinking-level').value = profile.options.some(([value]) => value === preferredThinkingLevel) ? preferredThinkingLevel : '';
}
async function refreshMetrics() {
  const ready = status?.state === 'ready';
  let data = null;
  if (ready) { try { data = await api('/daemon/metrics?format=json'); } catch {} }
  const engine = data?.engine;
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
  const result = await api(`/daemon/logs?download=${$('log-kind').value === 'download'}`);
  const text = result.text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\r(?!\n)/g, '\n');
  const atBottom = $('logs').scrollTop + $('logs').clientHeight >= $('logs').scrollHeight - 30;
  $('logs').textContent = text || t("No activity.");
  if (atBottom) $('logs').scrollTop = $('logs').scrollHeight;
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
  const selected = $('load-recipe').value;
  const root = status?.models_dir;
  try {
    const [catalog, local] = await Promise.allSettled([api(`/daemon/recipes?refresh=${refresh}&bundle=${bundleCatalog}`), api('/daemon/local-models')]);
    if (request !== catalogRequest) return;
    if (root !== status?.models_dir) return;
    if (catalog.status === 'fulfilled') {
      recipes = catalog.value.recipes; $('catalog-error').hidden = true;
      renderRecipeOptions(selected);
      renderRecipes(); recipeDescription();
    } else {
      $('catalog-error').hidden = false; $('catalog-error').textContent = t("Catalog unavailable: {0}. You can still load a local checkpoint from Overview.", catalog.reason.message);
      if (!recipes.length) { $('recipe-list').replaceChildren(element('p', 'empty-state', t("Could not load the catalog. Try refreshing, or use a local model."))); $('load-recipe').replaceChildren(new Option(t("Catalog unavailable \u00b7 use Local path"), '')); }
    }
    if (local.status === 'fulfilled') { localModels = local.value.models; renderLocal(); }
    else $('local-list').replaceChildren(element('p', 'muted', local.reason.message));
  } finally {
    if (request === catalogRequest) {
      catalogLoading = false; $('refresh-recipes').disabled = false;
      if (root !== status?.models_dir) refreshCatalog(false);
    }
  }
}
function renderRecipeOptions(selected = $('load-recipe').value) {
      $('load-recipe').replaceChildren(new Option(t("Choose a recipe\u2026"), ''));
      for (const recipe of recipes) {
        const option = new Option(`${recipe.id}${recipe.loadable ? t(" \u00b7 local") : ''}${recipe.status === 'coming soon' ? ` · ${t('Coming soon')}` : !recipe.servable ? t(" \u00b7 upgrade required") : ''}`, recipe.id);
        option.classList.toggle('local', Boolean(recipe.loadable));
        option.disabled = !recipe.servable; $('load-recipe').append(option);
      }
      if (recipes.some(recipe => recipe.id === selected)) $('load-recipe').value = selected;
}
function renderRecipes() {
  const search = $('model-search').value.toLowerCase();
  const visible = recipes.filter(recipe => (!search || [recipe.id, recipe.model, recipe.precision].join(' ').toLowerCase().includes(search)) && (!$('installed-only').checked || recipe.loadable));
  $('recipe-count').textContent = t("{0} recipes", visible.length);
  $('recipe-list').replaceChildren();
  for (const recipe of visible) {
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
      use.addEventListener('click', () => { chooseSource('recipe'); $('load-recipe').value = recipe.id; restoreLoadSettings(); recipeDescription(); location.hash = 'workspace'; $('load-recipe').focus(); }); buttons.append(use);
    }
    const download = element('button', `button ${recipe.loadable ? 'secondary' : 'primary'}`, recipe.loadable ? t("Update files") : t("Download"));
    download.dataset.download = recipe.id;
    download.addEventListener('click', () => action(download, async () => { renderStatus(await post('/daemon/pull', { recipe: recipe.id })); toast(t("Downloading {0}", recipe.id)); }));
    buttons.append(download); footer.append(buttons); card.append(footer); $('recipe-list').append(card);
  }
  if (!visible.length) $('recipe-list').append(element('p', 'empty-state', t("No recipes match your filters.")));
  controls();
}
function renderLocal() {
  const selected = $('local-model').value;
  $('nav-count').textContent = localModels.length;
  $('local-list').replaceChildren(); $('local-model').replaceChildren(new Option(t("Choose a local model\u2026"), ''));
  for (const model of localModels) {
    if (model.loadable !== false) $('local-model').append(new Option(model.name, model.path));
    const row = element('div', 'local-row'), text = element('div');
    text.append(element('strong', '', model.name), element('p', '', model.path), element('small', 'muted', model.bytes == null ? t('Storage size unavailable') : `${(model.bytes / 2 ** 30).toFixed(2)} GiB`));
    const use = element('button', 'button secondary', t("Use checkpoint")); use.dataset.use = model.path;
    use.disabled = model.loadable === false;
    if (model.loadable === false) { use.textContent = t('Partial download'); delete use.dataset.use; }
    use.addEventListener('click', () => { chooseSource('path'); $('load-path').value = model.path; restoreLoadSettings(); location.hash = 'workspace'; });
    const buttons = element('div', 'local-actions'); buttons.append(use);
    if (model.deletable !== false) {
      const remove = element('button', 'button secondary danger', t('Delete model'));
      remove.addEventListener('click', () => { if (confirm(t('Delete {0}? This permanently removes its files.', model.name))) action(remove, async () => { await post('/daemon/models/delete', {path: model.path}); await refreshCatalog(false); storageSpace(); toast(t('Model deleted')); }); });
      buttons.append(remove);
    }
    row.append(text, buttons); $('local-list').append(row);
  }
  $('local-model').value = selected;
  if (!localModels.length) $('local-list').append(element('p', 'muted', t("No checkpoints found here. Download a recipe or choose another storage directory.")));
  controls();
}
function chooseSource(next) {
  source = next; $('recipe-input').hidden = next !== 'recipe'; $('path-input').hidden = next !== 'path';
  document.querySelectorAll('[data-source]').forEach(button => { button.classList.toggle('selected', button.dataset.source === next); button.setAttribute('aria-pressed', String(button.dataset.source === next)); });
  restoreLoadSettings();
}
function recipeDescription() { const recipe = recipes.find(r => r.id === $('load-recipe').value); $('load-recipe').classList.toggle('local-selected', Boolean(recipe?.loadable)); $('recipe-description').textContent = recipe ? `${recipe.loadable ? t("Local copy available.") : t("Download this recipe from Models before loading.")}` : ''; }
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
function openDirectory(target) { directoryTarget = target; $('directory-dialog').showModal(); browseDirectory($(target).value || undefined); }
function saveConversation() {
  const item = {id: sessionId, title: conversation.find(message => message.role === 'user')?.content.slice(0, 48) || '',
    messages: conversation.slice(-100), model: currentChatModel, system: $('system-prompt').value,
    thinkingLevel: preferredThinkingLevel, temperature: $('temperature').value, maxTokens: $('max-tokens').value};
  const index = sessions.findIndex(session => session.id === sessionId);
  if (index >= 0) sessions[index] = item; else sessions.push(item);
  sessions = sessions.slice(-50);
  try {
    localStorage.setItem('snowllm.conversation', JSON.stringify(item));
    localStorage.setItem('snowllm.sessions', JSON.stringify({current: sessionId, sessions}));
  } catch { $('chat-status').textContent = t('Browser storage unavailable · this conversation is temporary'); }
  renderSessions();
}
// Construct DOM nodes from model output, never HTML supplied by the model.
function markdown(container, text) {
  container.replaceChildren();
  // Fences can carry a language label; close fences have no trailing newline requirement.
  const blocks = text.split(/(```[^\n]*\n[\s\S]*?(?:```|$))/g);
  for (const block of blocks) {
    if (!block) continue;
    if (block.startsWith('```')) { const pre = element('pre'); const code = element('code', '', block.replace(/^```[^\n]*\n/, '').replace(/```$/, '')); const copy = element('button', 'button secondary code-copy', t('Copy code')); copy.type = 'button'; copy.addEventListener('click', () => copyText(code.textContent)); pre.append(copy, code); container.append(pre); }
    else {
      for (const paragraph of block.split(/\n\s*\n/)) {
        if (!paragraph.trim()) continue;
        const heading = paragraph.match(/^#{1,6}\s+(.*)$/);
        const node = element(heading ? 'h3' : 'p');
        const pieces = (heading ? heading[1] : paragraph).split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g);
        for (const piece of pieces) {
          if (piece.startsWith('`') && piece.endsWith('`')) node.append(element('code', '', piece.slice(1, -1)));
          else if (piece.startsWith('**') && piece.endsWith('**')) node.append(element('strong', '', piece.slice(2, -2)));
          else node.append(document.createTextNode(piece));
        }
        container.append(node);
      }
    }
  }
}
function messageNode(message) {
  const root = element('article', `message ${message.role}`);
  root.append(element('div', 'message-role', message.role === 'user' ? t("YOU") : 'SNOWLLM'));
  const reasoning = element('details'); reasoning.append(element('summary', '', t("Thinking")), element('div', 'reasoning-content', message.reasoning || '')); reasoning.hidden = !message.reasoning;
  const content = element('div', 'message-content');
  root.append(reasoning, content);
  function update() { if (message.role === 'user') content.textContent = message.content; else markdown(content, message.content); reasoning.hidden = !message.reasoning; reasoning.lastElementChild.textContent = message.reasoning || ''; }
  if (message.role === 'assistant') { const copy = element('button', 'button secondary answer-copy', t('Copy answer')); copy.addEventListener('click', () => copyText(message.content)); root.append(copy); }
  if (message.stats) root.append(element('div', 'chat-token-line', message.stats));
  update(); return { root, update };
}
function renderConversation() {
  if (!conversation.length) { $('messages').replaceChildren(emptyChat); return; }
  $('messages').replaceChildren(); for (const message of conversation) $('messages').append(messageNode(message).root);
  $('messages').scrollTop = $('messages').scrollHeight;
}
const emptyChat = $('chat-empty');
async function sendMessage(event) {
  event.preventDefault();
  const text = $('chat-input').value.trim();
  if (!text || chatController || status?.state !== 'ready') return;
  if (currentChatModel && currentChatModel !== status.model) newSession();
  currentChatModel = status.model;
  chatController = new AbortController(); const controller = chatController;
  const user = { role: 'user', content: text };
  conversation.push(user); $('chat-input').value = ''; saveConversation(); renderConversation();
  const assistant = { role: 'assistant', content: '', reasoning: '' };
  const node = messageNode(assistant); $('messages').append(node.root); conversation.push(assistant);
  $('stop-chat').hidden = false; $('send-chat').hidden = true; $('chat-status').textContent = t("Generating\u2026"); controls();
  const started = performance.now(); let firstToken = null, usage = null;
  const messages = conversation.slice(0, -1).map(({ role, content }) => ({ role, content }));
  if ($('system-prompt').value.trim()) messages.unshift({ role: 'system', content: $('system-prompt').value.trim() });
  const payload = { model: aliases[0]?.id || status.model, messages, stream: true, stream_options: { include_usage: true } };
  if ($('max-tokens').value) payload.max_tokens = Number($('max-tokens').value);
  if ($('temperature').value) payload.temperature = Number($('temperature').value);
  try {
    Object.assign(payload, thinkingParameters($('thinking-level').value, aliases[0]?.thinking));
    const response = await fetch('/daemon/inference/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
    if (!response.ok) { let detail = t("Generation failed ({0})", response.status); try { const data = await response.json(); detail = data.detail || data.error?.message || detail; } catch {} throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)); }
    let framePending = false;
    await readEvents(response.body, (chunk) => {
      if (chunk.error) throw new Error(chunk.error.message || t("Generation failed."));
      const delta = chunk.choices?.[0]?.delta || {};
      if (delta.content || delta.reasoning_content) { if (firstToken === null) firstToken = performance.now(); }
      assistant.content += delta.content || ''; assistant.reasoning += delta.reasoning_content || '';
      if (chunk.usage) usage = chunk.usage;
      if (!framePending) {
        framePending = true;
        requestAnimationFrame(() => { framePending = false; const atBottom = $('messages').scrollTop + $('messages').clientHeight >= $('messages').scrollHeight - 100; node.update(); if (atBottom) $('messages').scrollTop = $('messages').scrollHeight; });
      }
    });
    node.update();
    if (firstToken !== null) $('chat-ttft').textContent = `${((firstToken - started) / 1000).toFixed(2)} s`;
    if (usage) {
      $('chat-tokens').textContent = usage.completion_tokens;
      const timing = usage.timings || {};
      const rate = value => Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—';
      assistant.stats = t("{0} output tokens \u00b7 {1} s \u00b7 Prefill {2} \u00b7 Decode {3}", usage.completion_tokens, ((performance.now() - started) / 1000).toFixed(1), rate(timing.prefill_tok_s), rate(timing.decode_tok_s));
      node.root.append(element('div', 'chat-token-line', assistant.stats));
    }
  } catch (error) {
    node.update(); const message = error.name === 'AbortError' ? t("Generation stopped.") : t(error.message);
    node.root.append(element('p', 'message-error', message));
    if (error.name !== 'AbortError') toast(message);
  } finally {
    if (!assistant.content && !assistant.reasoning) conversation.pop();
    chatController = null; $('stop-chat').hidden = true; $('send-chat').hidden = false;
    $('chat-status').textContent = t("Enter to send"); saveConversation(); controls(); refreshStatus().catch(() => {});
  }
}
async function copyApi(target = 'api-url') {
  if (!status?.inference_url) { toast(t("Wait for the local service connection before copying the API address.")); return; }
  const value = $(target).value;
  try { await navigator.clipboard.writeText(value); toast(t("API address copied")); }
  catch { toast(t("API address: {0}", value)); }
}

translatePage();
$('language').value = language();
$('language').addEventListener('change', () => {
  setLanguage($('language').value);
  translatePage(); navigate(); updateThinkingLevels();
  if (!chatController) renderConversation();
  else {
    document.querySelectorAll('.message-role').forEach(node => { if (node.closest('.user')) node.textContent = t('YOU'); });
    document.querySelectorAll('.message details summary').forEach(node => node.textContent = t('Thinking'));
  }
  if (status) renderStatus(status);
  renderRecipeOptions(); renderRecipes(); renderLocal(); recipeDescription();
  $('chat-status').textContent = chatController ? t('Generating…') : t('Enter to send');
  refreshMetrics(); refreshLogs().catch(() => {});
});
$('api-address').textContent = t("Connecting\u2026"); $('api-url').value = '';
$('copy-api').addEventListener('click', () => copyApi());
document.querySelectorAll('[data-copy-api]').forEach(button => button.addEventListener('click', () => copyApi(button.dataset.copyApi)));
window.addEventListener('hashchange', navigate);
document.querySelectorAll('[data-source]').forEach(button => button.addEventListener('click', () => chooseSource(button.dataset.source)));
$('load-recipe').addEventListener('change', () => { restoreLoadSettings(); recipeDescription(); });
$('load-path').addEventListener('change', restoreLoadSettings);
$('save-load-settings').addEventListener('click', () => saveLoadSettings());
$('reset-load-settings').addEventListener('click', () => { for (const id of loadFieldIds) $(id).value = ''; saveLoadSettings(); });
$('cancel-load').addEventListener('click', () => action($('cancel-load'), async () => { renderStatus(await post('/daemon/load/cancel')); toast(t('Model loading cancelled')); }));
$('retry-load').addEventListener('click', () => action($('retry-load'), async () => { renderStatus(await post('/daemon/load', status.last_load)); }));
$('local-model').addEventListener('change', () => { if ($('local-model').value) { $('load-path').value = $('local-model').value; restoreLoadSettings(); } });
$('model-search').addEventListener('input', renderRecipes); $('installed-only').addEventListener('change', renderRecipes);
$('refresh-state').addEventListener('click', () => action($('refresh-state'), refreshStatus));
$('refresh-recipes').addEventListener('click', () => refreshCatalog(true));
document.querySelectorAll('[data-catalog-source]').forEach(select => select.addEventListener('change', () => {
  bundleCatalog = select.value === 'bundle';
  document.querySelectorAll('[data-catalog-source]').forEach(control => { control.value = select.value; });
  recipes = []; renderRecipeOptions(); renderRecipes(); recipeDescription(); restoreLoadSettings();
  refreshCatalog(false, true);
}));
$('refresh-logs').addEventListener('click', () => action($('refresh-logs'), refreshLogs)); $('log-kind').addEventListener('change', () => refreshLogs().catch(error => toast(error.message)));
$('adjust-load-settings').addEventListener('click', () => {
  location.hash = 'workspace';
  navigate();
  document.querySelector('#load-form .advanced').open = true;
  $('context-length').scrollIntoView({ behavior: 'smooth', block: 'center' });
  $('context-length').focus({ preventScroll: true });
});
$('load-error-logs').addEventListener('click', () => action($('load-error-logs'), () => showLogs('inference')));
$('load-form').addEventListener('submit', (event) => {
  event.preventDefault(); action($('load-button'), async () => {
    const model = (source === 'recipe' ? $('load-recipe').value : $('load-path').value).trim();
    if (!model) throw new Error(t("Choose a recipe or enter a checkpoint directory."));
    const options = [];
    for (const [id, flag] of [['context-length', '--max-model-len'], ['gpu-budget', '--max-gpu-mem'], ['spec-depth', '--num-spec'], ['kv-type', '--kv-cache-dtype']]) if ($(id).value.trim()) options.push(flag, $(id).value.trim());
    options.push(...parseOptions($('extra-options').value));
    saveLoadSettings(false);
    const result = await post('/daemon/load', { model, options, force: $('force-switch').checked }); renderStatus(result);
    $('load-feedback').textContent = t("Model change requested. Watch engine status and activity for progress."); toast(t("Model change requested"));
  });
});
$('unload-button').addEventListener('click', () => action($('unload-button'), async () => { renderStatus(await post(`/daemon/unload?force=${$('force-switch').checked}`)); toast(t("Model unload requested")); }));
$('cancel-download').addEventListener('click', () => action($('cancel-download'), async () => { renderStatus(await post('/daemon/download/cancel')); toast(t("Download stopped. Partial files are kept for resuming.")); }));
$('download-logs-button').addEventListener('click', () => action($('download-logs-button'), () => showLogs('download')));
$('storage-form').addEventListener('submit', (event) => { event.preventDefault(); action(event.submitter, async () => { renderStatus(await api('/daemon/config', { method: 'PUT', body: JSON.stringify({ models_dir: $('models-dir').value.trim() }) })); await refreshCatalog(false); toast(['command-line', 'environment'].includes(status.models_dir_source) ? t("Directory saved to JSON. The startup override is still active.") : t("Model directory saved")); }); });
$('recipes-settings-form').addEventListener('submit', (event) => { event.preventDefault(); action(event.submitter, async () => { renderStatus(await api('/daemon/config', { method: 'PUT', body: JSON.stringify({ recipes_url: $('recipes-url').value.trim() || null }) })); await refreshCatalog(true); toast(status.recipes_url_source === 'environment' ? t("Catalogue saved. SNOWLLM_RECIPES_URL still overrides it.") : t("Catalogue saved")); }); });
$('library-settings-form').addEventListener('submit', (event) => { event.preventDefault(); action(event.submitter, async () => { renderStatus(await api('/daemon/config', { method: 'PUT', body: JSON.stringify({ library_path: $('library-path').value.trim() || null }) })); toast(status.library_path_source === 'environment' ? t("Library saved. SNOWLLM_LIB still overrides it.") : t("Library saved for the next model load")); }); });
$('browse-storage').addEventListener('click', () => openDirectory('models-dir')); $('browse-model').addEventListener('click', () => openDirectory('load-path'));
$('close-directory').addEventListener('click', () => $('directory-dialog').close());
$('directory-form').addEventListener('submit', (event) => { event.preventDefault(); browseDirectory($('directory-path').value); });
$('directory-up').addEventListener('click', () => browseDirectory(directoryData.parent));
$('select-directory').addEventListener('click', () => { $(directoryTarget).value = directoryData.path; $('directory-dialog').close(); $(directoryTarget).focus(); if (directoryTarget === 'load-path') restoreLoadSettings(); });
$('chat-form').addEventListener('submit', sendMessage);
$('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$('send-chat').disabled) $('chat-form').requestSubmit(); } });
$('stop-chat').addEventListener('click', () => chatController?.abort());
$('new-chat').addEventListener('click', () => { if (!chatController) newSession(); });
$('chat-session').addEventListener('change', () => { if (!chatController) { const next = $('chat-session').value; saveConversation(); selectSession(next); } });
$('delete-chat').addEventListener('click', () => {
  if (chatController || !confirm(t('Delete this conversation?'))) return;
  sessions = sessions.filter(item => item.id !== sessionId);
  selectSession(sessions.at(-1)?.id);
});
$('export-chat').addEventListener('click', () => {
  const text = conversation.map(message => `${message.role === 'user' ? 'User' : 'SnowLLM'}\n\n${message.reasoning ? `[Thinking]\n${message.reasoning}\n\n` : ''}${message.content}`).join('\n\n---\n\n');
  const url = URL.createObjectURL(new Blob([text], {type: 'text/plain;charset=utf-8'}));
  const link = element('a'); link.href = url; link.download = `snowllm-chat-${new Date().toISOString().slice(0, 10)}.txt`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('regenerate-chat').addEventListener('click', () => {
  if (chatController || status?.state !== 'ready' || conversation.at(-1)?.role !== 'assistant' || conversation.at(-2)?.role !== 'user') return;
  conversation.pop(); const user = conversation.pop(); $('chat-input').value = user.content; sendMessage({preventDefault() {}});
});
$('temperature').addEventListener('change', saveConversation);
$('max-tokens').addEventListener('change', saveConversation);
$('system-prompt').addEventListener('change', saveConversation);
$('thinking-level').addEventListener('change', () => { preferredThinkingLevel = $('thinking-level').value; saveConversation(); });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { $('chat-input').value = t(button.dataset.prompt); $('chat-input').focus(); }));
if (sessions.some(item => item.id === sessionId)) selectSession(sessionId);
else saveConversation();
renderConversation(); updateThinkingLevels(); chooseSource('recipe'); navigate();
async function poll() {
  if (polling) return;
  polling = true;
  try { await refreshStatus(); if (page === 'workspace') await Promise.allSettled([refreshLogs(), refreshMetrics()]); }
  catch { /* The connection banner communicates daemon availability. */ }
  finally { polling = false; }
}
poll(); setInterval(poll, 2000);
