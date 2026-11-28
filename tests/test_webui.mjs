// SPDX-License-Identifier: Apache-2.0
// CPU-only: node --test tests/test_webui.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import { spawnSync } from 'node:child_process';
const source = await readFile(new URL('../snowllm/daemon/static/stream.js', import.meta.url), 'utf8');
const { parseOptions, readEvents } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const commands = await readFile(new URL('../snowllm/daemon/static/commands.js', import.meta.url), 'utf8');
const { modelLoadCommand } = await import('data:text/javascript;base64,' + Buffer.from(commands).toString('base64'));
const thinking = await readFile(new URL('../snowllm/daemon/static/thinking.js', import.meta.url), 'utf8');
const { thinkingProfile, thinkingParameters } = await import('data:text/javascript;base64,' + Buffer.from(thinking).toString('base64'));

test('copied load command preserves the recipe, target, ID, service address and option boundary', () => {
  assert.equal(modelLoadCommand({model: 'b/example', target: 'gpu', id: 'main', options: ['--max-model-len', '32k']}, 'http://127.0.0.1:9001/#workspace'),
    'snowllm load --admin-host 127.0.0.1 --admin-port 9001 --id=main gpu:b/example -- --max-model-len 32k');
  assert.equal(modelLoadCommand({model: '/models/example', target: 'npu', id: null, options: []}, 'http://[::1]:8001/'),
    'snowllm load --admin-host ::1 --admin-port 8001 npu:/models/example');
  assert.equal(modelLoadCommand({model: 'b/example', target: 'gpu', options: []}, 'https://models.example/#workspace'),
    'snowllm load --admin-url https://models.example gpu:b/example');
});

test('copied load commands preserve shell metacharacters as literal arguments', () => {
  if (process.platform === 'win32') return;
  const request = {model: "/models/John's model; $(printf injected >&2)", target: 'gpu', id: '-main model',
    options: ['--dflash', 'C:\\Models\\Draft Model', '--served-model-name', '雪 `printf injected >&2` $HOME\nnext', '--stop', '']};
  const command = modelLoadCommand(request, 'http://localhost:8001/');
  const shell = spawnSync('/bin/sh', ['-c', 'snowllm() { printf "%s\\0" "$@"; }\n' + command], {encoding: 'utf8'});
  assert.equal(shell.status, 0, shell.stderr);
  assert.equal(shell.stderr, '');
  assert.deepEqual(shell.stdout.split('\0').slice(0, -1), ['load', '--admin-host', 'localhost', '--admin-port', '8001',
    '--id=-main model', 'gpu:' + request.model, '--', ...request.options]);
});

test('required thinking cannot be switched off, including stale saved levels', () => {
  const required = { required: true, switch: true, levels: ['none', 'low'] };
  assert.deepEqual(thinkingProfile(required).options, [['', 'Always on'], ['effort:low', 'low']]);
  assert.deepEqual(thinkingParameters('', required), {});
  assert.throws(() => thinkingParameters('off', required), /not supported/);
  assert.throws(() => thinkingParameters('effort:none', required), /not supported/);
  assert.deepEqual(thinkingParameters('off', { switch: true }), { enable_thinking: false });
  assert.deepEqual(thinkingProfile({ switch: false }).options, [['', 'Default']]);
});

test('thinking options offer On only when there is no enabling effort level', () => {
  const capabilities = { switch: true, levels: ['xhigh', 'medium', 'low'] };
  assert.deepEqual(thinkingProfile(capabilities).options.map(([value]) => value),
    ['', 'off', 'effort:xhigh', 'effort:medium', 'effort:low']);
  assert.deepEqual(thinkingParameters('effort:medium', capabilities), { reasoning_effort: 'medium' });
  for (const levels of [[], ['none']]) {
    assert.deepEqual(thinkingParameters('on', { switch: true, levels }), { enable_thinking: true });
  }
});

test('extra inference options preserve Windows paths and quoted spaces', () => {
  assert.deepEqual(parseOptions('--device-map auto --dflash "C:\\Models\\Draft Model"'), ['--device-map', 'auto', '--dflash', 'C:\\Models\\Draft Model']);
  assert.deepEqual(parseOptions("--served-model-name 'model name' --num-spec 2"), ['--served-model-name', 'model name', '--num-spec', '2']);
  assert.deepEqual(parseOptions('--dflash C:\\Models\\Draft'), ['--dflash', 'C:\\Models\\Draft']);
  assert.throws(() => parseOptions('--dflash "unclosed'), /quotation/);
});

test('SSE survives one-byte UTF-8 chunks and CRLF boundaries', async () => {
  const text = ': keepalive\r\ndata: {"choices":[{"delta":{"content":"雪 ❄"}}]}\r\n\r\ndata: {"usage":{"completion_tokens":3}}\n\ndata: [DONE]\n\n';
  const encoded = new TextEncoder().encode(text), events = [];
  let canceled = false;
  const stream = new ReadableStream({ start(controller) { for (const byte of encoded) controller.enqueue(new Uint8Array([byte])); }, cancel() { canceled = true; } });
  assert.equal(await readEvents(stream, value => events.push(value)), true);
  assert.equal(events[0].choices[0].delta.content, '雪 ❄');
  assert.equal(events[1].usage.completion_tokens, 3);
  assert.equal(events.length, 2);
  assert.equal(canceled, true);
});

test('malformed events fail visibly and close the stream', async () => {
  let canceled = false;
  const stream = new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode('data: invalid\n\n')); }, cancel() { canceled = true; } });
  await assert.rejects(readEvents(stream, () => {}), SyntaxError);
  assert.equal(canceled, true);
});

test('a stream closed without the completion marker preserves partial events', async () => {
  const partial = { choices: [{ delta: { content: 'Partial reply' } }] };
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(partial)}\n\n`));
    controller.close();
  } });
  const events = [];
  assert.equal(await readEvents(stream, value => events.push(value)), false);
  assert.deepEqual(events, [partial]);
});

test('update banner renders both channels, their notes and exact commands without HTML interpretation', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const render = app.slice(app.indexOf('function renderUpdate('), app.indexOf('function renderStatus('));
  function element(tag, className = '', textContent = '') {
    return { tag, className, textContent, dataset: {}, children: [],
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren(...nodes) { this.children = nodes; } };
  }
  const nodes = new Map(['update-notice', 'update-title', 'update-releases'].map(id => [id, element('div')]));
  let chinese = false;
  const context = vm.createContext({ $: id => nodes.get(id), element,
    t: (key, ...values) => {
      const label = chinese ? { Stable: '稳定版', Dev: '开发版' }[key] || key : key;
      return label.replace(/\{(\d+)\}/g, (_, index) => values[Number(index)]);
    } });
  vm.runInContext(render, context);
  const update = { current: '0.3.19.dev2', checking: true, releases: [
    { channel: 'stable', version: '0.3.19', changelog: '<script>untrusted notes</script>', command: 'python -m pip install --upgrade "snowllm==0.3.19"' },
    { channel: 'dev', version: '0.3.20.dev1', changelog: null, command: 'python -m pip install --upgrade "snowllm==0.3.20.dev1"' },
  ] };
  context.renderUpdate(update);
  assert.equal(nodes.get('update-notice').hidden, false);
  assert.match(nodes.get('update-title').textContent, /0\.3\.19\.dev2/);
  const rows = nodes.get('update-releases').children;
  assert.equal(rows.length, 2);
  assert.equal(rows[0].children[0].textContent, 'Stable (0.3.19)');
  assert.equal(rows[1].children[0].textContent, 'Dev (0.3.20.dev1)');
  assert.equal(rows[0].children[1].tag, 'pre');
  assert.equal(rows[0].children[1].textContent, '<script>untrusted notes</script>');
  assert.equal(rows[1].children[1].textContent, 'Loading release notes…');
  assert.equal(rows[1].children[2].textContent, update.releases[1].command);
  context.renderUpdate(update);
  assert.equal(nodes.get('update-releases').children[0], rows[0]);
  update.checking = false;
  context.renderUpdate(update);
  assert.equal(nodes.get('update-releases').children[1].children[1].textContent, 'Release notes are not available for this version.');
  chinese = true;
  context.renderUpdate(update);
  assert.equal(nodes.get('update-releases').children[0].children[0].textContent, '稳定版 (0.3.19)');
  assert.equal(nodes.get('update-releases').children[1].children[0].textContent, '开发版 (0.3.20.dev1)');
  context.renderUpdate({ current: '0.3.19', latest: '0.3.20', changelog: 'Legacy notes' });
  assert.equal(nodes.get('update-releases').children.length, 1);
  assert.match(nodes.get('update-title').textContent, /SnowLLM 0\.3\.20 is available/);
  assert.equal(nodes.get('update-releases').children[0].children[1].textContent, 'Legacy notes');
  context.renderUpdate({ current: '0.3.20', releases: [] });
  assert.equal(nodes.get('update-notice').hidden, true);
  context.renderUpdate(undefined);
  assert.equal(nodes.get('update-notice').hidden, true);
});

test('switching to bundled recipes completes while the server is pending and ignores its late response', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const refresh = app.slice(app.indexOf('async function refreshCatalog('), app.indexOf('function renderRecipeOptions('));
  const elements = new Map(['refresh-recipes', 'load-recipe', 'catalog-error', 'local-list'].map(id => [id, { value: '', disabled: false }]));
  const requests = [];
  let finishServer;
  const server = new Promise(resolve => { finishServer = resolve; });
  const context = vm.createContext({
    $: id => elements.get(id),
    api: async path => {
      requests.push(path);
      if (path === '/daemon/local-models') return { models: [] };
      if (path.includes('bundle=true')) return { recipes: [{ id: 'b/bundled', type: 'lfm' }], recipe_types: { lfm: 'LFM' } };
      return server;
    },
    renderRecipeOptions() {}, renderRecipes() {}, renderLoadHints() {}, renderLocal() {},
  });
  vm.runInContext(`let catalogLoading = false, catalogRequest = 0, bundleCatalog = false;
    const loadCards = [];
    let status = {models_dir: '/models'}, recipes = [], recipeTypes = {}, localModels = [];
    ${refresh}`, context);
  const pending = vm.runInContext('refreshCatalog(false)', context);
  await vm.runInContext('bundleCatalog = true; refreshCatalog(false, true)', context);
  assert.equal(vm.runInContext('recipes[0].id', context), 'b/bundled');
  assert.equal(vm.runInContext('recipeTypes.lfm', context), 'LFM');
  assert.equal(elements.get('refresh-recipes').disabled, false);
  finishServer({ recipes: [{ id: 'server' }], recipe_types: { glm: 'GLM' } });
  await pending;
  assert.equal(vm.runInContext('recipes[0].id', context), 'b/bundled');
  assert.equal(vm.runInContext('recipeTypes.lfm', context), 'LFM');
  assert.ok(requests.includes('/daemon/recipes?refresh=false&bundle=false'));
  assert.ok(requests.includes('/daemon/recipes?refresh=false&bundle=true'));
});

test('benchmark commands preserve the selected model, settings and shell quoting', async () => {
  const { benchmarkCommand } = await import('data:text/javascript;base64,' + Buffer.from(commands).toString('base64'));
  const config = { model_id: "main model'; $(printf injected >&2)", isl: [1024, 8192], osl: 256, conc: [1, 2], reps: 4, warm: 1, thinking: 'auto' };
  const command = benchmarkCommand({...config, workload: 'file-edit'}, 'http://localhost:9001/#benchmark');
  if (process.platform === 'win32') return;
  const result = spawnSync('/bin/sh', ['-c', 'snowllm() { printf "%s\\0" "$@"; }\n' + command], {encoding: 'utf8'});
  assert.equal(result.status, 0);
  assert.equal(result.stderr, '');
  assert.deepEqual(result.stdout.split('\0').slice(0, -1), ['bench', 'run', '--admin-host', 'localhost', '--admin-port', '9001',
    '--model-id=' + config.model_id, '--workload', 'file-edit', '--isl', '1024,8192', '--osl', '256', '--conc', '1,2', '--reps', '4', '--warm', '1', '--thinking', 'auto']);
});

test('benchmark step metrics use measured step time and only derive it when AL is known', async () => {
  const source = (await readFile(new URL('../snowllm/daemon/static/benchmark-results.js', import.meta.url), 'utf8')).replace(/^import .*;\n/gm, '');
  const { resultMetrics } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
  const row = {decode_tok_s: 100, accept_unit: 'AL', accept_mean: 2.5, prefill_tok_s: 2000, ttft_s: 0.5};
  assert.deepEqual(resultMetrics({...row, step_s: 0.02}), {
    decode_tok_s: 100, step_hz: 50, step_ms: 20, al: 2.5, prefill_tok_s: 2000, ttft_s: 0.5,
  });
  assert.equal(resultMetrics(row).step_ms, 25);
  assert.equal(resultMetrics({...row, accept_length_mean: 2, accept_unit: 'accept%'}).step_hz, 50);
  const missing = resultMetrics({...row, accept_unit: 'accept%', accept_mean: 0.5});
  assert.equal(missing.al, null);
  assert.equal(missing.step_hz, null);
  assert.equal(missing.step_ms, null);
});

test('benchmark comparisons expose outdated, incomplete and mismatched measurements', async () => {
  const source = (await readFile(new URL('../snowllm/daemon/static/benchmark.js', import.meta.url), 'utf8')).replace(/^import .*;\n/gm, '');
  const { comparisonDifferences, parseLengths } = await import('data:text/javascript;base64,' + Buffer.from('const t = key => key;\n' + source).toString('base64'));
  assert.deepEqual(parseLengths('1k, 8K,1m'), [1024,8192,1048576]);
  assert.throws(() => parseLengths('1k,0'), /positive/);
  assert.throws(() => parseLengths('1k,,8k'), /comma/);
  const left = {state: 'completed', outdated: false, config: {isl:[1024], conc:[1], osl:256, reps:4, warm:1, thinking:'auto'},
    model: {runtime_config: {native: {sha256:'native'}, prefill_chunk:8192}}, model_identity: {files:[{path:'weights.gguf', size:1024}]},
    events:[{type:'validated', corpus_sha256:'corpus'}], provenance:{client_sha256:'client'}};
  const right = structuredClone(left);
  right.config.workload = 'pg19';
  assert.deepEqual(comparisonDifferences(left, right), []);
  right.config.workload = 'file-edit';
  assert.deepEqual(comparisonDifferences(left, right), ['Workload']);
  right.config.workload = 'pg19';
  right.outdated = true; right.state = 'cancelled'; right.model.runtime_config.prefill_chunk = 512;
  right.events[0].corpus_sha256 = 'different'; right.config.reps = 1;
  assert.deepEqual(comparisonDifferences(left, right), ['Incomplete run', 'Outdated result', 'reps', 'Inference settings', 'Corpus']);
});

test('model load sends explicit spec overrides and inherits recipe defaults', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const fields = Object.fromEntries(['context-length', 'gpu-budget', 'vision-placement', 'host-budget', 'disk-budget', 'weight-map', 'disk-expert-slots', 'kv-type', 'max-num-seqs',
    'max-num-batched-tokens', 'load-target', 'load-id', 'extra-options'].map(id => [id, {value: ''}]));
  const specInputs = {mtp_prompt_lookup: {value: ''}, mtp_tool_draft: {value: ''}, dflash_block: {value: ''}};
  const context = vm.createContext({loadKey: () => 'model', parseOptions: () => []});
  vm.runInContext(app.slice(app.indexOf('const inferenceOptions'), app.indexOf('const loadFieldIds')) +
    app.slice(app.indexOf('function modelLoadRequest('), app.indexOf('function renderLoadCards(')), context);
  const options = () => Array.from(context.modelLoadRequest({fields, specInputs}).options);
  assert.deepEqual(options(), []);
  specInputs.mtp_prompt_lookup.value = 'off';
  specInputs.mtp_tool_draft.value = 'on';
  specInputs.dflash_block.value = '4';
  assert.deepEqual(options(), ['--no-mtp-prompt-lookup', '--mtp-tool-draft', '--dflash-block', '4']);
  for (const input of Object.values(specInputs)) input.disabled = true;
  assert.deepEqual(options(), []);
  for (const value of ['16GiB', '0']) {
    fields['disk-budget'].value = value;
    assert.deepEqual(options(), ['--max-disk-offload', value]);
  }
  fields['disk-budget'].disabled = true;
  assert.deepEqual(options(), []);
  for (const value of ['disk', 'gpu']) {
    fields['vision-placement'].value = value;
    assert.deepEqual(options(), ['--vision-placement', value]);
  }
  fields['vision-placement'].disabled = true;
  fields['host-budget'].value = '4GiB';
  fields['weight-map'].value = '*=gpu,experts[2]=host,experts[3-5]=disk';
  fields['disk-expert-slots'].value = '316';
  assert.deepEqual(options(), ['--max-host-mem', '4GiB', '--device-map', '*=gpu,experts[2]=host,experts[3-5]=disk',
    '--disk-offload-gpu-cache-experts', '316']);
});

test('model library shows only ordered recommendation candidates for v3 models', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const context = vm.createContext({});
  vm.runInContext(app.slice(app.indexOf('function modelChoices('), app.indexOf('function presetSelect(')), context);
  const groups = context.modelChoices([
    {model: 'Flash', id: 'old'},
    {model: 'Flash', model_id: 'flash', preset: 'q3', id: 'flash:q3', recommend_order: 1},
    {model: 'Flash', model_id: 'flash', id: 'flash:q3:dflash'},
    {model: 'Flash', model_id: 'flash', preset: 'other', id: 'flash:other'},
    {model: 'Flash', model_id: 'flash', preset: 'iq4', id: 'flash:iq4', recommend_order: 0, recommended: true},
    {model: 'Custom', id: 'custom-q4'},
    {model: 'Custom', id: 'custom-q8'},
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(groups.map(m => m.choices.map(r => r.id)))),
    [['flash:iq4', 'flash:q3'], ['custom-q4', 'custom-q8']]);
});

test('v3 preset selection recommends by memory and allows an explicit manual choice', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const context = vm.createContext({manualPreset: '@manual', t: key => key,
    Option: class { constructor(label, value) { this.value = value; } }});
  vm.runInContext(app.slice(app.indexOf('function presetSelect('), app.indexOf('function precisionRows(')), context);
  const select = {replaceChildren() { this.options = []; }, append(option) { this.options.push(option); }};
  const row = {id: 'flash:q3', model_id: 'flash', preset: 'q3', variant: 'q3', base_defaults: {}, servable: true};
  context.presetSelect(select, [row]);
  assert.deepEqual(select.options.map(o => o.value), [row.id, '@manual']);
  assert.equal(select.value, '@manual');
  row.recommended = true;
  context.presetSelect(select, [row]);
  assert.equal(select.value, row.id);
  context.presetSelect(select, [row], '@manual');
  assert.equal(select.value, '@manual');
  context.presetSelect(select, [{id: 'legacy'}]);
  assert.deepEqual(select.options.map(o => o.value), ['legacy']);
});

test('library and loader group models by family across architectures', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const context = vm.createContext({recipeTypes: {glm: 'GLM'}, t: text => text});
  vm.runInContext(app.slice(app.indexOf('function groupRecipes('), app.indexOf('function presetSelect(')), context);
  const rows = [{id: 'flash', family: 'Qwen', type: 'qwen38fn'}, {id: 'glm', type: 'glm'},
    {id: 'dense', type: 'qwen35dense'}, {id: 'moe', family: 'Qwen', type: 'qwen35moe'}];
  const groups = context.groupRecipes(context.modelChoices(rows));
  assert.deepEqual(JSON.parse(JSON.stringify(groups.map(g => [g.label, g.rows.map(row => row.name)]))),
    [['Qwen', ['flash', 'dense', 'moe']], ['GLM', ['glm']]]);
});

test('inheriting a preset and explicitly choosing its draft method remain distinct', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const rows = ['none', 'dflash2', 'mtp-adaptive'].map(spec => ({id: 'test:' + spec, model_id: 'test', variant: 'q3', precision: 'Q3', spec}));
  rows[2].preset = 'fast';
  rows.push({id: 'test:q4-dflash', model_id: 'test', variant: 'q4', precision: 'Q4', spec: 'dflash2', preset: 'q4-dflash'},
    {id: 'test:q4-mtp', model_id: 'test', variant: 'q4-head', precision: 'Q4', spec: 'mtp', preset: 'q4-mtp'},
    {id: 'test:old', model_id: 'test', variant: 'old-q4', precision: 'Q4', spec: 'none'});
  class Option {
    constructor(label, value) { Object.assign(this, {label, value, dataset: {}}); }
  }
  const context = vm.createContext({recipes: rows, Option, t: (key, name) => key.replace('{0}', name)});
  vm.runInContext(app.slice(app.indexOf('function precisionRows('), app.indexOf('function renderMethodOptions(')), context);
  const select = {
    replaceChildren() { this.options = []; this.value = ''; },
    append(option) { this.options.push(option); },
  };
  context.methodOptions(select, rows[2], '');
  assert.deepEqual(select.options.map(o => [o.label, o.value, o.dataset.recipe]), [
    ['Default (MTP)', '', rows[2].id], ['MTP', 'mtp', rows[2].id],
    ['DFlash', 'dflash', rows[1].id], ['Off', 'none', rows[0].id],
  ]);
  context.precisionOptions(select, rows[2], 'Q4');
  assert.deepEqual(select.options.map(o => o.label), ['Default (Q3)', 'Q3', 'Q4']);
  assert.equal(select.value, 'Q4');
  context.methodOptions(select, rows[2], 'dflash', 'Q4');
  assert.deepEqual(select.options.map(o => o.dataset.recipe), ['test:q4-mtp', 'test:q4-mtp', 'test:q4-dflash']);
  assert.equal(select.value, 'dflash');
  context.precisionOptions(select, rows[2], 'Q4', true);
  assert.deepEqual(select.options.map(o => o.label), ['Choose a weight precision…', 'Q3', 'Q4']);
  context.methodOptions(select, rows[2], 'dflash', 'Q4', true);
  assert.deepEqual(select.options.map(o => o.label), ['MTP', 'DFlash']);
  assert.equal(select.value, 'dflash');
  context.methodOptions(select, rows[2], '', 'Q3', true);
  assert.equal(select.value, 'none');
  context.methodOptions(select, rows[0], 'mtp');
  assert.equal(select.value, 'mtp');
  assert.equal(select.options[0].dataset.recipe, rows[0].id);
  assert.equal(select.options.find(o => o.value === select.value).dataset.recipe, rows[2].id);
  const mtpWeights = rows.find(row => row.variant === 'q4-head');
  rows.push({...mtpWeights, id: 'test:q4-head:dflash', preset: null, spec: 'dflash2'});
  context.methodOptions(select, mtpWeights, 'dflash');
  assert.equal(select.options.find(o => o.value === 'dflash').dataset.recipe, 'test:q4-head:dflash');

  const current = {model_id: 'flash', id: 'flash:iq4', preset: 'iq4', variant: 'iq4-ud3.1', precision: 'IQ4', spec: 'mtp-adaptive'};
  const old = {...current, id: 'flash:iq4-pre-ud3.1:dflash2', variant: 'iq4-pre-ud3.1', spec: 'dflash2', preset: null};
  rows.push(current, old);
  context.precisionOptions(select, current, 'IQ4');
  assert.deepEqual(select.options.map(o => o.label), ['Default (UD3.1 · IQ4)', 'UD3.1 · IQ4', 'IQ4']);
  assert.equal(select.value, 'UD3.1 · IQ4');
  context.precisionOptions(select, current, 'pre-UD3.1 · IQ4', true);
  assert.equal(select.options.find(o => o.value === select.value).label, 'IQ4');
  context.methodOptions(select, current, 'dflash');
  assert.deepEqual(select.options.map(o => o.dataset.recipe), [current.id, current.id]);
  context.methodOptions(select, current, 'dflash', 'pre-UD3.1 · IQ4');
  assert.deepEqual(select.options.map(o => o.dataset.recipe), [old.id, old.id]);
  assert.equal(select.value, 'dflash');

  rows[1].spec_label = 'DFlash2@original';
  const continued = {...rows[1], id: 'test:continued', spec: 'another-head', spec_label: 'DFlash2@continued'};
  rows.push(continued);
  for (const manual of [false, true]) {
    context.methodOptions(select, rows[2], continued.spec_label, 'Q3', manual);
    assert.deepEqual(select.options.filter(o => o.label.startsWith('DFlash2@')).map(o => [o.label, o.dataset.recipe]),
      [['DFlash2@original', rows[1].id], ['DFlash2@continued', continued.id]]);
    assert.equal(select.value, continued.spec_label);
    context.methodOptions(select, rows[2], 'dflash', 'Q3', manual);
    assert.equal(select.value, 'DFlash2@original');
  }
});

test('vision selection follows the draft and the resolved recipe is the load key', async () => {
  const app = await readFile(new URL('../snowllm/daemon/static/app.js', import.meta.url), 'utf8');
  const base = {id: 'b/ink:dspark', vision_head: 'none', vision_heads: {BF16: {}}};
  const plain = {...base, id: 'b/ink:plain'};
  const rows = [base, plain, ...[base, plain].map(row =>
    ({id: row.id + ':vision=bf16', vision_head: 'BF16', vision_base: row.id}))];
  class Option {
    constructor(label, value) { Object.assign(this, {label, value, dataset: {}}); }
  }
  const context = vm.createContext({recipes: rows, Option, t: (key, value) => key.replace('{0}', value)});
  vm.runInContext(app.slice(app.indexOf('function visionOptions('), app.indexOf('function renderMethodOptions(')), context);
  vm.runInContext(app.slice(app.indexOf('function loadKey('), app.indexOf('function selectLoadModel(')), context);
  const vision = {
    replaceChildren() { this.options = []; this.value = ''; },
    append(option) { this.options.push(option); },
    get selectedOptions() { return this.options.filter(option => option.value === this.value); },
  };
  const method = {selectedOptions: [{dataset: {recipe: base.id}}]};
  const card = {source: 'recipe', fields: {'vision-head': vision, 'spec-method': method}};
  context.visionOptions(vision, method);
  assert.equal(context.loadKey(card), base.id);
  vision.value = 'BF16';
  assert.equal(context.loadKey(card), base.id + ':vision=bf16');
  method.selectedOptions[0].dataset.recipe = plain.id;
  context.visionOptions(vision, method, vision.value);
  assert.equal(context.loadKey(card), plain.id + ':vision=bf16');
  method.selectedOptions[0].dataset.recipe = 'text-only';
  context.visionOptions(vision, method, vision.value);
  assert.equal(vision.options.length, 0);
  assert.equal(context.loadKey(card), 'text-only');
});
