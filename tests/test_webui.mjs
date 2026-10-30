// SPDX-License-Identifier: Apache-2.0
// CPU-only: node --test tests/test_webui.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
const source = await readFile(new URL('../snowllm/daemon/static/stream.js', import.meta.url), 'utf8');
const { parseOptions, readEvents } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

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
  await readEvents(stream, value => events.push(value));
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
      if (path.includes('bundle=true')) return { recipes: [{ id: 'b/bundled' }] };
      return server;
    },
    renderRecipeOptions() {}, renderRecipes() {}, recipeDescription() {}, renderLocal() {},
  });
  vm.runInContext(`let catalogLoading = false, catalogRequest = 0, bundleCatalog = false;
    let status = {models_dir: '/models'}, recipes = [], localModels = [];
    ${refresh}`, context);
  const pending = vm.runInContext('refreshCatalog(false)', context);
  await vm.runInContext('bundleCatalog = true; refreshCatalog(false, true)', context);
  assert.equal(vm.runInContext('recipes[0].id', context), 'b/bundled');
  assert.equal(elements.get('refresh-recipes').disabled, false);
  finishServer({ recipes: [{ id: 'server' }] });
  await pending;
  assert.equal(vm.runInContext('recipes[0].id', context), 'b/bundled');
  assert.ok(requests.includes('/daemon/recipes?refresh=false&bundle=false'));
  assert.ok(requests.includes('/daemon/recipes?refresh=false&bundle=true'));
});
