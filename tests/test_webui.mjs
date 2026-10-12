// SPDX-License-Identifier: Apache-2.0
// CPU-only: node --test tests/test_webui.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
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
