// SPDX-License-Identifier: Apache-2.0
// Preserve Windows path separators; quoting groups arguments without a shell.
export function parseOptions(text) {
  const result = [];
  let token = '', quote = '', started = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quote) {
      if (char === quote) quote = '';
      else if (char === '\\' && quote === '"' && ['"', '\\'].includes(text[i + 1])) token += text[++i];
      else token += char;
    } else if (char === '"' || char === "'") {
      quote = char; started = true;
    } else if (/\s/.test(char)) {
      if (started) { result.push(token); token = ''; started = false; }
    } else { token += char; started = true; }
  }
  if (quote) throw new Error('Close the quotation mark in additional options.');
  if (started) result.push(token);
  return result;
}

// SSE boundaries can span network chunks, UTF-8 characters, and CRLF pairs.
export async function readEvents(body, onEvent) {
  if (!body) throw new Error('The response has no readable stream.');
  const reader = body.getReader(), decoder = new TextDecoder();
  let pending = '', data = [], done = false;
  function line(value) {
    if (value === '') {
      if (data.length) {
        const payload = data.join('\n'); data = [];
        if (payload === '[DONE]') done = true;
        else onEvent(JSON.parse(payload));
      }
    } else if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /, ''));
  }
  function consume(final = false) {
    let index;
    while (!done && (index = pending.search(/[\r\n]/)) >= 0) {
      if (pending[index] === '\r' && index === pending.length - 1 && !final) break;
      const length = pending[index] === '\r' && pending[index + 1] === '\n' ? 2 : 1;
      const value = pending.slice(0, index); pending = pending.slice(index + length); line(value);
    }
    if (final && !done) { if (pending) line(pending); line(''); }
  }
  try {
    while (!done) {
      const chunk = await reader.read();
      if (chunk.done) { pending += decoder.decode(); consume(true); break; }
      pending += decoder.decode(chunk.value, { stream: true }); consume();
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
