// SPDX-License-Identifier: Apache-2.0
const rendered = new WeakMap();
const levels = { CRITICAL: 'error', FATAL: 'error', ERROR: 'error', WARNING: 'warning', WARN: 'warning', INFO: 'info', DEBUG: 'muted', TRACE: 'muted' };
const tokens = /\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)?\b|\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b|\[(?:snowllm|planner)[^\]\n]*\]|\b(?:CRITICAL|FATAL|ERROR|WARNING|WARN|INFO|DEBUG|TRACE)\b|\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b|\b\d+(?:\.\d+)?(?:\s?(?:GiB|MiB|KiB|GB|MB|KB|ms|tok\/s|tokens\/s)\b|%)?/g;

export function renderLogs(target, raw, emptyText) {
  const text = raw.replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, '').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\r\n?/g, '\n') || emptyText;
  if (rendered.get(target) === text) return;
  const atBottom = target.scrollTop + target.clientHeight >= target.scrollHeight - 30;
  const top = target.scrollTop;
  const doc = target.ownerDocument, fragment = doc.createDocumentFragment();
  for (const [index, line] of text.split('\n').entries()) {
    if (index) fragment.append('\n');
    const row = doc.createElement('span');
    const level = line.match(/\b(?:CRITICAL|FATAL|ERROR|WARNING|WARN)\b/)?.[0];
    row.className = `log-line${level ? ` log-${levels[level]}` : /^\s*(?:Traceback\b|[\w.]+(?:Error|Exception):)/.test(line) ? ' log-error' : ''}`;
    let offset = 0;
    for (const match of line.matchAll(tokens)) {
      row.append(line.slice(offset, match.index));
      const token = doc.createElement('span'), value = match[0];
      const kind = levels[value] || (/^\[/.test(value) ? 'tag' : /\d{2}:\d{2}:\d{2}|^\d{4}-\d{2}-\d{2}/.test(value) ? 'muted' : /^\d/.test(value) ? 'number' : 'info');
      token.className = `log-${kind}`;
      token.textContent = value;
      row.append(token);
      offset = match.index + value.length;
    }
    row.append(line.slice(offset));
    fragment.append(row);
  }
  target.replaceChildren(fragment);
  rendered.set(target, text);
  target.scrollTop = atBottom ? target.scrollHeight : top;
}
