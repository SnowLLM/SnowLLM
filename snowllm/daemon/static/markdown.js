// SPDX-License-Identifier: Apache-2.0

const inline = /\\([\\`*\[\]|])|`([^`\n]+)`|\*\*([^*\n]+)\*\*|(?<!!)\[([^\]\n]+)\]\(([^\s()]*(?:\([^\s()]*\)[^\s()]*)*)\)/g;
const fenceStart = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const headingStart = /^ {0,3}#{1,6}[ \t]+(.*)$/;
const listStart = /^( *)(?:(\d{1,9})[.)]|[-+*])[ \t]+(.*)$/;

function appendInline(parent, text, allowLinks = true) {
  let end = 0;
  for (const match of text.matchAll(inline)) {
    const [source, escaped, code, bold, label, href] = match;
    parent.append(text.slice(end, match.index));
    end = match.index + source.length;
    if (escaped) {
      parent.append(escaped);
    } else if (code || bold) {
      const node = document.createElement(code ? 'code' : 'strong');
      if (code) node.textContent = code;
      else appendInline(node, bold, allowLinks);
      parent.append(node);
    } else {
      let url;
      if (allowLinks) {
        try { url = new URL(href); } catch {}
      }
      if (url?.protocol === 'https:' || url?.protocol === 'http:') {
        const link = document.createElement('a');
        link.href = url.href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        appendInline(link, label, false);
        parent.append(link);
      } else parent.append(source);
    }
  }
  parent.append(text.slice(end));
}

function tableCells(line) {
  const text = line.trim();
  const cells = [];
  let start = 0;
  for (const match of text.matchAll(/\\.|(`+)[^`]*\1|\|/g)) {
    if (match[0] !== '|') continue;
    if (match.index) cells.push(text.slice(start, match.index).trim());
    start = match.index + 1;
  }
  if (start < text.length) cells.push(text.slice(start).trim());
  return cells;
}

function tableAlignment(line) {
  if (!line || !/^\s*[|:\-]/.test(line)) return null;
  const cells = tableCells(line);
  return cells.length && cells.every(cell => /^:?-{3,}:?$/.test(cell)) ? cells.map(cell => {
    if (!cell.endsWith(':')) return 'start';
    return cell.startsWith(':') ? 'center' : 'end';
  }) : null;
}

function appendTable(parent, lines, index, header, alignment) {
  const wrapper = document.createElement('div');
  wrapper.className = 'table-wrap';
  const table = document.createElement('table');
  wrapper.append(table);
  parent.append(wrapper);
  function appendRow(section, cells, headerRow = false) {
    const row = section.insertRow();
    alignment.forEach((align, column) => {
      const cell = document.createElement(headerRow ? 'th' : 'td');
      if (headerRow) cell.scope = 'col';
      cell.style.textAlign = align;
      appendInline(cell, cells[column] || '');
      row.append(cell);
    });
  }
  appendRow(table.createTHead(), header, true);
  const body = table.createTBody();
  while (lines[index]?.includes('|') && !fenceStart.test(lines[index]) && !listStart.test(lines[index])) {
    appendRow(body, tableCells(lines[index++]));
  }
  return index;
}

function appendList(parent, lines, index) {
  const levels = [];
  let item;
  while ((item = lines[index]?.match(listStart))) {
    const [, spaces, number, text] = item;
    const indent = spaces.length;
    const tag = number ? 'ol' : 'ul';
    while (levels.length && (levels.at(-1).indent > indent ||
      (levels.at(-1).indent === indent && levels.at(-1).tag !== tag))) levels.pop();
    if (!levels.length || levels.at(-1).indent < indent) {
      const list = document.createElement(tag);
      if (number && Number(number) !== 1) list.start = Number(number);
      (levels.at(-1)?.list.lastElementChild || parent).append(list);
      levels.push({indent, tag, list});
    }
    const li = document.createElement('li');
    appendInline(li, text);
    levels.at(-1).list.append(li);
    index++;
  }
  return index;
}

export function renderMarkdown(container, text, codeButton) {
  const fragment = document.createDocumentFragment();
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  let paragraph = [];
  function flushParagraph() {
    if (!paragraph.length) return;
    const node = document.createElement('p');
    appendInline(node, paragraph.join('\n'));
    fragment.append(node);
    paragraph = [];
  }
  // Model output is inserted as DOM text. Only explicit HTTP(S) links become interactive.
  for (let i = 0; i < lines.length;) {
    const line = lines[i++];
    const fence = line.match(fenceStart);
    const heading = line.match(headingStart);
    const header = line.includes('|') ? tableCells(line) : [];
    const alignment = header.length ? tableAlignment(lines[i]) : null;
    if (fence) {
      flushParagraph();
      const marker = fence[1];
      const close = new RegExp('^ {0,3}' + marker[0] + '{' + marker.length + ',}\\s*$');
      const start = i;
      while (i < lines.length && !close.test(lines[i])) i++;
      const text = lines.slice(start, i).join('\n') + (i < lines.length && i > start ? '\n' : '');
      i++;
      const block = document.createElement('div');
      block.className = 'code-block';
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = text;
      pre.append(code);
      if (codeButton) block.append(codeButton(text));
      block.append(pre);
      fragment.append(block);
    } else if (heading) {
      flushParagraph();
      const node = document.createElement('h3');
      appendInline(node, heading[1]);
      fragment.append(node);
    } else if (alignment?.length === header.length) {
      flushParagraph();
      i = appendTable(fragment, lines, i + 1, header, alignment);
    } else if (listStart.test(line)) {
      flushParagraph();
      i = appendList(fragment, lines, i - 1);
    } else if (!line.trim()) flushParagraph();
    else paragraph.push(line);
  }
  flushParagraph();
  container.replaceChildren(fragment);
}
