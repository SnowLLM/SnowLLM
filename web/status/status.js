import { t, getLanguage, onLanguageChange } from '../i18n.js'
import { localDate, localTimestamp, bucket, advance, selectedRange } from './time.js'

const tr = key => t('status.' + key)
const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
const url = new URL(location.href)
url.searchParams.delete('theme')
const api = document.body.dataset.apiBase.replace(/\/$/, '')
const periods = ['hour', 'day', 'week', 'month'];
let period = periods.includes(url.searchParams.get('period')) ? url.searchParams.get('period') : 'day';
const form = document.querySelector('.date-filter');
const startInput = form.elements.from, endInput = form.elements.to;
const datePickers = Array.from(form.querySelectorAll('[data-date-input]'), picker => {
  const input = document.getElementById(picker.dataset.dateInput);
  input.addEventListener('input', () => { picker.value = input.value; });
  picker.addEventListener('change', () => { input.value = picker.value; });
  picker.addEventListener('click', event => {
    if (typeof picker.showPicker === 'function') { event.preventDefault(); picker.showPicker(); }
  });
  return { input, picker };
});
function isDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(value + 'T00:00:00Z');
  return Number.isFinite(+date) && date.toISOString().slice(0, 10) === value;
}
const windows = {hour: 48, day: 30, week: 12, month: 12};
const iso = date => date.toISOString().replace('.000Z', 'Z');
const compact = value => {
  for (const [scale, suffix] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
    if (value >= scale) return (value / scale).toFixed(2).replace(/\.?0+$/, '') + suffix;
  }
  return String(value);
};
function chart(points, width) {
  const left = 52, right = width - 10, top = 20, bottom = 256;
  const max = points.reduce((value, p) => Math.max(value, p.prefill_tokens + p.decode_tokens), 0);
  const unit = max ? 10 ** Math.floor(Math.log10(max)) : 1;
  const ceiling = max ? Math.ceil(max / unit / .5) * unit * .5 : 100;
  const scale = (bottom - top) / ceiling, step = (right - left) / points.length;
  const barWidth = step * .65, every = Math.ceil(points.length / (width < 600 ? 4 : 7));
  let svg = `<svg class="plot plot-${width}" viewBox="0 0 ${width} 300" role="img" aria-labelledby="chart-title-${width} chart-desc-${width}"><title id="chart-title-${width}">${tr('chart')}</title><desc id="chart-desc-${width}">${tr('chartDesc')}</desc>`;
  for (let i = 0; i < 5; i++) {
    const value = ceiling * i / 4, y = bottom - value * scale;
    svg += `<line class="grid" x1="${left}" x2="${right}" y1="${y}" y2="${y}"/><text class="plot-axis" text-anchor="end" x="${left - 10}" y="${y + 4}">${compact(Math.round(value))}</text>`;
  }
  points.forEach((p, i) => {
    const x = left + step * i + (step - barWidth) / 2;
    const hp = p.prefill_tokens * scale, hd = p.decode_tokens * scale;
    const caption = `${localTimestamp(p.period_start, { offset: true })}: prefill ${p.prefill_tokens.toLocaleString('en-US')}, decode ${p.decode_tokens.toLocaleString('en-US')}${p.partial ? ' · ' + tr('partial') : ''}`;
    svg += `<g class="bar" data-index="${i}" tabindex="0" role="img" aria-label="${caption}"><rect class="bar-hit" x="${left + step * i}" y="${top}" width="${step}" height="${bottom - top}"/><rect class="prefill" x="${x}" y="${bottom - hp}" width="${barWidth}" height="${hp}"/><rect class="decode" x="${x}" y="${bottom - hp - hd}" width="${barWidth}" height="${hd}"/></g>`;
    if (i % every === 0) {
      const label = period === 'hour' ? localTimestamp(p.period_start).slice(5) : localDate(p.period_start).slice(5);
      svg += `<text class="plot-axis" text-anchor="middle" x="${x + barWidth / 2}" y="280">${label}</text>`;
    }
  });
  return svg + '</svg>';
}
function updateLinks() {
  document.querySelectorAll('.periods a').forEach(anchor => {
    const next = new URL(url);
    next.searchParams.set('period', new URL(anchor.href).searchParams.get('period'));
    anchor.href = next;
  });
}
async function loadSources() {
  for (const cache of ['default', 'reload']) {
    const entries = await Promise.all(periods.map(async key => {
      const response = await fetch(api + '/data/' + key + '.json', {
        credentials: 'omit', cache, signal: AbortSignal.timeout(10000)
      });
      if (!response.ok) throw Error('Cannot load usage data');
      const data = await response.json();
      if (data.period !== key || !Array.isArray(data.points) || !data.points.length || data.points.length > 1000) throw Error('Invalid usage data');
      return [key, data];
    }));
    // Separate HTTP requests may straddle an hourly publication or cached versions.
    if (new Set(entries.map(([, data]) => data.generated_at)).size === 1) return Object.fromEntries(entries);
  }
  throw Error('Usage snapshots do not match');
}
updateLinks();
loadSources().then(sources => {
  let data, now, firstDate, lastDate;
  const plot = document.querySelector('.plot-container');
  const tooltip = document.getElementById('usage-tooltip');
  let plottedPoints = [], activeBar;
  function hideTooltip() {
    tooltip.hidden = true;
    activeBar?.removeAttribute('aria-describedby');
    activeBar = null;
  }
  function showTooltip(bar, x, y) {
    const point = plottedPoints[Number(bar.dataset.index)];
    if (!point) return;
    if (activeBar !== bar) {
      activeBar?.removeAttribute('aria-describedby');
      activeBar = bar;
      bar.setAttribute('aria-describedby', tooltip.id);
      tooltip.querySelector('.tooltip-date').textContent = localTimestamp(point.period_start, { offset: true });
      for (const [key, value] of Object.entries({ prefill: point.prefill_tokens, decode: point.decode_tokens,
        total: point.prefill_tokens + point.decode_tokens })) {
        tooltip.querySelector(`[data-tooltip-value="${key}"]`).textContent = value.toLocaleString('en-US');
      }
      tooltip.querySelector('.tooltip-partial').hidden = !point.partial;
    }
    tooltip.hidden = false;
    const rect = tooltip.getBoundingClientRect();
    tooltip.style.left = Math.max(8, Math.min(x + 12, innerWidth - rect.width - 8)) + 'px';
    tooltip.style.top = Math.max(8, Math.min(y + 12, innerHeight - rect.height - 8)) + 'px';
  }
  function hoverTooltip(event) {
    const bar = event.target.closest('.bar');
    if (bar) showTooltip(bar, event.clientX, event.clientY);
    else hideTooltip();
  }
  plot.addEventListener('pointermove', hoverTooltip);
  plot.addEventListener('pointerdown', hoverTooltip);
  plot.addEventListener('pointerleave', hideTooltip);
  plot.addEventListener('focusin', event => {
    const bar = event.target.closest('.bar');
    if (!bar) return;
    const rect = bar.getBoundingClientRect();
    showTooltip(bar, rect.x + rect.width / 2, rect.y);
  });
  plot.addEventListener('focusout', hideTooltip);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') hideTooltip(); });
  document.addEventListener('pointerdown', event => { if (!plot.contains(event.target)) hideTooltip(); });
  window.addEventListener('wheel', hideTooltip, { passive: true });
  window.addEventListener('touchmove', hideTooltip, { passive: true });
  window.addEventListener('resize', hideTooltip);
  function updateBounds() {
    data = sources[period];
    now = new Date(data.generated_at);
    firstDate = localDate(data.available_from);
    lastDate = localDate(data.generated_at);
    for (const { input, picker } of datePickers) {
      picker.min = firstDate; picker.max = lastDate; picker.value = input.value;
    }
  }
  function covers(key, from, to) {
    return isDate(from) && isDate(to)
      && from <= to && from >= localDate(sources[key].available_from)
      && to <= localDate(sources[key].generated_at);
  }
  function validRange() { return covers(period, startInput.value, endInput.value); }
  function showError(key = '') {
    const error = form.querySelector('.filter-error');
    error.dataset.message = key; error.textContent = key ? tr(key) : '';
  }
  updateBounds();
  startInput.value = url.searchParams.get('from') || '';
  endInput.value = url.searchParams.get('to') || '';
  if ((startInput.value || endInput.value) && !validRange()) {
    const coarser = periods.slice(periods.indexOf(period) + 1).find(key => covers(key, startInput.value, endInput.value));
    if (coarser) { period = coarser; updateBounds(); }
    else {
      startInput.value = endInput.value = '';
      url.searchParams.delete('from'); url.searchParams.delete('to');
    }
    showError('rangeUnavailable');
  }
  function draw() {
    updateBounds();
    const custom = url.searchParams.has('from') && url.searchParams.has('to');
    const current = bucket(now, period);
    const selected = custom ? selectedRange(url.searchParams.get('from'), url.searchParams.get('to'), period) : null;
    let from = selected ? selected.from : advance(current, period, 1 - windows[period]);
    let until = selected ? selected.until : advance(current, period);
    // Snapshots contain whole UTC buckets, including those crossing local date boundaries.
    if (custom && from < new Date(data.available_from)) from = new Date(data.available_from);
    if (until > advance(current, period)) until = advance(current, period);
    const indexed = new Map(data.points.map(point => [point.period_start, point]));
    const points = [];
    for (let time = from; time < until; time = advance(time, period)) {
      const key = iso(time);
      points.push(indexed.get(key) || {period_start: key, prefill_tokens: 0, decode_tokens: 0, partial: +time === +current});
    }
    const prefill = points.reduce((sum, p) => sum + p.prefill_tokens, 0);
    const decode = points.reduce((sum, p) => sum + p.decode_tokens, 0);
    document.querySelectorAll('.stat dd').forEach((dd, i) => {
      const value = [prefill + decode, prefill, decode][i];
      dd.textContent = compact(value); dd.title = value.toLocaleString('en-US');
    });
    document.querySelector('.range').textContent = custom ? `${localTimestamp(from, { offset: true })} — ${localTimestamp(until - 1, { offset: true })}` : tr('range.' + period);
    document.querySelector('.timezone').textContent = t('status.timezone', { timezone });
    const coverage = document.querySelector('.coverage');
    coverage.textContent = `${tr('coverage')} ${localTimestamp(data.available_from, { offset: true })} — ${localTimestamp(data.generated_at, { offset: true })} · ${tr('coarseHint')}`;
    coverage.hidden = false;
    document.querySelectorAll('.periods a').forEach(anchor => {
      const key = new URL(anchor.href).searchParams.get('period');
      if (key === period) anchor.setAttribute('aria-current', 'page');
      else anchor.removeAttribute('aria-current');
      const unavailable = custom && !covers(key, url.searchParams.get('from'), url.searchParams.get('to'));
      if (unavailable) { anchor.setAttribute('aria-disabled', 'true'); anchor.title = tr('rangeUnavailable'); }
      else { anchor.removeAttribute('aria-disabled'); anchor.removeAttribute('title'); }
    });
    hideTooltip();
    plottedPoints = points;
    plot.innerHTML = prefill || decode ? chart(points, 960) + chart(points, 400)
      : `<div class="empty"><h3>${tr('empty')}</h3><p>${tr('emptyDesc')}</p></div>`;
    document.querySelector('.usage-table tbody').innerHTML = points.slice().reverse().map(p => {
      const moment = localTimestamp(p.period_start, { offset: true });
      return `<tr><th scope="row">${moment}${p.partial ? '<span class="partial">' + tr('partial') + '</span>' : ''}</th><td>${p.prefill_tokens.toLocaleString('en-US')}</td><td>${p.decode_tokens.toLocaleString('en-US')}</td><td>${(p.prefill_tokens + p.decode_tokens).toLocaleString('en-US')}</td></tr>`;
    }).join('');
    document.querySelectorAll('.freshness time').forEach((time, i) => {
      const value = i ? data.next_update_at : data.generated_at;
      time.dateTime = value; time.textContent = localTimestamp(value, { seconds: !i, offset: true });
    });
    document.querySelector('.all-time strong').textContent = compact(data.all_time.prefill_tokens + data.all_time.decode_tokens);
    url.searchParams.set('period', period);
    if (url.searchParams.has('lang')) url.searchParams.set('lang', getLanguage());
    history.replaceState(null, '', url);
    updateLinks();
  }
  document.querySelectorAll('.periods a').forEach(anchor => anchor.addEventListener('click', event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (anchor.getAttribute('aria-disabled') === 'true') {
      showError('rangeUnavailable'); return;
    }
    showError();
    period = new URL(anchor.href).searchParams.get('period'); draw();
  }));
  form.addEventListener('submit', event => {
    event.preventDefault();
    showError(validRange() ? '' : 'invalidRange');
    if (!validRange()) return;
    url.searchParams.set('from', startInput.value); url.searchParams.set('to', endInput.value); draw();
  });
  form.querySelector('.reset-range').addEventListener('click', () => {
    startInput.value = endInput.value = '';
    showError();
    url.searchParams.delete('from'); url.searchParams.delete('to'); draw();
  });
  document.querySelector('.demo').hidden = !sources[period].demo;
  document.querySelector('.loading').hidden = true;
  document.getElementById('usage-report').hidden = false;
  form.hidden = false;
  onLanguageChange(() => {
    const error = form.querySelector('.filter-error');
    if (error.dataset.message) error.textContent = tr(error.dataset.message);
    draw();
  });
  draw();
}).catch(() => { document.querySelector('.loading').hidden = true; document.querySelector('.load-error').hidden = false; });
