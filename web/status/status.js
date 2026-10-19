import { t, getLanguage, onLanguageChange } from '../i18n.js'

const tr = key => t('status.' + key)
const url = new URL(location.href)
url.searchParams.delete('theme')
const api = document.body.dataset.apiBase.replace(/\/$/, '')
const periods = ['hour', 'day', 'week', 'month'];
let period = periods.includes(url.searchParams.get('period')) ? url.searchParams.get('period') : 'day';
const form = document.querySelector('.date-filter');
const startInput = form.elements.from, endInput = form.elements.to;
const windows = {hour: 48, day: 30, week: 12, month: 12};
const iso = date => date.toISOString().replace('.000Z', 'Z');
const compact = value => {
  for (const [scale, suffix] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
    if (value >= scale) return (value / scale).toFixed(2).replace(/\.?0+$/, '') + suffix;
  }
  return String(value);
};
function bucket(date, unit) {
  const result = new Date(date);
  result.setUTCMinutes(0, 0, 0);
  if (unit !== 'hour') result.setUTCHours(0);
  if (unit === 'week') result.setUTCDate(result.getUTCDate() - (result.getUTCDay() + 6) % 7);
  if (unit === 'month') result.setUTCDate(1);
  return result;
}
function advance(date, unit, steps = 1) {
  const result = new Date(date);
  if (unit === 'month') result.setUTCMonth(result.getUTCMonth() + steps);
  else if (unit === 'hour') result.setUTCHours(result.getUTCHours() + steps);
  else result.setUTCDate(result.getUTCDate() + steps * (unit === 'week' ? 7 : 1));
  return result;
}
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
    const caption = `${p.period_start}: prefill ${p.prefill_tokens.toLocaleString('en-US')}, decode ${p.decode_tokens.toLocaleString('en-US')}${p.partial ? ' · ' + tr('partial') : ''}`;
    svg += `<g class="bar" tabindex="0" role="img" aria-label="${caption}"><title>${caption}</title><rect class="prefill" x="${x}" y="${bottom - hp}" width="${barWidth}" height="${hp}"/><rect class="decode" x="${x}" y="${bottom - hp - hd}" width="${barWidth}" height="${hd}"/></g>`;
    if (i % every === 0) {
      const label = period === 'hour' ? p.period_start.slice(5, 10) + ' ' + p.period_start.slice(11, 16) : period === 'month' ? p.period_start.slice(0, 7) : p.period_start.slice(5, 10);
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
  function updateBounds() {
    data = sources[period];
    now = new Date(data.generated_at);
    firstDate = data.available_from.slice(0, 10);
    lastDate = data.generated_at.slice(0, 10);
    for (const input of [startInput, endInput]) { input.min = firstDate; input.max = lastDate; }
  }
  function covers(key, from, to) {
    return /^\d{4}-\d{2}-\d{2}$/.test(from) && /^\d{4}-\d{2}-\d{2}$/.test(to)
      && from <= to && from >= sources[key].available_from.slice(0, 10)
      && to <= sources[key].generated_at.slice(0, 10);
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
    let from = custom ? bucket(new Date(url.searchParams.get('from') + 'T00:00:00Z'), period === 'hour' ? 'day' : period) : advance(current, period, 1 - windows[period]);
    let until = custom ? advance(bucket(new Date(url.searchParams.get('to') + 'T00:00:00Z'), period === 'hour' ? 'day' : period), period === 'hour' ? 'day' : period) : advance(current, period);
    // The first retained hour may begin part-way through a UTC day.
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
    document.querySelector('.range').textContent = custom ? `${iso(from).slice(0, period === 'hour' ? 16 : 10).replace('T', ' ')} — ${new Date(until - 1).toISOString().slice(0, period === 'hour' ? 16 : 10).replace('T', ' ')} · UTC` : tr('range.' + period);
    const coverage = document.querySelector('.coverage');
    coverage.textContent = `${tr('coverage')} ${data.available_from.slice(0, 16).replace('T', ' ')} — ${data.generated_at.slice(0, 16).replace('T', ' ')} UTC · ${tr('coarseHint')}`;
    coverage.hidden = false;
    document.querySelectorAll('.periods a').forEach(anchor => {
      const key = new URL(anchor.href).searchParams.get('period');
      if (key === period) anchor.setAttribute('aria-current', 'page');
      else anchor.removeAttribute('aria-current');
      const unavailable = custom && !covers(key, url.searchParams.get('from'), url.searchParams.get('to'));
      if (unavailable) { anchor.setAttribute('aria-disabled', 'true'); anchor.title = tr('rangeUnavailable'); }
      else { anchor.removeAttribute('aria-disabled'); anchor.removeAttribute('title'); }
    });
    document.querySelector('.plot-container').innerHTML = prefill || decode ? chart(points, 960) + chart(points, 400)
      : `<div class="empty"><h3>${tr('empty')}</h3><p>${tr('emptyDesc')}</p></div>`;
    document.querySelector('.usage-table tbody').innerHTML = points.slice().reverse().map(p => {
      const moment = period === 'hour' ? p.period_start.slice(0, 16).replace('T', ' ') : p.period_start.slice(0, 10);
      return `<tr><th scope="row">${moment}${p.partial ? '<span class="partial">' + tr('partial') + '</span>' : ''}</th><td>${p.prefill_tokens.toLocaleString('en-US')}</td><td>${p.decode_tokens.toLocaleString('en-US')}</td><td>${(p.prefill_tokens + p.decode_tokens).toLocaleString('en-US')}</td></tr>`;
    }).join('');
    document.querySelectorAll('.freshness time').forEach((time, i) => {
      const value = i ? data.next_update_at : data.generated_at;
      time.dateTime = value; time.textContent = i ? value.slice(11, 16) + ' UTC' : value.replace('T', ' ').replace('Z', ' UTC');
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
