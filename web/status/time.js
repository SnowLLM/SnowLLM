const pad = value => String(value).padStart(2, '0');

export function localDate(value) {
  const date = new Date(value);
  return `${String(date.getFullYear()).padStart(4, '0')}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function localTimestamp(value, { seconds = false, offset = false } = {}) {
  const date = new Date(value);
  let text = `${localDate(date)} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  if (seconds) text += ':' + pad(date.getSeconds());
  if (offset) {
    const minutes = -date.getTimezoneOffset();
    text += ` UTC${minutes < 0 ? '-' : '+'}${pad(Math.floor(Math.abs(minutes) / 60))}:${pad(Math.abs(minutes) % 60)}`;
  }
  return text;
}

export function bucket(date, unit) {
  const result = new Date(date);
  result.setUTCMinutes(0, 0, 0);
  if (unit !== 'hour') result.setUTCHours(0);
  if (unit === 'week') result.setUTCDate(result.getUTCDate() - (result.getUTCDay() + 6) % 7);
  if (unit === 'month') result.setUTCDate(1);
  return result;
}

export function advance(date, unit, steps = 1) {
  const result = new Date(date);
  if (unit === 'month') result.setUTCMonth(result.getUTCMonth() + steps);
  else if (unit === 'hour') result.setUTCHours(result.getUTCHours() + steps);
  else result.setUTCDate(result.getUTCDate() + steps * (unit === 'week' ? 7 : 1));
  return result;
}

export function selectedRange(from, to, unit) {
  const start = new Date(from + 'T00:00:00');
  const end = new Date(to + 'T00:00:00');
  end.setDate(end.getDate() + 1);
  const until = bucket(end, unit);
  return { from: bucket(start, unit), until: +until === +end ? until : advance(until, unit) };
}
