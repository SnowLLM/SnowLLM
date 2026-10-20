import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../web/status/time.js', import.meta.url), 'utf8');
const { localDate, localTimestamp, selectedRange, advance } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

function inZone(zone, callback) {
  const previous = process.env.TZ;
  process.env.TZ = zone;
  try { callback(); }
  finally {
    if (previous === undefined) delete process.env.TZ;
    else process.env.TZ = previous;
  }
}

test('timestamps and date bounds follow the browser time zone across midnight', () => {
  for (const [zone, date, timestamp] of [
    ['UTC', '2026-10-20', '2026-10-20 00:15:30 UTC+00:00'],
    ['America/Los_Angeles', '2026-10-19', '2026-10-19 17:15:30 UTC-07:00'],
    ['Asia/Shanghai', '2026-10-20', '2026-10-20 08:15:30 UTC+08:00'],
    ['Asia/Kathmandu', '2026-10-20', '2026-10-20 06:00:30 UTC+05:45'],
  ]) inZone(zone, () => {
    assert.equal(localDate('2026-10-20T00:15:30Z'), date);
    assert.equal(localTimestamp('2026-10-20T00:15:30Z', { seconds: true, offset: true }), timestamp);
  });
  inZone('Asia/Shanghai', () => assert.equal(localDate('2026-12-31T20:00:00Z'), '2027-01-01'));
});

test('the repeated hour at the end of daylight saving time remains distinguishable', () => {
  inZone('America/Los_Angeles', () => {
    assert.equal(localTimestamp('2026-11-01T08:00:00Z', { offset: true }), '2026-11-01 01:00 UTC-07:00');
    assert.equal(localTimestamp('2026-11-01T09:00:00Z', { offset: true }), '2026-11-01 01:00 UTC-08:00');
  });
});

test('local day filters include 23 or 25 hourly buckets on daylight saving transitions', () => {
  inZone('America/Los_Angeles', () => {
    for (const [date, start, end, hours] of [
      ['2026-03-08', '2026-03-08T08:00:00.000Z', '2026-03-09T07:00:00.000Z', 23],
      ['2026-11-01', '2026-11-01T07:00:00.000Z', '2026-11-02T08:00:00.000Z', 25],
    ]) {
      const { from, until } = selectedRange(date, date, 'hour');
      assert.equal(from.toISOString(), start);
      assert.equal(until.toISOString(), end);
      let count = 0;
      for (let time = from; time < until; time = advance(time, 'hour')) count++;
      assert.equal(count, hours);
    }
  });
});

test('positive offsets and fractional offsets include the intersecting UTC hours', () => {
  inZone('Asia/Shanghai', () => {
    const { from, until } = selectedRange('2026-10-20', '2026-10-20', 'hour');
    assert.equal(from.toISOString(), '2026-10-19T16:00:00.000Z');
    assert.equal(until.toISOString(), '2026-10-20T16:00:00.000Z');
  });
  inZone('Asia/Kathmandu', () => {
    const { from, until } = selectedRange('2026-10-20', '2026-10-20', 'hour');
    assert.equal(from.toISOString(), '2026-10-19T18:00:00.000Z');
    assert.equal(until.toISOString(), '2026-10-20T19:00:00.000Z');
  });
});

test('coarse filters retain whole UTC reporting periods without adding an aligned extra period', () => {
  inZone('UTC', () => {
    for (const [unit, start, end, expectedEnd] of [
      ['day', '2026-10-20', '2026-10-20', '2026-10-21T00:00:00.000Z'],
      ['week', '2026-10-19', '2026-10-25', '2026-10-26T00:00:00.000Z'],
      ['month', '2024-02-01', '2024-02-29', '2024-03-01T00:00:00.000Z'],
    ]) {
      const { from, until } = selectedRange(start, end, unit);
      assert.equal(from.toISOString(), start + 'T00:00:00.000Z');
      assert.equal(until.toISOString(), expectedEnd);
    }
  });
  inZone('Asia/Shanghai', () => {
    const { from, until } = selectedRange('2026-10-20', '2026-10-20', 'day');
    assert.equal(from.toISOString(), '2026-10-19T00:00:00.000Z');
    assert.equal(until.toISOString(), '2026-10-21T00:00:00.000Z');
  });
});
