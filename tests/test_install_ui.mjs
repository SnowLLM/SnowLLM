import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const source = await readFile(new URL('../web/install-releases.js', import.meta.url), 'utf8')
const { latestReleases, supportsTarget } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'))
const wheel = { filename: 'snowllm-0.3.9-py3-none-any.whl', yanked: false }
const report = versions => ({ releases: Object.fromEntries(versions.map(version => [version, [wheel]])) })

test('other GPU targets require 0.3.19.dev2, including correct dev and final release ordering', () => {
  for (const target of ['gfx1103', 'gfx1150', 'gfx1152', 'gfx1153']) {
    for (const version of [null, 'invalid', '0.3.9.dev99', '0.3.18', '0.3.18.post1', '0.3.19.dev1']) {
      assert.equal(supportsTarget(version, target), false, `${target}: ${version}`)
    }
    for (const version of ['0.3.19.dev2', '0.3.19.dev10', '0.3.19', '0.3.19.post1', '0.3.20.dev1', '0.4.0']) {
      assert.equal(supportsTarget(version, target), true, `${target}: ${version}`)
    }
  }
  for (const version of [null, '0.3.18', '0.3.19.dev1', '0.3.19.dev2']) {
    assert.equal(supportsTarget(version, 'gfx1151'), true)
  }
})

test('stable and dev are selected independently using numeric version ordering', () => {
  assert.deepEqual(latestReleases(report([
    '0.3.9.dev1', '0.3.9.dev10', '0.3.9.dev2', '0.3.9', '0.3.8',
  ])), { stable: '0.3.9', dev: '0.3.9.dev10' })
  assert.deepEqual(latestReleases(report([
    '0.3.9', '0.3.10.dev2', '0.3.9.dev99', '0.3.10', '0.3.10.post1',
  ])), { stable: '0.3.10.post1', dev: '0.3.10.dev2' })
})

test('unpublished, yanked, prerelease and incompatible artifacts are excluded', () => {
  const data = report(['0.3.9', '0.3.10.dev2', '0.3.11rc1', 'invalid'])
  Object.assign(data.releases, {
    '0.3.12': [],
    '0.3.13': [{ ...wheel, yanked: true }],
    '0.3.14': [{ filename: 'snowllm-0.3.14.tar.gz' }],
    '0.3.15': [{ filename: 'snowllm-0.3.15-cp314-cp314-win_amd64.whl' }],
    '0.3.16.dev1': [null],
  })
  assert.deepEqual(latestReleases(data), { stable: '0.3.9', dev: '0.3.10.dev2' })
  data.releases['0.3.13'].push(wheel)
  assert.equal(latestReleases(data).stable, '0.3.13')
})

test('missing channels remain empty and invalid metadata fails visibly', () => {
  assert.deepEqual(latestReleases(report(['0.3.9'])), { stable: '0.3.9', dev: null })
  assert.deepEqual(latestReleases(report(['0.3.9.dev1'])), { stable: null, dev: '0.3.9.dev1' })
  assert.deepEqual(latestReleases(report([])), { stable: null, dev: null })
  for (const data of [null, {}, { releases: [] }, { releases: 'invalid' }]) {
    assert.throws(() => latestReleases(data), /Invalid PyPI/)
  }
})
