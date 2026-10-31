import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'

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

test('dev is hidden when stable catches up or moves past it', () => {
  assert.deepEqual(latestReleases(report([
    '0.3.9.dev1', '0.3.9.dev10', '0.3.9.dev2', '0.3.9', '0.3.8',
  ])), { stable: '0.3.9', dev: null })
  assert.deepEqual(latestReleases(report([
    '0.3.9', '0.3.10.dev2', '0.3.9.dev99', '0.3.10', '0.3.10.post1',
  ])), { stable: '0.3.10.post1', dev: null })
  assert.deepEqual(latestReleases(report([
    '0.3.19.dev4', '0.3.19', '0.3.20',
  ])), { stable: '0.3.20', dev: null })
})

test('newer dev releases remain visible with numeric version ordering', () => {
  assert.deepEqual(latestReleases(report([
    '0.3.9', '0.3.10.dev2', '0.3.10.dev10', '0.3.10.dev9',
  ])), { stable: '0.3.9', dev: '0.3.10.dev10' })
  assert.deepEqual(latestReleases(report([
    '0.3.19', '0.3.20.dev1',
  ])), { stable: '0.3.19', dev: '0.3.20.dev1' })
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

test('the install selector hides superseded dev and switches its commands to stable', async () => {
  const source = await readFile(new URL('../web/install-ui.js', import.meta.url), 'utf8')
  const render = source.slice(source.indexOf('  function renderInstall()'),
    source.indexOf('  document.querySelectorAll(".opts[data-axis] .opt")'))
  function button(value) {
    return { dataset: { value }, setAttribute(name, value) { this[name] = value } }
  }
  for (const [versions, target, expected] of [
    [['0.3.19.dev4', '0.3.19'], 'gfx1151', 'stable'],
    [['0.3.19.dev4', '0.3.20'], 'gfx1103', 'stable'],
    [['0.3.18.dev1', '0.3.18'], 'gfx1103', 'stable'],
    [['0.3.19', '0.3.20.dev1'], 'gfx1151', 'dev'],
    [['0.3.20.dev1'], 'gfx1103', 'dev'],
  ]) {
    const channels = ['stable', 'dev'].map(button)
    const script = button('script')
    const versionsById = Object.fromEntries(['stable', 'dev'].map(channel => [
      channel + '-version', { removeAttribute() {} },
    ]))
    const context = vm.createContext({
      releases: latestReleases(report(versions)), releaseState: 'ready', supportsTarget,
      choice: { channel: 'dev', os: 'linux', how: 'pip', gpu: target, apu: 'ryzen-ai-max' },
      document: {
        querySelectorAll(selector) {
          if (selector.includes('data-axis="channel"')) return channels
          assert.equal(selector, '.opts[data-axis]')
          return [{ dataset: { axis: 'channel' }, querySelectorAll() { return channels } }]
        },
        querySelector() { return script },
        getElementById(id) { return versionsById[id] },
      },
      cmdEl: {}, noteEl: {}, channelNoteEl: {}, copyBtn: {}, targetEl: {},
      TORCH_LINE: '', SNOWLLM_LINE: 'python -m pip install --upgrade ', CURL_LINE: 'curl | sh',
      t: key => key, applyTranslations() {},
    })
    vm.runInContext(render + '\nrenderInstall()', context)
    assert.equal(context.choice.channel, expected, versions.join(', '))
    assert.equal(channels[1].hidden, expected !== 'dev')
    if (target === 'gfx1103' && versions.includes('0.3.18')) {
      assert.equal(context.cmdEl.innerHTML, '')
      assert.equal(context.copyBtn.disabled, true)
    } else if (expected === 'stable') {
      assert.equal(channels[0]['aria-pressed'], 'true')
      assert.match(context.cmdEl.innerHTML, /snowllm==/)
      assert.doesNotMatch(context.cmdEl.innerHTML, /--pre/)
    } else {
      assert.equal(channels[1]['aria-pressed'], 'true')
      assert.match(context.cmdEl.innerHTML, /--pre snowllm/)
    }
  }
})
