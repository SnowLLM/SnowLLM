function versionRank(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+)|\.post(\d+))?$/.exec(version)
  if (!match) return null
  const dev = match[4] !== undefined
  return [...match.slice(1, 4).map(Number), dev ? 0 : match[5] ? 2 : 1,
    Number(match[4] ?? match[5] ?? 0)]
}

function compareRanks(left, right) {
  return left.map((part, index) => part - right[index]).find(value => value !== 0) ?? 0
}

export function supportsTarget(version, target) {
  if (target === 'gfx1151') return true
  const rank = versionRank(version)
  return rank !== null && compareRanks(rank, [0, 3, 19, 0, 2]) >= 0
}

export function installReleases(report) {
  if (!report?.releases || typeof report.releases !== 'object' || Array.isArray(report.releases)) {
    throw new Error('Invalid PyPI release metadata')
  }
  const releases = []
  for (const [version, files] of Object.entries(report.releases)) {
    const rank = versionRank(version)
    if (!rank || !Array.isArray(files)) continue
    const wheel = files.find(file => file && !file.yanked && file.filename?.endsWith('-py3-none-any.whl'))
    if (!wheel) continue
    const date = (wheel.upload_time_iso_8601 || wheel.upload_time || '').slice(0, 10)
    releases.push({ version, date, rank })
  }
  releases.sort((a, b) => compareRanks(b.rank, a.rank))
  return {
    stable: releases.filter(release => release.rank[3] !== 0),
    dev: releases[0]?.rank[3] === 0 ? releases[0].version : null,
  }
}
