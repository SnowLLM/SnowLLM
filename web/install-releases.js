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

export function latestReleases(report) {
  if (!report?.releases || typeof report.releases !== 'object' || Array.isArray(report.releases)) {
    throw new Error('Invalid PyPI release metadata')
  }
  const releases = { stable: null, dev: null }
  const ranks = {}
  for (const [version, files] of Object.entries(report.releases)) {
    const rank = versionRank(version)
    if (!rank || !Array.isArray(files) || !files.some(file =>
      file && !file.yanked && file.filename?.endsWith('-py3-none-any.whl'))) continue
    const channel = rank[3] === 0 ? 'dev' : 'stable'
    if (!releases[channel] || compareRanks(rank, ranks[channel]) > 0) {
      releases[channel] = version
      ranks[channel] = rank
    }
  }
  if (releases.dev && releases.stable && compareRanks(ranks.dev, ranks.stable) <= 0) {
    releases.dev = null
  }
  return releases
}
