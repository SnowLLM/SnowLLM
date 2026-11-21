import { t } from './i18n.js'

const repo = 'https://github.com/SnowLLM/SnowLLM'
const icons = {
  stars: '<path d="m8 1.5 2 4.2 4.7.7-3.4 3.3.8 4.7L8 12.2l-4.1 2.2.8-4.7L1.3 6.4 6 5.7 8 1.5Z"/>',
  open_issues: '<circle cx="8" cy="8" r="6.25"/><path d="M8 4.5v4m0 2.75h.01"/>',
  closed_issues: '<circle cx="8" cy="8" r="6.25"/><path d="m5 8 2 2 4-4"/>',
  discussions: '<path d="M14.25 7.5a5 5 0 0 1-5 5h-2l-3.5 2v-2.7A5 5 0 0 1 6.75 2.5h2.5a5 5 0 0 1 5 5Z"/><path d="M5 7.5h6"/>',
}
const destinations = {
  stars: '',
  open_issues: '/issues?q=is%3Aissue+is%3Aopen',
  closed_issues: '/issues?q=is%3Aissue+is%3Aclosed',
  discussions: '/discussions',
}
let countsRequest

export function githubBadgeMarkup(name, locale, { compact = false, label = name } = {}) {
  return `<a class="github-badge github-badge-${name}${compact ? ' github-badge-compact' : ''}" href="${repo}${destinations[name]}">
    <svg aria-hidden="true" viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">${icons[name]}</svg>
    <span class="github-badge-label" data-i18n="github.${label}">${t(`github.${label}`, {}, locale)}</span>
    <span class="github-badge-count" data-github-count="${name}" hidden></span>
  </a>`
}

export function githubCommunityMarkup({ stars }) {
  return `<div class="github-issues" role="group" aria-label="${t('github.issues')}" data-i18n-aria-label="github.issues">
    <span class="github-issues-label" aria-hidden="true" data-i18n="github.issues">${t('github.issues')}</span>
    ${githubBadgeMarkup('open_issues', undefined, { label: 'open' })}
    ${githubBadgeMarkup('closed_issues', undefined, { label: 'closed' })}
  </div>${githubBadgeMarkup('discussions')}${stars ? githubBadgeMarkup('stars') : ''}`
}

export function setupGithubBadges(root) {
  const nodes = root.querySelectorAll('[data-github-count]')
  if (!nodes.length) return
  if (!countsRequest) {
    const base = document.querySelector('meta[name="snowllm-status-api"]')?.content || 'https://status.snowllm.dev'
    countsRequest = fetch(`${base}/data/github.json`, { credentials: 'omit' })
      .then(response => response.ok ? response.json() : null)
      .catch(() => null)
  }
  countsRequest.then(counts => {
    for (const node of nodes) {
      const value = counts?.[node.dataset.githubCount]
      if (Number.isSafeInteger(value) && value >= 0) {
        node.textContent = new Intl.NumberFormat(document.documentElement.lang).format(value)
        node.hidden = false
      }
    }
  })
}
