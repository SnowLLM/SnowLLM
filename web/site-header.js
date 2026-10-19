import { t, getLanguage, setLanguage, onLanguageChange, applyTranslations, rememberLanguage } from './i18n.js'
const pages = [['home', '/'], ['benchmark', '/benchmark/'], ['recipe', '/recipe/'], ['docs', '/docs/'], ['status', '/status/']]

export function headerMarkup(current = '/', locale = getLanguage()) {
  const text = key => t(key, {}, locale)
  const logo = current === '/docs/' ? '/docs/logo.svg' : '/logo.svg'
  return `<header class="snow-header"><div class="snow-header-inner">
    <a class="snow-brand" target="_self" href="/" data-i18n-aria-label="nav.homeLabel" aria-label="${text('nav.homeLabel')}"><img src="${logo}" alt="" width="28" height="28">SnowLLM</a>
    <nav class="snow-navigation" data-i18n-aria-label="nav.main" aria-label="${text('nav.main')}">${pages.map(([key, path]) => {
      const url = key === 'docs' && locale === 'zh_cn' ? '/docs/zh/' : path
      return `<a target="_self" href="${url}" data-i18n="nav.${key}"${path === current ? ' aria-current="page"' : ''}>${text('nav.' + key)}</a>`
    }).join('')}</nav>
    <div class="snow-header-tools"><a href="https://github.com/SnowLLM/SnowLLM">GitHub ↗</a>
      <select class="snow-language" aria-label="${text('language.label')}" data-i18n-aria-label="language.label"><option value="en"${locale === 'en' ? ' selected' : ''}>English</option><option value="zh_cn"${locale === 'zh_cn' ? ' selected' : ''}>简体中文</option></select>
      <button type="button" class="snow-theme" aria-label="${text('theme.toDark')}"><span aria-hidden="true">◐</span><span class="snow-theme-label">${text('theme.dark')}</span></button>
    </div>
  </div></header>`
}

export function setupHeader(node, options = {}) {
  const root = document.documentElement
  const media = matchMedia('(prefers-color-scheme: dark)')
  const button = node.querySelector('.snow-theme')
  const select = node.querySelector('.snow-language')
  const dark = options.isDark || (() => root.dataset.theme ? root.dataset.theme === 'dark' : media.matches)
  function refresh() {
    button.setAttribute('aria-label', t(dark() ? 'theme.toLight' : 'theme.toDark'))
    button.querySelector('.snow-theme-label').textContent = t(dark() ? 'theme.light' : 'theme.dark')
    select.value = getLanguage()
  }
  function toggle() {
    if (options.onToggle) options.onToggle()
    else {
      const next = dark() ? 'light' : 'dark'
      root.dataset.theme = next
      try { localStorage.setItem('snowllm-theme', next) } catch {}
    }
    refresh()
  }
  function changeLanguage() {
    if (options.onLanguage) { rememberLanguage(select.value); options.onLanguage(select.value) }
    else setLanguage(select.value)
  }
  button.addEventListener('click', toggle)
  select.addEventListener('change', changeLanguage)
  media.addEventListener('change', refresh)
  const stop = onLanguageChange(refresh)
  applyTranslations(node)
  refresh()
  return { refresh, dispose() { stop(); button.removeEventListener('click', toggle); select.removeEventListener('change', changeLanguage); media.removeEventListener('change', refresh) } }
}
