import { initI18n, applyTranslations, rememberLanguage } from './i18n.js'
import { headerMarkup, setupHeader } from './site-header.js'
const current = location.pathname.startsWith('/benchmark/') ? '/benchmark/'
  : location.pathname.startsWith('/recipe/') ? '/recipe/'
  : location.pathname.startsWith('/status/') ? '/status/' : '/'
const params = new URLSearchParams(location.search)
const language = params.get('lang')
await initI18n(language === 'en' || language === 'zh_cn' ? language : undefined)
if (language === 'en' || language === 'zh_cn') rememberLanguage(language)
const theme = current === '/status/' ? params.get('theme') : null
if (theme === 'light' || theme === 'dark') {
  document.documentElement.dataset.theme = theme
  try { localStorage.setItem('snowllm-theme', theme) } catch {}
}
const host = document.getElementById('site-header')
host.innerHTML = headerMarkup(current)
setupHeader(host)
applyTranslations()
if (current === '/') await import('./install-ui.js')
else if (current === '/status/') await import('./status/status.js')
else if (current === '/benchmark/') await import('./site.js?v=benchmark-unified-1')
else await import('./site.js')
if (current === '/') {
  function legacyAnchor() {
    const destination = { '#benchmarks': '/benchmark/', '#recipes': '/recipe/' }[location.hash]
    if (destination) location.replace(destination)
  }
  legacyAnchor()
  window.addEventListener('hashchange', legacyAnchor)
}
