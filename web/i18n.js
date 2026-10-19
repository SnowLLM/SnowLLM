const messages = {}
let language = 'en'
const listeners = new Set()
export function registerMessages(catalogues) { Object.assign(messages, catalogues) }
export function getLanguage() { return language }
export function t(key, values = {}, locale = language) {
  const text = messages[locale]?.[key] ?? messages.en?.[key] ?? key
  return text.replace(/\{(\w+)\}/g, (match, name) => String(values[name] ?? match))
}
export function onLanguageChange(callback) { listeners.add(callback); return () => listeners.delete(callback) }
export function rememberLanguage(locale) {
  try { localStorage.setItem('snowllm-language', locale) } catch {}
}
export function applyTranslations(root = document) {
  root.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n) })
  root.querySelectorAll('[data-i18n-html]').forEach(node => { node.innerHTML = t(node.dataset.i18nHtml) })
  for (const attr of ['content', 'aria-label', 'placeholder']) {
    root.querySelectorAll(`[data-i18n-${attr}]`).forEach(node => {
      node.setAttribute(attr, t(node.getAttribute(`data-i18n-${attr}`)))
    })
  }
  // Links to docs follow the selected language, including translated HTML copy.
  root.querySelectorAll('a[href^="/docs/"]').forEach(node => {
    const english = node.getAttribute('href').replace(/^\/docs\/zh\//, '/docs/')
    node.setAttribute('href', language === 'zh_cn' ? english.replace(/^\/docs\//, '/docs/zh/') : english)
  })
  document.documentElement.lang = language === 'zh_cn' ? 'zh-CN' : 'en'
  document.querySelector('meta[property="og:locale"]')?.setAttribute('content', language === 'zh_cn' ? 'zh_CN' : 'en_US')
}
export function setLanguage(locale, { notify = true, remember = true } = {}) {
  if (locale !== 'en' && locale !== 'zh_cn') return
  language = locale
  if (typeof document !== 'undefined') {
    if (remember) rememberLanguage(locale)
    applyTranslations()
    if (notify) Array.from(listeners).forEach(callback => callback(locale))
  }
}
export async function initI18n(locale) {
  if (!messages.en || !messages.zh_cn) {
    const pairs = await Promise.all(['en', 'zh_cn'].map(async name => {
      const response = await fetch(`/locales/${name}.json`, { cache: 'no-cache' })
      if (!response.ok) throw new Error(`Cannot load locale ${name}: ${response.status}`)
      return [name, await response.json()]
    }))
    registerMessages(Object.fromEntries(pairs))
  }
  if (!locale) {
    try { locale = localStorage.getItem('snowllm-language') } catch {}
    if (locale !== 'en' && locale !== 'zh_cn') locale = navigator.language.toLowerCase().startsWith('zh') ? 'zh_cn' : 'en'
  }
  setLanguage(locale, { notify: false, remember: false })
}
