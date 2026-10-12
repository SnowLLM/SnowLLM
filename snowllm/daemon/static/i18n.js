// SPDX-License-Identifier: Apache-2.0
// Language packs are local Web UI assets; missing entries fall back to English.
const packs = Object.fromEntries(await Promise.all(['en', 'zh_CN'].map(async name => {
  try {
    const response = await fetch(new URL(`./locales/${name}.json`, import.meta.url));
    if (!response.ok) throw new Error('Language pack unavailable');
    return [name, await response.json()];
  } catch { return [name, {}]; }
})));
let preference = 'auto';
try { const saved = localStorage.getItem('snowllm.language'); if (['auto', 'en', 'zh_CN'].includes(saved)) preference = saved; } catch {}
export function language() { return preference; }
function locale() {
  if (preference !== 'auto') return preference;
  const first = navigator.languages?.[0] || navigator.language || 'en';
  return /^zh(?:[-_]|$)/i.test(first) ? 'zh_CN' : 'en';
}
export function setLanguage(value) {
  preference = ['auto', 'en', 'zh_CN'].includes(value) ? value : 'auto';
  try { if (preference === 'auto') localStorage.removeItem('snowllm.language'); else localStorage.setItem('snowllm.language', preference); } catch {}
}
export function t(key, ...values) {
  const text = packs[locale()]?.[key] ?? packs.en?.[key] ?? key;
  return text.replace(/\{(\d+)\}/g, (_, index) => String(values[Number(index)] ?? ''));
}
export function translatePage() {
  document.documentElement.lang = locale() === 'zh_CN' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(node => node.textContent = t(node.dataset.i18n));
  for (const attr of ['placeholder', 'aria-label', 'title']) {
    document.querySelectorAll(`[data-i18n-${attr}]`).forEach(node => node.setAttribute(attr, t(node.getAttribute(`data-i18n-${attr}`))));
  }
}
window.addEventListener('languagechange', () => { if (preference === 'auto') document.getElementById('language')?.dispatchEvent(new Event('change')); });
