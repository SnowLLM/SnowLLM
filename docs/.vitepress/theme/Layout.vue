<script setup>
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme-without-fonts'
import { headerMarkup, setupHeader } from '../../../web/site-header.js'
const { Layout } = DefaultTheme
import en from '../../../web/locales/en.json'
import zh from '../../../web/locales/zh_cn.json'
import { registerMessages, initI18n, rememberLanguage, t } from '../../../web/i18n.js'
registerMessages({ en, zh_cn: zh })
const { isDark, lang } = useData()
const locale = computed(() => lang.value === 'zh-CN' ? 'zh_cn' : 'en')
const text = key => t(key, {}, locale.value)
const header = ref(null)
let controls
onMounted(async () => {
  await initI18n(locale.value)
  rememberLanguage(locale.value)
  try {
    const saved = localStorage.getItem('snowllm-theme')
    if (saved === 'light' || saved === 'dark') isDark.value = saved === 'dark'
  } catch {}
  controls = setupHeader(header.value, { isDark: () => isDark.value, onToggle: () => { isDark.value = !isDark.value }, onLanguage: (next) => {
    const english = location.pathname.replace(/^\/docs\/zh\//, '/docs/')
    const destination = next === 'zh_cn' ? english.replace(/^\/docs\//, '/docs/zh/') : english
    location.assign(destination + location.search + location.hash)
  } })
})
onBeforeUnmount(() => controls?.dispose())
watch(isDark, (dark) => {
  try { localStorage.setItem('snowllm-theme', dark ? 'dark' : 'light') } catch {}
  controls?.refresh()
})
function search() {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }))
}
</script>

<template>
  <Layout>
    <template #layout-top>
      <div ref="header" v-html="headerMarkup('/docs/', locale)" />
    </template>
    <template #sidebar-nav-before>
      <div class="docs-brand">
        <div class="docs-caption">{{ text('docs.title') }}</div>
        <button class="docs-search" type="button" @click="search">
          <span>{{ text('docs.search') }}</span><kbd>⌘ / Ctrl K</kbd>
        </button>
      </div>
    </template>
    <template #sidebar-nav-after>
      <div class="docs-rail-footer">
        <a href="https://github.com/SnowLLM/SnowLLM/discussions">{{ text('docs.discuss') }}</a>
      </div>
    </template>
    <template #doc-before>
      <div class="doc-eyebrow">SnowLLM / {{ text('docs.title') }}</div>
      <button class="docs-search docs-mobile-search" type="button" @click="search"><span>{{ text('docs.search') }}</span><kbd>⌘ / Ctrl K</kbd></button>
    </template>
  </Layout>
</template>
