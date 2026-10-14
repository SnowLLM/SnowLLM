<script setup>
import { onMounted, watch } from 'vue'
import { useData, withBase } from 'vitepress'
import ThemeToggle from './ThemeToggle.vue'
import DefaultTheme from 'vitepress/theme-without-fonts'
const { Layout } = DefaultTheme
const { isDark } = useData()
onMounted(() => {
  try {
    const saved = localStorage.getItem('snowllm-theme')
    if ((saved === 'light' || saved === 'dark') && isDark.value !== (saved === 'dark')) isDark.value = saved === 'dark'
  } catch {}
})
watch(isDark, (dark) => {
  try { localStorage.setItem('snowllm-theme', dark ? 'dark' : 'light') } catch {}
})
function search() {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }))
}
</script>

<template>
  <Layout>
    <template #sidebar-nav-before>
      <div class="docs-brand">
        <a class="brand-link" href="/" aria-label="SnowLLM home">
          <img :src="withBase('/logo.svg')" alt="" />
          <span>SnowLLM</span>
        </a>
        <a class="brand-caption" :href="withBase('/')">Documentation</a>
        <ThemeToggle class="rail-theme" />
        <button class="docs-search" type="button" @click="search">
          <span>Search docs</span><kbd>⌘ / Ctrl K</kbd>
        </button>
      </div>
    </template>
    <template #sidebar-nav-after>
      <div class="docs-rail-footer">
        <div class="rail-links">
          <a href="/">Home ↗</a>
          <a href="https://github.com/SnowLLM/SnowLLM">GitHub ↗</a>
          <a href="https://github.com/SnowLLM/SnowLLM/discussions">Discuss ↗</a>
        </div>

      </div>
    </template>
    <template #nav-bar-content-after>
      <ThemeToggle class="mobile-theme" />
    </template>
    <template #doc-before>
      <div class="doc-eyebrow">SnowLLM / Documentation</div>
    </template>
  </Layout>
</template>
