import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'SnowLLM',
  description: 'Local LLM inference, hand-tuned for Ryzen AI Max. Installation, guides, and reference.',
  base: '/docs/',
  cleanUrls: true,
  head: [
    ['link', { rel: 'icon', href: '/docs/favicon.svg', type: 'image/svg+xml' }],
    ['script', {}, `(function () {
      try {
        var theme = localStorage.getItem('snowllm-theme');
        if (theme === 'light' || theme === 'dark') {
          localStorage.setItem('vitepress-theme-appearance', theme);
          document.documentElement.classList.toggle('dark', theme === 'dark');
        }
      } catch (_) {}
    })()`],
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'SnowLLM',
    search: { provider: 'local' },
    outline: { level: [2, 3], label: 'On this page' },
    nav: [{ text: 'Home', link: 'https://snowllm.dev/' }],
    sidebar: [
      { text: 'Getting started', items: [
        { text: 'Overview', link: '/' },
        { text: 'Installation', link: '/getting-started/installation' },
        { text: 'Quick start', link: '/getting-started/quick-start' },
      ] },
      { text: 'Guides', items: [
        { text: 'Model downloads', link: '/guides/models' },
        { text: 'Web UI', link: '/webui' },
        { text: 'Integrations', link: '/guides/integrations' },
        { text: 'Performance tuning', link: '/guides/performance' },
        { text: 'FAQ', link: '/faq' },
      ] },
      { text: 'Reference', items: [
        { text: 'CLI options', link: '/reference/cli' },
        { text: 'Configuration', link: '/reference/configuration' },
        { text: 'Supported models', link: '/reference/models' },
      ] },
      { text: 'Development', items: [
        { text: 'Benchmark methodology', link: '/development/benchmarks' },
      ] },
    ],
    editLink: { pattern: 'https://github.com/SnowLLM/SnowLLM/edit/main/docs/:path', text: 'Edit this page on GitHub' },
    docFooter: { prev: 'Previous', next: 'Next' },
  },
})
