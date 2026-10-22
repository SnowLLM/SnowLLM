import { defineConfig } from 'vitepress'
import type { DefaultTheme } from 'vitepress'
import en from '../../web/locales/en.json'
import zh from '../../web/locales/zh_cn.json'

function docsTheme(messages: Record<string, string>, prefix = ''): DefaultTheme.Config {
  const text = (key: string) => messages['docs.' + key]
  const item = (key: string, path: string) => ({ text: text(key), link: prefix + path })
  return {
    outline: { level: [2, 3], label: text('outline') },
    sidebarMenuLabel: text('menu'), returnToTopLabel: text('top'), darkModeSwitchLabel: messages['theme.dark'],
    sidebar: [
      { text: text('gettingStarted'), items: [item('overview', '/'), item('installation', '/getting-started/installation'), item('quickStart', '/getting-started/quick-start')] },
      { text: text('guides'), items: [item('models', '/guides/models'), item('webui', '/webui'), item('integrations', '/guides/integrations'), item('performance', '/guides/performance'), item('windowsPagefile', '/guides/windows-pagefile'), item('faq', '/faq')] },
      { text: text('reference'), items: [item('cli', '/reference/cli'), item('configuration', '/reference/configuration'), item('supportedModels', '/reference/models'), item('telemetry', '/reference/telemetry')] },
      { text: text('development'), items: [item('benchmarks', '/development/benchmarks')] },
    ],
    editLink: { pattern: 'https://github.com/SnowLLM/SnowLLM/edit/main/docs/:path', text: text('edit') },
    docFooter: { prev: text('previous'), next: text('next') },
  }
}

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
  locales: {
    root: { label: 'English', lang: 'en', themeConfig: docsTheme(en) },
    zh: { label: '简体中文', lang: 'zh-CN', description: zh['home.description'], themeConfig: docsTheme(zh, '/zh') },
  },
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'SnowLLM',
    search: { provider: 'local', options: { locales: { zh: { translations: {
      button: { buttonText: zh['docs.search'], buttonAriaLabel: zh['docs.search'] },
      modal: {
        displayDetails: zh['docs.details'], resetButtonTitle: zh['docs.reset'], backButtonTitle: zh['docs.close'],
        noResultsText: zh['docs.noResults'],
        footer: { selectText: zh['docs.select'], navigateText: zh['docs.navigate'], closeText: zh['docs.closeHint'] },
      },
    } } } } },
  },
})
