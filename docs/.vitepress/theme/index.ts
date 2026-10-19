import DefaultTheme from 'vitepress/theme-without-fonts'
import Layout from './Layout.vue'
import '../../../web/typography.css'
import '../../../web/palette.css'
import './style.css'
import '../../../web/site-header.css'

export default { extends: DefaultTheme, Layout }
