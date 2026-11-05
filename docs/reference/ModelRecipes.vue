<script setup>
import { computed } from 'vue'
import { data } from './models.data'

const props = defineProps({ chinese: Boolean })
const labels = computed(() => props.chinese
  ? { model: '模型', precision: '格式', input: '输入', recipes: '方案', text: '文本', image: '图片', variants: n => `${n} 个方案` }
  : { model: 'Model', precision: 'Format', input: 'Input', recipes: 'Recipes', text: 'Text', image: 'Images', variants: n => `${n} recipes` })
const groups = computed(() => {
  const models = new Map()
  for (const recipe of data) {
    if (!models.has(recipe.model)) models.set(recipe.model, [])
    models.get(recipe.model).push(recipe)
  }
  return [...models].map(([name, recipes]) => ({
    name,
    recipes,
    formats: [...new Set(recipes.map(recipe => recipe.precision))].map(format =>
      format === 'BF16 + original auxiliary weights'
        ? (props.chinese ? 'BF16 + 辅助模型' : 'BF16 + auxiliary models') : format),
    inputs: [...new Set(recipes.flatMap(recipe => recipe.modalities ?? ['text']))]
      .map(input => labels.value[input] ?? input),
  }))
})
</script>

<template>
  <table class="model-recipes">
    <thead><tr><th>{{ labels.model }}</th><th>{{ labels.precision }}</th><th>{{ labels.input }}</th><th>{{ labels.recipes }}</th></tr></thead>
    <tbody>
      <tr v-for="group in groups" :key="group.name">
        <td>{{ group.name }}</td>
        <td>{{ group.formats.join(', ') }}</td>
        <td>{{ group.inputs.join(' / ') }}</td>
        <td>
          <details>
            <summary>{{ labels.variants(group.recipes.length) }}</summary>
            <ul><li v-for="recipe in group.recipes" :key="recipe.id"><code>{{ recipe.id }}</code></li></ul>
          </details>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.model-recipes td { vertical-align: top; }
.model-recipes summary { cursor: pointer; white-space: nowrap; color: var(--vp-c-brand-1); }
.model-recipes ul { margin: 10px 0 0; padding-left: 16px; }
.model-recipes code { overflow-wrap: anywhere; }
</style>
