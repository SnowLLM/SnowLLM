import { readFileSync } from 'node:fs'

export default {
  watch: ['../../recipes/*.json'],
  load(files: string[]) {
    return files.map(file => JSON.parse(readFileSync(file, 'utf8')))
      .filter(recipe => (recipe.status ?? 'supported') === 'supported')
      .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0) || a.id.localeCompare(b.id))
  },
}
