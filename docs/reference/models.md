<script setup>
import ModelRecipes from './ModelRecipes.vue'
</script>

# Supported models {#supported-models}

Expand **Recipes** to see recipe IDs. See [Model downloads](../guides/models) for installation and [Benchmarks](https://snowllm.dev/benchmark/) for measured speed.

## Model list {#models}

<ModelRecipes />

## Limitations {#limitations}

| Model | Notes |
| --- | --- |
| Qwen3.8-Flash-Next | PLE lookup tables are read on demand; keep them on a local SSD. |
| DeepSeek V4 / V4.1 | Text only; keep the matching Engram files on a local SSD. |
| MiniCPM-V 4.7 | Single images, multiple images and high-resolution slices are supported. Video and LES expert streaming are not implemented. |
| LFM2.5-8B-A1B | Thinking must stay enabled. Requests that disable it return HTTP 400. |
| MinerU | `mineru-vlm` serves the VLM alone; `mineru` serves the full bundle. See [MinerU integration](../integrations/mineru). |

Available context and concurrency depend on the [memory configuration](../guides/memory-management). Download size is not inference memory usage.

## List and run recipes {#usage}

```sh
snowllm recipes --bundle
snowllm pull b/RECIPE_ID
snowllm b/RECIPE_ID
```

This list follows the bundled recipes. Run `snowllm recipes` to see recipes in the online catalogue.
