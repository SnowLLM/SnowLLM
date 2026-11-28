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
| DeepSeek V4 / V4.1 | V4 serves text. V4.1 Flash 1017 accepts images with the BF16 vision head included in the download. Keep its matching Engram tables on a local SSD. |
| GLM-5.3-Flash | Select the BF16 or F16 vision head to accept images. The recipes default to text. |
| Inkling-2-Mini (GGUF) | The Q4 and Q8 vision recipes accept one or more images through chat `image_url` parts. Text generation uses BF16 KV and supports batched requests. Both DSpark recipes pair Q4_K_M or Q8_K_XL with a Q8_0 draft and enable prompt lookup. For single-request greedy decoding, the draft selects candidate vocabulary tiles and the target adapts its verification width up to eight positions, including the anchor. Sampled drafts use full-vocabulary scores and rejection sampling. For a smaller memory footprint use `--no-dflash-adaptive --dflash-block 4`. |
| MiniCPM-V 4.7 | Single images, multiple images and high-resolution slices are supported. Video and LES expert streaming are not implemented. |
| LFM2.5-8B-A1B | Thinking must stay enabled. Requests that disable it return HTTP 400. |
| MinerU | `mineru-vlm` serves the VLM alone; `mineru` serves the full bundle. See [MinerU integration](../integrations/mineru). |

DeepSeek V4.1 and GLM require SnowLLM-Kernels with 64-wide vision attention. Streaming text MoE experts from disk also requires disk expert operators in that build. GLM also uses the Transformers `Glm5NextImageProcessorPil` image processor.

Inkling 2 Mini, MiniCPM-V 4.7, Qwen vision models, DeepSeek V4.1 and GLM-5.3-Flash default to `--vision-placement disk` when a vision head is loaded. MinerU defaults to GPU placement to reuse its vision graphs. Selecting disk placement streams its Qwen2-VL encoder using eager execution. The document pipeline keeps its own layout and OCR models.

On first load, disk placement packs the visual projections into `$XDG_CACHE_HOME/snowllm/vision` (by default `~/.cache/snowllm/vision`). Allow about 925 MiB of extra disk space for the DeepSeek BF16 head or 1.05 GiB for the GLM F16 head. Subsequent loads reuse the cache. Text requests skip the projections. Image encoding streams them into temporary GPU memory and releases the buffer afterwards. Small parameters such as normalization weights and biases stay resident. Selecting a vision head enables this independently of text expert disk offload.

For image input, select the head in the WebUI or append `:vision=bf16` to a recipe ID for both download and launch, for example `b/glm-5.3-flash:ud-iq2-xxs-mtp:vision=bf16` or `b/deepseek-v4.1-flash-1017:vq-q4-k-none:vision=bf16`. Image tokens share the model context with text. For DeepSeek V4.1 and GLM-5.3-Flash, the startup planner budgets for the largest image allowed by the context and for the embeddings of up to `--limit-mm-per-prompt` images per request. Images from one request are grouped within the vision token budget. The encoder reads each projection once per group and checks workspace and host memory at runtime. Lowering the image count reduces the embedding budget. Reducing the context can also reduce the maximum image size.

Run `snowllm pull b/inkling-2-mini:ud-q4-k-m-vision` for Q4 vision with DSpark, or use `b/inkling-2-mini:ud-q8-k-xl-vision` for Q8. If you already have the text GGUF, place the vision weights (612 MiB) alongside it and restart the server with a SnowLLM runtime and kernel library that support Inkling vision. [Download Unsloth’s mmproj-BF16.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/8078d149286eec7193795413b9e0643da6d11fa1/mmproj-BF16.gguf)

On the tested Radeon 780M with a 16 GiB dedicated GPU allocation, the Inkling Q4 recipe ran with expert weights mapped into host memory. Use `snowllm b/inkling-2-mini-ud-q4-k-m --max-gpu-mem 15.5GiB --max-host-mem 7GiB --host-offload 6.5GiB` for a 32K context.

Available context and concurrency depend on the [memory configuration](../guides/memory-management). Download size is not inference memory usage.

## List and run recipes {#usage}

```sh
snowllm recipes --bundle
snowllm pull b/RECIPE_ID
snowllm b/RECIPE_ID
```

This list follows the bundled recipes. Run `snowllm recipes` to see recipes in the online catalogue.
