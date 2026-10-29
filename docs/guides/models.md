# Model downloads

A recipe is one model at one precision. `snowllm pull` fetches it:

```sh
snowllm recipes                      # what there is
snowllm pull qwen3.6-35b-a3b-fp8     # or just `snowllm pull`, and pick from the list
```

The catalogue comes from the server by default. Use `snowllm recipes --bundle` to
view recipes shipped with the installed wheel. `snowllm pull --bundle RECIPE`
and `snowllm --bundle RECIPE` use that bundled catalogue, as do
`snowllm pull b/RECIPE` and `snowllm b/RECIPE`. Only the recipe definitions are
bundled; pulling still downloads model files. In the WebUI, select **Bundled** under **Recipe source** on Overview or Models.

- Fetches over several connections at once, and resumes where it stopped
- Verifies SHA-256 for files whose repository or catalogue provides a digest
- A recipe that wants a newer snowllm downloads anyway and says which version it wants

Checkpoints live in `~/models` by default. Set `models_dir` in the JSON configuration
or override it with `SNOWLLM_MODELS`. `snowllm <recipe>` looks there;
`snowllm pull --dir PATH` downloads directly into that checkpoint directory; it does not change the configured model root. Serve that path directly if it is outside the root.

See [Configuration](../reference/configuration#environment-variables) for catalogue URLs, mirrors and download credentials.

Or fetch it yourself. Use a local checkpoint directory with a supported architecture and weight format:

```sh
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
```

A GGUF works too — `snowllm pull qwen3.6-35b-a3b-q4-k-xl-dflash` gets the UD-Q4_K_XL quant with
its DFlash drafter beside it: a smaller download, and llama.cpp serves the same file.

## Choosing a model

[See supported models and resident weight sizes](../reference/models). Leave room for workspaces and caches beyond the listed weight memory.

For custom fine-tuned checkpoints, see the [FAQ](../faq#_1-how-do-i-run-a-custom-fine-tuned-model).
