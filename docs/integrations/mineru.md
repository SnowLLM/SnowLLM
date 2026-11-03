# MinerU

MinerU can use SnowLLM for VLM inference alone, or use `snowllm-mineru` for the full VLM, layout, OCR, table and formula model bundle. Document parsing and interfaces use native MinerU implementations.

## VLM only

Download and start the VLM:

```sh
snowllm pull b/mineru-vlm
snowllm b/mineru-vlm
```

Set MinerU's VLM API address to `http://127.0.0.1:8000` and model name to `MinerU-VLM`. No adapter launcher is required. For an OpenAI client, use `http://127.0.0.1:8000/v1` as the base URL.

## Full model bundle

Install in the same Python environment that runs MinerU (Python 3.11+):

```sh
python -m pip install --upgrade --pre 'snowllm[mineru]'
```

Start the full model service. For an existing server, load `b/mineru` through the [Web UI](../webui):

```sh
snowllm b/mineru --max-gpu-mem 12GiB
```

In another terminal, launch MinerU through the adapter launcher:

```sh
snowllm-mineru webui
```

The default server address is `http://127.0.0.1:8000`. To select a server and model ID:

```sh
snowllm-mineru --snowllm-url http://server:8000 --snowllm-model MinerU webui
```

You can also set `SNOWLLM_MINERU_URL` and `SNOWLLM_MINERU_MODEL`. An existing server must have the full bundle loaded first.

## CLI and parsing API

The adapter launcher also supports MinerU's parsing commands and API service:

```sh
snowllm-mineru parse report.pdf -o report.md --tier standard
snowllm-mineru api-server --host 127.0.0.1 --port 8002 --tier advanced
```

Send PDFs to the MinerU parsing API (`8002` in this example). The adapter launcher forwards model calls to SnowLLM (port `8000` by default). See each command's `--help` for options and [HTTP API](./http-api#snowllm-extensions) for model endpoints.
