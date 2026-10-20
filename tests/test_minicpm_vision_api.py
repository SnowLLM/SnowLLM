# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import gc
import io
import json
import os
from pathlib import Path
import sys
import time

import httpx
from PIL import Image, ImageDraw, ImageFont
import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
from snowllm.engine.async_engine import AsyncEngine
from snowllm.models import multimodal
from snowllm.serve.api import app
from snowllm.serve.state import Alias, ServerState, install


def encode_image(image):
    data = io.BytesIO()
    image.save(data, format="PNG")
    return base64.b64encode(data.getvalue()).decode()


async def api_checks(engine, model, tok, processor, check):
    name = "MiniCPM-V-4.7-35B-A3B"
    async_engine = AsyncEngine(engine)
    install(ServerState(engine=async_engine, model=model, tokenizer=tok, processor=processor,
        model_name=name, aliases={name: Alias(1, 4096)}, created=int(time.time()),
        think_open_id=tok.convert_tokens_to_ids("<think>"),
        think_close_id=tok.convert_tokens_to_ids("</think>"), default_max_tokens=32,
        allow_image_urls=False, limit_mm_per_prompt=4, sampling={}))
    data = encode_image(Image.new("RGB", (224, 224), "yellow"))
    url = "data:image/png;base64," + data
    question = "What color is this image? Answer with one word."
    messages = [dict(role="user", content=[dict(type="image_url", image_url=dict(url=url)),
                 dict(type="text", text=question)])]
    options = dict(enable_thinking=False, downsample_mode="4x", max_slice_nums=1)
    async_engine.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://asgi") as client:
            response = await client.post("/v1/chat/completions", json=dict(model=name,
                messages=messages, temperature=0, max_tokens=16, stream=True,
                chat_template_kwargs=options))
            check("image chat stream HTTP 200", response.status_code == 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            text = "".join(event["choices"][0].get("delta", {}).get("content", "") for event in events)
            check("image chat stream sees yellow", "yellow" in text.lower(), repr(text))
            check("image chat stream ends cleanly", "data: [DONE]" in response.text
                  and any(event["choices"][0].get("finish_reason") == "stop" for event in events))
            response = await client.post("/v1/responses", json=dict(model=name, temperature=0,
                max_output_tokens=16, chat_template_kwargs=options,
                input=[dict(role="user", content=[dict(type="input_image", image_url=url),
                       dict(type="input_text", text=question)])]))
            check("Responses image HTTP 200", response.status_code == 200, response.text[:160])
            if response.status_code == 200:
                text = "".join(part.get("text", "") for item in response.json()["output"]
                               for part in item.get("content", []))
                check("Responses image sees yellow", "yellow" in text.lower(), repr(text))
            response = await client.post("/v1/messages", json=dict(model=name, temperature=0,
                max_tokens=16, thinking=dict(type="disabled"), messages=[dict(role="user", content=[dict(type="image",
                    source=dict(type="base64", media_type="image/png", data=data)),
                    dict(type="text", text=question)])]))
            check("Messages image HTTP 200", response.status_code == 200, response.text[:160])
            if response.status_code == 200:
                text = "".join(part.get("text", "") for part in response.json()["content"])
                check("Messages image sees yellow", "yellow" in text.lower(), repr(text))
            for invalid in (dict(downsample_mode="8x"), dict(max_slice_nums=0),
                            dict(max_slice_nums=1.5), dict(max_slice_nums=True)):
                response = await client.post("/v1/chat/completions", json=dict(model=name,
                    messages=messages, max_tokens=16, chat_template_kwargs=options | invalid))
                check(f"invalid visual option returns HTTP 400: {invalid}", response.status_code == 400)
            response = await client.post("/v1/chat/completions", json=dict(model=name,
                messages=[dict(role="user", content=[dict(type="image_url",
                    image_url=dict(url="data:image/png;base64,broken"))])], max_tokens=16))
            check("invalid image bytes return HTTP 400", response.status_code == 400)
    finally:
        await async_engine.close()


def main():
    root = _harness.checkpoint("MiniCPM-V-4.7-35B-A3B-Q4_K_M", env="SNOWLLM_MINICPM_DIR")
    draft = Path(os.environ.get("SNOWLLM_MINICPM_DSPARK", root / "dspark"))
    if not draft.exists():
        _harness.skip(f"no DSpark checkpoint at {draft}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    check = _harness.Checks(66)
    model = loader.load(root, mtp=False)
    tok, eos = loader.load_tokenizer(root)
    processor = loader.load_processor(root, tok)
    engine = Engine(model, max_num_seqs=2, num_kv_blocks=512, max_model_len=4096,
                    stop_token_ids=eos, num_spec=0, dflash_path=str(draft), dflash_block=8,
                    prefill_chunk=128, prefix_memory_ratio=0, account=False)
    params = SamplingParams(temperature=0, max_new_tokens=32)
    red, blue = [Image.new("RGB", (224, 224), color) for color in ("red", "blue")]
    conversation = [dict(role="user", content=[dict(type="image"), dict(type="text", text="Remember this first image.")]),
                    dict(role="assistant", content="The first image is red."),
                    dict(role="user", content=[dict(type="image"), dict(type="text",
                         text="What are the colors of the first and second images? Answer in order with two color names.")])]
    requests = []
    for mode in ("16x", "4x"):
        mm = multimodal.prepare(model, processor, conversation, [red, blue],
                                enable_thinking=False, downsample_mode=mode)
        check(f"multi-turn {mode}: placeholders and embeddings agree", len(mm["embed_rows"]) == mm["embeds"].shape[0])
        requests.append(engine.add(mm.pop("prompt"), params, **mm))
    engine.run()
    for mode, request in zip(("16x", "4x"), requests):
        text = tok.decode(request.out).lower()
        check(f"mixed image batch and multi-turn order {mode}", "red" in text and "blue" in text
              and text.index("red") < text.index("blue"), repr(text))
    image = Image.new("RGB", (1200, 900), "white")
    font = ImageFont.truetype("DejaVuSans.ttf", 84)
    ImageDraw.Draw(image).text((140, 340), "SNOWLLM 4727", fill="black", font=font)
    messages = [dict(role="user", content=[dict(type="image"), dict(type="text",
                     text="Read the text in this image. Output only the text.")])]
    for mode in ("16x", "4x"):
        mm = multimodal.prepare(model, processor, messages, [image], enable_thinking=False,
                                downsample_mode=mode, max_slice_nums=4)
        check(f"DSpark OCR {mode}: detail slices present", len(mm["embed_rows"]) > (64 if mode == "16x" else 256))
        request = engine.add(mm.pop("prompt"), params, **mm)
        engine.run()
        text = tok.decode(request.out).lower()
        check(f"DSpark sliced OCR {mode}", "snowllm" in text and "4727" in text, repr(text))
    check("batched main and draft graphs exercised", engine.runner.n_graph_replays > 0
          and engine.runner.n_draft_replays > 0)
    asyncio.run(api_checks(engine, model, tok, processor, check))
    engine.runner.resources.close()
    gc.collect()
    return check.done()


if __name__ == "__main__":
    sys.exit(main())
