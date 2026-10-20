# SPDX-License-Identifier: Apache-2.0

import gc
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from PIL import Image

import _harness
from snowllm import ops
from snowllm.errors import SnowLLMError
from snowllm.checkpoint import loader
from snowllm.checkpoint.gguf import GGUF, names
from snowllm.checkpoint.gguf.minicpmv import vision_config, vision_names
from snowllm.checkpoint.gguf.source import GGUFReader, find_gguf, find_mmproj
from snowllm.engine import Engine, SamplingParams
from snowllm.models import multimodal
from snowllm.models.minicpmv import validate_config
from snowllm.models.minicpmv.vision import MiniCPMVision


class NativeVisionOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def cuda(value):
            if isinstance(value, torch.Tensor):
                return value.is_cuda
            if isinstance(value, (tuple, list)):
                return any(cuda(item) for item in value)
            if isinstance(value, dict):
                return any(cuda(item) for item in value.values())
            return False
        out = func(*args, **(kwargs or {}))
        if cuda((args, kwargs, out)) and not str(func).startswith((
            "aten.empty", "aten.view.", "aten.slice.", "aten.as_strided.", "aten.detach.",
            "aten.copy_.", "aten._to_copy.", "aten.alias.", "aten.lift_fresh.",
        )):
            raise AssertionError(f"visual computation fell back to PyTorch GPU: {func}")
        return out


def vision_checks(root, check):
    from types import SimpleNamespace
    from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import MiniCPMV4_7VisionConfig
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Merger, MiniCPMV4_7VisionModel

    with GGUFReader(find_mmproj(root)) as rd:
        cfg = vision_config(rd.gguf)
        sd = {hf: rd.tensor(gg) for hf, gg in vision_names(rd.gguf).items()}
    native = MiniCPMVision(sd, cfg)
    vc = MiniCPMV4_7VisionConfig(**{k: v for k, v in cfg.items()
                                   if k not in ("downsample_mode", "scale_resolution", "max_slice_nums",
                                                "out_hidden_size", "model_type")})
    vc._attn_implementation = "eager"
    reference = MiniCPMV4_7VisionModel(vc).eval().float().cuda()
    reference.load_state_dict({k.removeprefix("model.vision_tower."): v.float()
                               for k, v in sd.items() if k.startswith("model.vision_tower.")})
    merger = MiniCPMV4_7Merger(SimpleNamespace(
        merge_kernel_size=(2, 2), merger_times=1, vision_config=vc,
        text_config=SimpleNamespace(hidden_size=cfg["out_hidden_size"]))).eval().float().cuda()
    merger.load_state_dict({k.removeprefix("model.merger."): v.float()
                            for k, v in sd.items() if k.startswith("model.merger.")})
    torch.manual_seed(4727)
    for mode in ("16x", "4x"):
        for sizes in ([[4, 4]], [[8, 12]], [[16, 16]], [[32, 32]], [[8, 12], [4, 8]]):
            grids = torch.tensor(sizes)
            m = sum(h * w for h, w in sizes)
            pixels = torch.randn(1, 3, 14, 14*m)
            with torch.no_grad():
                want = reference(pixels.cuda(), target_sizes=grids.cuda(), use_vit_merger=mode == "16x")
                want = torch.cat(merger(want.last_hidden_state, grids // (2 if mode == "16x" else 1)))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["requested_bytes.all.current"]
                torch.cuda.reset_peak_memory_stats()
                with NativeVisionOnly():
                    got = native.forward(pixels, grids, mode)
                peak = torch.cuda.memory_stats()["requested_bytes.all.peak"] - before
            check.close(f"HF FP32 vision {mode} {sizes}", got, want, .035)
            check(f"all GPU visual compute uses native kernels {mode} {sizes}", True)
            need = native.workspace_bytes(grids, mode)
            check(f"planned workspace {mode} {sizes}", peak == need,
                  f"peak {peak}, planned {need}")
            if len(sizes) > 1:
                n = sizes[0][0] * sizes[0][1]
                saved = got.clone()
                first = native.forward(pixels[..., :14*n], grids[:1], mode)
                check.exact(f"crop isolation {mode}", got[:first.shape[0]], first)
                native.forward(pixels, grids, "4x" if mode == "16x" else "16x")
                check.exact(f"embeddings survive later Arena allocations {mode}", got, saved)
    for bad in (torch.tensor([[3, 4]]), torch.tensor([[0, 4]]), torch.empty(0, 2)):
        try:
            native.forward(torch.zeros(1, 3, 14, 224), bad)
        except SnowLLMError:
            check("reject malformed patch grids", True)
        else:
            check("reject malformed patch grids", False)


def main():
    root = _harness.checkpoint("MiniCPM-V-4.7-35B-A3B-Q4_K_M", env="SNOWLLM_MINICPM_DIR")
    check = _harness.Checks(58)
    raw = names.config(GGUF(find_gguf(root)))
    check("detect MiniCPM GGUF", raw["architectures"] == ["MiniCPMV4_7ForConditionalGeneration"])
    validate_config(raw["text_config"])
    tok, eos = loader.load_tokenizer(root)
    processor = loader.load_processor(root, tok)
    check("native vocabulary", len(tok) == 248144)
    message = [dict(role="user", content="Hi")]
    thinking = tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    check("thinking template available", thinking.endswith("<think>\n"))
    vision_checks(root, check)
    gc.collect()
    torch.cuda.empty_cache()
    model = loader.load(root, mtp=False)
    print(f"Resident weights: {model.device_map.total_bytes / (1 << 30):.3f} GiB", flush=True)
    check("padded compute vocabulary", model.geo.vocab_size == ops.geo().vocab_size)
    check("original vocabulary preserved", model.config["vocab_size"] == 248144)
    engine = Engine(model, num_kv_blocks=256, max_num_seqs=2, max_model_len=2048,
                    stop_token_ids=eos, prefill_chunk=128, prefix_memory_ratio=0,
                    enforce_eager=False)
    with torch.no_grad(), GGUFReader(find_gguf(root)) as rd:
        hidden = torch.randn(2, model.geo.hidden, dtype=torch.bfloat16, device="cuda")
        logits = torch.empty(2, model.geo.vocab_size, dtype=torch.float32, device="cuda")
        model.lm_head(hidden, logits)
        weight = rd.tensor("output.weight", dtype=torch.float32)
        want = hidden.float() @ weight.T
        check.close("quantized output head vs FP32 dequant", logits[:, :248144], want, .015)
        check("all padded logits are negative infinity", torch.isneginf(logits[:, 248144:]).all())
        del hidden, logits, weight, want
    greedy = SamplingParams(temperature=0, max_new_tokens=24)
    prompt = tok.apply_chat_template([dict(role="user", content="What is 17 + 25? Reply with just the number.")],
                                      tokenize=True, return_dict=False, add_generation_prompt=True,
                                      enable_thinking=False)
    r = engine.add(prompt, greedy)
    engine.run()
    text = tok.decode(r.out)
    check("text generation", "42" in text, repr(text))
    check("padding cannot be sampled", all(t < 248144 for t in r.out))
    messages = [dict(role="user", content=[dict(type="image"), dict(type="text", text=
                "What is the background color of this image? Answer with one word.")])]
    for mode, color in (("16x", "red"), ("4x", "blue"), ("16x", "green")):
        mm = multimodal.prepare(model, processor, messages, [Image.new("RGB", (224, 224), color)],
                                enable_thinking=False, downsample_mode=mode)
        check(f"canvas positions {mode}", mm["mrope"].shape == (3, len(mm["prompt"])))
        r = engine.add(mm.pop("prompt"), greedy, **mm)
        engine.run()
        text = tok.decode(r.out)
        check(f"image generation {mode} {color}", color in text.lower(), repr(text))
        check("image output uses real tokens", all(t < 248144 for t in r.out))
    check("decode graphs exercised", engine.runner.n_graph_replays > 0)
    from PIL import ImageDraw, ImageFont
    image = Image.new("RGB", (1200, 900), "white")
    font = ImageFont.truetype("DejaVuSans.ttf", 84)
    ImageDraw.Draw(image).text((140, 340), "SNOWLLM 4727", fill="black", font=font)
    messages[0]["content"][1]["text"] = "Read the text in this image. Output only the text."
    mm = multimodal.prepare(model, processor, messages, [image], enable_thinking=False)
    check("high resolution uses detail slices", mm["embeds"].shape[0] > 64)
    r = engine.add(mm.pop("prompt"), greedy, **mm)
    engine.run()
    text = tok.decode(r.out)
    check("sliced image OCR", "4727" in text and "snowllm" in text.lower(), repr(text))
    messages[0]["content"] = [dict(type="image"), dict(type="image"), dict(type="text",
        text="What are the colors of the first and second images? Answer in order, with two color names.")]
    mm = multimodal.prepare(model, processor, messages,
                            [Image.new("RGB", (224, 224), c) for c in ("red", "blue")],
                            enable_thinking=False)
    r = engine.add(mm.pop("prompt"), greedy, **mm)
    engine.run()
    text = tok.decode(r.out).lower()
    check("multi image order", "red" in text and "blue" in text and text.index("red") < text.index("blue"), repr(text))
    import asyncio
    asyncio.run(server_check(engine, model, tok, processor, check))
    engine.runner.resources.close()
    return check.done()


async def server_check(engine, model, tok, processor, check):
    import base64
    import io
    import time
    import httpx
    from snowllm.engine.async_engine import AsyncEngine
    from snowllm.serve.api import app
    from snowllm.serve.state import Alias, ServerState, install

    name = "MiniCPM-V-4.7-35B-A3B"
    async_engine = AsyncEngine(engine)
    install(ServerState(engine=async_engine, model=model, tokenizer=tok, processor=processor,
                        model_name=name, aliases={name: Alias(1, 2048)}, created=int(time.time()),
                        think_open_id=tok.convert_tokens_to_ids("<think>"),
                        think_close_id=tok.convert_tokens_to_ids("</think>"),
                        default_max_tokens=24, allow_image_urls=False, limit_mm_per_prompt=4,
                        sampling=loader.sampling_defaults(_harness.checkpoint_path(
                            "MiniCPM-V-4.7-35B-A3B-Q4_K_M", env="SNOWLLM_MINICPM_DIR"))))
    image = io.BytesIO()
    Image.new("RGB", (224, 224), "yellow").save(image, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    async_engine.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://asgi") as client:
            response = await client.post("/v1/chat/completions", json=dict(
                model=name, temperature=0, max_tokens=24,
                chat_template_kwargs=dict(enable_thinking=False, downsample_mode="4x"),
                messages=[dict(role="user", content=[dict(type="image_url", image_url=dict(url=url)),
                    dict(type="text", text="What color is this image? Answer with one word.")])]))
            check("image_url chat completion HTTP 200", response.status_code == 200, response.text[:180])
            if response.status_code == 200:
                data = response.json()
                check("image_url chat completion sees yellow",
                      "yellow" in (data["choices"][0]["message"]["content"] or "").lower())
                check("chat completion strips stop tokens",
                      "<|im_end|>" not in (data["choices"][0]["message"]["content"] or "")
                      and data["choices"][0]["finish_reason"] == "stop")
    finally:
        await async_engine.close()


if __name__ == "__main__":
    sys.exit(main())
