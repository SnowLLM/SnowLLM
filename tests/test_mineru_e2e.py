# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import json
import os
from pathlib import Path
import tempfile


async def main():
    import httpx
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from snowllm.ops.image import allocate
    from snowllm.serve.api import app
    from snowllm.serve.registry import ModelRegistry

    root = Path(
        os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
    ).expanduser()
    if not (root / "model.safetensors").is_file():
        raise SystemExit(
            "download the MinerU recipe before running the GPU integration test"
        )
    source = json.loads(
        (Path(__file__).resolve().parents[1] / "recipes/mineru.json").read_text()
    )
    source["status"] = "supported"
    occupied = None
    if os.environ.get("SNOWLLM_MINERU_OCCUPIED_GIB"):
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        target = int(float(os.environ["SNOWLLM_MINERU_OCCUPIED_GIB"]) * (1 << 30))
        occupied = torch.empty(
            max(0, target - (total - free)), dtype=torch.uint8, device="cuda"
        )
        print("EXISTING_GPU_BYTES", total - torch.cuda.mem_get_info()[0], flush=True)
    with tempfile.TemporaryDirectory(prefix="snowllm-mineru-test-") as directory:
        temporary = Path(directory)
        catalogue = temporary / "catalogue.json"
        catalogue.write_text(json.dumps({"schema": 2, "recipes": [source]}))
        os.environ["SNOWLLM_MODELS"] = str(root.parent)
        os.environ["SNOWLLM_RECIPES_URL"] = str(catalogue)
        image = Image.new("RGB", (800, 1100), "white")
        drawing = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        for y, line in [
            (75, "SnowLLM MinerU integration"),
            (160, "This document tests GPU document parsing."),
            (215, "The answer is 42."),
        ]:
            drawing.text((60, y), line, font=font, fill="black")
        image_path = temporary / "document.png"
        image.save(image_path)
        registry = ModelRegistry()
        app.state.registry = registry
        ceiling = float(os.environ.get("SNOWLLM_MINERU_GPU_LIMIT_GIB", "12"))
        options = ["--max-gpu-mem", f"{ceiling:g}GiB", "--stats-interval", "0"]
        if os.environ.get("SNOWLLM_MINERU_EAGER", "1") == "1":
            options.append("--enforce-eager")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test", timeout=600
        ) as client:
            try:
                await registry.load("mineru", options, "documents")
                state = registry.records["documents"].state
                runtime = state.engine.mineru
                assert runtime.weight_bytes > 0
                assert (
                    state.engine.memory["weights"]
                    == state.model.device_map.device_bytes + runtime.weight_bytes
                )
                assert state.engine.memory["weights"] > (2.5 * (1 << 30))
                if occupied is not None:
                    probe = int(
                        (ceiling - float(os.environ["SNOWLLM_MINERU_OCCUPIED_GIB"]) + 1)
                        * (1 << 30)
                    )
                    try:
                        await state.engine.call(
                            lambda: runtime.gpu_call(
                                lambda: allocate(
                                    (probe,),
                                    dtype=torch.uint8,
                                    device="cuda",
                                    admit=runtime.graphs.admit,
                                )
                            )
                        )
                    except MemoryError:
                        pass
                    else:
                        raise AssertionError(
                            "allocation must use the remaining shared budget"
                        )
                print(
                    "MINERU_LOADED",
                    json.dumps(registry.snapshot()["models"]),
                    flush=True,
                )
                if os.environ.get("SNOWLLM_MINERU_AUXILIARY") == "1":
                    import numpy as np

                    formula = Image.new("RGB", (600, 160), "white")
                    ImageDraw.Draw(formula).text(
                        (30, 35),
                        "E = mc²",
                        fill="black",
                        font=ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 60
                        ),
                    )
                    pixels = np.array(formula)[:, :, ::-1].copy()
                    formulas = await asyncio.to_thread(
                        runtime.context.mfr_model.predict,
                        [dict(label="display_formula", bbox=[0, 0, 600, 160])],
                        pixels,
                    )
                    assert formulas and formulas[0]["latex"], formulas
                    print("FORMULA", formulas, flush=True)
                    seals = await asyncio.to_thread(
                        runtime.context.seal_model.ocr, pixels
                    )
                    print("SEAL", seals, flush=True)
                image_url = (
                    "data:image/png;base64,"
                    + base64.b64encode(image_path.read_bytes()).decode()
                )
                answer = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "documents",
                        "temperature": 0,
                        "max_tokens": 128,
                        "skip_special_tokens": False,
                        "vllm_xargs": {"no_repeat_ngram_size": 100},
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url},
                                    },
                                    {"type": "text", "text": "\nText Recognition:"},
                                ],
                            }
                        ],
                    },
                )
                assert answer.status_code == 200, answer.text
                print("VISION", answer.json()["choices"], flush=True)
                assert "42" in answer.json()["choices"][0]["message"]["content"]
                for tier in ("basic", "standard", "advanced"):
                    response = await client.post(
                        "/v1/documents/parse",
                        json={
                            "model": "documents",
                            "filename": "document.png",
                            "file": base64.b64encode(image_path.read_bytes()).decode(),
                            "tier": tier,
                        },
                    )
                    assert response.status_code == 200, response.text
                    print("DOCUMENT", tier, response.json()["markdown"], flush=True)
                    assert "42" in response.json()["markdown"]
                pending = asyncio.create_task(
                    client.post(
                        "/v1/documents/parse",
                        json={
                            "model": "documents",
                            "filename": "document.png",
                            "file": base64.b64encode(image_path.read_bytes()).decode(),
                            "tier": "standard",
                        },
                    )
                )
                for _ in range(500):
                    if registry.records["documents"].active:
                        break
                    await asyncio.sleep(0.01)
                await registry.unload("documents", False)
                drained = await pending
                assert drained.status_code == 200, drained.text
                assert runtime.disposed
                if os.environ.get("SNOWLLM_MINERU_COEXIST") == "1":
                    await registry.load(
                        str(root.parent / "LFM2.5-8B-A1B-Q4_K_M"),
                        [
                            "--max-gpu-mem",
                            f"{ceiling:g}GiB",
                            "--max-model-len",
                            "512",
                            "--max-num-seqs",
                            "1",
                            "--max-num-batched-tokens",
                            "128",
                            "--num-spec",
                            "0",
                            "--stats-interval",
                            "0",
                        ],
                        "chat",
                    )

                    async def chat():
                        response = await client.post(
                            "/v1/chat/completions",
                            json={
                                "model": "chat",
                                "temperature": 0,
                                "max_tokens": 48,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "What is the capital of France?",
                                    }
                                ],
                            },
                        )
                        assert response.status_code == 200, response.text
                        return response.json()["choices"][0]

                    baseline = await chat()
                    await registry.load("mineru", options, "documents")
                    body = dict(
                        model="documents",
                        filename="document.png",
                        tier="standard",
                        file=base64.b64encode(image_path.read_bytes()).decode(),
                    )
                    parsed, answer = await asyncio.gather(
                        client.post("/v1/documents/parse", json=body), chat()
                    )
                    assert parsed.status_code == 200, parsed.text
                    assert answer == baseline, (answer, baseline)
                    pending = asyncio.create_task(
                        client.post("/v1/documents/parse", json=body)
                    )
                    for _ in range(500):
                        if registry.records["documents"].active:
                            break
                        await asyncio.sleep(0.01)
                    await asyncio.wait_for(
                        registry.unload("documents", True), timeout=30
                    )
                    await asyncio.gather(pending, return_exceptions=True)
                    assert await chat() == baseline
                    print(
                        "PASS: MinerU and LFM coexist, interleave, and unload independently",
                        flush=True,
                    )
                print(
                    "PASS: MinerU GPU recipe, vision, all parsing tiers and draining unload",
                    flush=True,
                )
            finally:
                await registry.close()


if __name__ == "__main__":
    asyncio.run(main())
