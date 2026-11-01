# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from pathlib import Path

import httpx


async def main():
    from snowllm.serve.api import app
    from snowllm.serve.registry import ModelRegistry

    paths = [
        Path(
            os.environ.get("SNOWLLM_MULTI_MODEL_A", "~/models/LFM2.5-8B-A1B-Q4_K_M")
        ).expanduser(),
        Path(
            os.environ.get("SNOWLLM_MULTI_MODEL_B", "~/models/Qwen3.6-27B-Q4_K_S")
        ).expanduser(),
    ]
    if any(not path.is_dir() for path in paths):
        raise SystemExit("both multimodel checkpoints are required")
    registry = ModelRegistry()
    app.state.registry = registry
    options = [
        "--max-model-len",
        "2048" if os.environ.get("SNOWLLM_MULTI_IMAGE_B") == "1" else "512",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "128",
        "--num-spec",
        "0",
        "--seed",
        "7",
        "--stats-interval",
        "0",
    ]
    if os.environ.get("SNOWLLM_MULTI_PROFILE_DIR"):
        options.extend(["--profile-dir", os.environ["SNOWLLM_MULTI_PROFILE_DIR"]])
    if os.environ.get("SNOWLLM_MULTI_EAGER", "1") == "1":
        options.append("--enforce-eager")
    options_a, options_b = list(options), list(options)
    if (
        os.environ.get("SNOWLLM_MULTI_EAGER_A") == "1"
        and "--enforce-eager" not in options_a
    ):
        options_a.append("--enforce-eager")
    if (
        os.environ.get("SNOWLLM_MULTI_EAGER_B") == "1"
        and "--enforce-eager" not in options_b
    ):
        options_b.append("--enforce-eager")
    for name, target in (
        ("SNOWLLM_MULTI_DRAFT_A", options_a),
        ("SNOWLLM_MULTI_DRAFT_B", options_b),
    ):
        if os.environ.get(name):
            target.extend(["--dflash", os.environ[name]])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test", timeout=600
    ) as client:

        async def generate(name):
            response = await client.post(
                "/v1/chat/completions",
                json=dict(
                    model=name,
                    messages=[
                        dict(
                            role="user",
                            content="What is the capital of France? Answer briefly.",
                        )
                    ],
                    temperature=0,
                    max_tokens=24,
                ),
            )
            assert response.status_code == 200, response.text
            response.raise_for_status()
            return response.json()["choices"][0]

        try:
            await registry.load(str(paths[0]), options_a, "small")
            baseline_a = await generate("small")
            await registry.load(str(paths[1]), options_b, "main")
            baseline_b = await generate("main")
            profiler = (
                registry.records["small"].state.engine
                if os.environ.get("SNOWLLM_MULTI_PROFILE_DIR")
                else None
            )
            if profiler:
                await profiler.start_profile()
            try:
                actual_a, actual_b = await asyncio.gather(
                    generate("small"), generate("main")
                )
            finally:
                if profiler:
                    await profiler.stop_profile()
            assert actual_a == baseline_a, (actual_a, baseline_a)
            assert actual_b == baseline_b, (actual_b, baseline_b)
            assert (
                registry.records["small"].state.engine.context.geometry
                != registry.records["main"].state.engine.context.geometry
            )
            for name in ("small", "main"):
                for path, body in (
                    ("/v1/responses", {"input": "Hi", "max_output_tokens": 8}),
                    (
                        "/v1/messages",
                        {
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 8,
                        },
                    ),
                ):
                    response = await client.post(
                        path, json={"model": name, "temperature": 0, **body}
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["model"] == name, response.text
            if os.environ.get("SNOWLLM_MULTI_IMAGE_B") == "1":
                import base64
                import io
                from PIL import Image

                image = io.BytesIO()
                Image.new("RGB", (32, 32), "red").save(image, format="PNG")
                image_url = (
                    "data:image/png;base64,"
                    + base64.b64encode(image.getvalue()).decode()
                )

                async def vision():
                    response = await client.post(
                        "/v1/chat/completions",
                        json=dict(
                            model="main",
                            temperature=0,
                            max_tokens=24,
                            chat_template_kwargs={
                                "max_slice_nums": 1,
                                "downsample_mode": "16x",
                            },
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "What color is the image? Answer briefly.",
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": image_url},
                                        },
                                    ],
                                }
                            ],
                        ),
                    )
                    assert response.status_code == 200, response.text
                    return response.json()["choices"][0]

                expected_image = await vision()
                actual_image, alongside = await asyncio.gather(
                    vision(), generate("small")
                )
                assert actual_image == expected_image, (actual_image, expected_image)
                assert alongside == baseline_a, (alongside, baseline_a)
                print(
                    "PASS: image encoding and MiniCPM generation interleaved with LFM",
                    flush=True,
                )
            models = (await client.get("/v1/models")).json()
            assert {row["id"] for row in models["data"]} == {"small", "main"}
            assert (
                await client.post(
                    "/v1/chat/completions", json={"model": "missing", "messages": []}
                )
            ).status_code == 404
            await registry.unload("main")
            assert await generate("small") == baseline_a
            print(
                "PASS: two geometries, deterministic interleaving, routing and targeted unload",
                flush=True,
            )
            print(
                {
                    row["id"]: {
                        "engine": row["engine"],
                        "memory": row["memory"],
                        "scheduling": row["scheduling"],
                    }
                    for row in registry.snapshot()["models"]
                },
                flush=True,
            )
        finally:
            await registry.close()
            del app.state.registry


if __name__ == "__main__":
    asyncio.run(main())
