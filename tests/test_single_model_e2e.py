# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
from pathlib import Path

import httpx


async def main():
    from snowllm.cli import build
    from snowllm.serve.api import app
    from snowllm.serve.state import install

    root = Path(
        os.environ.get("SNOWLLM_SINGLE_MODEL", "~/models/LFM2.5-8B-A1B-Q4_K_M")
    ).expanduser()
    state = build(
        str(root),
        1,
        512,
        num_kv_blocks=None,
        default_max_tokens=16,
        seed=0,
        profile_dir=None,
        num_spec=0,
        prefill_chunk=128,
        stats_interval=0,
        served_names=["single"],
    )
    assert getattr(state.engine, "scheduler", None) is None
    assert getattr(state.engine, "context", None) is None
    assert getattr(app.state, "registry", None) is None
    install(state)
    state.engine.start()
    report = {}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            models = await client.get("/v1/models")
            assert models.status_code == 200, models.text
            assert [row["id"] for row in models.json()["data"]] == ["single"]
            messages = [{"role": "user", "content": "Say hello briefly."}]
            cases = [
                (
                    "chat",
                    "/v1/chat/completions",
                    {"messages": messages, "max_tokens": 16},
                ),
                (
                    "responses",
                    "/v1/responses",
                    {"input": "Say hello briefly.", "max_output_tokens": 16},
                ),
                ("messages", "/v1/messages", {"messages": messages, "max_tokens": 16}),
            ]
            for name, path, body in cases:
                for stream in (False, True):
                    response = await client.post(
                        path,
                        json={
                            "model": "single",
                            "temperature": 0,
                            "stream": stream,
                            **body,
                        },
                    )
                    assert response.status_code == 200, response.text
                    if stream:
                        assert "text/event-stream" in response.headers["content-type"]
                        assert "data:" in response.text
                    elif name == "chat":
                        report[name] = response.json()["choices"]
                    elif name == "responses":
                        report[name] = response.json()["output"]
                    else:
                        report[name] = response.json()["content"]
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "unlisted-legacy-alias",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 16,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"] == report["chat"]
            print(
                "PASS: original standalone executor, three protocols, streaming and legacy model fallback"
            )
            if os.environ.get("SNOWLLM_SINGLE_REPORT"):
                Path(os.environ["SNOWLLM_SINGLE_REPORT"]).write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)
                )
    finally:
        await state.engine.close()


if __name__ == "__main__":
    asyncio.run(main())
