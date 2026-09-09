# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import base64
import io
import sys
import threading

import numpy as np
from PIL import Image

import _harness

CKPT = _harness.checkpoint()

import uvicorn  # noqa: E402
from openai import OpenAI  # noqa: E402

from snowllm import cli  # noqa: E402
from snowllm.serve import api as server  # noqa: E402
from snowllm.serve.state import install, serving  # noqa: E402

PORT = 8123


def data_uri(rgb: tuple[int, int, int], h: int = 224, w: int = 224) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), rgb, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    install(cli.build(str(CKPT), max_num_seqs=4, max_model_len=4096, num_kv_blocks=None,
                      default_max_tokens=32, seed=0, profile_dir=None, enforce_eager=True,
                      num_spec=0))
    if serving().processor is None:
        _harness.skip("checkpoint carries no vision tower")

    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="error")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        pass

    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="none")
    ok = True

    def ask(content: object, **kw: object) -> str:
        return client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": content}],
            temperature=0.0, max_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **kw).choices[0].message.content.strip().lower()

    for rgb, want in [((220, 30, 30), "red"), ((30, 120, 220), "blue")]:
        got = ask([{"type": "image_url", "image_url": {"url": data_uri(rgb)}},
                   {"type": "text", "text": "What color is this image? One word."}])
        ok &= want in got
        print(f"  image_url {want:<5} -> {got!r:<24} {'PASS' if want in got else 'FAIL'}")

    text = ask("The capital of France is")
    ok &= "paris" in text
    print(f"  text-only        -> {text!r:<24} {'PASS' if 'paris' in text else 'FAIL'}")

    try:
        ask([{"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}},
             {"type": "text", "text": "?"}])
        print("  http URL refused -> FAIL (it was accepted)")
        ok = False
    except Exception as e:
        refused = "--allow-image-urls" in str(e)
        ok &= refused
        print(f"  http URL refused -> {'PASS' if refused else f'FAIL ({e})'}")

    srv.should_exit = True
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
