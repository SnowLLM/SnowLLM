# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import json
import pathlib
import sys

from .. import term
from .._capi import SnowLLMError
from . import recipes

MODELS_JSON = "~/.pi/agent/models.json"


def main(cmd: str, argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"snowllm {cmd}",
        description="Point Pi at a local SnowLLM server by writing its models file.")
    p.add_argument("--port", type=int, default=8000,
                   help="port the server listens on (default: 8000)")
    p.add_argument("--catalogue", metavar="URL", default=None,
                   help=f"where the recipe list comes from (default: {recipes.CATALOGUE_URL}, "
                        f"or $SNOWLLM_RECIPES_URL). A path works too.")
    p.add_argument("--refresh", action="store_true",
                   help="refetch the catalogue instead of reusing the cached copy")
    a = p.parse_args(argv)

    try:
        known = recipes.catalogue(a.catalogue, a.refresh)
    except SnowLLMError as e:
        print(f"snowllm {cmd}: {e}", file=sys.stderr)
        return 1

    models, seen = [], set()
    for r in known:
        model_id = r.serve_as or r.dir
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        # Pi cannot infer that a custom OpenAI-compatible endpoint is a reasoning model, so
        # mark it. reasoning alone is not enough: compat carries the endpoint's thinking
        # on/off switch (Qwen reads enable_thinking, DeepSeek reads thinking), and Pi sends
        # the system prompt as the OpenAI developer role for any reasoning model unless
        # supportsDeveloperRole is false. The Qwen chat template only accepts
        # system/user/assistant/tool, so it must stay "system".
        models.append({"id": model_id, "reasoning": True,
                       "compat": {"thinkingFormat":
                                  "deepseek" if model_id.startswith("DeepSeek") else "qwen",
                                  "supportsDeveloperRole": False}})
    if not models:
        print(f"{term.stamp()} the catalogue names no model to serve", file=sys.stderr)
        return 1
    models.sort(key=lambda m: m["id"])

    models_json = pathlib.Path(MODELS_JSON).expanduser()
    models_json.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {}
    if models_json.exists():
        try:
            data = json.loads(models_json.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"{term.stamp()} warning: could not read {models_json}: {e}",
                  file=sys.stderr)
    if not isinstance(data, dict):
        data = {}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        providers = data["providers"] = {}
    providers["snowllm"] = {"baseUrl": f"http://localhost:{a.port}/v1",
                            "api": "openai-completions",
                            "apiKey": "snowllm",
                            "models": models}
    models_json.write_text(json.dumps(data, indent=2) + "\n")

    print(f"{term.stamp()} wrote {len(models)} model(s) to {models_json}")
    print(f"{term.stamp()} models: " + ", ".join(m["id"] for m in models))
    return 0
