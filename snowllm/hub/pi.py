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

# Fold Pi's level names onto what each template reads: Qwen3.8 xhigh/medium/low, DeepSeek max.
QWEN38_LEVELS = {"minimal": "low", "low": "low", "medium": "medium",
                 "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
DEEPSEEK_LEVELS = {"max": "max"}


def _has_vision(recipe: recipes.Recipe) -> bool:
    return any("mmproj" in name for source in recipe.sources for name in source.include)


def main(cmd: str, argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"snowllm {cmd}",
        description="Point Pi at a local SnowLLM server by writing its models file.")
    p.add_argument("--host", default="localhost",
                   help="host the server runs on (default: localhost)")
    p.add_argument("--port", type=int, default=8000,
                   help="port the server listens on (default: 8000)")
    p.add_argument("--catalogue", metavar="URL", default=None,
                   help=f"where the recipe list comes from (default: {recipes.CATALOGUE_URL}, "
                        f"or $SNOWLLM_RECIPES_URL). A path works too.")
    p.add_argument("--refresh", action="store_true",
                   help="refetch the catalogue instead of reusing the cached copy")
    p.add_argument("--api", choices=("openai-completions", "openai-responses"),
                   default="openai-responses",
                   help="Pi wire API for the endpoint (default: openai-responses)")
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
        model = {"id": model_id, "reasoning": True,
                 "contextWindow": int(r.defaults.get("max_model_len",
                                                      recipes.DEFAULT_CONTEXT)),
                 # Pi would send the system prompt as the developer role, which Qwen rejects.
                 "compat": {"supportsDeveloperRole": False}}
        if model_id.startswith("Qwen3.8"):
            model["thinkingLevelMap"] = QWEN38_LEVELS
        elif model_id.startswith("DeepSeek"):
            model["thinkingLevelMap"] = DEEPSEEK_LEVELS
        if _has_vision(r):
            model["input"] = ["text", "image"]
        if a.api == "openai-completions":
            # Mark the model a reasoning one: compat carries its thinking switch and effort.
            model["compat"]["thinkingFormat"] = (
                "deepseek" if model_id.startswith("DeepSeek") else "qwen")
            if "thinkingLevelMap" in model:
                model["compat"]["supportsReasoningEffort"] = True
        models.append(model)
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
    providers["snowllm"] = {"baseUrl": f"http://{a.host}:{a.port}/v1",
                            "api": a.api,
                            "apiKey": "snowllm",
                            "models": models}
    models_json.write_text(json.dumps(data, indent=2) + "\n")

    print(f"{term.stamp()} wrote {len(models)} model(s) to {models_json}")
    print(f"{term.stamp()} models: " + ", ".join(m["id"] for m in models))
    return 0
