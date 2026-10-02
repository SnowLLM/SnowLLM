# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import json
import pathlib
import shutil
import sys
import tempfile

import _harness

from snowllm.hub import pi, recipes  # noqa: E402
from snowllm.serve.protocol import ChatRequest, ResponsesRequest  # noqa: E402

check = _harness.Checks()


def catalogue(tmp: pathlib.Path) -> pathlib.Path:
    raw = {"schema": 1, "recipes": [
        {"id": "deepseek-fp8", "dir": "DeepSeek-V4-Flash-0731", "repo": "a/b",
         "serve_as": "DeepSeek-V4-Flash-0731"},
        {"id": "qwen27-fp8", "dir": "Qwen3.6-27B-FP8", "repo": "a/b",
         "serve_as": "Qwen3.6-27B", "defaults": {"max_model_len": 131072}},
        {"id": "qwen38-fp8", "dir": "Qwen3.8-27B-FP8", "repo": "a/b",
         "include": ["Qwen3.8-27B-Q4_K_XL.gguf", "mmproj-BF16.gguf"],
         "serve_as": "Qwen3.8-27B"},
    ]}
    p = tmp / "recipe.json"
    p.write_text(json.dumps(raw))
    return p


def run(tmp: pathlib.Path, cat: pathlib.Path, *extra: str) -> tuple[int, dict]:
    models = tmp / "models.json"
    if not models.exists():
        models.write_text(json.dumps(
            {"providers": {"remote": {"api": "openai", "models": [{"id": "r"}]}}}))
    saved, pi.MODELS_JSON = pi.MODELS_JSON, str(models)
    try:
        rc = pi.main("pi", ["--catalogue", str(cat), *extra])
    finally:
        pi.MODELS_JSON = saved
    return rc, json.loads(models.read_text())


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        cat = catalogue(tmp)
        rc, data = run(tmp, cat)
        prov = data["providers"]["snowllm"]
        by = {m["id"]: m for m in prov["models"]}
        check("exit 0", rc == 0)
        check("other providers preserved", "remote" in data["providers"])
        check("responses api by default", prov["api"] == "openai-responses", prov["api"])
        check("baseUrl points at the port", prov["baseUrl"].endswith(":8000/v1"),
              prov["baseUrl"])
        check("catalogue models emitted",
              set(by) == {"DeepSeek-V4-Flash-0731", "Qwen3.6-27B", "Qwen3.8-27B"}, sorted(by))
        check("context window defaults to 256K",
              by["DeepSeek-V4-Flash-0731"]["contextWindow"] == recipes.DEFAULT_CONTEXT == 262144,
              by["DeepSeek-V4-Flash-0731"]["contextWindow"])
        check("a recipe's max_model_len wins", by["Qwen3.6-27B"]["contextWindow"] == 131072,
              by["Qwen3.6-27B"]["contextWindow"])
        check("Qwen3.8 folds high onto xhigh",
              by["Qwen3.8-27B"]["thinkingLevelMap"]["high"] == "xhigh",
              by["Qwen3.8-27B"]["thinkingLevelMap"])
        check("Qwen3.8 folds minimal onto low",
              by["Qwen3.8-27B"]["thinkingLevelMap"]["minimal"] == "low")
        check("Qwen3.6 needs no level map", "thinkingLevelMap" not in by["Qwen3.6-27B"])
        check("DeepSeek exposes max",
              by["DeepSeek-V4-Flash-0731"]["thinkingLevelMap"] == {"max": "max"},
              by["DeepSeek-V4-Flash-0731"]["thinkingLevelMap"])
        check("vision models take images", by["Qwen3.8-27B"]["input"] == ["text", "image"],
              by["Qwen3.8-27B"]["input"])
        check("text-only models need no input field",
              "input" not in by["Qwen3.6-27B"]
              and "input" not in by["DeepSeek-V4-Flash-0731"])
        check("responses needs no thinkingFormat by default",
              all("thinkingFormat" not in m["compat"] for m in prov["models"]))
        check("system prompt stays system",
              all(m["compat"]["supportsDeveloperRole"] is False for m in prov["models"]))

        rc, data = run(tmp, cat, "--api", "openai-completions")
        prov = data["providers"]["snowllm"]
        by = {m["id"]: m for m in prov["models"]}
        check("completions api selected", prov["api"] == "openai-completions", prov["api"])
        check("DeepSeek reads thinking",
              by["DeepSeek-V4-Flash-0731"]["compat"]["thinkingFormat"] == "deepseek")
        check("Qwen reads enable_thinking",
              by["Qwen3.6-27B"]["compat"]["thinkingFormat"] == "qwen")
        check("completions keeps the Qwen3.8 map",
              by["Qwen3.8-27B"]["thinkingLevelMap"]["max"] == "xhigh")
        check("completions sends reasoning_effort when the template reads it",
              by["Qwen3.8-27B"]["compat"]["supportsReasoningEffort"] is True
              and by["DeepSeek-V4-Flash-0731"]["compat"]["supportsReasoningEffort"] is True)
        check("Qwen3.6 has no reasoning_effort to send",
              "supportsReasoningEffort" not in by["Qwen3.6-27B"]["compat"])

        none = ResponsesRequest(input="x", reasoning={"effort": "none"}).template_kwargs()
        med = ResponsesRequest(input="x", reasoning={"effort": "medium"}).template_kwargs()
        check("effort none turns thinking off",
              none["enable_thinking"] is False and none["thinking"] is False, none)
        check("effort passes through", med["reasoning_effort"] == "medium")
        check("effort medium turns thinking on",
              med["enable_thinking"] is True and med["thinking"] is True, med)
        off = ChatRequest(messages=[], reasoning_effort="none").template_kwargs()
        check("chat none turns thinking off",
              off["enable_thinking"] is False and off["thinking"] is False, off)
        explicit = ChatRequest(messages=[], reasoning_effort="medium",
                               chat_template_kwargs={"enable_thinking": False}).template_kwargs()
        check("an explicit enable_thinking wins",
              explicit["enable_thinking"] is False and explicit["thinking"] is False, explicit)
        alias = ChatRequest(messages=[], thinking={"type": "disabled"}).template_kwargs()
        check("thinking aliases enable_thinking",
              alias["enable_thinking"] is False and alias["thinking"] is False, alias)
        return check.done()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
