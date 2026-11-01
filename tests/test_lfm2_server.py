# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import httpx
import _harness

from snowllm import cli
from snowllm.engine import SamplingParams
from snowllm.serve.api import app
from snowllm.serve.state import install, serving


WEATHER = {"name": "get_weather", "description": "Look up current weather in a city.",
           "parameters": {"type": "object", "required": ["city"],
                          "properties": {"city": {"type": "string"}}}}
QUESTION = "What's the weather in Paris? Use get_weather to look it up."
KNOWLEDGE = [
    ("法国的首都是哪里？请用中文简短回答。", ("巴黎",)),
    ("水分子的化学式是什么？简短回答。", ("H2O", "H₂O")),
    ("太阳系中体积最大的行星是什么？请用中文简短回答。", ("木星",)),
    ("光合作用主要吸收二氧化碳还是氧气？简短回答。", ("二氧化碳",)),
    ("计算 17×23，只回答结果。", ("391",)),
]


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"]


async def main():
    root = _harness.checkpoint("LFM2.5-8B-A1B-Q4_K_M", env="SNOWLLM_LFM_DIR")
    draft = _harness.checkpoint("LFM2.5-8B-A1B-DSpark-Q8_0", env="SNOWLLM_LFM_DSPARK_DIR")
    context = int(os.environ.get("SNOWLLM_LFM_CONTEXT", "8192"))
    mode = os.environ.get("SNOWLLM_LFM_SPEC", "fixed")
    if mode not in ("plain", "fixed", "adaptive"):
        raise ValueError("SNOWLLM_LFM_SPEC must be plain, fixed or adaptive")
    install(cli.build(str(root), max_num_seqs=4, max_model_len=context, num_spec=0,
        num_kv_blocks=None, default_max_tokens=512, profile_dir=None,
        prefill_chunk=512, batch_prefill=True, prefix_memory_ratio=.04, seed=0,
        dflash=str(draft) if mode != "plain" else None, dflash_block=9,
        dspark_adaptive=mode == "adaptive", exact_stats=True))
    async_engine = serving().engine
    engine = async_engine.engine
    tok = serving().tokenizer
    check = _harness.Checks(52)
    report = dict(model=root.name, mode=mode, context=context, slots=4, prefill_chunk=512,
                  timestamp=datetime.now(timezone.utc).isoformat(), cases=[])
    completed = False
    async_engine.start()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                base_url="http://asgi", timeout=1800) as http:
        async def post(label, path="/v1/chat/completions", **body):
            start = time.perf_counter()
            r = await http.post(path, json={"model": serving().model_name, **body})
            check(label + " HTTP 200", r.status_code == 200, r.text[:200] if r.status_code != 200 else "")
            r.raise_for_status()
            value = events(r.text) if body.get("stream") else r.json()
            report["cases"].append(dict(label=label, seconds=time.perf_counter()-start, response=value))
            print("CASE", label, json.dumps(value, ensure_ascii=False)[:1800], flush=True)
            return value

        async def chat(label, question, **kwargs):
            return await post(label, messages=[{"role": "user", "content": question}],
                              max_tokens=512, **kwargs)

        try:
            for i, (question, answers) in enumerate(KNOWLEDGE):
                r = await chat(f"knowledge-{i}", question, temperature=0)
                content = r["choices"][0]["message"]["content"].split("</think>")[-1]
                check(f"knowledge-{i} answer", any(a in content for a in answers), repr(content))
                check(f"knowledge-{i} reaches EOS", r["choices"][0]["finish_reason"] == "stop")
                check(f"knowledge-{i} reasoning separated", "<think>" not in content and
                      "</think>" not in r["choices"][0]["message"]["content"])

            chat_tools = [{"type": "function", "function": WEATHER}]
            r = await chat("tool-chat", QUESTION, temperature=0, tools=chat_tools)
            choice = r["choices"][0]
            calls = choice["message"].get("tool_calls", [])
            check("chat returns structured tool calls", len(calls) == 1, calls)
            check("chat tool finish_reason", choice["finish_reason"] == "tool_calls")
            if calls:
                fn = calls[0]["function"]
                check("chat tool name and typed arguments", fn["name"] == "get_weather"
                      and json.loads(fn["arguments"]) == {"city": "Paris"}, fn)
                r = await post("tool-chat-result", temperature=0, max_tokens=512, tools=chat_tools,
                    messages=[{"role": "user", "content": QUESTION}, choice["message"],
                              {"role": "tool", "tool_call_id": calls[0]["id"],
                               "content": "Paris: 22 Celsius, sunny."}])
                check("chat tool result round trip", "22" in (r["choices"][0]["message"]["content"] or ""))

            chunks = await chat("tool-chat-stream", QUESTION, temperature=0, tools=chat_tools,
                                stream=True, stream_options={"include_usage": True})
            streamed = [c for e in chunks for choice in e["choices"]
                        for c in choice["delta"].get("tool_calls", [])]
            check("SSE chat structured tool call", len(streamed) == 1 and
                  json.loads(streamed[0]["function"]["arguments"]) == {"city": "Paris"})
            check("SSE tool finish_reason", any(c["finish_reason"] == "tool_calls"
                  for e in chunks for c in e["choices"]))
            check("SSE reasoning separated", any(c["delta"].get("reasoning_content")
                  for e in chunks for c in e["choices"]) and not any("<think>" in
                  c["delta"].get("content", "") for e in chunks for c in e["choices"]))

            response_tools = [{"type": "function", **WEATHER}]
            r = await post("tool-responses", "/v1/responses", input=QUESTION,
                           temperature=0, max_output_tokens=512, tools=response_tools)
            calls = [c for c in r["output"] if c["type"] == "function_call"]
            check("Responses structured tool call", len(calls) == 1 and
                  json.loads(calls[0]["arguments"]) == {"city": "Paris"})
            check("Responses reasoning item", any(c["type"] == "reasoning" for c in r["output"]))
            if calls:
                r = await post("tool-responses-result", "/v1/responses", temperature=0,
                    max_output_tokens=512, tools=response_tools,
                    input=[{"role": "user", "content": QUESTION}, *r["output"],
                           {"type": "function_call_output", "call_id": calls[0]["call_id"],
                            "output": "Paris: 22 Celsius, sunny."}])
                check("Responses tool result round trip", any("22" in c.get("text", "")
                      for item in r["output"] for c in item.get("content", [])))
            chunks = await post("tool-responses-stream", "/v1/responses", input=QUESTION,
                temperature=0, max_output_tokens=512, tools=response_tools, stream=True)
            completed = [e["response"] for e in chunks if e["type"] == "response.completed"]
            check("Responses SSE completes with tool call", len(completed) == 1 and any(
                c["type"] == "function_call" for c in completed[0]["output"]))

            message_body = dict(temperature=0, max_tokens=512,
                messages=[{"role": "user", "content": QUESTION}], tools=[{
                    "name": WEATHER["name"], "description": WEATHER["description"],
                    "input_schema": WEATHER["parameters"]}])
            r = await post("tool-messages", "/v1/messages", **message_body)
            calls = [c for c in r["content"] if c["type"] == "tool_use"]
            check("Messages structured tool call", r["stop_reason"] == "tool_use" and
                  len(calls) == 1 and calls[0]["input"] == {"city": "Paris"})
            chunks = await post("tool-messages-stream", "/v1/messages", **message_body, stream=True)
            check("Messages SSE tool call", any(e["type"] == "content_block_start" and
                  e["content_block"]["type"] == "tool_use" for e in chunks))

            for i, settings in enumerate((dict(temperature=.7, top_k=40, top_p=.9),
                    dict(temperature=1.1, top_k=80, top_p=.95, repetition_penalty=1.1,
                         presence_penalty=.2, frequency_penalty=.1))):
                r = await chat(f"sampling-{i}", KNOWLEDGE[0][0], **settings)
                content = r["choices"][0]["message"]["content"].split("</think>")[-1]
                check(f"sampling-{i} answer and EOS", "巴黎" in content
                      and r["choices"][0]["finish_reason"] == "stop", repr(content))

            lengths = [int(n) for n in os.environ.get("SNOWLLM_LFM_LONG_TOKENS", "4096").split(",")]
            filler = "这是一条普通的仓库记录，内容为日常库存更新。\n"
            unit = len(tok.encode(filler, add_special_tokens=False))
            corpus = os.environ.get("SNOWLLM_LFM_LONG_CORPUS", str(
                Path(__file__).resolve().parents[1]/"benchmarks/data/pg19_context_bench.txt"))
            corpus = None if corpus == "repeat" else corpus
            corpus_ids = tok.encode(Path(corpus).read_text(), add_special_tokens=False) if corpus else None
            report["long_corpus"] = corpus or "repeated Chinese inventory records"
            for length in lengths:
                if corpus_ids is None:
                    records = filler * max(1, (length-150)//unit)
                else:
                    ids = (corpus_ids * (1+(length//len(corpus_ids))))[:length-150]
                    records = tok.decode(ids)
                mid = len(records)//2
                document = records[:mid] + "\n唯一校验码是73924151。\n" + records[mid:]
                question = ("从下面记录中找出唯一校验码。\n" + document +
                            "\n上面的唯一校验码是多少？只回答校验码。")
                r = await chat(f"long-{length}", question, temperature=0)
                content = r["choices"][0]["message"]["content"].split("</think>")[-1]
                check(f"long-{length} needle recovered", "73924151" in content, repr(content))
                check(f"long-{length} actual prompt length", r["usage"]["prompt_tokens"] >= length-256,
                      r["usage"]["prompt_tokens"])

            async def mixed(i):
                await asyncio.sleep(.03 * i)
                question, answers = KNOWLEDGE[0 if i%2 else i % len(KNOWLEDGE)]
                question = "请阅读背景并回答最后的问题。背景：" + "今天库存正常。" * (17*i) + "\n" + question
                r = await chat(f"concurrent-{i}", question, temperature=0 if i%2 == 0 else .7,
                               top_p=.9, top_k=40)
                text = r["choices"][0]["message"]["content"].split("</think>")[-1]
                check(f"concurrent-{i} answer", any(a in text for a in answers), repr(text))
                check(f"concurrent-{i} EOS", r["choices"][0]["finish_reason"] == "stop")

            peaks = {"running": 0, "waiting": 0}
            jobs = [asyncio.create_task(mixed(i)) for i in range(8)]
            while not all(job.done() for job in jobs):
                stats = engine.stats()
                for key in peaks:
                    peaks[key] = max(peaks[key], getattr(stats, key))
                await asyncio.sleep(.005)
            await asyncio.gather(*jobs)
            report["concurrency_peaks"] = peaks
            check("four active requests observed", peaks["running"] == 4, peaks)
            check("queued requests observed", peaks["waiting"] > 0, peaks)

            prompt = tok.encode("List all integers from 1 to 10000, separated by commas.")
            r = async_engine.submit(prompt, SamplingParams(temperature=.7, max_new_tokens=512))
            stream = async_engine.stream(r)
            await anext(stream)
            await stream.aclose()
            deadline = time.monotonic()+30
            while (not r.done or engine.running or async_engine._live) and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            check("cancellation retires request and slots", r.done and r.finish_reason == "abort"
                  and engine.stats().free_slots == 4 and not async_engine._live)
            r = await chat("after-cancel", KNOWLEDGE[0][0], temperature=0)
            check("generation recovers after cancellation", "巴黎" in r["choices"][0]["message"]["content"])
            check("target graph replayed", engine.runner.n_graph_replays > 0)
            if mode != "plain":
                check("draft graph replayed", engine.runner.n_draft_replays > 0)
            invalid = await http.post("/v1/chat/completions", json={
                "model": serving().model_name, "max_tokens": context,
                "messages": [{"role": "user", "content": "Hi"}]})
            check("API rejects requests beyond context", invalid.status_code == 400)
            report["stats"] = asdict(engine.stats())
            report["accept_hist"] = list(engine.dflash.accept_hist) if engine.dflash else []
            completed = True
        finally:
            await async_engine.close()
            report["passed"] = completed and check.ok
            output = os.environ.get("SNOWLLM_LFM_REPORT")
            if output:
                Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return check.done()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
