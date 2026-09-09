# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import asyncio
import collections
import json
import pathlib
import re
import statistics as st
import time
import urllib.request
from collections.abc import Callable

ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=MathArena%2Faime_2026"
            "&config=default&split=train&offset=0&length=30")
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
N_PROBLEMS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1, metavar="K",
                   help="samples per problem, the k of avg@k. 1 by default because that is one "
                        "round -- a complete 30-problem result in ~1.8 h. Deeper is a multiple of "
                        "that and should be asked for, not arrived at by typing nothing")
    p.add_argument("--conc", type=int, default=16,
                   help="requests in flight; match the server's --max-num-seqs")
    p.add_argument("--max-tokens", type=int, default=81920,
                   help="the card's competition-math cap. A smaller one scores truncation as wrong")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=None, help="default: whatever /v1/models serves first")
    p.add_argument("--out", default="benchmarks/results/results_aime.jsonl",
                   help="one line per completed sample, appended as it lands; reread to resume")
    p.add_argument("--log", default=None, help="default: --out with a .log suffix")
    p.add_argument("--summary", default=None, help="default: --out with a .summary.json suffix")
    p.add_argument("--every", type=float, default=120.0, help="seconds between progress lines")
    p.add_argument("--rescore", action="store_true",
                   help="re-judge the existing JSONL and reprint the report; generates nothing and "
                        "needs no server")
    return p.parse_args()


class Log:
    def __init__(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", buffering=1)

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")


def problems() -> list[dict]:
    raw = json.load(urllib.request.urlopen(ROWS_URL))["rows"]
    out = []
    for rec in raw:
        r = rec["row"]
        out.append({"idx": int(r["problem_idx"]), "answer": int(r["answer"]),
                    "problem": r["problem"]})
    if len(out) != N_PROBLEMS:
        raise SystemExit(f"expected {N_PROBLEMS} problems, the dataset served {len(out)}")
    return out


def boxed(text: str) -> str | None:
    at = text.rfind("\\boxed{")
    if at < 0:
        return None
    depth, out = 1, []
    for ch in text[at + len("\\boxed{"):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


def scored(got: str | None, gold: int) -> bool:
    if got is None:
        return False
    m = re.fullmatch(r"\s*(-?\d+)\s*", got.replace(",", "").replace("\\!", ""))
    return m is not None and int(m.group(1)) == gold


def health(base_url: str) -> dict:
    try:
        with urllib.request.urlopen(base_url.rstrip("/").removesuffix("/v1") + "/health") as f:
            return json.load(f).get("accounting") or {}
    except Exception:  # noqa: BLE001 -- a server without /health still evaluates fine
        return {}


def throughput(before: dict, after: dict) -> dict:
    if not before or not after:
        return {}
    d = {k: after[k] - before[k] for k in after if isinstance(after[k], (int, float))}
    return dict(
        prefill_tokens=d["prefill_tokens"], prefill_seconds=d["prefill_seconds"],
        prefill_tok_s=d["prefill_tokens"] / d["prefill_seconds"] if d["prefill_seconds"] else 0.0,
        decode_tokens=d["decode_tokens"], decode_seconds=d["decode_seconds"],
        decode_tok_s=d["decode_tokens"] / d["decode_seconds"] if d["decode_seconds"] else 0.0,
        decode_steps=d["decode_steps"],
        accept_len=d["decode_tokens"] / d["decode_rows"] if d["decode_rows"] else 0.0)


def done_already(path: pathlib.Path) -> set:
    if not path.exists():
        return set()
    got = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        got.add((r["problem"], r["sample"]))
    return got


async def main() -> None:
    a = parse_args()
    out = pathlib.Path(a.out)
    log = Log(pathlib.Path(a.log) if a.log else out.with_suffix(".log"))
    summary_path = pathlib.Path(a.summary) if a.summary else out.with_suffix(".summary.json")

    if a.rescore:
        log(f"\n=== rescoring {out} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
        report(a, out, summary_path, log, 0.0, 0, {})
        return

    probs = problems()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=14400)
    model = a.model or (await client.models.list()).data[0].id

    acct0 = health(a.base_url)
    health_path = out.with_suffix(".health.jsonl")
    snap = health_path.open("a", buffering=1) if acct0 else None
    if snap:
        snap.write(json.dumps({"t": time.time(), **acct0}) + "\n")

    have = done_already(out)
    todo = [(s, p) for s in range(a.n) for p in probs if (p["idx"], s) not in have]

    log(f"\n=== AIME 2026, avg@{a.n}, {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    log(f"model {model}  conc {a.conc}  max_tokens {a.max_tokens}")
    log(f"sampling: temperature {a.temperature}, top_p {a.top_p}, top_k {a.top_k}, "
        f"presence_penalty {a.presence_penalty}")
    log(f"{len(probs)} problems x {a.n} = {len(probs) * a.n} samples; "
        f"{len(have)} already in {out}, {len(todo)} to run")
    if not todo:
        log("nothing to do")
        return

    sink = out.open("a", buffering=1)
    sem = asyncio.Semaphore(a.conc)
    state = {"done": 0, "hits": 0, "tokens": 0, "flying": 0, "failed": 0,
             "t0": time.perf_counter()}

    async def run_one(s: int, p: dict) -> None:
        async with sem:
            state["flying"] += 1
            t0 = time.perf_counter()
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": p["problem"] + SUFFIX}],
                    max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p,
                    presence_penalty=a.presence_penalty, extra_body={"top_k": a.top_k})
            except Exception as e:  # noqa: BLE001 -- anything at all, for 3 hours
                state["failed"] += 1
                log(f"  p{p['idx']:<2} s{s}  ERR {type(e).__name__}: {str(e)[:120]}")
                return
            finally:
                state["flying"] -= 1
            sec = time.perf_counter() - t0

        msg = r.choices[0].message
        think = getattr(msg, "reasoning_content", None) or ""
        content = msg.content or ""
        got = boxed(content) or boxed(think + content)
        ok = scored(got, p["answer"])
        rec = dict(problem=p["idx"], sample=s, ok=ok, got=got, gold=p["answer"],
                   sec=round(sec, 2), out_tokens=r.usage.completion_tokens,
                   prompt_tokens=r.usage.prompt_tokens, finish=r.choices[0].finish_reason,
                   reasoning=think, content=content)
        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")

        state["done"] += 1
        state["hits"] += ok
        state["tokens"] += r.usage.completion_tokens
        log(f"  p{p['idx']:<2} s{s}  {'OK ' if ok else 'BAD'} {str(got)[:12]:>12} "
            f"(gold {p['answer']:>3})  {r.usage.completion_tokens:6d} tok  {sec:7.1f}s  "
            f"{r.choices[0].finish_reason}")

    async def progress() -> None:
        while True:
            await asyncio.sleep(a.every)
            if snap:
                cur = health(a.base_url)
                if cur:
                    snap.write(json.dumps({"t": time.time(), **cur}) + "\n")
            d, n = state["done"], len(todo)
            if d == 0:
                continue
            el = time.perf_counter() - state["t0"]
            log(f"[{d}/{n}  {state['flying']} in flight  {state['hits'] / d:.3f} correct  "
                f"{state['tokens'] / el:.0f} tok/s  elapsed {el / 60:.0f}m  "
                f"eta {el / d * (n - d) / 60:.0f}m"
                + (f"  {state['failed']} FAILED" if state["failed"] else "") + "]")

    ticker = asyncio.create_task(progress())
    try:
        await asyncio.gather(*(run_one(s, p) for s, p in todo))
    finally:
        ticker.cancel()
        sink.close()

    report(a, out, summary_path, log, time.perf_counter() - state["t0"], state["tokens"],
           throughput(acct0, health(a.base_url)))


def report(a: argparse.Namespace, out: pathlib.Path, summary_path: pathlib.Path,
           log: Callable[..., None], wall: float, tokens: int, engine: dict) -> None:
    recs = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    by_problem = collections.defaultdict(list)
    for r in recs:
        r["got"] = boxed(r["content"]) or boxed(r["reasoning"] + r["content"])
        r["ok"] = scored(r["got"], r["gold"])
        by_problem[r["problem"]].append(r)

    per = {p: sum(v["ok"] for v in rs) / len(rs) for p, rs in by_problem.items()}
    score = st.mean(per.values()) if per else 0.0
    stderr = (st.stdev(per.values()) / len(per) ** 0.5) if len(per) > 1 else float("nan")
    trunc = sum(r["finish"] == "length" for r in recs)
    lens = [r["out_tokens"] for r in recs]

    prompt_toks = sum(r["prompt_tokens"] for r in recs)
    out_toks = sum(r["out_tokens"] for r in recs)

    depths = sorted(len(v) for v in by_problem.values())
    k = f"avg@{depths[0]}" if depths[0] == depths[-1] else f"avg@{depths[0]}-{depths[-1]}"
    short = [p for p, v in sorted(by_problem.items()) if len(v) < depths[-1]]

    log("")
    log("--- accuracy ---")
    log(f"{k} over {len(by_problem)} problems, {len(recs)} samples: "
        f"{score * 100:.1f}%  (+/- {stderr * 100:.1f} between problems)")
    if depths[0] != depths[-1]:
        log(f"  uneven: {len(short)} problem(s) have fewer than {depths[-1]} samples "
            f"-- each problem's rate is over its own samples, so the mean is still unbiased, "
            f"but the deeper problems carry less noise than the shallow ones")
    log(f"  Qwen3.6-35B-A3B reports 92.7 on AIME26 -- avg@8, and a different harness, so read the "
        f"gap as a harness gap until a bf16 run says otherwise")
    if depths[-1] < 8:
        log(f"  DEPTH-ASYMMETRIC: {k} against the card's avg@8 -- this side carries the larger "
            f"sampling error, so the sign of the gap is not evidence on its own")
    hard = sorted(per.items(), key=lambda kv: kv[1])[:5]
    log(f"  worst problems: {', '.join(f'p{p}={v:.2f}' for p, v in hard)}")

    log("")
    log("--- work ---")
    log(f"  input {prompt_toks} tok, output {out_toks} tok "
        f"(median {st.median(lens):.0f}/sample, max {max(lens)}, "
        f"{trunc}/{len(recs)} truncated at the cap)")
    log(f"  wall {wall / 3600:.2f} h this run, {tokens / max(wall, 1e-9):.0f} tok/s end to end")

    source = "differenced /health"
    if not engine:
        hp = out.with_suffix(".health.jsonl")
        if hp.exists():
            snaps = [json.loads(x) for x in hp.read_text().splitlines() if x.strip()]
            if len(snaps) >= 2:
                engine = throughput(snaps[0], snaps[-1])
                wall = wall or snaps[-1]["t"] - snaps[0]["t"]
                source = f"{hp.name}, {len(snaps)} snapshots over {wall / 3600:.2f} h"

    if engine:
        log("")
        log(f"--- engine, this run only ({source}) ---")
        log(f"  prefill {engine['prefill_tok_s']:8.1f} tok/s   "
            f"({engine['prefill_tokens']} tok in {engine['prefill_seconds']:.1f}s of prefill steps)")
        log(f"  decode  {engine['decode_tok_s']:8.1f} tok/s   "
            f"({engine['decode_tokens']} tok in {engine['decode_seconds']:.1f}s of decode steps)")
        log(f"  accepted length {engine['accept_len']:.3f} over {engine['decode_steps']} steps "
            f"(ceiling num_spec + 1)")
    else:
        log("")
        log("  no engine accounting: start the server with --stats-interval so that /health "
            "carries it and this run's snapshots land beside the results")

    json.dump(dict(args=vars(a), score=score, stderr=stderr, n_samples=len(recs), depth=k,
                   samples_per_problem={p: len(v) for p, v in sorted(by_problem.items())},
                   per_problem=per, truncated=trunc, median_tokens=st.median(lens),
                   prompt_tokens=prompt_toks, output_tokens=out_toks,
                   wall_s=wall, end_to_end_tok_s=tokens / max(wall, 1e-9), engine=engine),
              summary_path.open("w"), indent=1)
    log(f"\n  summary -> {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
