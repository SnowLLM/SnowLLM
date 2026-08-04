# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Throughput/interactivity sweep at a fixed ISL/OSL, mirroring InferenceMAX; `--prompts speedbench`
swaps in real text (nvidia/SPEED-Bench) since random ids flatter this engine's MTP/MoE behavior.

Usage:  python benchmarks/bench_isl_osl.py --isl 8192 --osl 1024 --conc 1,4,8,16
"""

import argparse
import collections
import json
import pathlib
import random
import statistics as st
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from snowllm import _capi, loader
from snowllm._capi import build_geometry
from snowllm.engine import Engine, SamplingParams

BUCKETS = {1024: "1k", 2048: "2k", 8192: "8k", 16384: "16k", 32768: "32k"}
# Part of the SPEED-Bench parquet is a manifest rather than a corpus: the row carries this string
# in place of the prompt, plus the `source` URL it must be fetched from.
SENTINEL = "SHOULD BE FETCHED FROM THE SOURCE"


def random_prompts(tok, isl: int, n: int, rng: random.Random, jitter: float):
    """{"random": [ids]} the way vLLM's RandomDataset builds them, which is what InferenceMAX runs.

    The ramp is generated longer than needed and the RE-ENCODED sequence truncated to the drawn
    length, so `want` is delivered exactly. vLLM instead tops up after the fact and reports a
    `token_mismatch` count; the two agree on length, which is all the engine sees.
    """
    special = set(tok.all_special_ids)
    allowed = np.array([t for t in range(tok.vocab_size) if t not in special])
    out = []
    for i in range(n):
        want = rng.randrange(int(jitter * isl), isl + 1)
        offset = rng.randrange(len(allowed))
        ramp = allowed[(offset + i + np.arange(int(want * 1.5))) % len(allowed)].tolist()
        ids = tok.encode(tok.decode(ramp), add_special_tokens=False)[:want]
        out.append(ids)
    return {"random": out}


def prompts(tok, isl: int, n: int, rng: random.Random, jitter: float):
    """{tier: [ids]} from the SPEED-Bench bucket nearest `isl`, each truncated to its own length.

    The length is drawn per request from [jitter*isl, isl] and the row is skipped unless it can
    supply that many tokens, so the realized ISL is the drawn one rather than however short the
    document happened to be.
    """
    name = BUCKETS[min(BUCKETS, key=lambda b: abs(b - isl))]
    table = pq.read_table(hf_hub_download(
        "nvidia/SPEED-Bench", f"throughput_{name}/test-00000-of-00001.parquet",
        repo_type="dataset"))
    out = collections.defaultdict(list)
    dropped = collections.Counter()
    for tier, turns in zip(table.column("category").to_pylist(),
                           table.column("turns").to_pylist()):
        if len(out[tier]) >= n:
            continue
        if SENTINEL in turns[0]:
            dropped[tier] += 1
            continue
        want = rng.randrange(int(jitter * isl), isl + 1)
        ids = tok.encode(turns[0])
        if len(ids) >= want:
            out[tier].append(ids[:want])
    if dropped:
        print(f"  unshipped rows skipped: {dict(dropped)}", flush=True)
    return {t: v for t, v in out.items() if v}


def run_cell(eng, batch, osl):
    """One concurrency level end to end. Returns per-request (ttft_s, e2el_s, n_out) and the wall.

    No stop tokens, so every request ends on max_new_tokens and generates exactly `osl`. That is
    what keeps the batch at full width for the whole decode: with EOS honoured, requests retire at
    different steps and the measured concurrency would be an average, not the stated one.
    """
    params = SamplingParams(temperature=0.0, max_new_tokens=osl)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reqs = [eng.add(ids, params) for ids in batch]
    ttft, steps = {}, collections.Counter()
    while any(not r.done for r in reqs):
        # Snapshotted BEFORE the step: a request that finishes inside it still took part in it, and
        # charging it afterwards would divide its tokens by one step too few.
        active = [i for i, r in enumerate(reqs) if r.prefilled and not r.done]
        kind = eng.step()
        torch.cuda.synchronize()
        now = time.perf_counter()
        if kind == "decode":
            for i in active:
                steps[i] += 1
        for i, r in enumerate(reqs):
            if i not in ttft and (r.out or r.done):   # osl == 1: first and last token coincide
                ttft[i] = now - t0
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    # Per-request end times are not observable at step granularity once a request is done, so E2EL
    # is charged at the wall clock: with a fixed osl every request ends within one step of the last.
    # out[0] is sampled by the PREFILL step, so only the rest is what the decode steps produced;
    # charging it to them would put k=0 at 1.03 tokens/step instead of the 1.000 it is by definition.
    return [(ttft[i], wall, len(r.out), (len(r.out) - 1) / max(steps[i], 1))
            for i, r in enumerate(reqs)], wall


def cell_metrics(per, wall, isl_lens):
    """Prefill and decode throughput, kept apart.

    The wall clock splits at the moment the LAST request has its first token: before it the machine
    is doing prompt processing, after it decode. So `prefill_tok_s` is every prompt token over the
    first phase and `output_tok_s` every generated token over the second, which is
    llama-batched-bench's S_PP and S_TG. One blended tokens/s over the whole wall is the number
    llama-bench refuses to print, and at 220K it is dominated by a 157 s prefill amortised over
    1024 output tokens -- a figure that describes neither half. `blended_tok_s` is kept only
    because it is what a caller timing the whole request would see.

    TPOT and TTFT follow vLLM: TPOT = (E2EL - TTFT)/(n-1), per request, then the median.

    `accept_len` is tokens emitted per decode step -- 1.0 with speculation off, up to num_spec+1
    with it on. It is the mechanism behind every MTP number here, and a property of the text rather
    than of the timing, so it is the field to look at when two cells disagree.
    """
    ttfts = [p[0] for p in per]
    tpots = [(p[1] - p[0]) / max(p[2] - 1, 1) for p in per]
    out_tokens = sum(p[2] for p in per)
    tpot = st.median(tpots)
    t_pp = max(ttfts)
    t_tg = max(wall - t_pp, 1e-9)
    return dict(ttft_ms=st.median(ttfts) * 1e3,
                ttft_max_ms=t_pp * 1e3,
                tpot_ms=tpot * 1e3,
                t_pp_s=t_pp,
                t_tg_s=t_tg,
                prefill_tok_s=sum(isl_lens) / t_pp,
                # Minus one per request: out[0] is sampled by the prefill step, not by a decode one.
                output_tok_s=(out_tokens - len(per)) / t_tg,
                tok_s_user=1.0 / tpot,
                blended_tok_s=out_tokens / wall,
                accept_len=st.median(p[3] for p in per),
                out_tokens=out_tokens,
                isl_mean=st.mean(isl_lens),
                isl_min=min(isl_lens),
                isl_max=max(isl_lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--isl", default="8192", help="one ISL, or a comma list to sweep")
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--conc", default="1,4,8,16", help="concurrency levels to sweep")
    ap.add_argument("--num-spec", default="2", help="MTP depths to sweep; 0 is the baseline")
    ap.add_argument("--jitter", type=float, default=0.8,
                    help="per-request ISL is drawn from [jitter*isl, isl]")
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="0 = isl + osl + 64. Sizing it to the test is what InferenceMAX is "
                         "criticised for (its configs run DeepSeek at a ~2K ceiling): KV is "
                         "provisioned worst-case over it, so a tight ceiling buys memory no real "
                         "deployment has. Pass the context you would actually serve.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", choices=("random", "speedbench"), default="random",
                    help="random = InferenceMAX's own synthetic prompts, for comparability with "
                         "their charts; speedbench = real text, for a number about the engine")
    ap.add_argument("--tiers", default="high_entropy,low_entropy",
                    help="SPEED-Bench tiers; ignored when --prompts random")
    ap.add_argument("--out", default="benchmarks/results/isl_osl_results.json")
    args = ap.parse_args()

    isls = [int(x) for x in args.isl.split(",")]
    concs = [int(c) for c in args.conc.split(",")]
    ks = [int(k) for k in args.num_spec.split(",")]
    tiers = args.tiers.split(",")
    ckpt = pathlib.Path("~/models").expanduser() / args.model

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    t0 = time.perf_counter()
    model = loader.load(ckpt)
    print(f"loaded {ckpt.name} in {time.perf_counter() - t0:.1f}s", flush=True)

    block = _capi.build_geometry().block_size
    rows = []
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for isl in isls:
        # Reseeded per ISL, so a point's prompts do not depend on which other points share the run.
        rng = random.Random(args.seed)
        need = max(concs) * args.reps
        if args.prompts == "random":
            pool = random_prompts(tok, isl, need, rng, args.jitter)
        else:
            pool = prompts(tok, isl, need, rng, args.jitter)
            pool = {t: v for t, v in pool.items() if t in tiers}
            missing = [t for t in tiers if t not in pool]
            if missing:
                raise SystemExit(f"no prompt survived for {missing} at ISL {isl}; SPEED-Bench "
                                 f"ships `mixed` as placeholders, and a tier needs rows of "
                                 f">= {int(args.jitter * isl)} tokens")
        for tier, v in pool.items():
            print(f"  ISL {isl} {tier}: {len(v)} prompts, "
                  f"{min(map(len, v))}-{max(map(len, v))} tokens", flush=True)

        mml = args.max_model_len or isl + args.osl + 64
        for k in ks:
            # A fresh engine per depth: num_spec > 0 retires the decode graphs (engine.py), so
            # sharing one across depths would run the k=0 arm un-graphed and flatter MTP.
            eng = Engine(model, num_kv_blocks=max(concs) * -(-mml // block),
                         max_num_seqs=max(concs), max_model_len=mml, seed=args.seed, num_spec=k)
            warm = next(iter(pool.values()))[0]
            run_cell(eng, [warm[:512]], 8)                # discard: cold caches, first capture
            cells = [(tier, c) for tier in pool for c in concs]
            acc = collections.defaultdict(list)
            for rep in range(args.reps):
                # Palindrome order across reps: this part is thermally biased, so whichever cell
                # runs first tends to win. Reversing every other rep cancels the drift out of the
                # medians.
                for tier, c in (cells if rep % 2 == 0 else list(reversed(cells))):
                    batch = pool[tier][rep * c:(rep + 1) * c]
                    if len(batch) < c:
                        batch = pool[tier][:c]
                    per, wall = run_cell(eng, batch, args.osl)
                    m = cell_metrics(per, wall, [len(b) for b in batch])
                    acc[(tier, c)].append(m)
                    print(f"  ISL {isl} k={k} {tier:14s} C={c:3d} rep{rep}  "
                          f"TTFT {m['ttft_ms']:8.1f} ms  prefill {m['prefill_tok_s']:7.0f} tok/s  "
                          f"output {m['output_tok_s']:7.1f} tok/s  "
                          f"({m['tok_s_user']:5.1f}/user)", flush=True)
            for (tier, c), ms in acc.items():
                # graph_sizes belongs to the row, not to the run: num_spec > 0 disables capture, so
                # the k arms of one sweep do not share it.
                row = dict(k=k, tier=tier, conc=c, isl=isl, osl=args.osl, max_model_len=mml,
                           reps=len(ms), graph_sizes=list(eng.graph_sizes),
                           # Per-rep, in order: each rep is a DIFFERENT batch of prompts, so this
                           # is the prompt-to-prompt spread, which is what decides whether two
                           # cells differ or merely differ in their sample.
                           accept_len_reps=[m["accept_len"] for m in ms],
                           tok_s_user_reps=[m["tok_s_user"] for m in ms],
                           **{f: st.median(m[f] for m in ms) for f in ms[0]})
                rows.append(row)
                out_path.write_text(json.dumps(
                    dict(config=vars(args), block_size=block, abi=_capi.ABI_VERSION, rows=rows),
                    indent=1))
            del eng
            torch.cuda.empty_cache()

    # ACHIEVED ISL, not the nominal one. Drawing from [jitter*isl, isl] and then advertising `isl`
    # is the InferenceMAX bug its own issue #356 declined to fix ("we are adding real datasets soon
    # so it won't matter anyway"): every published number is against ~10% less input than its label.
    print(f"\nOSL {args.osl}, jitter {args.jitter:g}, prompts {args.prompts}\n")
    print("| ISL | achieved | tier | k | C | TTFT ms | prefill tok/s | output tok/s | per user | "
          "AL |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = min(r["accept_len_reps"]), max(r["accept_len_reps"])
        print(f"| {r['isl']} | {r['isl_mean']:.0f} | {r['tier']} | {r['k']} | {r['conc']} | "
              f"{r['ttft_ms']:.0f} | {r['prefill_tok_s']:.0f} | {r['output_tok_s']:.1f} | "
              f"{r['tok_s_user']:.1f} | {r['accept_len']:.3f} [{lo:.2f},{hi:.2f}] |")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
