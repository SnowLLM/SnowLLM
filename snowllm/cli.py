# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import json
import os
import pathlib
import re
import time

from .async_engine import AsyncEngine
from .engine import SPEC_MAX_STEP_ROWS, Engine
from .model import DEFAULT_GPU_UTIL, DEFAULT_MAX_NUM_SEQS, MAX_NUM_SEQS
from .prefix_cache import CKPT_EVERY, DEFAULT_PREFIX_CACHE_GIB, StateStore
from .state import Alias, ServerState, install
from . import loader, ops


def _size(x: str) -> int:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmM]?)\s*", x)
    if not m:
        raise argparse.ArgumentTypeError(f"{x!r} is not a size (try 4096, 32k, 1M)")
    return int(float(m.group(1)) * {"": 1, "k": 1000, "K": 1000, "m": 10**6, "M": 10**6}[m.group(2)])


def _log(msg: str) -> None:
    print(f"[snowllm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(model_path: str, max_num_seqs: int, max_model_len: int, num_kv_blocks: int | None,
          default_max_tokens: int | None, seed: int | None, profile_dir: str | None,
          enforce_eager: bool = False, num_spec: int = 0, kv_int8: bool = False,
          mtp_window: int = 0,
          mtp_sinks: int = 64,
          prefill_chunk: "int | str" = "auto",
          served_names: list[str] | None = None, batch_prefill: bool = False,
          allow_image_urls: bool = False, limit_mm_per_prompt: int = 4,
          stats_interval: float = 0.0,
          gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
          prefix_cache_gib: float = DEFAULT_PREFIX_CACHE_GIB) -> ServerState:
    from transformers import AutoTokenizer

    root = pathlib.Path(model_path).expanduser()
    tok = AutoTokenizer.from_pretrained(str(root))

    gen = json.loads((root / "generation_config.json").read_text())
    eos = gen.get("eos_token_id", tok.eos_token_id)
    eos = tuple(eos) if isinstance(eos, list) else (eos,)

    def _single(s: str) -> int | None:
        e = tok.encode(s, add_special_tokens=False)
        return e[0] if len(e) == 1 else None

    _log(f"loading {root.name} ...")
    t0 = time.time()
    model = loader.load(root)
    _log(f"loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    engine = Engine(model, num_kv_blocks=num_kv_blocks, max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len, stop_token_ids=eos, seed=seed,
                    enforce_eager=enforce_eager, num_spec=num_spec, kv_int8=kv_int8,
                    mtp_window=mtp_window, mtp_sinks=mtp_sinks,
                    prefill_chunk=prefill_chunk, batch_prefill=batch_prefill,
                    account=stats_interval > 0,
                    gpu_memory_utilization=gpu_memory_utilization,
                    prefix_cache_gib=prefix_cache_gib)
    blocks = engine.blocks.total
    _log(f"decode graphs: {engine.graph_sizes or 'off (eager)'} "
         f"(engine init + capture {time.time() - t1:.1f}s total)")

    orig_max = int(model.config.get("max_position_embeddings", 262144))
    aliases = {}
    for s in served_names or [root.name]:
        name, _, fac = s.partition("=")
        factor = float(fac) if fac else 1.0
        aliases[name] = Alias(factor, min(max_model_len, int(factor * orig_max)))
    proc = None
    if model.visual is not None:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(root))
        _log("vision tower loaded; /v1/chat/completions accepts image_url parts")

    state = ServerState(
        engine=AsyncEngine(engine, profile_dir, stats_interval=stats_interval),
        tokenizer=tok, model=model, processor=proc,
        model_name=next(iter(aliases)), aliases=aliases,
        think_open_id=_single("<think>"), think_close_id=_single("</think>"),
        created=int(time.time()),
        default_max_tokens=default_max_tokens, allow_image_urls=allow_image_urls,
        limit_mm_per_prompt=limit_mm_per_prompt)

    tokens = blocks * ops.KV_BLOCK_SIZE
    _log(f"{max_num_seqs} slots, {blocks} KV blocks = {tokens} tokens "
         f"({engine.runner.kv_bytes / (1 << 30):.1f} GiB, {tokens // max_num_seqs} per slot at full "
         f"concurrency), prefill chunk {engine.prefill_chunk}, ctx {max_model_len}, stop {eos}")
    if engine.cache is None:
        _log("prefix cache off")
    else:
        each = StateStore.bytes_each(engine.runner.linear_mods)
        _log(f"prefix cache: {engine.cache.store.capacity} checkpoints x "
             f"{each / (1 << 20):.1f} MiB of pinned host RAM, one every {CKPT_EVERY} tokens")
    _log(f"served: {', '.join(f'{n} (factor {a.factor}, ctx {a.ctx})' for n, a in aliases.items())}")
    if profile_dir:
        _log(f"profiling armed -> {profile_dir} (POST /start_profile, /stop_profile)")
    return state


def main() -> None:
    import uvicorn

    from .server import app

    p = argparse.ArgumentParser(
        prog="snowllm",
        usage="snowllm PATH [options]",
        description="An OpenAI-compatible server for Qwen3.6-35B-A3B on one gfx1151.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("serving")
    g.add_argument("model", nargs="?", metavar="PATH", default=argparse.SUPPRESS,
                   help="the checkpoint directory")
    g.add_argument("--model", dest="model_flag", metavar="PATH", help=argparse.SUPPRESS)
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--served-model-name", action="append", metavar="NAME=FACTOR",
                   help="expose the model under NAME with YaRN context factor FACTOR (default 1.0); "
                        "repeatable, e.g. --served-model-name Qwen3.6[512K]=2.0. The first is the "
                        "default when a request names no model.")
    g.add_argument("--default-max-tokens", type=int, default=None, metavar="N",
                   help="cap on generated tokens when a request omits max_tokens; default: fill the "
                        "remaining context (recommended for thinking -- a small cap truncates it)")
    g.add_argument("--seed", type=int, default=None)

    g = p.add_argument_group("capacity", "What the engine reserves, and what it will accept. Both "
                                         "are memory choices; no kernel imposes either.")
    g.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS, metavar="N",
                   help=f"concurrency ceiling, up to {MAX_NUM_SEQS}. Speculating does not lower it: "
                        f"a step whose batch is too big to verify decodes plainly instead. Each "
                        f"request pins (num-spec + 1) x 61.4 MiB of linear-attn state for its whole "
                        f"life, so 16 costs 2.9 GiB at --num-spec 2 and 256 costs 46.1 GiB "
                        f"against a ~40 GiB model. That pool is the one thing here provisioned "
                        f"worst-case, so this is the knob that decides whether the engine starts.")
    g.add_argument("--max-model-len", type=_size, default=262144, metavar="N",
                   help="context length; this defaults to the 256K the model supports. "
                        "It is a per-request ceiling, not a reservation: "
                        "the KV pool is shared, and a request that outgrows what is left is "
                        "preempted and re-prefilled rather than pre-allocated for.")
    g.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_UTIL, metavar="F",
                   help="fraction of the device the whole process may occupy. The KV pool is "
                        "whatever is left under it once the weights, the state pool, the prefill "
                        "workspace and the logits are down -- measured, not modelled. Raise it "
                        "toward 1.0 if the GPU is yours alone; what it holds back is fragmentation "
                        "headroom for the caching allocator.")
    g.add_argument("--prefix-cache-gib", type=float, default=DEFAULT_PREFIX_CACHE_GIB,
                   metavar="G",
                   help=f"host RAM for cached prefixes, 0 to disable. A shared prompt is only "
                        f"skippable if the linear-attn state at its end was kept, and that state is "
                        f"61.4 MiB whatever the prefix length -- so this budget, not the KV pool, is "
                        f"what bounds how many prefixes are remembered. Their KV stays in the pool "
                        f"and is shared, not copied. A checkpoint is taken every {CKPT_EVERY} "
                        f"tokens, so a hit lands at the last multiple of {CKPT_EVERY} at or before "
                        f"the shared prefix rather than at a coarse boundary. A prefix shorter than "
                        f"{CKPT_EVERY} tokens is never cached.")
    g.add_argument("--num-kv-blocks", type=_size, default=None, metavar="N",
                   help=f"pin the KV pool at N blocks of {ops.KV_BLOCK_SIZE} tokens instead of "
                        f"sizing it from --gpu-memory-utilization. It must still hold one "
                        f"max-model-len sequence; below that the engine refuses to start.")

    g = p.add_argument_group("throughput", "Every one of these is measured; the numbers and the "
                                           "shapes they were taken at are in docs/ablations.md.")
    g.add_argument("--num-spec", "--num-speculative-tokens", type=int, default=2,
                   metavar="K", dest="num_spec",
                   help=f"MTP draft tokens per step (0 = off). Applied only while the batch is "
                        f"<= {SPEC_MAX_STEP_ROWS} // (num-spec + 1); bigger batches draft shallower")
    g.add_argument("--kv-cache-dtype", choices=("bf16", "auto", "int8"), default="bf16",
                   help="bf16 (or vLLM's `auto`, the same thing here), or int8. int8 halves KV "
                        "bytes and buys ~16%% long-context decode, at some prefill -- for "
                        "decode/long-context-heavy workloads. Its layout is per (token, kv-head, "
                        "d-group) for K and per (token, kv-head) for V, so it is not vLLM's "
                        "int8_per_token_head; int8 beat fp8 by 3.2x on this model's KV "
                        "(docs/ablations.md).")
    g.add_argument("--max-num-batched-tokens", default="auto", metavar="N",
                   help="tokens per launch, or 'auto' to pick the largest whose workspace stays "
                        "under a fifth of what the state pool leaves -- the rest goes to KV. A "
                        "step is one prefill chunk or one decode, so this is the "
                        "prefill chunk: it bounds every M-sized buffer -- 30 GB of scratch for a "
                        "one-shot 200K prefill against 1 GB at 8192 -- and changes no arithmetic.")
    g.add_argument("--batch-prefill", action="store_true",
                   help="pack several fresh short prompts into one prefill launch (up to +8%% at "
                        "high concurrency, --num-spec 0 only). NOT batch-invariant: a prompt's "
                        "output can depend on who it batched with (bf16 tiling).")
    g.add_argument("--mtp-window", type=int, default=0, metavar="W",
                   help="cap the MTP draft layer's attention at --mtp-sinks leading tokens plus the "
                        "most recent W (0 = off, full context). Try 4032. Cannot change output, "
                        "only acceptance.")
    g.add_argument("--mtp-sinks", type=int, default=64, metavar="N",
                   help="leading tokens the draft layer always sees, with --mtp-window")

    g = p.add_argument_group("multimodal")
    g.add_argument("--limit-mm-per-prompt", type=int, default=4, metavar="N",
                   help="images a single request may carry. Each one runs the vision tower and adds "
                        "its tokens to the prompt, so an unbounded count is a way to make one "
                        "request occupy the GPU.")
    g.add_argument("--allow-image-urls", action="store_true",
                   help="let an image_url part name an http(s) URL, which this server will then "
                        "FETCH. Off by default: that is an egress from your network chosen by "
                        "whoever sends the request. data: URIs always work.")

    g = p.add_argument_group("diagnostics")
    g.add_argument("--enforce-eager", action="store_true", help="skip CUDA-graph capture")
    g.add_argument("--profile-dir", default=os.environ.get("SNOWLLM_TORCH_PROFILER_DIR"),
                   metavar="DIR",
                   help="enable POST /start_profile and /stop_profile, writing traces here. "
                        "Unless this, SNOWLLM_TRACE or HSA_TOOLS_LIB is set, "
                        "HSA_TOOLS_DISABLE_REGISTER=1 is set so ROCm's profiler does not "
                        "intercept queue creation -- its interception leaves a signal the "
                        "runtime cannot sleep on, and one core then spins for the process's "
                        "whole life, idle or not. Anything that profiles or traces needs it "
                        "back, and costs that core.")
    g.add_argument("--stats-interval", type=float, default=0.0, metavar="SECONDS",
                   help="log prefill/decode throughput and accepted length every SECONDS, and "
                        "carry the cumulative pair on /health. OFF by default because splitting "
                        "the two costs a torch.cuda.synchronize() per step -- without it a prefill "
                        "chunk that emits no token never syncs and its time lands on a later "
                        "decode step (engine.py's StepAccounting).")

    a = p.parse_args()
    positional = getattr(a, "model", None)
    if positional and a.model_flag:
        p.error("the checkpoint directory was given twice")
    model = positional or a.model_flag
    if not model:
        p.error("the checkpoint directory is required: `snowllm PATH`")

    if not (a.profile_dir or os.environ.get("SNOWLLM_TRACE")
            or os.environ.get("HSA_TOOLS_LIB")):
        os.environ.setdefault("HSA_TOOLS_DISABLE_REGISTER", "1")

    install(build(model, a.max_num_seqs, a.max_model_len, num_kv_blocks=a.num_kv_blocks,
          default_max_tokens=a.default_max_tokens, seed=a.seed, profile_dir=a.profile_dir,
          stats_interval=a.stats_interval,
          enforce_eager=a.enforce_eager, num_spec=a.num_spec,
          mtp_window=a.mtp_window, mtp_sinks=a.mtp_sinks, served_names=a.served_model_name,
          batch_prefill=a.batch_prefill, allow_image_urls=a.allow_image_urls,
          prefill_chunk=a.max_num_batched_tokens, kv_int8=a.kv_cache_dtype == "int8",
          gpu_memory_utilization=a.gpu_memory_utilization,
          prefix_cache_gib=a.prefix_cache_gib,
          limit_mm_per_prompt=a.limit_mm_per_prompt))
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
