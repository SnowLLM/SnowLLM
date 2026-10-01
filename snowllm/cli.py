# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import json
import os
import pathlib
import re
import sys
import time

from . import term

SUBCOMMANDS = ("pull", "recipes", "pi")


def _size(x: str) -> int:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmM]?)\s*", x)
    if not m:
        raise argparse.ArgumentTypeError(f"{x!r} is not a size (try 4096, 32k, 1M)")
    return int(float(m.group(1)) * {"": 1, "k": 1024, "K": 1024, "m": 2**20, "M": 2**20}[m.group(2)])


def _log(msg: str) -> None:
    print(f"{term.stamp(time.strftime('%H:%M:%S'))} {msg}", flush=True)


def _draft_bytes(dflash: str | None, geo: object) -> int:
    from .engine import _dspark
    if not dflash or not _dspark(geo):
        return 0
    from .checkpoint.gguf.source import find_dspark_gguf
    p = pathlib.Path(dflash).expanduser()
    side = find_dspark_gguf(p) if p.is_dir() else p
    return side.stat().st_size if side else 0


def build(model_path: str, max_num_seqs: int, max_model_len: int, num_kv_blocks: int | None,
          default_max_tokens: int | None, seed: int | None, profile_dir: str | None,
          enforce_eager: bool = False, num_spec: int = 0, kv_int8: bool = False,
          mtp_window: int = 0,
          mtp_sinks: int = 64,
          dflash: str | None = None, dflash_block: int = 0,
          dflash_p_min: float | None = None,
          prefill_chunk: int | str = "auto",
          served_names: list[str] | None = None, batch_prefill: bool = False,
          allow_image_urls: bool = False, limit_mm_per_prompt: int = 4,
          stats_interval: float = 0.0, exact_stats: bool = False,
          gpu_memory_utilization: float | None = None,
          device_map: str = "",
          serve_as: str | None = None,
          prefix_memory_ratio: float | None = None) -> "ServerState":
    from .engine.async_engine import AsyncEngine
    from .engine import Engine
    from .engine.dspark_decode import DEFAULT_DSPARK_P_MIN
    from .engine.runner import DEFAULT_GPU_UTIL
    from .engine.prefix_cache import DEFAULT_PREFIX_MEMORY_RATIO
    from .checkpoint.reader import read_bytes
    from .serve.state import Alias, ServerState
    from .checkpoint import loader
    from ._capi import SnowLLMError
    from . import ops

    if gpu_memory_utilization is None:
        gpu_memory_utilization = DEFAULT_GPU_UTIL
    if prefix_memory_ratio is None:
        prefix_memory_ratio = DEFAULT_PREFIX_MEMORY_RATIO

    root = pathlib.Path(model_path).expanduser()
    tok, eos = loader.load_tokenizer(root)

    def _single(s: str) -> int | None:
        e = tok.encode(s, add_special_tokens=False)
        return e[0] if len(e) == 1 else None

    _log(f"loading {root.name} ...")
    t0 = time.time()
    from .engine import draft_carve_out
    draft_res, draft_taps, draft_spec, draft_geo = draft_carve_out(dflash, dflash_block,
                                                                   max_num_seqs, max_model_len)
    model = loader.load(root, device_map=device_map,
                        mtp=bool(num_spec) and not dflash,
                        kv=dict(ctx=max_model_len, slots=max_num_seqs,
                                util=gpu_memory_utilization, chunk=prefill_chunk,
                                reserve=draft_res, taps=draft_taps, draft_geo=draft_geo,
                                prefix_ratio=prefix_memory_ratio, kv_int8=kv_int8,
                                num_spec=draft_spec if dflash else num_spec,
                                host_taken=_draft_bytes(dflash, draft_geo)))
    dt = time.time() - t0
    gib = read_bytes() / (1 << 30)
    _log(f"loaded in {dt:.1f}s"
         + (f", {gib:.1f} GiB off disk at {gib * (1 << 30) / dt / 1e9:.2f} GB/s" if gib else ""))

    if dflash:
        if num_spec:
            _log(f"--num-spec {num_spec} ignored: --dflash proposes a whole block "
                 f"in one forward instead of a chain of {num_spec}")
        num_spec = 0
    elif num_spec and model.mtp is None:
        _log(f"--num-spec {num_spec} ignored: {root.name} carries no MTP head "
             f"(scripts/make-mtp-gguf.py builds one for a GGUF), decoding plainly")
        num_spec = 0

    t1 = time.time()
    engine = Engine(model, num_kv_blocks=num_kv_blocks, max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len, stop_token_ids=eos, seed=seed,
                    enforce_eager=enforce_eager, num_spec=num_spec, kv_int8=kv_int8,
                    mtp_window=mtp_window, mtp_sinks=mtp_sinks,
                    dflash_path=dflash, dflash_block=dflash_block,
                    dflash_p_min=(DEFAULT_DSPARK_P_MIN if dflash_p_min is None
                                  else dflash_p_min),
                    prefill_chunk=prefill_chunk, batch_prefill=batch_prefill,
                    account=stats_interval > 0, exact_stats=exact_stats,
                    gpu_memory_utilization=gpu_memory_utilization,
                    prefix_memory_ratio=prefix_memory_ratio)
    blocks = engine.blocks.total
    _log(f"decode graphs: {engine.graph_sizes or 'off (eager)'} "
         f"(engine init + capture {time.time() - t1:.1f}s total)")

    orig_max = int(model.config.get("max_position_embeddings", 262144))
    aliases = {}
    for s in served_names or [serve_as or root.name]:
        name, _, fac = s.partition("=")
        factor = float(fac) if fac else 1.0
        if factor != 1.0 and not engine.runner.tunable_rope:
            raise SnowLLMError(
                f"--served-model-name {s}: {root.name} cannot be served at a YaRN factor. Its rope "
                f"is built at load out of one table per compression flavour, with the ramp already "
                f"folded into the frequencies, and the engine has no handle on it. Drop the "
                f"=FACTOR.")
        aliases[name] = Alias(factor, min(max_model_len, int(factor * orig_max)))
    proc = None
    if model.visual is not None:
        proc = loader.load_processor(root, tok)
        _log("vision tower loaded; /v1/chat/completions accepts image_url parts")

    sampling = loader.sampling_defaults(root)
    _log(f"sampling defaults from the checkpoint: {sampling or 'none'}")
    state = ServerState(
        engine=AsyncEngine(engine, profile_dir, stats_interval=stats_interval),
        tokenizer=tok, model=model, processor=proc,
        model_name=next(iter(aliases)), aliases=aliases,
        think_open_id=_single("<think>"), think_close_id=_single("</think>"),
        created=int(time.time()),
        default_max_tokens=default_max_tokens, allow_image_urls=allow_image_urls,
        limit_mm_per_prompt=limit_mm_per_prompt, sampling=sampling)

    page = engine.runner.block_size
    tokens = blocks * page
    where = (f"{blocks} KV blocks, a {engine.ring_blocks * page}-token window each"
             if engine.ring_blocks else
             f"{blocks} KV blocks = {tokens} tokens ({tokens // max_num_seqs} per slot at full "
             f"concurrency)")
    act = getattr(engine.runner, "act_bytes", 0)
    _log(f"{max_num_seqs} slots, {where}, "
         f"{engine.runner.kv_bytes / (1 << 30):.1f} GiB of KV in all, "
         f"prefill chunk {engine.prefill_chunk}"
         f"{f' ({act / (1 << 30):.1f} GiB of activations, allocated up front)' if act else ''}, "
         f"ctx {max_model_len}, stop {eos}")
    if engine.dflash is not None:
        pool = engine.dflash.blocks
        g = engine.dflash_geo
        stages = getattr(g, "stack", g).num_layers
        p_min = getattr(engine.dflash, "p_min", 0.0)
        cut = f", cut at confidence {p_min}" if p_min else ", whole block every step"
        _log(f"dflash: block {engine.dflash.block}{cut}, {pool.total} draft KV blocks "
             f"({engine._dflash_pool_bytes(pool.total) / (1 << 30):.1f} GiB in {stages} pools of "
             f"its own) -- held back from the context above, not taken on top of it")
    if engine.cache is None:
        _log("prefix cache off")
    else:
        _log(f"prefix cache: {engine.cache.store.describe()}")
    _log(f"served: {', '.join(f'{n} (factor {a.factor}, ctx {a.ctx})' for n, a in aliases.items())}")
    if profile_dir:
        _log(f"profiling armed -> {profile_dir} (POST /start_profile, /stop_profile)")
    return state


def parser() -> argparse.ArgumentParser:
    from .engine import SPEC_MAX_STEP_ROWS
    from .engine.dspark_decode import DEFAULT_DSPARK_P_MIN
    from .engine.runner import DEFAULT_GPU_UTIL, DEFAULT_MAX_NUM_SEQS, MAX_NUM_SEQS
    from .engine.prefix_cache import CKPT_EVERY, DEFAULT_PREFIX_MEMORY_RATIO
    from . import ops

    p = argparse.ArgumentParser(
        prog="snowllm",
        usage="snowllm PATH|RECIPE [options]\n       snowllm pull [RECIPE]\n       "
              "snowllm recipes",
        description="An OpenAI-compatible server for Qwen3.6 on one gfx1151.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("serving")
    g.add_argument("model", nargs="?", metavar="PATH", default=argparse.SUPPRESS,
                   help="the checkpoint directory, or the name of a recipe `snowllm pull` has "
                        "already fetched (`snowllm recipes` lists them)")
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
                        f"request pins (num-spec + 1) slots of linear-attn state for its whole "
                        f"life; under --dflash it is one slot flat whatever the block, the state "
                        f"being rolled forward rather than snapshotted per row. That pool is the "
                        f"one thing here provisioned worst-case, so this is the knob that decides "
                        f"whether the engine starts.")
    g.add_argument("--max-model-len", type=_size, default=262144, metavar="N",
                   help="context length. A per-request ceiling, not a reservation: the KV pool is "
                        "shared, and a request that outgrows what is left is preempted and "
                        "re-prefilled rather than pre-allocated for.")
    g.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_UTIL, metavar="F",
                   help="fraction of the device the whole process may occupy. The KV pool is "
                        "whatever is left under it once the weights, the state pool, the prefill "
                        "workspace and the logits are down -- measured, not modelled. Raise it "
                        "toward 1.0 if the GPU is yours alone; what it holds back is fragmentation "
                        "headroom for the caching allocator.")
    g.add_argument("--device-map", type=str, default="", metavar="SPEC",
                   help="which weight groups live in pinned HOST memory instead of the carve-out, "
                        "trading a slower decode step for KV pool. On a unified-memory part both "
                        "are the same DRAM; the cost is measured, and it goes with the BYTES "
                        "moved, not with which layers move. `auto` frees what the KV pools AND "
                        "the prefill chunk are short of at your --max-model-len, a layer at a "
                        "time, cheapest group first, buying the widest chunk this host can pay "
                        "for and falling back to a narrower one when it cannot; `all` fills the "
                        "free system RAM, experts included; `off` is the default; or name groups, "
                        "optionally with a layer count: dense,head or dense:12. "
                        "GGUF DeepSeek-V4 only today.")
    g.add_argument("--prefix-memory-ratio", type=float, default=DEFAULT_PREFIX_MEMORY_RATIO,
                   metavar="R",
                   help=f"the share of the memory budget that goes to remembering prefixes, 0 to "
                        f"disable. What it decides is HOW MANY are remembered: a prefix's KV "
                        f"pages stay in the pool and are shared rather than copied, and what has "
                        f"a fixed cost is the state at its end -- the linear-attn state on a "
                        f"hybrid model, the compressor carry on DeepSeek-V4 -- one of which is the "
                        f"same size whatever the prefix length. Whether that share is really "
                        f"taken OFF the KV pool depends on what is spare when it is asked for. A "
                        f"SHARE rather than a byte figure because what a remembered prefix is "
                        f"worth is relative to what the pools get, which moves with the device, "
                        f"the checkpoint and --gpu-memory-utilization. The startup line prints "
                        f"the count it works out to, and how often a checkpoint is taken -- every "
                        f"{CKPT_EVERY} tokens on a linear-attention checkpoint, one prefill chunk "
                        f"on DeepSeek-V4, where a mark has to land where every compression ratio "
                        f"has filled whole blocks. A hit lands at the last mark at or before the "
                        f"shared prefix, so a prefix shorter than one spacing is never cached.")
    g.add_argument("--num-kv-blocks", type=_size, default=None, metavar="N",
                   help=f"pin the KV pool at N blocks of {ops.KV_BLOCK_SIZES[0]} tokens instead of "
                        f"sizing it from --gpu-memory-utilization. It must still hold one "
                        f"max-model-len sequence; below that the engine refuses to start.")

    g = p.add_argument_group("throughput", "Every one of these is measured; what it measured at "
                                           "depends on the checkpoint and the device, so try it.")
    g.add_argument("--num-spec", "--num-speculative-tokens", type=int, default=2,
                   metavar="K", dest="num_spec",
                   help=f"MTP draft tokens per step (0 = off). Applied only while the batch is "
                        f"<= {SPEC_MAX_STEP_ROWS} // (num-spec + 1); bigger batches draft shallower. "
                        f"Ignored under --dflash, which proposes differently")
    g.add_argument("--dflash", metavar="PATH", default=None,
                   help="speculate with a block-diffusion draft instead of the MTP head, given "
                        "the draft's directory. A Qwen DFlash draft is a directory of safetensors "
                        "whose config.json says how wide it is; DeepSeek-V4's DSpark draft is the "
                        "dspark-*.gguf that sits beside the model, so pointing this at the model "
                        "directory finds it. Either way it is a whole second model with KV pools "
                        "of its own, conditioned on the target's hidden states at the layers it "
                        "names, and it proposes a whole block in ONE forward rather than a token "
                        "at a time -- DSpark a token for every row of its block, DFlash one fewer. "
                        "Output is unchanged either way: every proposed token is verified. The "
                        "draft's KV pool is held back from the one above rather than taken on top "
                        "of it, and every target forward carries the tapped hidden states per row. "
                        "On Qwen it does NOT cost linear-attn state per block row: that is rolled "
                        "forward, so a request pins one slot rather than --dflash-block of them.")
    g.add_argument("--dflash-block", type=int, default=0, metavar="N",
                   help=f"tokens per draft block. The step is up to N rows wide -- flat at N on a "
                        f"DFlash draft, and as wide as --dflash-p-min leaves it on a DSpark one -- "
                        f"so this trades work per step against tokens per step, and a batch too "
                        f"step. 0 takes the draft's own width: a fixed one on a DFlash draft, "
                        f"and on a DSpark draft the checkpoint's block_size, the width its Markov "
                        f"head was trained at -- that one is a FLOOR, and wider is legal")
    g.add_argument("--dflash-p-min", type=float, default=DEFAULT_DSPARK_P_MIN, metavar="P",
                   help="DSpark ONLY, and ignored by a draft with no confidence head: stop the "
                        "draft block at the first position where the drafter's running "
                        "confidence -- the chance the whole prefix up to it is accepted -- falls "
                        "below P, so a step is only as wide as the drafter can justify. OFF by "
                        "default because it measured that way here: a verify row costs a fraction "
                        "of a base step, so a doubtful row is usually still worth verifying. On a "
                        "machine where those two are closer, this is the knob")
    g.add_argument("--kv-cache-dtype", choices=("bf16", "auto", "int8"), default="bf16",
                   help="bf16 (or vLLM's `auto`, the same thing here), or int8. int8 halves KV "
                        "bytes and buys long-context decode at some prefill -- for "
                        "decode/long-context-heavy workloads. Its layout is per (token, kv-head, "
                        "d-group) for K and per (token, kv-head) for V, so it is not vLLM's "
                        "int8_per_token_head. It is worth most on a checkpoint whose KV is wide: "
                        "many full-attention layers, or many kv heads.")
    g.add_argument("--max-num-batched-tokens", default="auto", metavar="N",
                   type=lambda x: x if x == "auto" else _size(x),
                   help="tokens per launch, or 'auto' to take the widest candidate whose "
                        "activations -- WALKED, not estimated -- still leave the pools their room. "
                        "A share of the allowance is what this used to be, and a share is not a "
                        "measurement. A step is one prefill chunk or one decode, so this is the "
                        "prefill chunk: it bounds every M-sized buffer, and changes no "
                        "arithmetic.")
    g.add_argument("--batch-prefill", action="store_true",
                   help="pack several fresh short prompts into one prefill launch (--num-spec 0 "
                        "only), which pays at high concurrency. NOT batch-invariant: a prompt's "
                        "output can depend on who it batched with (bf16 tiling).")
    g.add_argument("--mtp-window", type=int, default=0, metavar="W",
                   help="cap the MTP draft layer's attention at --mtp-sinks leading tokens plus the "
                        "most recent W (0 = off, full context). Cannot change output, only "
                        "acceptance.")
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
    g.add_argument("--stats-interval", type=float, default=10.0, metavar="SECONDS",
                   help="log throughput, acceptance and pool occupancy every SECONDS, and carry "
                        "the cumulative counters on /health. 0 turns it off. The rates are token "
                        "counts over the interval's wall clock, which costs nothing to collect -- "
                        "so both read lower than their exclusive speed while the other phase is "
                        "running, which is what a server produces rather than what a kernel can.")
    g.add_argument("--stats-exact", action="store_true",
                   help="report prefill and decode at their EXCLUSIVE rates, by charging each step "
                        "its own GPU time. Costs a torch.cuda.synchronize() PER STEP: without it "
                        "the CPU returns first and a prefill chunk that emits no token never "
                        "syncs, so its time lands on a later decode step. For kernel work, not "
                        "for serving.")

    return p


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        if argv[0] == "pi":
            from .hub import pi
            raise SystemExit(pi.main(argv[0], argv[1:]))
        from .hub import recipes
        raise SystemExit(recipes.main(argv[0], argv[1:]))

    import uvicorn

    from .serve.api import app
    from .serve.state import install

    p = parser()
    a = p.parse_args()
    positional = getattr(a, "model", None)
    if positional and a.model_flag:
        p.error("the checkpoint directory was given twice")
    model = positional or a.model_flag
    if not model:
        p.error("a checkpoint directory or a recipe is required: `snowllm PATH`, or "
                "`snowllm recipes` to see what there is")

    from .hub import recipes
    from ._capi import SnowLLMError
    try:
        found, recipe = recipes.locate(model)
        model = str(found)
    except SnowLLMError as e:
        p.exit(2, f"{term.paint('snowllm:', term.BOLD, term.RED, stream=sys.stderr)} {e}\n")
    except KeyboardInterrupt:
        p.exit(130, "\n")

    if recipe and recipe.defaults:
        def _at(v: object) -> object:
            return model + v[len("@model"):] if isinstance(v, str) and v.startswith("@model") \
                else v

        unset = object()
        given = p.parse_args(namespace=argparse.Namespace(**dict.fromkeys(recipes.SETTABLE, unset)))
        used = {k: _at(v) for k, v in recipe.defaults.items()
                if k in recipes.SETTABLE and getattr(given, k) is unset}
        for k, v in used.items():
            setattr(a, k, v)
        if used:
            print(f"{term.stamp()} {recipe.id} sets " +
                  ", ".join(f"--{k.replace('_', '-')} {v}" for k, v in sorted(used.items())),
                  flush=True)

    install(build(model, a.max_num_seqs, a.max_model_len, num_kv_blocks=a.num_kv_blocks,
          default_max_tokens=a.default_max_tokens, seed=a.seed, profile_dir=a.profile_dir,
          stats_interval=a.stats_interval, exact_stats=a.stats_exact,
          enforce_eager=a.enforce_eager, num_spec=a.num_spec,
          mtp_window=a.mtp_window, mtp_sinks=a.mtp_sinks, served_names=a.served_model_name,
          dflash=a.dflash, dflash_block=a.dflash_block,
          dflash_p_min=a.dflash_p_min,
          batch_prefill=a.batch_prefill, allow_image_urls=a.allow_image_urls,
          prefill_chunk=a.max_num_batched_tokens, kv_int8=a.kv_cache_dtype == "int8",
          gpu_memory_utilization=a.gpu_memory_utilization,
          device_map=a.device_map,
          serve_as=(recipe.serve_as if recipe else None),
          prefix_memory_ratio=a.prefix_memory_ratio,
          limit_mm_per_prompt=a.limit_mm_per_prompt))
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
