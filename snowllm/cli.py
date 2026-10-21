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
from ._version import __version__
from .telemetry import enabled as telemetry_enabled

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
    from .defaults import DEFAULT_GPU_UTIL
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
    from .engine import plan_draft, draft_num_spec, _dspark
    draft = plan_draft(dflash)
    draft_kv = dict(draft_resident=draft.device_bytes) if _dspark(draft.geo) else {}
    model = loader.load(root, device_map=device_map,
                        mtp=bool(num_spec) and not dflash,
                        kv=dict(ctx=max_model_len, slots=max_num_seqs,
                                util=gpu_memory_utilization, chunk=prefill_chunk,
                                reserve=draft.load_peak_bytes,
                                taps=draft.geo.tap_layers if dflash else (), draft_geo=draft.geo,
                                prefix_ratio=prefix_memory_ratio, kv_int8=kv_int8,
                                num_spec=draft_num_spec(draft.geo, dflash_block) if dflash else num_spec,
                                host_taken=_draft_bytes(dflash, draft.geo), **draft_kv))
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
                    account=stats_interval > 0 or telemetry_enabled(), exact_stats=exact_stats,
                    gpu_memory_utilization=gpu_memory_utilization,
                    prefix_memory_ratio=prefix_memory_ratio, draft_plan=draft)
    engine.runtime_config = dict(model_path=str(root.resolve()), max_model_len=engine.max_model_len,
                                 max_num_seqs=engine.max_num_seqs, kv_cache_dtype="int8" if engine.runner.kv_int8 else "bf16",
                                 gpu_memory_utilization=gpu_memory_utilization,
                                 num_spec=0 if engine.dflash else engine.num_spec,
                                 dflash_path=str(pathlib.Path(dflash).resolve()) if dflash else None,
                                 dflash_block=engine.dflash_block if engine.dflash else None,
                                 enforce_eager=enforce_eager, prefill_chunk=engine.prefill_chunk)
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


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                     argparse.RawDescriptionHelpFormatter):
    pass


def parser() -> argparse.ArgumentParser:
    from .defaults import (DEFAULT_DSPARK_P_MIN, DEFAULT_GPU_UTIL,
                           DEFAULT_MAX_NUM_SEQS,
                           DEFAULT_PREFIX_MEMORY_RATIO)

    p = argparse.ArgumentParser(
        prog="snowllm",
        usage="snowllm webui [options]\n       snowllm PATH|RECIPE [options]\n       "
              "snowllm COMMAND [options]",
        description="Run models locally. Start with: snowllm webui\n\n"
                    "Commands:\n"
                    "  webui             Open model management and chat in your browser\n"
                    "  pull [RECIPE]     Download a model\n"
                    "  recipes           List available models\n"
                    "  pi                Start the coding agent\n"
                    "  load MODEL        Load a model in the running service\n"
                    "  unload            Unload the current model\n"
                    "  status            Show service status",
        epilog="Use snowllm COMMAND --help for that command's options.",
        formatter_class=_HelpFormatter)
    p.add_argument("--version", action="version", version=f"snowllm {__version__}",
                   help="Show the installed version and exit.")

    g = p.add_argument_group("serving")
    g.add_argument("model", nargs="?", metavar="PATH", default=argparse.SUPPRESS,
                   help='Checkpoint directory or downloaded recipe.')
    g.add_argument("--model", dest="model_flag", metavar="PATH", help=argparse.SUPPRESS)
    g.add_argument("--host", default="127.0.0.1", help="API listener address.")
    g.add_argument("--port", type=int, default=8000, help="API listener port.")
    g.add_argument("--served-model-name", action="append", metavar="NAME=FACTOR",
                   help='API model name and context factor; repeatable.')
    g.add_argument("--default-max-tokens", type=int, default=None, metavar="N",
                   help='Default reply limit; unset uses remaining context.')
    g.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")

    g = p.add_argument_group('capacity')
    g.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS, metavar="N",
                   help='Maximum concurrent requests (up to 256).')
    g.add_argument("--max-model-len", type=_size, default=262144, metavar="N",
                   help='Maximum context length, e.g. 32k.')
    g.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_UTIL, metavar="F",
                   help='Fraction of GPU memory available to the server.')
    g.add_argument("--device-map", type=str, default="", metavar="SPEC",
                   help='Offload GGUF weights: off, auto, all, or groups such as experts.down:1.')
    g.add_argument("--prefix-memory-ratio", type=float, default=DEFAULT_PREFIX_MEMORY_RATIO,
                   metavar="R",
                   help='Memory share for prompt caching; 0 disables caching.')
    g.add_argument("--num-kv-blocks", type=_size, default=None, metavar="N",
                   help='Set the KV cache block count manually.')

    g = p.add_argument_group('throughput')
    g.add_argument("--num-spec", "--num-speculative-tokens", type=int, default=2,
                   metavar="K", dest="num_spec",
                   help='MTP draft tokens per step; 0 disables MTP.')
    g.add_argument("--dflash", metavar="PATH", default=None,
                   help='Draft checkpoint for DFlash or DSpark; replaces MTP.')
    g.add_argument("--dflash-block", type=int, default=0, metavar="N",
                   help='Draft block size; 0 uses the checkpoint default.')
    g.add_argument("--dflash-p-min", type=float, default=DEFAULT_DSPARK_P_MIN, metavar="P",
                   help='DSpark confidence threshold; 0 disables early stopping.')
    g.add_argument("--kv-cache-dtype", choices=("bf16", "auto", "int8"), default="bf16",
                   help='KV cache precision; auto uses bf16.')
    g.add_argument("--max-num-batched-tokens", default="auto", metavar="N",
                   type=lambda x: x if x == "auto" else _size(x),
                   help='Prefill chunk size, e.g. 8k; auto selects a size.')
    g.add_argument("--batch-prefill", action="store_true",
                   help='Batch fresh prompts; requires --num-spec 0, may change output.')
    g.add_argument("--mtp-window", type=int, default=0, metavar="W",
                   help='Recent tokens visible to MTP; 0 uses full context.')
    g.add_argument("--mtp-sinks", type=int, default=64, metavar="N",
                   help='Leading tokens retained with --mtp-window.')

    g = p.add_argument_group("multimodal")
    g.add_argument("--limit-mm-per-prompt", type=int, default=4, metavar="N",
                   help='Maximum images per request.')
    g.add_argument("--allow-image-urls", action="store_true",
                   help='Allow fetching HTTP(S) images; data URLs always work.')

    g = p.add_argument_group("diagnostics")
    g.add_argument("--enforce-eager", action="store_true", help='Disable GPU graph execution.')
    g.add_argument("--profile-dir", default=os.environ.get("SNOWLLM_TORCH_PROFILER_DIR"),
                   metavar="DIR",
                   help='Directory for traces captured via profiling endpoints.')
    g.add_argument("--stats-interval", type=float, default=10.0, metavar="SECONDS",
                   help='Seconds between throughput logs; 0 disables logs.')
    g.add_argument("--stats-exact", action="store_true",
                   help='Synchronize each step for GPU timing; slows inference.')

    return p


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("webui", "load", "unload", "status"):
        from .daemon.cli import main as daemon_main
        raise SystemExit(daemon_main(argv))
    if argv and argv[0] in SUBCOMMANDS:
        if argv[0] == "pi":
            from .hub import pi
            raise SystemExit(pi.main(argv[0], argv[1:]))
        from .hub import recipes
        raise SystemExit(recipes.main(argv[0], argv[1:]))

    p = parser()
    if not argv:
        p.print_help()
        return
    a = p.parse_args()
    positional = getattr(a, "model", None)
    if positional and a.model_flag:
        p.error("the checkpoint directory was given twice")
    model = positional or a.model_flag
    if not model:
        p.error("use `snowllm webui` to open the browser interface, or "
                "`snowllm PATH|RECIPE` to serve a model directly")

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

    import uvicorn
    from .serve.api import app
    from .serve.state import install

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
