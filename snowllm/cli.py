# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time

from . import term
from ._version import __version__
from ._platform import release_device_cache
from .defaults import DEFAULT_GPU_UTIL, DEFAULT_PREFILL_CHUNK
from .telemetry import enabled as telemetry_enabled

SUBCOMMANDS = ("pull", "recipes", "pi")


def _size(x: str) -> int:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmM]?)\s*", x)
    if not m:
        raise argparse.ArgumentTypeError(f"{x!r} is not a size (try 4096, 32k, 1M)")
    return int(float(m.group(1)) * {"": 1, "k": 1024, "K": 1024, "m": 2**20, "M": 2**20}[m.group(2)])


def _fraction(x: str) -> float:
    from .capacity import fraction
    from .errors import SnowLLMError
    try:
        return fraction(float(x), 'memory utilization')
    except (ValueError, SnowLLMError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _offload_size(x: str, option='--host-offload') -> int:
    from .capacity import memory_bytes
    from .errors import SnowLLMError
    try:
        return memory_bytes(x, option=option, allow_zero=True)
    except SnowLLMError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _memory_limit(x: str, *, capacity_only=False):
    from .capacity import memory_bytes
    from .errors import SnowLLMError
    try:
        float(x)
    except ValueError:
        try:
            return None, memory_bytes(x, allow_zero=capacity_only) / (1 << 30)
        except SnowLLMError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    if capacity_only:
        if float(x) == 0:
            return None, 0.
        raise argparse.ArgumentTypeError('specify a capacity with units, for example 20GB or 20GiB')
    return _fraction(x), None


class _MemoryLimit(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        util, gib = values
        setattr(namespace, self.dest + '_util', util)
        setattr(namespace, self.dest + '_gib', gib)


def _log(msg: str) -> None:
    print(f"{term.stamp(time.strftime('%H:%M:%S'))} {msg}", flush=True)


def _draft_bytes(dflash: str | None, geo: object) -> int:
    from .engine import _dspark
    if not dflash or not _dspark(geo):
        return 0
    if geo.stack.v41:
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
          dspark_adaptive: bool = False,
          prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
          served_names: list[str] | None = None, batch_prefill: bool = False,
          allow_image_urls: bool = False, limit_mm_per_prompt: int = 4,
          stats_interval: float = 0.0, exact_stats: bool = False,
          max_gpu_mem_util: float | None = None, max_gpu_mem_gib: float | None = None,
          max_host_mem_util: float | None = None, max_host_mem_gib: float | None = None,
          max_disk_mem_gib: float | None = None,
          device_map: str = "",
          moe_cache_experts: int = 0, moe_cache_mode: str = 'all',
          host_offload: int | str | None = None,
          serve_as: str | None = None,
          prefix_memory_ratio: float | None = None,
          offload_ratio: float | None = None, host_alloc: bool = False) -> "ServerState":
    from .capacity import MemoryBudget, configure, placement_options
    limits = MemoryBudget(max_gpu_mem_util, max_host_mem_util, max_host_mem_gib,
                          max_gpu_mem_gib=max_gpu_mem_gib, max_disk_mem_gib=max_disk_mem_gib)
    from ._platform import host_total
    limits.host_limit(host_total())
    if max_model_len <= 0 or prefill_chunk <= 0:
        from .errors import SnowLLMError
        raise SnowLLMError('context and prefill chunk must be positive')
    host_offload, _ = placement_options(host_alloc=host_alloc, device_map=device_map,
        host_offload=host_offload, offload_ratio=offload_ratio, max_disk_bytes=limits.disk_limit(),
        moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode)
    memory_auto = (not host_alloc and device_map in ('', 'auto') and host_offload is None
                   and offload_ratio is None and not moe_cache_experts)
    from . import host_alloc as allocation
    configure(limits)
    if host_alloc:
        allocation.enable()
    from .engine.async_engine import AsyncEngine
    from .engine import Engine
    from .engine.dspark_decode import DEFAULT_DSPARK_P_MIN
    from .engine.prefix_cache import DEFAULT_PREFIX_MEMORY_RATIO
    from .checkpoint.reader import read_bytes
    from .serve.state import Alias, ServerState
    from .checkpoint import loader
    from ._capi import SnowLLMError
    from . import ops
    from ._platform import device_mem_info
    _, physical = device_mem_info()
    gpu_memory_utilization = 1. if host_alloc else limits.gpu_utilization(physical)
    host_limit = limits.host_limit(host_total())
    host_ceiling = 'unset' if host_limit is None else f'{host_limit / (1 << 30):.3f} GiB'
    _log(f'memory ceilings: GPU {limits.gpu_limit(physical) / (1 << 30):.3f} GiB; host {host_ceiling}')
    if host_alloc:
        _log('host allocation: hipHostMalloc-backed tensors; weights, KV and workspace use system RAM')
    if prefix_memory_ratio is None:
        prefix_memory_ratio = DEFAULT_PREFIX_MEMORY_RATIO
    if not 0 <= prefix_memory_ratio < 1:
        raise SnowLLMError('--prefix-memory-ratio must be in [0, 1)')

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
    kv = dict(ctx=max_model_len, slots=max_num_seqs,
              util=gpu_memory_utilization,
              chunk=prefill_chunk, reserve=draft.load_peak_bytes,
              taps=draft.geo.tap_layers if dflash else (), draft_geo=draft.geo,
              draft_workspace=draft.workspace,
              prefix_ratio=prefix_memory_ratio, kv_int8=kv_int8,
              num_spec=draft_num_spec(draft.geo, dflash_block) if dflash else num_spec,
              host_taken=_draft_bytes(dflash, draft.geo), blocks=num_kv_blocks,
              dflash_block=dflash_block,
              dflash_p_min=DEFAULT_DSPARK_P_MIN if dflash_p_min is None else dflash_p_min,
              dspark_adaptive=dspark_adaptive, mtp_window=mtp_window, mtp_sinks=mtp_sinks,
              offload_ratio=offload_ratio,
              memory_auto=memory_auto,
              host_offload_allowed=limits.host_offload_enabled,
              max_disk_bytes=limits.disk_limit(),
              **draft_kv)
    model = loader.load(root, device_map=device_map, moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode,
                        host_offload=host_offload,
                        mtp=bool(num_spec) and not dflash,
                        kv=kv)
    dt = time.time() - t0
    gib = read_bytes() / (1 << 30)
    _log(f"loaded in {dt:.1f}s"
         + (f", {gib:.1f} GiB off disk at {gib * (1 << 30) / dt / 1e9:.2f} GB/s" if gib else ""))
    store = getattr(model, 'moe_offload_plan', None)
    if store is not None:
        mode = store.mode if store.active else 'host'
        selected = f'auto -> {mode}' if moe_cache_mode == 'auto' else mode
        _log(f'MoE offload ({selected}): {store.capacity if store.active else 0} routed experts on each of '
             f'{len(store.specs) if store.active else 0} disk-cached layers; {len(store.host_layers)} host layers; '
             f'{len(store.resident_layers)} hash layers and shared experts stay on GPU; '
             + ('target forwards run eager for cache transfers' if store.active else 'no disk cache transfers'))
        _log(f'MoE offload: net {store.saved_bytes} bytes ({store.saved_bytes / (1 << 30):.3f} GiB) '
             f'saved, host {store.host_bytes / (1 << 30):.3f} GiB + disk net '
             f'{store.disk_saved_bytes / (1 << 30):.3f} GiB' +
             (f'; disk target {store.offload_target} bytes' if store.offload_target is not None else ''))
        disk_cache = store.disk_cache if store.active else None
        if disk_cache is not None:
            _log('MoE prepacked cache: ' + ', '.join(map(str, disk_cache.directories)) +
                 '; records are built on first use and striped across directories')

    if dflash:
        if num_spec:
            _log(f"--num-spec {num_spec} ignored: --dflash proposes a whole block "
                 f"in one forward instead of a chain of {num_spec}")
        num_spec = 0
    elif num_spec and model.mtp is None:
        _log(f"--num-spec {num_spec} ignored: {root.name} carries no MTP head "
             f"(scripts/make-mtp-gguf.py builds one for a GGUF), decoding plainly")
        num_spec = 0
        kv['num_spec'] = 0

    t1 = time.time()
    engine = Engine(model, num_kv_blocks=num_kv_blocks, max_num_seqs=max_num_seqs,
            max_model_len=max_model_len, stop_token_ids=eos, seed=seed,
            enforce_eager=enforce_eager, num_spec=num_spec, kv_int8=kv_int8,
            mtp_window=mtp_window, mtp_sinks=mtp_sinks,
            dflash_path=dflash, dflash_block=dflash_block,
            dflash_p_min=(DEFAULT_DSPARK_P_MIN if dflash_p_min is None
                          else dflash_p_min),
            dspark_adaptive=dspark_adaptive,
            prefill_chunk=prefill_chunk, batch_prefill=batch_prefill,
            account=stats_interval > 0 or telemetry_enabled(), exact_stats=exact_stats,
            gpu_memory_utilization=kv['util'],
            prefix_memory_ratio=prefix_memory_ratio, draft_plan=draft)
    from ._platform import device_mem_info
    release_device_cache()
    free, total = device_mem_info()
    if dspark_adaptive and getattr(model.geo, "v41", False):
        # V4.1 has no captured target graphs, so post-profile weight moves are safe.
        from .models.placement import ensure_room
        while total-free > limits.gpu_limit(total) and ensure_room(
                model, total-free-limits.gpu_limit(total)):
            release_device_cache()
            free, total = device_mem_info()
    if total-free > limits.gpu_limit(total):
        engine.runner._discard_pools()
        from .capacity import memory_shortage
        raise memory_shortage(total-free-limits.gpu_limit(total))
    if host_alloc:
        _log(f'Host allocation: {allocation.allocated_bytes() / (1 << 30):.3f} GiB in tensor pools; '
             f'physical VRAM in use {(total-free) / (1 << 20):.1f} MiB (driver/runtime included)')
    dmap = getattr(model, 'device_map', None)
    actual_offload = dmap.host_bytes / dmap.total_bytes if dmap and dmap.total_bytes else 0.
    if host_offload is not None:
        _log(f'Host offload: {dmap.host_bytes} bytes ({dmap.host_bytes / (1 << 30):.3f} GiB); '
             f'host target {host_offload} bytes')
    _log(f'capacity: context {max_model_len} tokens (input + output), '
         f'offload {actual_offload:.4%}, '
         f'{max_num_seqs} concurrent requests; chunk {engine.prefill_chunk}')
    engine.runtime_config = dict(model_path=str(root.resolve()), moe_cache_experts=moe_cache_experts, moe_cache_mode=moe_cache_mode,
                                 max_disk_mem_gib=limits.max_disk_mem_gib,
                                 disk_offloaded_weight_bytes=getattr(store, "disk_weight_bytes", 0),
                                 host_offload=host_offload,
                                 host_alloc=host_alloc,
                                 max_model_len=engine.max_model_len,
                                 max_num_seqs=engine.max_num_seqs, kv_cache_dtype="int8" if engine.runner.kv_int8 else "bf16",
                                 max_gpu_mem_util=((limits.max_gpu_mem_util or DEFAULT_GPU_UTIL) if limits.max_gpu_mem_gib is None else None),
                                 max_gpu_mem_gib=limits.max_gpu_mem_gib,
                                 max_gpu_mem_bytes=limits.gpu_limit(total),
                                 max_host_mem_util=limits.max_host_mem_util,
                                 automatic_host_weight_offload=memory_auto and limits.host_offload_enabled,
                                 max_host_mem_gib=limits.max_host_mem_gib,
                                 max_host_mem_bytes=limits.host_limit(host_total()),
                                 offload_ratio=actual_offload,
                                 num_spec=0 if engine.dflash else engine.num_spec,
                                 dflash_path=str(pathlib.Path(dflash).resolve()) if dflash else None,
                                 dflash_block=engine.dflash_block if engine.dflash else None,
                                 dspark_adaptive=dspark_adaptive,
                                 enforce_eager=enforce_eager, prefill_chunk=engine.prefill_chunk)
    blocks = engine.blocks.total
    _log(f"decode graphs: {engine.graph_sizes or 'off (eager)'} "
         f"(engine init + capture {time.time() - t1:.1f}s total)")

    orig_max = int(model.config.get("max_position_embeddings", 262144))
    aliases = {}
    for s in served_names or [serve_as or root.name]:
        name, _, fac = s.partition("=")
        factor = float(fac) if fac else engine.default_rope_factor
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
    from .defaults import (DEFAULT_DSPARK_P_MIN, DEFAULT_GPU_UTIL, DEFAULT_CONTEXT_TARGET,
                           DEFAULT_MAX_NUM_SEQS, DEFAULT_PREFILL_CHUNK,
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
    g.add_argument("--max-model-len", type=_size, default=DEFAULT_CONTEXT_TARGET, metavar="N",
                   help='Maximum context input + output tokens; never increased to fill memory.')
    p.set_defaults(max_gpu_mem_util=None, max_gpu_mem_gib=None,
                   max_host_mem_util=None, max_host_mem_gib=None)
    budget = g.add_mutually_exclusive_group()
    budget.add_argument('--max-gpu-mem', type=_memory_limit, action=_MemoryLimit,
                        default=argparse.SUPPRESS, metavar='LIMIT',
                        help='GPU memory ceiling: fraction (0.9) or capacity (20GB, 20GiB); default 0.9.')
    budget.add_argument('--gpu-memory-utilization', dest='max_gpu_mem_util', type=_fraction,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    g.add_argument('--max-host-mem', type=_memory_limit, action=_MemoryLimit,
                        default=argparse.SUPPRESS, metavar='LIMIT',
                        help='Host buffer ceiling: fraction (0.5) or capacity (20GB, 20GiB); enables automatic host weight offload.')
    g.add_argument("--device-map", type=str, default="", metavar="SPEC",
                   help='Offload weights: off, auto, all, or groups such as experts.down:1.')
    g.add_argument('--host-alloc', action=argparse.BooleanOptionalAction, default=False,
                   help='Allocate GPU tensors in mapped host RAM using hipHostMalloc; excludes all offload modes.')
    host_size = g.add_mutually_exclusive_group()
    host_size.add_argument('--host-offload', type=lambda x: _offload_size(x, '--host-offload'),
                   default=None, metavar='SIZE',
                   help='Move at least SIZE of model weights to host memory, e.g. 8GiB; independent of disk offload.')
    host_size.add_argument('--offload-ratio', type=float, default=None, metavar='F',
                   help='Fraction of total weight bytes to offload; enable with --device-map auto.')
    moe_size = g.add_mutually_exclusive_group()
    moe_size.add_argument('--moe-cache-experts', type=int, default=0, metavar='N',
                   help='DeepSeek V4 Flash: cache N routed experts per non-hash layer with LRU; '
                        'shared experts stay on GPU, misses read from GGUF; 0 disables.')
    moe_size.add_argument('--max-disk-offload', dest='max_disk_mem_gib',
                   type=lambda x: _memory_limit(x, capacity_only=True)[1],
                   default=None, metavar='SIZE',
                   help='Maximum expert weights offloaded to disk when GPU/host budgets cannot fit; no cache file quota.')
    g.add_argument('--moe-cache-mode', choices=('all', 'down', 'auto'), default='all',
                   help='Cache both projections, only down, or auto-select within the all-mode GPU cache budget.')
    g.add_argument("--prefix-memory-ratio", type=float, default=DEFAULT_PREFIX_MEMORY_RATIO,
                   metavar="R",
                   help='Prefix snapshot memory budget as a fraction of base cache memory.')
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
    g.add_argument("--dspark-adaptive", action=argparse.BooleanOptionalAction, default=False,
                   help='Profile DSpark step costs at startup and schedule confidence prefixes.')
    g.add_argument("--kv-cache-dtype", choices=("bf16", "auto", "int8"), default="bf16",
                   help='KV cache precision; auto uses bf16.')
    g.add_argument("--max-num-batched-tokens", default=DEFAULT_PREFILL_CHUNK, metavar="N",
                   type=_size, help='Prefill chunk size, e.g. 8k; fixed independently of memory budgets.')
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


def _equivalent_command(p, a, model, recipe):
    from .capacity import gibibytes

    def capacity_arg(gib):
        nbytes = gibibytes(gib) if gib else 0
        return f'{nbytes / (1 << 30)}GiB' if nbytes % (1 << 20) == 0 else f'{nbytes}B'

    values = vars(a).copy()
    for pool in ('gpu', 'host'):
        gib = values[f'max_{pool}_mem_gib']
        values[f'max_{pool}_mem'] = (capacity_arg(gib) if gib is not None
                                     else values[f'max_{pool}_mem_util'])
    if values['max_gpu_mem'] is None:
        values['max_gpu_mem'] = DEFAULT_GPU_UTIL
    if values['max_disk_mem_gib'] is not None:
        values['max_disk_mem_gib'] = capacity_arg(values['max_disk_mem_gib'])
    if not values['served_model_name'] and recipe.serve_as:
        values['served_model_name'] = [recipe.serve_as]
    important = set(recipe.defaults) | {'max_model_len', 'max_num_seqs',
        'max_num_batched_tokens', 'num_spec', 'kv_cache_dtype', 'prefix_memory_ratio'}
    command = ['snowllm', str(pathlib.Path(model).resolve())]
    for action in p._actions:
        if not action.option_strings or action.dest in (
                'model_flag', 'max_gpu_mem_util', 'max_gpu_mem_gib', 'max_host_mem_util', 'max_host_mem_gib'):
            continue
        value = values.get(action.dest)
        if value is None or (value == action.default and action.dest not in important):
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            command.append(flag if value else '--no-' + flag[2:])
        elif action.nargs == 0:
            if value:
                command.append(flag)
        else:
            for item in value if isinstance(value, list) else [value]:
                command.extend((flag, str(item)))
    return subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)


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
    from .capacity import MemoryBudget, placement_options
    from .errors import SnowLLMError
    def check_allocation(*, complete=True):
        try:
            limits = MemoryBudget(a.max_gpu_mem_util, a.max_host_mem_util, a.max_host_mem_gib,
                                  max_gpu_mem_gib=a.max_gpu_mem_gib, max_disk_mem_gib=a.max_disk_mem_gib)
            a.host_offload, _ = placement_options(host_alloc=a.host_alloc, device_map=a.device_map,
                host_offload=a.host_offload, offload_ratio=a.offload_ratio,
                max_disk_bytes=limits.disk_limit(), moe_cache_experts=a.moe_cache_experts,
                moe_cache_mode=a.moe_cache_mode, complete=complete)
        except SnowLLMError as exc:
            p.error(str(exc))
    # Recipe defaults can still supply device_map and the cache size.
    check_allocation(complete=False)
    positional = getattr(a, "model", None)
    if positional and a.model_flag:
        p.error("the checkpoint directory was given twice")
    model = positional or a.model_flag
    if not model:
        p.error("use `snowllm webui` to open the browser interface, or "
                "`snowllm PATH|RECIPE` to serve a model directly")

    from .hub import recipes
    try:
        found, recipe = recipes.locate(model)
        model = str(found)
    except SnowLLMError as e:
        p.exit(2, f"{term.paint('snowllm:', term.BOLD, term.RED, stream=sys.stderr)} {e}\n")
    except KeyboardInterrupt:
        p.exit(130, "\n")

    unset = object()
    given = p.parse_args(namespace=argparse.Namespace(**dict.fromkeys(recipes.SETTABLE, unset)))
    if recipe and recipe.defaults:
        def _at(v: object) -> object:
            return model + v[len("@model"):] if isinstance(v, str) and v.startswith("@model") \
                else v

        used = {k: _at(v) for k, v in recipe.defaults.items()
                if k in recipes.SETTABLE and getattr(given, k) is unset}
        for group in (('max_gpu_mem_util', 'max_gpu_mem_gib'),
                      ('max_host_mem_util', 'max_host_mem_gib'),
                      ('moe_cache_experts', 'max_disk_mem_gib')):
            if any(getattr(given, key) is not unset for key in group):
                for key in group:
                    used.pop(key, None)
        if given.host_offload is not unset:
            used.pop('offload_ratio', None)
            used.pop('device_map', None)
        if given.offload_ratio is not unset or given.device_map is not unset:
            used.pop('host_offload', None)
        if given.host_alloc is not unset and a.host_alloc:
            for key in ('device_map', 'host_offload', 'offload_ratio',
                        'moe_cache_experts', 'moe_cache_mode', 'max_disk_mem_gib'):
                used.pop(key, None)
        if any(getattr(given, key) is not unset for key in
               ('device_map', 'host_offload', 'offload_ratio', 'moe_cache_experts', 'moe_cache_mode', 'max_disk_mem_gib')):
            used.pop('host_alloc', None)
        for k, v in used.items():
            setattr(a, k, v)

    if a.max_model_len <= 0 or a.max_num_batched_tokens <= 0:
        p.error('context and prefill chunk must be positive')
    check_allocation()
    if recipe:
        print(f'{term.stamp()} run {_equivalent_command(p, a, model, recipe)}', flush=True)

    import uvicorn
    from .serve.api import app
    from .serve.state import install

    try:
        state = build(model, a.max_num_seqs, a.max_model_len, num_kv_blocks=a.num_kv_blocks,
              default_max_tokens=a.default_max_tokens, seed=a.seed, profile_dir=a.profile_dir,
              stats_interval=a.stats_interval, exact_stats=a.stats_exact,
              enforce_eager=a.enforce_eager, num_spec=a.num_spec,
              mtp_window=a.mtp_window, mtp_sinks=a.mtp_sinks, served_names=a.served_model_name,
              dflash=a.dflash, dflash_block=a.dflash_block,
              dflash_p_min=a.dflash_p_min,
              dspark_adaptive=a.dspark_adaptive,
              batch_prefill=a.batch_prefill, allow_image_urls=a.allow_image_urls,
              prefill_chunk=a.max_num_batched_tokens, kv_int8=a.kv_cache_dtype == "int8",
              max_gpu_mem_util=a.max_gpu_mem_util, max_gpu_mem_gib=a.max_gpu_mem_gib,
              max_host_mem_util=a.max_host_mem_util, max_host_mem_gib=a.max_host_mem_gib,
              offload_ratio=a.offload_ratio,
              device_map=a.device_map,
              moe_cache_experts=a.moe_cache_experts, moe_cache_mode=a.moe_cache_mode,
              host_offload=a.host_offload,
              host_alloc=a.host_alloc,
              max_disk_mem_gib=a.max_disk_mem_gib,
              serve_as=(recipe.serve_as if recipe else None),
              prefix_memory_ratio=a.prefix_memory_ratio,
              limit_mm_per_prompt=a.limit_mm_per_prompt)
    except (RuntimeError, MemoryError) as exc:
        from .errors import is_capacity_error
        capacity_error = is_capacity_error(f'{type(exc).__name__}: {exc}')
        if not isinstance(exc, SnowLLMError) and not capacity_error:
            raise
        message = str(exc) or type(exc).__name__
        if capacity_error:
            if not is_capacity_error(message):
                message = f'Out of memory: {message}'
            if 'budget exceeds physical' in message:
                message += ' Lower --max-gpu-mem or --max-host-mem to fit physical memory.'
            else:
                context = max(1, a.max_model_len // 2)
                chunk = min(context, max(1, a.max_num_batched_tokens // 2))
                message += (f' Try smaller workload options, for example: --max-model-len {context} '
                            f'--max-num-seqs 1 --max-num-batched-tokens {chunk}.')
        p.exit(2, f'snowllm: {message}\n')
    install(state)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
