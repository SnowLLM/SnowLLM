# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from .. import ops
from .resources import draft_table_blocks


def record_memory(r, kv, *, weights, host_weights, load_peak, room):
    from ..memory_report import current
    report = current()
    if report is None:
        return
    from ..models.geometry import DeepSeekV4Geometry, Glm5NextGeometry
    plan = r._pool_plan(0)
    units = plan.minimum if plan.requested is None else plan.requested
    cache = cache_total = plan.bytes_for(units)
    if isinstance(r.geo, Glm5NextGeometry):
        cache -= r.sparse_state_bytes()
    step = 100000 if isinstance(r.geo, DeepSeekV4Geometry) else ops.kv_blocks_for(100000, r.block_size)
    cache_100k = plan.bytes_for(units + step) - cache_total
    linear, fixed_workspace = 0, 0
    modules = getattr(r.resources, 'linear', None)
    if modules is None:
        modules = [a for _, a in getattr(r, 'linear_mods', ())]
    for a in modules:
        tensors = (a.history, a.state) if hasattr(a, 'history') else a.state
        for t in tensors:
            unused = max(0, t.shape[0] - r.max_num_seqs*r.slots_per_request)
            overhead = t.nbytes // t.shape[0] * unused
            fixed_workspace += overhead
            linear += t.nbytes - overhead
    for a in getattr(r, 'ple_mods', ()):
        t = a.state
        unused = max(0, t.shape[0] - r.max_num_seqs*r.slots_per_request)
        overhead = t.nbytes // t.shape[0] * unused
        fixed_workspace += overhead
        linear += t.nbytes - overhead
    for a in getattr(r.resources, 'indexers', ()):
        for t in (a.carry, a.carry_pos):
            unused = max(0, t.shape[0] - r.max_num_seqs*r.slots_per_request)
            fixed_workspace += t.nbytes // t.shape[0] * unused
    workspace = r.act_bytes_for(r.max_prefill_tokens) + fixed_workspace
    aux = auxiliary_bytes(r, kv)
    fixed = r.fixed_bytes(r.max_prefill_tokens)
    other = fixed - cache - plan.snapshot_floor - linear - fixed_workspace + aux
    snapshots = plan.minimum_snapshots if plan.checkpoint_bytes else 0
    resident = int(kv.get('draft_weight_bytes', kv.get('draft_resident', kv.get('reserve'))) or 0)
    snapshot_each = plan.checkpoint_bytes
    if not isinstance(r.geo, DeepSeekV4Geometry) and hasattr(r.resources, 'checkpoints'):
        snapshot_each = r.resources.checkpoints.bytes_for(1)
    cache_snapshot_each = max(0, plan.checkpoint_bytes - snapshot_each)
    snapshot_bytes = snapshots * snapshot_each
    cache += snapshots * cache_snapshot_each
    report.values.update(context=int(kv.get('ctx') or r.ctx_cap), requests=r.max_num_seqs,
        chunk=r.max_prefill_tokens, slots=r.slots_per_request, weights=weights,
        host_weights=host_weights, cache=cache, cache_100k=cache_100k, linear=linear,
        snapshot_each=snapshot_each, cache_snapshot_each=cache_snapshot_each, fixed_workspace=fixed_workspace,
        other=other, snapshots=snapshot_bytes, snapshot_count=snapshots, workspace=workspace,
        total=weights + resident + cache + linear + other + snapshot_bytes + workspace,
        load_peak=load_peak, load_room=room, room=room)
    capacity = getattr(r, 'context_capacity', None)
    if capacity is not None:
        report.values['per_request'] = capacity.per_request
    report.values['weights'] += resident
    from ..host_alloc import enabled
    report.values['allocation'] = 'host' if enabled() else 'GPU'


def auxiliary_bytes(r, kv):
    from . import _dspark, draft_block
    from .sampler import Sampler
    from .spec_decode import SpecDecoder, DFlashDecoder
    from .dspark_decode import DSparkDecoder
    from ..checkpoint.loader import yarn_rope_table
    with ops.dry_load() as dry:
        sampler = object.__new__(Sampler)
        sampler.reserve_device(r.max_num_seqs*r.T, r.geo.vocab_size,arena=r.arena)
        tables, stage = r.create_inputs()
        spec, inv = None, None
        geo = kv.get('draft_geo')
        if geo is not None:
            block = draft_block(geo, kv.get('dflash_block', 0))
            width = draft_table_blocks(geo, r.ctx_cap, block, r.block_size)
            if _dspark(geo):
                draft = SimpleNamespace(geo=geo, tap_layers=geo.tap_layers,
                    conf_proj=object() if kv.get('dspark_adaptive') or kv.get('dflash_p_min') else None)
                spec = DSparkDecoder(r, sampler, None, draft, None, tables, stage,
                    max_blocks_per_seq=width, block=block,
                    p_min=kv.get('dflash_p_min', 0.), adaptive=kv.get('dspark_adaptive', False))
            else:
                from ..models.qwen3_5.dflash_draft import DFlashDraft
                draft = object.__new__(DFlashDraft)
                draft.geo, draft._scratch, draft._sel_ws = geo, None, None
                draft.w = {'conf_weight': None}
                spec = DFlashDecoder(r, sampler, None, draft, None, tables, stage,
                    block=block, max_blocks_per_seq=width,
                    p_min=kv.get('dflash_p_min', 0.), adaptive=kv.get('dspark_adaptive', False))
        elif r.num_spec:
            decoder = getattr(r, 'spec_decoder_type', SpecDecoder)
            spec = decoder(r, sampler, None, tables, stage, num_spec=r.num_spec,
                           window=kv.get('mtp_window', 0), sinks=kv.get('mtp_sinks', 64))
        if r.auto_yarn:
            from ..context import ContextCapacity
            capacity = r.context_capacity or ContextCapacity.for_model(r.model, r.ctx_cap)
            inv, _ = yarn_rope_table(r.model.config, capacity.rope_factor,
                                    orig_max_pos=capacity.original)
            inv = inv.cuda()
        return dry.device
