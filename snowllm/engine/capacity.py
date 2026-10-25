# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from .. import ops
from .resources import draft_table_blocks


def auxiliary_bytes(r, kv):
    from . import _dspark, draft_block
    from .sampler import Sampler
    from .spec_decode import SpecDecoder, DFlashDecoder
    from .dspark_decode import DSparkDecoder
    from ..checkpoint.loader import yarn_rope_table
    with ops.dry_load() as dry:
        sampler = object.__new__(Sampler)
        sampler.reserve_device(r.max_num_seqs*r.T, r.geo.vocab_size)
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
            native = int(r.model.config['max_position_embeddings'])
            inv, _ = yarn_rope_table(r.model.config, max(1., r.ctx_cap/native))
            inv = inv.cuda()
        return dry.device
