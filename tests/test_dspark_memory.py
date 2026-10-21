# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from snowllm import ops
from snowllm.checkpoint.gguf.source import GGUFReader, find_dspark_gguf
from snowllm.engine import plan_draft
from snowllm.engine.forward_context import Dsv4Walker
from snowllm.models.deepseek_v4 import dspark

import _harness


def main():
    root = Path(os.environ.get('SNOWLLM_DSV4_DIR',
                Path.home() / 'models/DeepSeek-V4-Flash-0731-UD-IQ2_XXS'))
    side = find_dspark_gguf(root)
    if side is None:
        _harness.skip(f'no DSpark checkpoint at {root}')
    ck = _harness.Checks()
    plan = plan_draft(str(side))
    geo, planned, peak = plan.geo, plan.device_bytes, plan.load_peak_bytes
    shared = torch.empty(0, dtype=torch.bfloat16, device='cuda')
    target = SimpleNamespace(embed=SimpleNamespace(weight=shared), head=object(),
                             geo=SimpleNamespace(num_layers=max(geo.tap_layers)))
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()['requested_bytes.all.current']
    torch.cuda.reset_peak_memory_stats()
    with GGUFReader(side) as rd:
        draft = dspark.load(rd, geo, target, Dsv4Walker())
    torch.cuda.synchronize()
    live = torch.cuda.memory_stats()['requested_bytes.all.current'] - before
    high = torch.cuda.memory_stats()['requested_bytes.all.peak'] - before
    ck('device weight storage matches the shared load plan', live == planned,
       f'planned={planned}, actual={live}')
    ck('loading and conversion peak matches the shared load plan', high == peak,
       f'planned={peak}, actual={high}')
    ck('routed and shared draft experts stay in mapped host memory',
       all(ops.host_mapped(layer.moe.gate_up.p0) and ops.host_mapped(layer.moe.down.p0)
           and ops.host_mapped(layer.moe.shared_gate_up.quant)
           and ops.host_mapped(layer.moe.shared_down.quant)
           for layer in draft.stack.layers))
    ck('transient loading storage is budgeted separately from resident weights', peak > planned)
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
