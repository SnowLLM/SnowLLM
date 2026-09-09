# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import collections
import re
import sys

from snowllm import ops
from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf
from snowllm.models.geometry import Qwen4ExpGeometry
from snowllm.models.qwen4exp import PLE_TABLE

import _harness

CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")
CARVE_OUT = 96.0
GIB = float(2 ** 30)
DIRECT = {"F32", "F16", "BF16"}
EXPERT_STEMS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def main() -> int:
    ck = _harness.Checks()
    g = GGUF(find_gguf(CKPT))
    cfg = config(g)["text_config"]
    geo = Qwen4ExpGeometry.from_config(cfg)

    readable = DIRECT | set(ops.KQUANT_FORMATS) | set(ops.LOWBIT_FORMATS)
    unknown = sorted({t.quant.name for t in g.tensors.values()} - readable)
    ck("every tensor is in a format this build can read", not unknown, f"{unknown}")

    counts = collections.Counter(t.quant.name for t in g.tensors.values())
    print(f"     formats: {dict(counts)}")

    by_role = collections.defaultdict(set)
    for name, t in g.tensors.items():
        m = re.match(r"blk\.\d+\.(ffn_(?:gate|up|down)_exps)\.weight$", name)
        if m:
            by_role[m.group(1)].add(t.quant.name)
    for role in EXPERT_STEMS:
        ck(f"{role} is k-quant or i-quant, never both families in one tensor",
           all(f in ops.KQUANT_FORMATS or f in ops.LOWBIT_FORMATS for f in by_role[role]),
           f"{sorted(by_role[role])}")
    mixed = any(f in ops.LOWBIT_FORMATS for f in by_role["ffn_gate_exps"]) and \
        all(f in ops.KQUANT_FORMATS for f in by_role["ffn_down_exps"])
    ck("gate/up and down cross families, which is what the mixed entry point is for", mixed,
       f"gate/up {sorted(by_role['ffn_gate_exps'])} over down {sorted(by_role['ffn_down_exps'])}")

    total = sum(t.nbytes for t in g.tensors.values())
    table = g[PLE_TABLE].nbytes
    resident = total - table
    ctx = int(cfg["max_position_embeddings"])
    full = len(geo.full_layers)
    kv = full * ctx * geo.num_kv_heads * geo.head_size * 2 * 2
    idx = full * ctx * geo.index_head_dim * 2
    cache = (kv + idx) / GIB
    print(f"     checkpoint {total / GIB:.2f} GiB = {resident / GIB:.2f} resident "
          f"+ {table / GIB:.2f} in {PLE_TABLE}")
    print(f"     cache at {ctx}: {kv / GIB:.2f} GiB KV over {full} full layers "
          f"+ {idx / GIB:.2f} GiB of raw indexer keys")

    biggest = max((t.nbytes, n) for n, t in g.tensors.items() if n != PLE_TABLE)
    ck(f"one tensor is {100 * table / total:.0f}% of the checkpoint and the next is "
       f"{biggest[0] / GIB:.2f} GiB", table > total / 4 and biggest[0] < table // 8,
       f"{PLE_TABLE} {table / GIB:.2f} GiB, then {biggest[1]}")

    ck(f"resident, the table would leave {CARVE_OUT - (total / GIB + cache):.2f} GiB of the "
       f"{CARVE_OUT:.0f} GiB carve-out for everything else",
       total / GIB + cache > CARVE_OUT * 0.9, f"{total / GIB + cache:.2f} GiB")
    ck(f"read from the file it leaves {CARVE_OUT - (resident / GIB + cache):.2f} GiB",
       resident / GIB + cache < CARVE_OUT * 0.7, f"{resident / GIB + cache:.2f} GiB")

    per_token = geo.ple_heads * (table // g[PLE_TABLE].rows)
    ck(f"and a token needs only {per_token} B of it",
       per_token * 1000000 < table, f"{per_token} B against {table / GIB:.2f} GiB")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
