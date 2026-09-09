# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import _harness  # noqa: E402

from snowllm import cli  # noqa: E402
from snowllm.engine import Engine, Request, SamplingParams  # noqa: E402

MODEL_DIR = pathlib.Path.home() / "models" / "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"
UTIL = 0.95
CTX = 8192
SHORT = "In 1815 the eruption of Mount Tambora threw enough ash into the stratosphere to"

def ptrs(eng: Engine) -> dict[str, int]:
    a = eng.runner.arena
    return {"<arena>": a.buf.data_ptr(), "<arena size>": a.buf.numel()}


def run(eng: Engine, ids: list[int], n: int) -> Request:
    r = eng.add(ids, SamplingParams(temperature=0.0, max_new_tokens=n))
    while not r.done:
        eng.step()
    return r


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"== skipped: {MODEL_DIR} is not here")
        return 0
    ck = _harness.Checks()

    st = cli.build(str(MODEL_DIR), max_num_seqs=1, max_model_len=CTX, num_kv_blocks=None,
                   default_max_tokens=8, seed=0, profile_dir=None, dflash=None,
                   enforce_eager=False, device_map="off", gpu_memory_utilization=UTIL,
                   prefill_chunk=2048)
    eng = st.engine.engine
    short = st.tokenizer.encode(SHORT, add_special_tokens=False)

    print("\n=== every shape is captured before the first request, under the ceiling ===")
    free, total = torch.cuda.mem_get_info()
    dev = total - free
    ck("the runner captured at startup, with no request in the engine yet",
       len(eng.runner.graphs) > 0, f"{len(eng.runner.graphs)} graphs")
    ck("and what the device holds is under the ceiling, not over it", dev <= int(total * UTIL),
       f"{dev / (1 << 30):.1f}G in use against {total * UTIL / (1 << 30):.1f}G")

    print("\n=== a captured graph bakes its pointers, so nothing it reads may be reallocated ===")
    run(eng, short, 12)
    res = torch.cuda.memory_reserved()
    eager = eng.runner.n_eager_forwards
    out = run(eng, short, 12)
    ck("a short request captured graphs and replayed them", len(eng.runner.graphs) > 0
       and eng.runner.n_graph_replays > 0,
       f"{len(eng.runner.graphs)} graphs, {eng.runner.n_graph_replays} replays")
    ck("and it took no more memory to serve it", torch.cuda.memory_reserved() == res,
       f"{(torch.cuda.memory_reserved() - res) / (1 << 20):+.1f} MiB reserved")
    ck("with every eager forward a prefill, so no decode step ran without a graph",
       eng.runner.n_eager_forwards - eager == 1,
       f"{eng.runner.n_eager_forwards - eager} eager forwards")
    before = ptrs(eng)
    base, graphs = torch.cuda.memory_allocated(), len(eng.runner.graphs)

    long = (short * (4096 // len(short) + 1))[:4096]
    run(eng, long, 4)
    after = ptrs(eng)

    moved = sorted(k for k in before if after.get(k) != before[k])
    grew = sorted(k for k in after if k not in before)
    served = eng.runner.arena.planner.served
    ck("a 4096-token prompt afterwards moves no buffer at all", not moved,
       f"moved: {', '.join(moved) or 'nothing'}")
    ck("and allocates no new one either", not grew, f"new: {', '.join(grew) or 'nothing'}")
    ck("and the long prompt really walked that buffer, so that is not vacuous",
       served > 0 and eng.runner.arena.buf.numel() > (1 << 20),
       f"{served} placements served from {eng.runner.arena.buf.numel() >> 20} MiB")
    took = torch.cuda.memory_allocated() - base
    ck("and the only bytes it took are the graphs the new shapes captured",
       took <= (len(eng.runner.graphs) - graphs) * (200 << 20) + (16 << 20),
       f"{took / (1 << 20):.1f} MiB over {len(eng.runner.graphs) - graphs} new graphs")

    ck("the graphs kept replaying after it", eng.runner.n_graph_replays > 0,
       f"{eng.runner.n_graph_replays} replays, {eng.runner.n_eager_forwards} eager")
    ck("and the short request still read as prose",
       len(st.tokenizer.decode(out.out).strip()) > 0, repr(st.tokenizer.decode(out.out))[:80])

    print("\n=== the ceiling bounds the graph pool, and a full device decodes eagerly ===")
    rn = eng.runner
    ck("with room, the sweep would capture another", rn._graph_room(), "room for another")
    was, rn.gpu_util = rn.gpu_util, 0.01
    try:
        ck("at a ceiling the device is already past, it refuses", not rn._graph_room(),
           f"gpu_util {rn.gpu_util}")
        ck("and says so exactly once", rn._graphs_full and not rn._graph_room(), "sticky log")
        n, eager = len(rn.graphs), rn.n_eager_forwards
        run(eng, short[:len(short) // 2], 6)
        ck("a request afterwards mints no graph of its own",
           len(rn.graphs) == n and rn.n_eager_forwards > eager,
           f"{len(rn.graphs)} graphs (was {n}), {rn.n_eager_forwards - eager} eager forwards")
    finally:
        rn.gpu_util = was

    rn.budget.full = False
    rn._said_miss = False
    rn._miss((99, 99))
    ck("and a shape nobody captured is reported, once",
       rn._said_miss and rn._miss((98, 98)) is None, "sticky")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
