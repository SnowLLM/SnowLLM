# SPDX-License-Identifier: Apache-2.0
import sys

import torch

import _harness
from test_dflash_sampling import geometry
from snowllm import ops


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(1743)
    ck = _harness.Checks(70)
    geo = geometry()
    ops.dflash.select(geo)
    for vocab in [67, 248320]:
        pred = torch.zeros(vocab, 256, dtype=torch.bfloat16, device='cuda')
        succ = torch.zeros_like(pred)
        for rows in ([3] if vocab == 67 else [3, 12, 16, 17]):
            hidden = torch.zeros(rows, 256, dtype=torch.bfloat16, device='cuda')
            logits = torch.empty(rows, vocab, device='cuda')
            anchor = torch.zeros(1, dtype=torch.int64, device='cuda')
            temp = torch.zeros(1, device='cuda')
            uniform = torch.full((1, rows), .371, device='cuda')
            slots = torch.tensor([1], dtype=torch.int32, device='cuda')
            path_alloc = torch.full((rows + 128,), -73, dtype=torch.int64, device='cuda')
            path = path_alloc[:rows].view(1, rows)
            ids_alloc = torch.full((3 * rows * 16 + 128,), -99, dtype=torch.int64, device='cuda')
            ids = ids_alloc[:3 * rows * 16].view(3, rows, 16)
            q_alloc = torch.full((3 * rows * 16 + 128,), -99., device='cuda')
            q = q_alloc[:3 * rows * 16].view(3, rows, 16)
            need = ops.dflash.select_scratch_bytes(rows)
            scratch = torch.full((need + 256,), 123, dtype=torch.uint8, device='cuda')

            def run():
                ops.dflash.select_path(hidden, logits, pred, succ, anchor, path, scratch[:need],
                    temperature=temp, uniform=uniform, slots=slots, candidate_ids=ids,
                    probabilities=q)

            for pattern in ['random', 'ties', 'nan-mix']:
                logits.normal_()
                if pattern == 'ties':
                    logits.zero_()
                    logits[:, ::2] = -0.
                elif pattern == 'nan-mix':
                    logits[:, ::257] = float('nan')
                clean = logits.cpu().nan_to_num(nan=-float('inf'))
                expected_ids = clean.argsort(dim=-1, descending=True, stable=True)[:, :16]
                expected_values = clean.gather(1, expected_ids)
                for temperature in [0., .7]:
                    temp.fill_(temperature)
                    expected_q = (torch.nn.functional.one_hot(torch.zeros(rows, dtype=torch.int64), 16).float()
                                  if temperature == 0 else (expected_values / temperature).softmax(-1))
                    picked = (torch.full((rows, 1), .371) >= expected_q.cumsum(-1)).sum(-1).clamp_max(15)
                    expected_path = expected_ids.gather(1, picked[:, None]).view(1, rows)
                    run()
                    ck.exact(f'V={vocab} M={rows} {pattern} T={temperature}: candidate order', ids[1].cpu(), expected_ids)
                    ck.close('selector probabilities preserve the top16 scores', q[1].cpu(), expected_q, 1e-6)
                    ck.exact('selector path matches independent stable-sort reference', path.cpu(), expected_path)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    graph.replay()
                    ck.exact('selector graph replay preserves the path', path.cpu(), expected_path)
                    ck('selector preserves workspace, output and unused-slot bounds',
                       bool((scratch[need:] == 123).all() and (path_alloc[rows:] == -73).all()
                            and (ids_alloc[3*rows*16:] == -99).all()
                            and (q_alloc[3*rows*16:] == -99).all()
                            and (ids[0] == -99).all() and (ids[2] == -99).all()
                            and (q[0] == -99).all() and (q[2] == -99).all()))
                    del graph
        del pred, succ
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
