import sys

import torch

import _harness
from snowllm import ops


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1743)
    ck = _harness.Checks(65)
    for rows, vocab, offset in [(b, 248320, 0) for b in (1, 3, 5, 8, 9, 16, 17)] + [
            (3, 248320, 1), (3, 1027, 0), (3, 31, 0), (0, 248320, 0)]:
        storage = torch.randn(rows * vocab + offset, device='cuda')
        logits = storage[offset:].view(rows, vocab)
        allocation = torch.full((rows + 32,), -73, dtype=torch.int64, device='cuda')
        output = allocation[:rows]
        for pattern in ('random', 'ties', 'nan', 'negative-inf'):
            if pattern == 'ties':
                logits.fill_(-1)
                logits[:, [min(vocab - 1, 31), min(vocab - 1, 1024), vocab - 1]] = 5
            elif pattern == 'nan':
                logits.fill_(float('nan'))
                if rows:
                    logits[0, vocab - 1] = 7
            elif pattern == 'negative-inf':
                logits.fill_(float('-inf'))
            expected = logits.nan_to_num(nan=float('-inf'), posinf=float('inf'),
                                         neginf=float('-inf')).argmax(-1)
            for capture in (False, True):
                output.fill_(-73)
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        ops.argmax(logits, out=output)
                    graph.replay()
                else:
                    ops.argmax(logits, out=output)
                torch.cuda.synchronize()
                label = f'{rows}x{vocab} offset={offset} {pattern} graph={capture}'
                ck.exact(label, output, expected)
                ck(f'{label} output bounds', bool((allocation[rows:] == -73).all()))
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
