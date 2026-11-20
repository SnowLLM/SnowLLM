# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from snowllm.checkpoint.gguf.sequential import SequentialReads, layout


def reader(requests):
    groups, pieces = layout(requests, window=16384)
    assert all(g.size <= 16384 for g in groups)
    for group in groups:
        assert [(s.path, s.start) for s in group.spans] == sorted((s.path, s.start) for s in group.spans)
    return SequentialReads(requests, groups, pieces, max(g.size for g in groups))


def main():
    with TemporaryDirectory() as folder:
        a, b = (Path(folder) / f'{i}.gguf' for i in range(2))
        data = {p: bytes((j * 9 + i) % 251 for j in range(80178)) for i, p in enumerate((a, b))}
        for path, value in data.items():
            path.write_bytes(value)
        requests = [('first', [(b, 77, 312), (a, 4011, 253), (a, 4022, 45)]),
                    ('large', [(a, 1005, 70001)]), ('eof', [(b, 80123, 55)])]
        reads = reader(requests)
        outputs = []
        try:
            for key, ranges in requests:
                expected = b''.join(data[p][offset:offset+n] for p, offset, n in ranges)
                out = torch.empty(len(expected) + 2, dtype=torch.uint8, device='cuda')
                out.fill_(244)
                assert reads.copy(key, out[1:-1])
                outputs.append((out, b'\xf4' + expected + b'\xf4'))
        finally:
            reads.close()
        for out, expected in outputs:
            assert out.cpu().numpy().tobytes() == expected
        reads = reader([('bad', [(a, 80174, 7)])])
        try:
            try:
                reads.copy('bad', torch.empty(7, dtype=torch.uint8, device='cuda'))
            except OSError:
                pass
            else:
                raise AssertionError('short GGUF read accepted')
        finally:
            reads.close()
    print('GGUF cross-shard reads, bank reuse, destination slices and truncated input PASS')


if __name__ == '__main__':
    main()
