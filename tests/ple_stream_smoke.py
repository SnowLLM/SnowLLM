# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from snowllm_kernels import LIB

os.environ['SNOWLLM_LIB'] = str(LIB)

from snowllm import _capi, ops
from snowllm.checkpoint.gguf.dequant import dequantize
from snowllm.models.qwen4exp.ple import PleTable
from snowllm.models.qwen4exp.ple_io import _row_reader
from snowllm.models.qwen4exp.ple_stream import PleReadStream


def main():
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    rng = np.random.default_rng(817)
    data = rng.integers(0, 256, (129, 90), dtype=np.uint8)
    scales = rng.normal(size=(129, 5)).astype(np.float16)
    data.reshape(-1, 18)[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    rows = rng.integers(0, len(data), (64, 16), dtype=np.int64)
    rows[:, 1] = rows[:, 0]
    next_rows = rows[:29, ::-1].copy()
    release = threading.Event()

    with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
        path = Path(directory) / 'table.gguf'
        path.write_bytes(bytes(192) + data.tobytes())
        reader = _row_reader(path, 192, len(data), 90)
        cleanup.callback(reader.close)

        class DelayedReader:
            def read(self, ids, target):
                if not release.wait(20):
                    raise TimeoutError('consumer submission blocked on the file read')
                reader.read(ids, target)

        table = SimpleNamespace(reader=DelayedReader(), row_bytes=90, rows=len(data),
                                geo=SimpleNamespace(ple_heads=16, ple_head_dim=160),
                                quant='IQ4_NL', _staging=True)
        stream = table.file_stream = PleReadStream.create(table, len(rows))
        assert stream is not None, 'required read gate missing from native wheel'
        try:
            main_stream = torch.cuda.current_stream()
            slot = stream.take(rows)
            with torch.cuda.stream(stream.stream):
                prepared = PleTable.gather(table, rows, dtype=torch.float32, raw=slot.device)
                packed = torch.empty(ops.ple_prepared_bytes(len(rows)), dtype=torch.uint8, device='cuda')
                PleTable.gather(table, rows, raw=slot.device, packed_out=packed)
                ready = stream.stream.record_event()
            stream.prefetch(next_rows)
            main_stream.wait_event(ready)
            assert not ready.query(), 'GPU consumed the rows before the file read finished'
            release.set()
            reference = torch.from_numpy(data[rows].copy()).cuda()
            expected = dequantize(reference.flatten(), 'IQ4_NL', rows.size * 160,
                                  torch.float32).reshape(len(rows), -1)
            expected_packed = torch.empty_like(packed)
            ops.ple_pack_fp32(expected, expected_packed, 0, len(rows))
            assert torch.equal(expected, prepared) and torch.equal(packed, expected_packed)
            stream.release(slot)
            pending = stream.pending
            slot = stream.take(next_rows.copy())
            assert slot is pending
            main_stream.wait_event(slot.copied)
            assert torch.equal(slot.device[:len(next_rows)].cpu(),
                               torch.from_numpy(data[next_rows].reshape(len(next_rows), -1)))
            stream.release(slot)
            stream.prefetch(rows)
            slot = stream.take(next_rows)
            stream.release(slot)

            class FailedReader:
                def read(self, ids, target):
                    raise OSError('injected read failure')

            stream.reader = FailedReader()
            slot = stream.take(rows)
            try:
                stream.release(slot)
            except OSError as exc:
                assert str(exc) == 'injected read failure' and stream.error is exc
            else:
                raise AssertionError('missing file read error')
        finally:
            release.set()
            stream.close()
    print('PASS', torch.cuda.get_device_name(), type(reader).__name__,
          'delayed read, consumer event, IQ4_NL, packed bytes, lookahead reuse/mismatch, failed read')


if __name__ == '__main__':
    main()
