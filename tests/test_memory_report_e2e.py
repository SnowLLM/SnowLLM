# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import sys
from unittest.mock import patch

import torch

import _harness
from snowllm import cli, memory_report
from snowllm.checkpoint import loader


def main():
    torch.set_num_threads(4)
    root = _harness.checkpoint('Qwen3.6-35B-A3B-UD-Q4_K_XL')
    ck = _harness.Checks()
    original = loader.load
    report = None
    emit = memory_report.MemoryReport.emit

    def observe(self, status):
        nonlocal report
        report = self
        emit(self, status)

    def four_layers(*args, **kwargs):
        return original(*args, layers=range(4), vision=False, **kwargs)

    for context in (1300000, 1100000):
        output = io.StringIO()
        with patch.object(loader, 'load', side_effect=four_layers), \
             patch.object(memory_report.MemoryReport, 'emit', observe), \
             contextlib.redirect_stdout(output):
            try:
                cli.build(str(root), 1, context, None, None, 23, None,
                          max_gpu_mem_gib=6, prefill_chunk=512, prefix_memory_ratio=0.)
            except RuntimeError as exc:
                failure = str(exc)
            else:
                raise AssertionError('oversized context unexpectedly fit')
        print(output.getvalue(), end='')
        ck('failure prints the planned breakdown',
           '[planner] failed GPU tensors=' in output.getvalue() and 'workspace=' in output.getvalue())
        ck('available GPU and host memory precedes model planning',
           output.getvalue().startswith('[planner] available GPU=') and 'host=' in output.getvalue().splitlines()[0])
        ck('failure identifies a GPU budget shortfall', 'GPU memory budget is short' in failure)
        ck('context suggestion stays near the fitting boundary',
           report is not None and report.suggestion is not None and context//2 < report.suggestion < context,
           str(report.suggestion))
        ck('failed candidate search preserves the requested report',
           report.values['context'] == context and report.values['chunk'] == 512)
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
