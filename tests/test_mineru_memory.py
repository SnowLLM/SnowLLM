# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch
    from snowllm._platform import DeviceMemory
    from snowllm.models.vision import VisionModel
    from snowllm.mineru.graphs import GraphCache
    from snowllm.ops import image as op

    class Large(VisionModel):
        def __init__(self, cache):
            self._cache = {}
            self._memory_admit = cache.admit

        def __call__(self):
            return self.execute_plan(
                ("oversized",),
                1,
                lambda: op.full((5 << 30,), 0, device="cuda", dtype=torch.uint8),
            )

    for limit, occupied in ((12, 8), (10, 7), (6, 2)):
        cache = GraphCache(lambda: limit << 30, enabled=False)
        memory = DeviceMemory(16 << 30, occupied << 30, 0, 16 << 30, occupied << 30)
        with patch("snowllm.mineru.graphs.device_memory_snapshot", return_value=memory):
            cache.admit((limit - occupied) << 30)
            before = torch.cuda.memory_allocated()
            try:
                Large(cache)()
            except MemoryError:
                pass
            else:
                raise AssertionError(
                    "static plans must use the remaining shared budget"
                )
            assert torch.cuda.memory_allocated() == before
    print(
        "PASS: Arena rejects an oversized forward before allocation, including eager mode",
        flush=True,
    )


if __name__ == "__main__":
    main()
