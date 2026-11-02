# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch


def main():
    import torch
    from snowllm._platform import DeviceMemory
    from snowllm.models.vision import VisionModel
    from snowllm.mineru.graphs import GraphCache
    from snowllm.ops import image as op

    class Model(VisionModel):
        def __init__(self, cache):
            self._cache = {}
            self._memory_admit = cache.admit

        def execute(self, x, position, out, *, arena=None, admit=True):
            def walk():
                value = op.activation(op.add(x, op.cast(position, x.dtype)), "gelu")
                return op.copy(out, value)

            return self.execute_plan(
                ("native-test", tuple(x.shape), x.dtype),
                x.shape[0],
                walk,
                arena=arena,
                admit=admit,
            )

    torch.manual_seed(19)
    cache = GraphCache(lambda: 12 << 30)
    model = Model(cache)
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(2, 32, device="cuda", dtype=dtype)
            position = torch.zeros(1, device="cuda", dtype=torch.int64)
            previous = snapshot = None
            for index in range(4):
                x.normal_()
                position.fill_(index)
                out = torch.empty_like(x)
                actual = cache.native(model.execute, (x, position), out)
                expected = torch.nn.functional.gelu(x + index)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=2e-6 if dtype == torch.float32 else 0,
                    rtol=1e-5 if dtype == torch.float32 else 0,
                )
                if previous is not None:
                    torch.testing.assert_close(previous, snapshot, atol=0, rtol=0)
                previous, snapshot = actual, actual.clone()
        assert cache.captures == 2 and cache.replays == 6, cache.status()
        print(
            "PASS: native Arena graphs refresh FP16, FP32 and integer inputs; outputs stay independent",
            flush=True,
        )
        old_pool = cache.pools[model.execute]

        def memory():
            occupied = (8 << 30) if cache.cases else 0
            return DeviceMemory(16 << 30, occupied, 0, 16 << 30, occupied)

        cache.limit = lambda: 1 << 30
        with patch("snowllm.mineru.graphs.device_memory_snapshot", side_effect=memory):
            cache.admit(64 << 20)
        assert not cache.cases and not cache.pools
        cache.limit = lambda: 12 << 30
        cache.native(model.execute, (x, position), torch.empty_like(x))
        assert cache.pools[model.execute] != old_pool
        cache.clear()
        print(
            "PASS: evicted graphs release their pool and can be captured again",
            flush=True,
        )
        refused = GraphCache(lambda: 12 << 30)
        denied_model = Model(refused)
        with patch.object(refused.budget, "room", return_value=False):
            result = refused.native(
                denied_model.execute, (x, position), torch.empty_like(x)
            )
        torch.testing.assert_close(
            result, torch.nn.functional.gelu(x + position), atol=0, rtol=0
        )
        assert refused.fallbacks == 1 and not refused.cases
        print(
            "PASS: a graph that does not fit falls back to the same Arena plan",
            flush=True,
        )


if __name__ == "__main__":
    main()
