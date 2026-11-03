# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import gc
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch


def main(args):
    import pypdfium2
    import torch
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
    from snowllm.checkpoint.loader import load_auxiliary
    from snowllm.mineru.graphs import GraphCache
    from snowllm.ops import image as op

    torch.set_num_threads(2)
    processor = PPDocLayoutV2LayoutModel.__new__(PPDocLayoutV2LayoutModel)
    processor.imgsz = (800, 800)
    processor.rescale_factor = 1 / 255
    images = []
    with pypdfium2.PdfDocument(str(args.pdf)) as document:
        for index in range(8):
            page = document[min(index * 10, len(document) - 1)]
            try:
                bitmap = page.render(scale=1.5)
                try:
                    images.append(processor._preprocess_single_image(bitmap.to_pil().convert("RGB"))[0])
                finally:
                    bitmap.close()
            finally:
                page.close()
    report = dict(gpu=torch.cuda.get_device_name(), pdf=str(args.pdf), rows=[])
    with torch.inference_mode():
        pages = torch.stack(images).cuda()
        for batch in (1, 8):
            expected = None
            for repetition in range(2):
                for mode in ("torch_copy", "tiled_transpose"):
                    gc.collect()
                    torch.cuda.empty_cache()
                    cache = GraphCache(lambda: 12 << 30)
                    model = load_auxiliary(args.model, "PPDocLayoutV2Model", admit=cache.admit)
                    model._memory_admit, model._native_graph_cache = cache.admit, cache
                    manager = patch.object(op, "transpose_f32", op.copy) if mode == "torch_copy" else contextlib.nullcontext()
                    with manager:
                        for _ in range(5):
                            result = model(pages[:batch])
                        torch.cuda.synchronize()
                        samples = []
                        for _ in range(9):
                            started = time.perf_counter()
                            result = model(pages[:batch])
                            torch.cuda.synchronize()
                            samples.append((time.perf_counter() - started) * 1000)
                        actual = [getattr(result, name).cpu().clone() for name in ("logits", "pred_boxes", "order_logits")]
                        if expected is None:
                            expected = actual
                        for value, reference in zip(actual, expected):
                            torch.testing.assert_close(value, reference, atol=0, rtol=0)
                    row = dict(batch=batch, repetition=repetition, mode=mode,
                               median_ms=statistics.median(samples), samples_ms=samples,
                               bitwise_equal=True, graphs=cache.status())
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    cache.clear()
                    del model, cache, result, actual
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
