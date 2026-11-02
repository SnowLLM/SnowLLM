# SPDX-License-Identifier: Apache-2.0
import gc
import os
from pathlib import Path


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import PPDocLayoutV2ForObjectDetection
    from mineru.model._internal.pytorchocr.infer.pytorchocr_utility import (
        get_arch_config,
    )
    from mineru.model._internal.pytorchocr.modeling.architectures.base_model import (
        BaseModel,
    )
    from snowllm.checkpoint.loader import load_auxiliary
    from snowllm.mineru.graphs import GraphCache

    root = Path(
        os.environ.get("SNOWLLM_MINERU_DIR", "~/models/MinerU-4.1")
    ).expanduser()
    ocr = root / "small-models/OCR/paddleocr"
    torch.manual_seed(127)
    with torch.inference_mode():
        for filename, options, shapes in (
            (
                "ch_PP-OCRv6_tiny_det_infer.safetensors",
                {"detection": True},
                ((1, 3, 128, 160), (2, 3, 96, 128)),
            ),
            (
                "ch_PP-OCRv6_small_rec_infer.safetensors",
                {},
                ((2, 3, 48, 320), (3, 3, 48, 324)),
            ),
            (
                "seal_PP-OCRv4_det_infer.pth",
                {"seal": True},
                ((1, 3, 128, 160), (2, 3, 64, 96)),
            ),
        ):
            path = ocr / filename
            detection = bool(options)
            dtype = torch.float32 if detection else torch.float16
            config = get_arch_config(str(path))
            if not detection:
                config["Head"]["out_channels_list"] = {"CTCLabelDecode": 18710}
            reference = BaseModel(config, out_channels=18710).eval()
            state = (
                load_file(path)
                if path.suffix == ".safetensors"
                else torch.load(path, map_location="cpu", weights_only=True)
            )
            reference.load_state_dict(
                {key.removeprefix("model."): value for key, value in state.items()}
            )
            reference = reference.to(device="cuda", dtype=dtype)
            if options.get("seal"):
                for layer in reference.modules():
                    if hasattr(layer, "rep"):
                        layer.rep()
            cache = GraphCache(lambda: 12 << 30)
            model = load_auxiliary(path, "PPOCRModel", admit=cache.admit, **options)
            model._memory_admit, model._native_graph_cache = cache.admit, cache
            assert model.load_peak_bytes > 0
            key = "maps" if detection else "ctc_logits"
            for shape in shapes:
                for _ in range(3):
                    x = torch.randn(shape, device="cuda", dtype=dtype)
                    actual, expected = model(x)[key], reference(x)[key]
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=0.003 if not detection else 0.0001,
                        rtol=0.003,
                    )
            assert cache.captures == 2 and cache.replays == 4, cache.status()
            print(
                "PASS: own checkpoint model, Arena and graph replay",
                filename,
                flush=True,
            )
            cache.clear()
            del model, reference, state, x, actual, expected
            gc.collect()
            torch.cuda.empty_cache()
        path = root / "small-models/Layout/PP-DocLayoutV2"
        model = load_auxiliary(path / "model.safetensors", "PPDocLayoutV2Model")
        reference = (
            PPDocLayoutV2ForObjectDetection.from_pretrained(
                path, attn_implementation="eager"
            )
            .cuda()
            .eval()
        )
        cache = GraphCache(lambda: 12 << 30)
        model._memory_admit, model._native_graph_cache = cache.admit, cache
        for _ in range(3):
            x = torch.randn(1, 3, 192, 192, device="cuda")
            actual, expected = model(x), reference(x)
            for name in ("logits", "pred_boxes", "order_logits"):
                torch.testing.assert_close(
                    getattr(actual, name),
                    getattr(expected, name),
                    atol=0.002,
                    rtol=0.003,
                )
        boxes = torch.rand(2, 300, 4, device="cuda") * 1000
        boxes[:, :, 2:] = (
            boxes[:, :, :2] + boxes[:, :, 2:] * (1000 - boxes[:, :, :2]) / 1000
        )
        labels = torch.randint(0, 20, (2, 300), device="cuda")
        mask = torch.rand(2, 300, device="cuda") > 0.5
        torch.testing.assert_close(
            model.reading_order(boxes, labels, mask),
            reference.reading_order(boxes=boxes, labels=labels, mask=mask),
            atol=0.003,
            rtol=0.003,
        )
        print(
            "PASS: own layout detection and reading-order models match reference",
            flush=True,
        )
        cache.clear()


if __name__ == "__main__":
    main()
