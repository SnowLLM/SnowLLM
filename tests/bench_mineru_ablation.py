# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


CASES = {
    "layout_b1": ("pointwise", "depthwise", "fusion"),
    "layout_b8": ("pointwise", "depthwise", "fusion"),
    "ocr_det": ("pointwise", "depthwise", "fusion"),
    "ocr_rec": ("fusion",),
    "seal_det": ("pointwise", "depthwise", "fusion"),
    "formula_b1": ("half", "fusion", "sync"),
    "formula_b8": ("half", "fusion", "sync"),
    "table_cls": ("pointwise", "depthwise", "fusion"),
    "table_unet": ("pointwise", "fusion"),
    "table_slanet": ("pointwise", "depthwise", "fusion"),
    "ocr_192": ("fusion",),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def ablation(mode):
    import torch
    from snowllm import ops
    from snowllm.ops import image as op
    from snowllm.ops.operation import operator
    from snowllm.models.vision import VisionModel
    from snowllm.models.mineru.pp_formulanet import DecodeLinear

    with contextlib.ExitStack() as stack:
        off = {mode.removeprefix("no_")}
        if mode in ("no_custom", "no_custom_no_graph"):
            off = {"pointwise", "depthwise", "fusion", "half"}
        if mode in ("no_graph", "no_custom_no_graph"):
            original_plan = VisionModel.execute_plan

            def retained_plan(self, key, rows, walk, *, arena=None, **kwargs):
                if arena is None and not getattr(self, "_native_eager", False):
                    arenas = self.__dict__.setdefault("_ablation_arenas", {})
                    arena = arenas.setdefault((key, rows), ops.Arena())
                return original_plan(self, key, rows, walk, arena=arena, **kwargs)

            stack.enter_context(patch.object(VisionModel, "execute_plan", retained_plan))
        elif mode == "eager_unplanned":
            stack.enter_context(patch.object(
                VisionModel, "execute_plan", lambda self, key, rows, walk, **kw: walk()
            ))

        @operator
        def general_conv(x, weight, bias, stride, padding, groups, scale, shift, act):
            from snowllm.ops.miopen import convolution

            spatial = tuple(
                (n + 2 * p - k) // s + 1
                for n, p, k, s in zip(x.shape[2:], padding, weight.shape[2:], stride)
            )
            out = op.empty((x.shape[0], weight.shape[0], *spatial),
                           dtype=x.dtype, device=x.device)
            x = op.dense(x)
            plan = convolution(tuple(x.shape), tuple(weight.shape), tuple(out.shape),
                               x.dtype, stride, padding, (1, 1), groups, False,
                               torch.cuda.current_device())
            arena = op.active_arena()
            with arena.frame() if arena is not None else contextlib.nullcontext():
                workspace = op.empty((plan.workspace,), dtype=torch.uint8, device=x.device)
                if out.device.type != "meta":
                    plan(x, weight, out, workspace)
            if bias is not None:
                out = op.add(out, bias[None, :, None, None])
            names = {0: None, 4: "relu", 5: "silu", 2: "gelu", 9: "sigmoid"}
            return op.activation(out, names.get(act, act), scale, shift)

        if "pointwise" in off:
            def pointwise(x, weight, bias=None, scale=None, shift=None, act=0, config=0):
                return general_conv(x, weight, bias, (1, 1), (0, 0), 1, scale, shift, act)
            stack.enter_context(patch.object(op, "pointwise_f32", pointwise))
        if "depthwise" in off:
            def depthwise(x, weight, bias=None, scale=None, shift=None, stride=1, act=0):
                padding = tuple(k // 2 for k in weight.shape[2:])
                return general_conv(x, weight, bias, (stride, stride), padding,
                                    x.shape[1], scale, shift, act)
            stack.enter_context(patch.object(op, "depthwise_conv", depthwise))
        if "half" in off:
            stack.enter_context(patch.object(
                DecodeLinear, "__call__", lambda self, x: op.linear(x, self.weight, self.bias)
            ))
        if "fusion" in off:
            original_conv = op.convolution

            def separate_conv(x, weight, bias=None, *, scale=None, shift=None, act=None, **kw):
                value = original_conv(x, weight, bias, scale=None, shift=None, act=None, **kw)
                return op.activation(value, act, scale, shift)

            @operator
            def separate_activation(x, kind, scale=None, shift=None):
                if scale is not None:
                    x = op.multiply(x, scale.view(1, -1, *([1] * (x.ndim - 2))))
                if shift is not None:
                    x = op.add(x, shift.view(1, -1, *([1] * (x.ndim - 2))))
                if kind is None or kind == 0:
                    return x
                names = {2: "gelu", 4: "relu", 5: "silu", 6: "hardswish",
                         9: "sigmoid", 10: "tanh", 11: "hardsigmoid"}
                kind = names.get(kind, kind)
                if kind in ("paddle_hsigmoid", 7):
                    return op.divide(op.clamp(op.add(op.multiply(x, 1.2), 3.0), 0.0, 6.0), 6.0)
                if kind in ("paddle_hswish", 12):
                    return op.divide(op.multiply(x, op.clamp(op.add(x, 3.0), 0.0, 6.0)), 6.0)
                function = getattr(torch.ops.aten, kind).out
                return op.evaluate(function, x.shape, (x,))

            stack.enter_context(patch.object(op, "activation", separate_activation))
            stack.enter_context(patch.object(op, "convolution", separate_conv))
        if mode == "no_sync":
            from snowllm.models.mineru.pp_formulanet import FormulaRunner
            original_step = FormulaRunner.generate_step

            def synced_step(self, state, position):
                result = original_step(self, state, position)
                state[-1].item()
                return result

            stack.enter_context(patch.object(FormulaRunner, "generate_step", synced_step))
        yield


def ocr_fixture():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    fonts = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
    lines = ["文档解析与知识检索：答案是 42，编号 A-2026-1101。",
             "GPU 多模型执行，共享内存预算，按步骤串行调度。",
             "小矩阵乘法、图捕获和原生算子需要共同验证。",
             "项目名称：研发费用；数量：128；金额：￥1,234.56。",
             "SnowLLM document parsing keeps layout, tables and formulas.",
             "The quick brown fox jumps over the lazy dog. Record 42.",
             "Data: 0.123456789, -42.00, 1.23e-5, ID abc-DEF_0123.",
             "If performance improves, verify both correctness and memory."]
    images = []
    for index in range(192):
        font = ImageFont.truetype(fonts[index % 2], (22, 28, 36, 42)[index % 4])
        text = lines[index % 8] + f" {index:03d}"
        box = font.getbbox(text)
        image = Image.new("RGB", (box[2] + 24, max(64, box[3] + 20)), "white")
        ImageDraw.Draw(image).text((12, 8), text, font=font, fill="black")
        if index % 3 == 0:
            image = image.filter(ImageFilter.GaussianBlur(0.4))
        images.append(np.asarray(image)[:, :, ::-1].copy())
    return images


def build_case(name, mode, args, stack):
    import numpy as np
    import torch
    from PIL import Image
    from snowllm.checkpoint.loader import load_auxiliary
    from snowllm.mineru.graphs import GraphCache

    cache = GraphCache(lambda: 12 << 30,
                       enabled=mode not in ("no_graph", "no_custom_no_graph", "eager_unplanned"))
    models = []
    stack.callback(cache.clear)

    def load(path, architecture="PPOCRModel", **options):
        model = load_auxiliary(path, architecture, **options)
        model._native_graph_cache = cache
        model._memory_admit = cache.admit
        if architecture == "PPOCRModel":
            model._native_eager = True
            model._native_graph_after = 3
        models.append(model)
        if hasattr(model, "close"):
            stack.callback(model.close)
        return model

    torch.manual_seed(127)
    root = args.model / "small-models"
    if name.startswith("layout"):
        model = load(root / "Layout/PP-DocLayoutV2/model.safetensors", "PPDocLayoutV2Model")
        batch = int(name.rsplit("b", 1)[1])
        x = torch.rand(batch, 3, 800, 800, device="cuda")
        function = lambda: model(x)
    elif name.startswith("formula"):
        from snowllm.mineru.mfr import Formula

        model = Formula(root / "MFR/pp_formulanet_plus_m", load)
        images = [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy()
                  for p in sorted(args.images.glob("*.png"))]
        batch = int(name.rsplit("b", 1)[1])
        assert len(images) >= batch
        selected = images[:batch]
        boxes = [[dict(label="display_formula", bbox=[0, 0, x.shape[1], x.shape[0]])]
                 for x in selected]
        function = lambda: model.batch_predict(boxes, selected, batch_size=batch)
    elif name == "ocr_192":
        from mineru.model.download import ModelPath
        from snowllm.mineru.ocr import OCR

        original = ModelPath.ensure

        def ensure(path, **kw):
            if path.repo.name == "MinerU-4_models_torch":
                candidate = root / path.relative_path
                assert candidate.exists(), candidate
                return candidate
            return original(path, **kw)

        with patch.object(ModelPath, "ensure", ensure):
            model = OCR(load, device="cuda", lang="ch")
        images = ocr_fixture()
        function = lambda: model.text_recognizer(images)
    elif name.startswith("table"):
        filenames = {"table_cls": "PP-LCNet_x1_0_table_cls.onnx",
                     "table_unet": "unet.onnx", "table_slanet": "slanet-plus.onnx"}
        kinds = {"table_cls": "classifier", "table_unet": "unet", "table_slanet": "slanet"}
        shapes = {"table_cls": (1, 3, 224, 224), "table_unet": (1, 3, 1024, 1024),
                  "table_slanet": (1, 3, 488, 488)}
        model = load(root / "Table" / filenames[name], "MinerUTableModel", kind=kinds[name])
        x = torch.rand(shapes[name], device="cuda")
        function = lambda: model(x)
    else:
        files = {"ocr_det": "ch_PP-OCRv6_tiny_det_infer.safetensors",
                 "ocr_rec": "ch_PP-OCRv6_small_rec_infer.safetensors",
                 "seal_det": "seal_PP-OCRv4_det_infer.pth"}
        shape = (6, 3, 48, 824) if name == "ocr_rec" else (1, 3, 960, 704)
        model = load(root / "OCR/paddleocr" / files[name],
                     detection=name == "ocr_det", seal=name == "seal_det")
        x = torch.rand(shape, dtype=model.dtype, device="cuda")
        function = lambda: model(x)
    return function, cache, models


def snapshot(name, result):
    import torch

    if name.startswith("formula"):
        return [items[0]["latex"] for items in result]
    if name == "ocr_192":
        return [(text, float(score)) for text, score in result[0]]
    if name.startswith("layout"):
        result = [result.logits, result.pred_boxes, result.order_logits]
    elif isinstance(result, dict):
        result = list(result.values())
    elif isinstance(result, torch.Tensor):
        result = [result]
    return [(value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
            .detach().cpu().clone() for value in result]


def compare(name, actual, expected):
    import torch

    if name.startswith("formula"):
        return dict(passed=actual == expected, actual=actual, expected=expected)
    if name == "ocr_192":
        mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a[0] != b[0]]
        delta = max(abs(a[1] - b[1]) for a, b in zip(actual, expected))
        return dict(passed=not mismatches and delta <= 0.002,
                    text_mismatches=mismatches, max_confidence_delta=delta)
    rows = []
    for a, b in zip(actual, expected):
        if a.shape != b.shape:
            rows.append(dict(passed=False, shape=list(a.shape), expected_shape=list(b.shape)))
            continue
        delta = (a.float() - b.float()).abs()
        equal = a == b if not a.is_floating_point() else torch.isclose(a, b, atol=0.003, rtol=0.003)
        rows.append(dict(shape=list(a.shape), passed=bool(equal.all()),
                         mismatches=int((~equal).sum()), max_error=float(delta.max()),
                         relative_l2=float(delta.norm() / b.float().norm().clamp_min(1e-20))))
    return dict(passed=len(actual) == len(expected) and all(x["passed"] for x in rows), tensors=rows)


def worker(args):
    import torch
    from snowllm import ops
    from snowllm._capi import GEO_MINERU_12B

    torch.set_num_threads(2)
    ops.select_geometry(GEO_MINERU_12B)
    report = dict(seed=args.seed, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  library=os.environ["SNOWLLM_LIB"], library_sha256=digest(os.environ["SNOWLLM_LIB"]),
                  warmups=args.warmups, repeats=args.repeats, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for name in args.cases:
        modes = ["no_graph", "eager_unplanned", "no_custom", "no_custom_no_graph"]
        modes += ["no_" + kind for kind in CASES[name]]
        if args.modes:
            modes = [mode for mode in modes if mode in args.modes]
        rng.shuffle(modes)
        modes = ["baseline", *modes, "baseline_end"]
        expected = None
        for label in modes:
            mode = "baseline" if label == "baseline_end" else label
            gc.collect()
            torch.cuda.empty_cache()
            started = time.perf_counter()
            with contextlib.ExitStack() as stack, ablation(mode), torch.inference_mode():
                function, cache, models = build_case(name, mode, args, stack)
                for _ in range(args.warmups):
                    value = function()
                torch.cuda.synchronize()
                before = cache.status()
                row = dict(case=name, mode=label, setup_seconds=time.perf_counter() - started,
                           samples_ms=[])
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    value = function()
                    torch.cuda.synchronize()
                    row["samples_ms"].append((time.perf_counter() - start) * 1000)
                after = cache.status()
                row["median_ms"] = statistics.median(row["samples_ms"])
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                row["graph_delta"] = {k: after[k] - before[k] for k in
                                      ("captures", "replays", "fallbacks", "capacity_rebuilds")}
                actual = snapshot(name, value)
                if expected is None:
                    expected = actual
                row["correctness"] = compare(name, actual, expected)
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps({k: row[k] for k in ("case", "mode", "median_ms", "graph_delta")},
                                 ensure_ascii=False), "correct", row["correctness"]["passed"], flush=True)
            for model in models:
                model.__dict__.pop("_ablation_arenas", None)
                model.__dict__.pop("_eager_arena", None)
            del function, models, cache, model, value, actual
        del expected
    report["completed"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main(args):
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.processes):
        output = args.output / f"run-{index + 1}.json"
        command = [str(args.withgpu), "--timeout", "600", sys.executable,
                   str(Path(__file__).resolve()), "--worker", "--model", str(args.model),
                   "--images", str(args.images), "--seed", str(args.seed + index),
                   "--warmups", str(args.warmups), "--repeats", str(args.repeats),
                   "--output", str(output), "--cases", *args.cases]
        if args.modes:
            command += ["--modes", *args.modes]
        print("START", index + 1, flush=True)
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                returncode = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if returncode:
            print(output.with_suffix(".log").read_text()[-6000:], flush=True)
            raise SystemExit(returncode)
        print("DONE", index + 1, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--withgpu", type=Path, default=Path.home() / "withgpu/withgpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    main(parser.parse_args())
