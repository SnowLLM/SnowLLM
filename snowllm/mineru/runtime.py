# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path

import torch

from .graphs import GraphCache


COMPONENTS = {
    'layout': ('Layout/PP-DocLayoutV2/model.safetensors', 'PPDocLayoutV2Model', {}, 'float32'),
    'formula': ('MFR/pp_formulanet_plus_m/PP-FormulaNet_plus-M.pth', 'PPFormulaNetModel', {}, 'float16'),
    'ocr_det': ('OCR/paddleocr/ch_PP-OCRv6_tiny_det_infer.safetensors', 'PPOCRModel', {'detection': True, 'seal': False}, 'float32'),
    'ocr_rec': ('OCR/paddleocr/ch_PP-OCRv6_small_rec_infer.safetensors', 'PPOCRModel', {'detection': False, 'seal': False}, 'float16'),
    'seal_det': ('OCR/paddleocr/seal_PP-OCRv4_det_infer.pth', 'PPOCRModel', {'detection': True, 'seal': True}, 'float32'),
    'table_cls': ('Table/PP-LCNet_x1_0_table_cls.onnx', 'MinerUTableModel', {'kind': 'classifier'}, 'float32'),
    'table_unet': ('Table/unet.onnx', 'MinerUTableModel', {'kind': 'unet'}, 'float32'),
    'table_slanet': ('Table/slanet-plus.onnx', 'MinerUTableModel', {'kind': 'slanet'}, 'float32'),
}


class MinerURuntime:
    def __init__(self, root, limits, *, enforce_eager=False):
        self.root = Path(root).resolve()
        self.limits = limits
        self.state = self.engine = self.loop = None
        self.closed = self.disposed = False
        self.tasks = set()
        self.native_models = {}
        self.models = {}
        self.graphs = GraphCache(lambda: self.limit, enabled=not enforce_eager)
        before = torch.cuda.memory_allocated()
        try:
            with torch.inference_mode():
                for name, (path, architecture, options, _) in COMPONENTS.items():
                    self.models[name] = self._load_native(self.root / 'small-models' / path,
                                                          architecture, **options)
            self.weight_bytes = max(0, torch.cuda.memory_allocated() - before)
        except BaseException:
            self.close()
            raise

    @property
    def limit(self):
        return self.limits.gpu_limit(torch.cuda.get_device_properties(0).total_memory)

    def _load_native(self, path, architecture, **options):
        from ..checkpoint.loader import load_auxiliary

        key = str(path), architecture
        if key not in self.native_models:
            model = load_auxiliary(path, architecture, admit=self.graphs.admit, **options)
            model._native_graph_cache = self.graphs
            if architecture == 'PPOCRModel':
                model._native_eager = True
                model._native_graph_after = 3
            model._memory_admit = self.graphs.admit
            self.native_models[key] = model
        return self.native_models[key]

    def bind(self, state):
        self.state, self.engine = state, state.engine
        state.engine.mineru = self
        state.model.visual._memory_admit = self.graphs.admit
        state.model.visual._native_graph_cache = self.graphs
        state.model.visual._native_eager = True
        state.model.visual._native_graph_after = 3
        state.model.visual._native_graph_max_rows = 1024
        state.engine.engine.runtime_config.update(
            runtime='mineru', auxiliary_weight_bytes=self.weight_bytes,
            auxiliary_inference_path='/v1/inference')
        state.engine.memory['auxiliary_weights'] = self.weight_bytes
        for name in ('weights', 'total', 'load_peak'):
            if name in state.engine.memory:
                state.engine.memory[name] += self.weight_bytes

    def start(self):
        self.loop = asyncio.get_running_loop()

    def check_cancelled(self):
        if self.closed:
            raise RuntimeError('auxiliary model service is closed')

    def gpu_call(self, function):
        self.check_cancelled()
        return function()

    def describe(self):
        components = {}
        for name, (path, architecture, options, dtype) in COMPONENTS.items():
            model = self.models[name]
            spec = dict(checkpoint=path, architecture=architecture, options=options, dtype=dtype)
            if name == 'layout':
                spec.update(kind='namespace', outputs=['logits', 'pred_boxes', 'order_logits'])
            elif name == 'formula':
                spec.update(kind='tensor', outputs=['output'])
            elif name in ('ocr_det', 'seal_det'):
                spec.update(kind='dict', outputs=['maps'])
            elif name == 'ocr_rec':
                spec.update(kind='dict', outputs=['ctc_logits', 'ctc_use_raw_logits'],
                            head_channels=model.head.weight.shape[0])
            else:
                spec.update(kind='tuple' if name == 'table_slanet' else 'tensor',
                            inputs=list(model.inputs), outputs=list(model.outputs),
                            metadata=model.metadata)
            components[name] = spec
        assets = {}
        base = self.root / 'small-models'
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.yml', '.yaml', '.txt'):
                assets[path.relative_to(base).as_posix()] = path.read_text()
        return dict(protocol=1, model=self.state.model_name, components=components, assets=assets)

    @torch.inference_mode()
    def forward(self, component, inputs):
        self.check_cancelled()
        if component not in COMPONENTS:
            raise ValueError(f'unknown auxiliary component: {component}')
        value = inputs.get('input')
        if value is None or value.ndim != 4 or min(value.shape) <= 0:
            raise ValueError('input must be a nonempty NCHW tensor')
        channels = (1, 3) if component == 'formula' else (3,)
        if value.shape[1] not in channels or value.dtype.kind != 'f':
            raise ValueError('input has unsupported channels or dtype')
        allowed = {'input', 'class_thresholds'} if component == 'layout' else {'input'}
        if set(inputs) - allowed:
            raise ValueError('unexpected model inputs')
        model = self.models[component]
        if component == 'layout':
            thresholds = inputs.get('class_thresholds')
            if thresholds is None:
                thresholds = model.config['class_thresholds']
            thresholds = torch.as_tensor(thresholds, dtype=torch.float32)
            if (thresholds.shape != model.thresholds.shape or not torch.isfinite(thresholds).all()
                    or bool(((thresholds < 0) | (thresholds > 1)).any())):
                raise ValueError('class_thresholds must match the model classes and be in [0, 1]')
            model.thresholds.copy_(thresholds)
        dtype = getattr(torch, COMPONENTS[component][3])
        self.graphs.admit(value.size * torch.empty((), dtype=dtype).element_size())
        x = torch.as_tensor(value, dtype=dtype, device='cuda')
        result = model(x)
        if component == 'layout':
            result = {name: getattr(result, name) for name in ('logits', 'pred_boxes', 'order_logits')}
        elif not isinstance(result, dict):
            names = model.outputs if component.startswith('table_') else ['output']
            values = result if isinstance(result, tuple) else (result,)
            result = dict(zip(names, values))
        return {name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                for name, value in result.items()}

    async def infer(self, component, inputs):
        self.check_cancelled()
        task = asyncio.create_task(self.engine.call(lambda: self.forward(component, inputs)))
        self.tasks.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise
        finally:
            self.tasks.discard(task)

    def cancel(self):
        self.closed = True

    async def aclose(self):
        self.closed = True
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)

    def close(self):
        if self.disposed:
            return
        self.disposed = self.closed = True
        self.graphs.clear()
        for model in self.native_models.values():
            close = getattr(model, 'close', None)
            if close is not None:
                close()
        self.models.clear()
        self.native_models.clear()
        self.state = None
