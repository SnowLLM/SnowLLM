# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from unittest.mock import patch

from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from mineru.model._internal.pytorchocr.base_ocr_v20 import BaseOCRV20
from mineru.model._internal.pytorchocr.infer.predict_det import TextDetector
from mineru.model._internal.pytorchocr.infer.predict_rec import TextRecognizer


class InferenceNetwork:
    def __init__(self, model):
        self.model = model

    def __call__(self, value):
        return self.model(value)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def modules(self):
        # Weight conversion and fusion belong to the inference backend.
        return ()


def OCR(load_model, **kwargs):
    def read(instance, path):
        if not isinstance(instance, (TextDetector, TextRecognizer)):
            raise ValueError('unsupported remote OCR architecture')
        detection = isinstance(instance, TextDetector)
        seal = detection and getattr(instance.args, 'lang', 'ch') in ('seal', 'seal_lite')
        return InferenceNetwork(load_model(path, detection=detection, seal=seal))

    def build(instance, **options):
        instance.net = read(instance, instance.weights_path)

    def assign(instance, network):
        instance.net = network

    with ExitStack() as stack:
        stack.enter_context(patch.object(BaseOCRV20, 'build_net', build))
        stack.enter_context(patch.object(BaseOCRV20, 'read_pytorch_weights', read))
        stack.enter_context(patch.object(BaseOCRV20, 'get_out_channels',
                                         lambda self, network: network.model.head.weight.shape[0]))
        stack.enter_context(patch.object(BaseOCRV20, 'load_state_dict', assign))
        stack.enter_context(patch.object(BaseOCRV20, 'load_pytorch_weights',
                                         lambda self, path: assign(self, read(self, path))))
        return PytorchPaddleOCR(**kwargs)
