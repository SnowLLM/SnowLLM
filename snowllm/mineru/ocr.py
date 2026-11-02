# SPDX-License-Identifier: Apache-2.0
import torch

from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from mineru.model._internal.pytorchocr.infer.predict_system import TextSystem
from mineru.model._internal.pytorchocr.infer.predict_det import TextDetector
from mineru.model._internal.pytorchocr.infer.predict_rec import TextRecognizer
from mineru.model._internal.pytorchocr.data import create_operators
from mineru.model._internal.pytorchocr.postprocess import build_post_process


class Detector(TextDetector):
    def __init__(self, args, load_model):
        self.args = args
        self.det_algorithm = args.det_algorithm
        if self.det_algorithm != "DB":
            raise ValueError("the bundled native OCR detector requires DB")
        self.device = "cpu"
        self.ocr_inference_dtype = torch.float32
        self.weights_path = args.det_model_path
        self.preprocess_op = create_operators(
            [
                {
                    "DetResizeForTest": {
                        "limit_side_len": args.det_limit_side_len,
                        "limit_type": args.det_limit_type,
                        "max_side_limit": args.det_max_side_limit,
                    }
                },
                {
                    "NormalizeImage": {
                        "std": [0.229, 0.224, 0.225],
                        "mean": [0.485, 0.456, 0.406],
                        "scale": "1./255.",
                        "order": "hwc",
                    }
                },
                {"ToCHWImage": None},
                {"KeepKeys": {"keep_keys": ["image", "shape"]}},
            ]
        )
        self.postprocess_op = build_post_process(
            {
                "name": "DBPostProcess",
                "thresh": args.det_db_thresh,
                "box_thresh": args.det_db_box_thresh,
                "max_candidates": 1000,
                "unclip_ratio": args.det_db_unclip_ratio,
                "use_dilation": args.use_dilation,
                "score_mode": args.det_db_score_mode,
                "box_type": args.det_box_type,
            }
        )
        self.net = load_model(
            self.weights_path,
            detection=True,
            seal=getattr(args, "lang", "ch") in ("seal", "seal_lite"),
        )


class Recognizer(TextRecognizer):
    def __init__(self, args, load_model):
        self.device = "cpu"
        self.rec_image_shape = [int(value) for value in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        if self.rec_algorithm not in ("CRNN", "SVTR_LCNet"):
            raise ValueError("the bundled native OCR recognizer requires SVTR_LCNet")
        self.max_text_length = args.max_text_length
        self.limited_max_width, self.limited_min_width = (
            args.limited_max_width,
            args.limited_min_width,
        )
        self.weights_path = args.rec_model_path
        self.ocr_inference_dtype = torch.float16
        self.postprocess_op = build_post_process(
            {
                "name": "CTCLabelDecode",
                "character_type": args.rec_char_type,
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        )
        self.net = load_model(self.weights_path)
        self.out_channels = self.net.head.weight.shape[0]


class NativeTextSystem(TextSystem):
    def __init__(self, args, **kwargs):
        self.text_detector = Detector(args, self._native_loader)
        self.text_recognizer = Recognizer(args, self._native_loader)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            raise ValueError(
                "the bundled native OCR recipe does not use angle classification"
            )


class OCR(PytorchPaddleOCR, NativeTextSystem):
    def __init__(self, load_model, **kwargs):
        self._native_loader = load_model
        super().__init__(**kwargs)
