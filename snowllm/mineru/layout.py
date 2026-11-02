# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
from mineru.model.layout.pp_doclayout_v2_base import (
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESCALE_FACTOR,
    PP_DOCLAYOUT_V2_LABEL_TO_ID,
    load_preprocess_config,
)
from transformers import PPDocLayoutV2Config


class Layout(PPDocLayoutV2LayoutModel):
    def __init__(self, path, load_model):
        self.device = "cpu"
        self.use_paddlex_filter_boxes = True
        self.model_dir = str(path)
        self.preprocess_config = load_preprocess_config(self.model_dir)
        size = self.preprocess_config.get("size", {})
        self.imgsz = (
            int(size.get("width", DEFAULT_IMAGE_SIZE[0])),
            int(size.get("height", DEFAULT_IMAGE_SIZE[1])),
        )
        self.rescale_factor = float(
            self.preprocess_config.get("rescale_factor", DEFAULT_RESCALE_FACTOR)
        )
        self.config = PPDocLayoutV2Config.from_pretrained(self.model_dir)
        inline_formula = PP_DOCLAYOUT_V2_LABEL_TO_ID["inline_formula"]
        self.config.class_thresholds[inline_formula] = DEFAULT_CLASS_THRESHOLDS[
            inline_formula
        ]
        model = load_model(
            Path(path) / "model.safetensors",
            "PPDocLayoutV2Model",
            config=self.config.to_dict(),
        )

        def predict(pixel_values):
            output = model(pixel_values)
            output.logits = output.logits.cpu()
            output.pred_boxes = output.pred_boxes.cpu()
            output.order_logits = output.order_logits.cpu()
            return output

        self.model = predict
