# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch
import yaml
from mineru.model.mfr.pp_formulanet.predict_formula import FormulaRecognizer
from mineru.model.mfr.pp_formulanet.processors import (
    LatexImageFormat,
    ToBatch,
    UniMERNetDecode,
    UniMERNetImgDecode,
    UniMERNetTestTransform,
)


class Formula(FormulaRecognizer):
    def __init__(self, path, load_model):
        path = Path(path)
        self.model_name = "PP-FormulaNet_plus-M"
        self.device = torch.device("cpu")
        self.ocr_inference_dtype = torch.float16
        self.net = load_model(path / (self.model_name + ".pth"), "PPFormulaNetModel")
        data = yaml.load(
            (path / (self.model_name + "_inference.yml")).read_text(),
            Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
        )
        self.pre_tfs = {
            "UniMERNetImgDecode": UniMERNetImgDecode(input_size=(384, 384)),
            "UniMERNetTestTransform": UniMERNetTestTransform(paddle_compatible=False),
            "LatexImageFormat": LatexImageFormat(),
            "ToBatch": ToBatch(),
        }
        self.post_op = UniMERNetDecode(
            character_list=data["PostProcess"]["character_dict"]
        )
