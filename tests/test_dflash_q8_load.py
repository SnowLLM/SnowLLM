# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from snowllm.checkpoint.gguf.dflash import DictDraftSource, OnLoadQ8DraftSource, prepare_draft_source
from snowllm.models.geometry import DFlashGeometry


class DFlashQ8LoadTests(unittest.TestCase):
    def test_flash_next_default_and_bf16_override(self):
        source = DictDraftSource({})
        dspark = DFlashGeometry(
            hidden=2560, num_target_layers=48, vocab_size=248320, markov_rank=256,
            num_layers=5, num_heads=32, num_kv_heads=8, head_size=128, intermediate=9728,
            sliding_window=0, num_sliding_layers=0, tap_layers=(1, 12, 23, 34, 45),
            mask_token_id=248070, block_size=7, rope_theta=1e7, eps=1e-6)
        dflash2 = replace(dspark, markov_rank=0, conv_taps=2, vocab_size=0)
        with patch.dict(os.environ):
            os.environ.pop("SNOWLLM_DFLASH_Q8_ON_LOAD", None)
            for geo in (dspark, dflash2):
                self.assertIsInstance(prepare_draft_source(source, geo), OnLoadQ8DraftSource)
            for geo in (replace(dspark, vocab_size=128000), replace(dflash2, num_target_layers=64)):
                self.assertIs(prepare_draft_source(source, geo), source)
            os.environ["SNOWLLM_DFLASH_Q8_ON_LOAD"] = "0"
            for geo in (dspark, dflash2):
                self.assertIs(prepare_draft_source(source, geo), source)

    def test_zero_blocks_and_signed_rounding(self):
        weights = torch.zeros(2, 32, dtype=torch.bfloat16)
        weights[1, :8] = torch.tensor([127, .5, -.5, 1.5, -1.5, 126.5, -126.5, .498046875])
        source = OnLoadQ8DraftSource({"w": weights})
        blocks = source.packed("w").cpu().reshape(2, 34)
        self.assertEqual(blocks[:, :2].contiguous().view(torch.float16).flatten().tolist(), [0, 1])
        self.assertEqual(blocks[0, 2:].tolist(), [0] * 32)
        self.assertEqual(blocks[1, 2:].view(torch.int8).tolist(), [127, 1, -1, 2, -2, 127, -127, 0] + [0] * 24)


if __name__ == "__main__":
    unittest.main()
