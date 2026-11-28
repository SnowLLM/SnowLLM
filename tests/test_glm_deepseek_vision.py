# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import Mock

import torch

from snowllm.models.deepseek_v4.vision import DeepSeekV41Vision
from snowllm.models.glm5_next.vision import Glm5NextVision


class VisionBudgetTest(unittest.TestCase):
    def test_encoding_peak_and_embeddings_across_slots(self):
        for tower_cls, merge, columns, channels, dtype in (
            (Glm5NextVision, 2, 12, 6, torch.float32), (DeepSeekV41Vision, 3, 9, 3, torch.bfloat16)
        ):
            with self.subTest(tower=tower_cls.__name__):
                tower = object.__new__(tower_cls)
                tower.max_images, tower.min_tokens, tower.max_tokens = 2, 4, 12
                tower.patch_size, tower.out_hidden = 14, 64
                tower._budgets = {}
                tower.encode = Mock(return_value=9000)
                expected = 9000 + 3 * 24 * 64 * 2
                self.assertEqual(tower.request_bytes(32, 3), expected)
                pixels, h, w, output = tower.encode.call_args.args
                self.assertEqual((h, w), (merge, columns * merge))
                self.assertEqual(pixels.shape, (h * w, 14**2 * channels))
                self.assertEqual(pixels.dtype, dtype)
                self.assertEqual(output.shape, (12, 64))
                self.assertTrue(tower.encode.call_args.kwargs['estimate'])
                self.assertEqual(tower.request_bytes(32, 3), expected)
                tower.encode.assert_called_once()
                self.assertEqual(tower.request_bytes(8, 1), 9000 + 7 * 64 * 2)
                self.assertEqual(tower.request_bytes(4, 1), 0)
                tower.max_images = 0
                self.assertEqual(tower.request_bytes(32, 3), 0)

    def test_image_layout_matches_token_count(self):
        image = torch.arange(4, dtype=torch.bfloat16, device="cuda").reshape(4, 1)
        for tower_cls, expected in (
            (Glm5NextVision, [0, 1, 2, 3]),
            (DeepSeekV41Vision, [-1, 0, 1, -2, 2, 3, -2, -3]),
        ):
            with self.subTest(tower=tower_cls.__name__):
                tower = object.__new__(tower_cls)
                tower.out_hidden = 1
                if tower_cls is DeepSeekV41Vision:
                    tower.start, tower.newline, tower.end = (image.new_tensor([n]) for n in (-1, -2, -3))
                out = image.new_empty(tower.token_count(2, 2), 1)
                tower.write_image(out, image, 2, 2)
                self.assertEqual(out.flatten().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
