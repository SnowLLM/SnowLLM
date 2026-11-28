# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from snowllm.models.inkling2.processing import patches, prepare


class InklingVisionTest(unittest.TestCase):
    def test_alpha_and_partial_patches(self):
        image = Image.new("RGBA", (41, 20), (255, 0, 0, 0))
        image.putpixel((40, 0), (0, 255, 0, 255))
        pixels = patches(image, 1, 2, 20, 2).view(8, 3, 20, 20)
        torch.testing.assert_close(pixels[0, 0, 0, :9], torch.tensor([0.6] * 8 + [0.2]).bfloat16())
        torch.testing.assert_close(pixels[4, :, 0, 0], torch.tensor([-1, 1, -1]).bfloat16())
        self.assertEqual(pixels[4, 0, 0, 1], torch.tensor(-1 - 2/255).bfloat16())
        self.assertTrue(torch.all(pixels[2] == pixels[4, 0, 0, 1]))

    def test_interleaved_images(self):
        marker = 200054
        tok = Mock(model_max_length=64)
        tok.apply_chat_template.return_value = {"input_ids": [1, marker, 2, marker, 3]}

        def encode(pixels, rows, cols, out):
            out.copy_(pixels[::4, 0, None])

        tower = SimpleNamespace(patch_size=20, merge=2, forward=encode,
                                allocate_output=lambda n: torch.empty(n, 1, dtype=torch.bfloat16))
        model = SimpleNamespace(visual=tower, image_token_id=marker)
        images = [Image.new("RGB", (80, 40), "red"), Image.new("RGB", (40, 40), "blue")]
        result = prepare(model, tok, [], images)
        self.assertEqual(result["prompt"], [1, marker, marker, 2, marker, 3])
        self.assertEqual(result["embed_rows"], [1, 2, 4])
        self.assertEqual(result["embeds"].flatten().tolist(), [1, 1, -1])
        with self.assertRaisesRegex(ValueError, "placeholder"):
            prepare(model, tok, [], images[:1])
        tok.model_max_length = 6
        with self.assertRaisesRegex(ValueError, "context"):
            prepare(model, tok, [], images)


if __name__ == "__main__":
    unittest.main()
