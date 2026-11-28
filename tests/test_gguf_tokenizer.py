# SPDX-License-Identifier: Apache-2.0
import json
import unittest

from tokenizers import Tokenizer
from snowllm.checkpoint.gguf.tokenizer import build, tokenizer_json


class Metadata(dict):
    need = dict.__getitem__


def metadata(arch, pre, section_order=None):
    g = Metadata({"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre,
                  "tokenizer.ggml.tokens": ["x", "y", "xy", "<slice>"], "tokenizer.ggml.merges": [],
                  "{arch}.rope.section_order": section_order})
    g.arch = arch
    return g


def tokenizer(arch, pre, section_order=None):
    return Tokenizer.from_str(json.dumps(tokenizer_json(metadata(arch, pre, section_order))))


class GGUFTokenizerTests(unittest.TestCase):
    def test_minicpm_tokenizer_preserves_multimodal_ids(self):
        g = metadata("qwen35moe", "qwen35", "zyxt")
        markers = {"image_token": "<|image_pad|>", "video_token": "<|video_pad|>",
                   "image_start_token": "<image>", "image_end_token": "</image>",
                   "slice_start_token": "<slice>", "slice_end_token": "</slice>",
                   "image_id_start_token": "<image_id>", "image_id_end_token": "</image_id>"}
        vocab = ["x", *markers.values()]
        g["tokenizer.ggml.tokens"] = vocab
        g["tokenizer.ggml.token_type"] = [1] + [3] * len(markers)
        tok = build(g)
        for key, token in markers.items():
            self.assertEqual(getattr(tok, key + "_id"), vocab.index(token))
            self.assertEqual(tok.encode(token, add_special_tokens=False), [vocab.index(token)])

    def test_whole_vocab_token_without_merge_rule(self):
        for arch, pre in (("glm5-next", "glm4"), ("lfm2moe", "lfm2"), ("inkling2_loop", "inkling2")):
            self.assertEqual(tokenizer(arch, pre).encode("xy").ids, [2])
        self.assertEqual(tokenizer("qwen35moe", "qwen35").encode("xy").ids, [0, 1])

    def test_minicpm_splits_combining_marks(self):
        word = "किताब"
        mini = tokenizer("qwen35moe", "qwen35", "zyxt")
        qwen = tokenizer("qwen35moe", "qwen35")
        self.assertEqual([span for _, span in mini.pre_tokenizer.pre_tokenize_str(word)],
                         [(0, 1), (1, 3), (3, 5)])
        self.assertEqual([span for _, span in qwen.pre_tokenizer.pre_tokenize_str(word)], [(0, 5)])


if __name__ == "__main__":
    unittest.main()
