# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


def main():
    path = os.environ.get('SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF')
    if not path:
        _harness.skip('set SNOWLLM_QWEN38_FLASH_NEXT_Q4_FAST_GGUF to run full-model generation')
    root = Path(path).expanduser().parent
    tokenizer, stops = loader.load_tokenizer(root)
    model = loader.load(root, mtp=False, vision=False)
    print(f'Loaded {len(model.model.layers)} layers; '
          f'Torch allocations: {torch.cuda.memory_allocated() / (1 << 30):.2f} GiB', flush=True)
    engine = Engine(model, num_kv_blocks=256, max_num_seqs=1, max_model_len=1024,
                    stop_token_ids=stops, seed=0, prefill_chunk=128, preempt=False)
    checks = _harness.Checks()
    for prompt, expected in (
        ('What is the capital of France? Answer briefly.', 'paris'),
        ('计算 12 + 30，直接回答结果。', '42'),
    ):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                            tokenize=True, add_generation_prompt=True)
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        request = engine.add(ids, SamplingParams(temperature=0., max_new_tokens=96))
        engine.run()
        text = tokenizer.decode(request.out)
        answer = text.rsplit('</think>', 1)[-1].lower()
        checks(prompt, expected in answer, repr(text))
    return checks.done()


if __name__ == '__main__':
    sys.exit(main())
