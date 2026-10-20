# SPDX-License-Identifier: Apache-2.0

import gc
import os
from pathlib import Path
import sys

import torch
from PIL import Image

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams, _draft_geometry, draft_block, draft_num_spec
from snowllm.models import multimodal
from snowllm.models.qwen3_5.dspark_draft import DSparkDraft


def main():
    root = _harness.checkpoint("MiniCPM-V-4.7-35B-A3B-Q4_K_M", env="SNOWLLM_MINICPM_DIR")
    draft = Path(os.environ.get("SNOWLLM_MINICPM_DSPARK", root / "dspark"))
    if not draft.exists():
        _harness.skip(f"no DSpark checkpoint at {draft}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks(65)
    geo = _draft_geometry(str(draft))
    ck("checkpoint selects fifteen proposals and sixteen verification rows",
       draft_block(geo, 0) == draft_num_spec(geo, 0) == 15)
    model = loader.load(root, mtp=False)
    tok, eos = loader.load_tokenizer(root)
    processor = loader.load_processor(root, tok)
    def prompt(text):
        return tok.apply_chat_template([dict(role="user", content=text)], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
    prompts = [prompt("Return exactly the integers from 1 to 30, separated by commas."),
               prompt("Write a Python function to compute Fibonacci numbers.")]
    params = SamplingParams(temperature=0, max_new_tokens=64)
    long_prompt = prompt("Background: The quick brown fox jumps over the lazy dog. " * 60
                         + "What is 17 + 25? Reply with just the number.")
    short = SamplingParams(temperature=0, max_new_tokens=8)
    def make(block=None, eager=False, confidence=0, prefix=0):
        return Engine(model, num_kv_blocks=256, max_num_seqs=2, max_model_len=2048,
            stop_token_ids=eos, enforce_eager=eager, num_spec=0,
            prefill_chunk=128, prefix_memory_ratio=prefix, preempt=False, account=False,
            dflash_path=str(draft) if block is not None else None,
            dflash_block=block or 0, dflash_p_min=confidence)
    def release(engine):
        engine.runner.resources.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    def ask(engine, force=False):
        rr = [engine.add(p, params) for p in prompts]
        while not all(r.done for r in rr):
            if force:
                for r in rr:
                    if r.prefilled and not r.done:
                        r.drafts = [geo.mask_token_id] * engine.dflash.block
            engine.step()
        if engine.dflash:
            ck("requests return all draft pages", len(engine.dflash.blocks.free) == engine.dflash.blocks.total)
        ck("requests return recurrent state slots", len(engine.slots.free) == 2)
        return [r.out for r in rr]
    base = make(eager=True)
    expected = ask(base)
    long_request = base.add(long_prompt, short)
    base.run()
    long_expected = long_request.out
    release(base)
    for block, eager in ((4, True), (15, False)):
        engine = make(block, eager, prefix=.002 if block == 15 else 0)
        ck("native Qwen DSpark drafter selected", isinstance(engine.dflash.draft, DSparkDraft))
        ck(f"block={block}: verification width includes anchor", engine.dflash.verify_rows(2) == block + 1)
        actual = ask(engine)
        ck(f"block={block}: greedy tokens agree with ordinary decoding", actual == expected,
           str([tok.decode(row)[:100] for row in actual]))
        hist = engine.dflash.accept_hist
        accepted = sum((i+1)*n for i,n in enumerate(hist)) / max(1, sum(hist))
        ck(f"block={block}: accepts multiple tokens per verification", accepted > 1.3, f"{accepted:.2f}")
        rejected = ask(engine, force=True)
        ck("forced first-token rejection preserves target tokens and recurrent state", rejected == expected)
        ck("forced rejection reaches the one-token acceptance bin", engine.dflash.accept_hist[0] > 0)
        if not eager:
            ck("main and draft HIP graphs replay", engine.runner.n_graph_replays > 0
               and engine.runner.n_draft_replays > 0)
            for batch in (1, 2, 1):
                requests = [engine.add(prompts[i], params) for i in range(batch)]
                engine.run()
                ck(f"graph batch transition B={batch}", [r.out for r in requests] == expected[:batch])
            requests = [engine.add(p, params) for p in prompts]
            while not all(r.prefilled for r in requests):
                engine.step()
            victim = engine._preempt()
            ck("preemption returns the interrupted draft pages", victim is not None
               and not victim.draft_blocks)
            engine.run()
            ck("preemption replay preserves target output", [r.out for r in requests] == expected)
            for rep in range(2):
                request = engine.add(long_prompt, short)
                engine.run()
                ck("chunked prefill / prefix reuse preserves output", request.out == long_expected)
                if rep:
                    ck("prefix reuse restores both target and DSpark KV", request.cached_tokens >= 512,
                       f"{request.cached_tokens} cached tokens")
            messages = [dict(role="user", content=[dict(type="image"), dict(type="text",
                text="What color is this image? Answer with one word.")])]
            for mode, color in (("16x", "red"), ("4x", "blue")):
                mm = multimodal.prepare(model, processor, messages,
                    [Image.new("RGB", (224,224), color)], enable_thinking=False, downsample_mode=mode)
                request = engine.add(mm.pop("prompt"), SamplingParams(temperature=0, max_new_tokens=16), **mm)
                engine.run()
                ck(f"image + DSpark mode={mode}", color in tok.decode(request.out).lower(), tok.decode(request.out))
            before = engine.dflash.workspace_bytes()
            for batch in (1, 2):
                engine.dflash._stage_values(batch, [20]*batch+[block]*batch+[geo.mask_token_id]*batch+[0]*batch)
                engine.dflash.d_bt[:batch].zero_()
                torch.cuda.synchronize()
                live = torch.cuda.memory_stats()["requested_bytes.all.current"]
                torch.cuda.reset_peak_memory_stats()
                engine.dflash._blocks_dev(batch)
                torch.cuda.synchronize()
                peak = torch.cuda.memory_stats()["requested_bytes.all.peak"] - live
                ck(f"draft B={batch}: allocation peak fits shared planner", peak <= before,
                   f"actual={peak}, planned={before}")
        release(engine)
    engine = make(4, confidence=.99)
    actual = ask(engine)
    ck("confidence truncation preserves greedy output", actual == expected)
    release(engine)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
