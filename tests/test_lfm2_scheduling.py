# SPDX-License-Identifier: Apache-2.0

import os
import sys

import _harness
from snowllm import cli
from snowllm.engine import SamplingParams


def main():
    root = _harness.checkpoint("LFM2.5-8B-A1B-Q4_K_M", env="SNOWLLM_LFM_DIR")
    draft = _harness.checkpoint("LFM2.5-8B-A1B-DSpark-Q8_0", env="SNOWLLM_LFM_DSPARK_DIR")
    mode = os.environ.get("SNOWLLM_LFM_SPEC", "fixed")
    state = cli.build(str(root), max_num_seqs=2, max_model_len=2048,
        num_kv_blocks=None, default_max_tokens=192, profile_dir=None, seed=0, num_spec=0,
        prefill_chunk=128, prefix_memory_ratio=.02,
        dflash=str(draft) if mode != "plain" else None, dflash_block=9,
        dspark_adaptive=mode == "adaptive")
    engine, tok = state.engine.engine, state.tokenizer
    check = _harness.Checks()
    stops = engine.stop_token_ids
    engine.sampler.stop_token_ids = frozenset()
    prompts = [tok.encode(f"Document {i}: " + "The warehouse is open today. " * 250)[:976]
               for i in range(2)]
    requests = [engine.add(p, SamplingParams(temperature=.7, top_k=40, top_p=.9,
                                           repetition_penalty=1.05, max_new_tokens=192))
                for p in prompts]
    engine.run()
    stats = engine.stats()
    check("KV pressure preempts a request", stats.preemptions > 0, stats.preemptions)
    check("replayed requests finish to token limit", all(
        r.finish_reason == "length" and len(r.out) == 192 for r in requests))
    check("request slots released", stats.free_slots == 2 and not engine.waiting and not engine.running)
    check("preempted request replayed", any(r.replay > 0 for r in requests))
    edge = (prompts[0]*3)[:2032]
    r = engine.add(edge, SamplingParams(temperature=.7, max_new_tokens=16))
    engine.run()
    check("generation reaches exact context boundary", len(r.prompt)+len(r.out) == 2048
          and r.finish_reason == "length")
    engine.sampler.stop_token_ids = stops
    prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content":
        "What is the capital of France? Answer briefly."}], tokenize=False,
        add_generation_prompt=True), add_special_tokens=False)
    r = engine.add(prompt, SamplingParams(temperature=0, max_new_tokens=256))
    engine.run()
    check("generation recovers after preemption", r.finish_reason == "stop" and
          "Paris" in tok.decode(r.out).split("</think>")[-1], tok.decode(r.out)[-100:])
    print("Scheduling stats:", engine.stats(), flush=True)
    state.engine._worker.shutdown(wait=True)
    return check.done()


if __name__ == "__main__":
    sys.exit(main())
