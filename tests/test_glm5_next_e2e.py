# SPDX-License-Identifier: Apache-2.0
"""Full local GLM checkpoint: host experts, chunked prefill and two concurrent requests."""
import os
import pathlib
import sys

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


def main():
    root = pathlib.Path(os.environ.get("SNOWLLM_GLM_DIR", pathlib.Path.home()/"models/UD-IQ2_XXS"))
    if not root.is_dir():
        _harness.skip("no GLM checkpoint")
    ck = _harness.Checks()
    tok, stops = loader.load_tokenizer(root)
    model = loader.load(root, mtp=False, vision=False, device_map="auto",
                        kv=dict(ctx=8192, slots=2, util=.95, chunk=128))
    ck("expert weights placed in mapped host memory",ops.host_pinned_bytes()>0 and "experts" in model.device_map.host)
    engine = Engine(model, max_num_seqs=2, max_model_len=8192, stop_token_ids=stops,
                    enforce_eager=True, gpu_memory_utilization=.95, prefill_chunk=128,
                    prefix_memory_ratio=0., batch_prefill=True)

    def prompt(text):
        rendered=tok.apply_chat_template([{"role":"user","content":text}],tokenize=False,
                                          add_generation_prompt=True,reasoning_effort="low")
        return tok.encode(rendered,add_special_tokens=False)

    params=SamplingParams(temperature=0,max_new_tokens=24)
    p=prompt("计算1+1，只回答结果。")
    r=engine.add(p,params);engine.run();baseline=list(r.out)
    ck("greedy text generation terminates",r.done and r.finish_reason=="stop" and "2" in tok.decode(r.out))
    engine.prefill_chunk=3
    r=engine.add(p,params);engine.run()
    ck("3-token chunks preserve generated tokens",r.out==baseline)
    engine.prefill_chunk=128
    r1=engine.add(p,params);r2=engine.add(prompt("计算2+3，只回答结果。"),params)
    engine.run()
    ck("concurrent requests finish independently",r1.done and r2.done and
       r1.finish_reason==r2.finish_reason=="stop" and "2" in tok.decode(r1.out) and "5" in tok.decode(r2.out))
    ck("batched prefill exercised",engine.n_batched_prefills>0)
    ck("all request slots and pages released",len(engine.slots.free)==2 and len(engine.blocks.free)==engine.blocks.total)
    # mmap-backed index/KV pools are addressed through noncontiguous physical pages.
    long=prompt("下面的数字是背景，请忽略："+"123 "*3750+"\n计算1+1，只回答结果。")
    r=engine.add(long,params);engine.run()
    ck("sparse selection beyond 2048 tokens works",len(long)>7000 and r.finish_reason=="stop" and "2" in tok.decode(r.out))
    print("memory allocated/reserved GiB",torch.cuda.memory_allocated()/2**30,torch.cuda.memory_reserved()/2**30)
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
