# SPDX-License-Identifier: Apache-2.0
"""Real NextN weights: greedy parity, chunking, rejection, concurrency and EOS."""
import gc
import json
from pathlib import Path
import sys
import time

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine,SamplingParams


def main():
    root=_harness.checkpoint_path('UD-IQ2_XXS', env='SNOWLLM_GLM_DIR')
    if not root.is_dir():
        _harness.skip("no GLM checkpoint")
    ck=_harness.Checks()
    tok,stops=loader.load_tokenizer(root)
    model=loader.load(root,mtp=True,vision=False,device_map="auto",
                      kv=dict(ctx=1024,slots=2,util=.95,chunk=128,num_spec=3))
    ck("appended NextN block loaded",model.mtp is not None and model.mtp.fc.k==8192)
    ck("MTP uses existing k-quant expert path",model.mtp.mlp.gate_up.fmt==2 and model.mtp.mlp.down.fmt==3)
    prompt=lambda text: tok.encode(tok.apply_chat_template([{"role":"user","content":text}],
        tokenize=False,add_generation_prompt=True,reasoning_effort="low"),add_special_tokens=False)
    prompts=[prompt("用 Python 写一个计算斐波那契数列的函数，并解释时间复杂度。"),
             prompt("从1开始列出前20个正整数，用逗号分隔。")]
    params=SamplingParams(temperature=0,max_new_tokens=128)
    results=[]

    def engine(depth,chunk=128,ctx=1024):
        gc.collect();torch.cuda.empty_cache()
        return Engine(model,max_num_seqs=2,max_model_len=ctx,num_spec=depth,stop_token_ids=stops,
                      seed=7,enforce_eager=True,gpu_memory_utilization=.95,prefill_chunk=chunk,
                      prefix_memory_ratio=0.,account=False)

    def run(depth,chunk=128,force=False):
        e=engine(depth,chunk)
        outputs=[]
        for p in prompts:
            r=e.add(p,params)
            while not r.prefilled and not r.done: e.step()
            if force and not r.done:
                r.drafts=[0]*depth
            n0=len(r.out);steps=0
            torch.cuda.synchronize();t0=time.perf_counter()
            while not r.done:
                e.step();steps+=1
            torch.cuda.synchronize();dt=time.perf_counter()-t0
            outputs.append(list(r.out))
            row=dict(depth=depth,chunk=chunk,forced_rejection=force,tokens=len(r.out)-n0,
                     steps=steps,decode_tok_s=(len(r.out)-n0)/dt,
                     tokens_per_step=(len(r.out)-n0)/steps,accept_hist=list(e.spec.accept_hist) if e.spec else [])
            results.append(row);print(json.dumps(row),flush=True)
        ck(f"depth {depth}, chunk {chunk} releases slots/pages",
           len(e.slots.free)==2 and len(e.blocks.free)==e.blocks.total)
        return outputs

    baseline=run(0)
    for depth in (1,2,3):
        out=run(depth)
        for i,(a,b) in enumerate(zip(baseline,out)):
            mismatch=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            ck(f"MTP depth {depth}, prompt {i} token parity",a==b,f"first difference {mismatch}")
    ck("forced rejection preserves tokens",run(3,force=True)==baseline)
    chunked_baseline=run(0,chunk=3)
    ck("chunked MTP matches chunked plain decode",run(2,chunk=3)==chunked_baseline)

    e=engine(2)
    r1,r2=[e.add(p,params) for p in prompts]
    while not (r1.prefilled and r2.prefilled):e.step()
    for _ in range(3):e.step()
    preempted=e._preempt()
    e.run()
    ck("concurrent MTP requests preserve tokens",[r1.out,r2.out]==baseline)
    ck("MTP preemption replay preserves state",preempted is not None and e.n_preemptions==1)
    short=e.add(prompt("计算1+1，只回答结果。"),SamplingParams(temperature=0,max_new_tokens=32));e.run()
    ck("MTP EOS stops inside accepted block",short.finish_reason=="stop" and "2" in tok.decode(short.out))
    # End of context must safely fall back to plain decode without indexing past pages.
    e=engine(2,ctx=len(prompts[0])+12)
    r=e.add(prompts[0],SamplingParams(temperature=0,max_new_tokens=12));e.run()
    ck("context boundary terminates safely",r.done and r.out==baseline[0][:len(r.out)])
    long=prompt("以下数字是背景，请忽略："+"123 "*1200+"\n计算1+1，只回答结果。")
    long_outputs=[]
    for depth in (0,2):
        e=engine(depth,ctx=4096)
        r=e.add(long,SamplingParams(temperature=0,max_new_tokens=24));e.run()
        long_outputs.append(list(r.out))
    ck("MTP beyond sparse top-k budget",len(long)>2048 and long_outputs[0]==long_outputs[1]
       and "2" in tok.decode(long_outputs[1]))
    Path("/tmp/snowllm-glm53-mtp-bench.json").write_text(json.dumps(results,indent=2))
    return ck.done()


if __name__=="__main__":
    sys.exit(main())
