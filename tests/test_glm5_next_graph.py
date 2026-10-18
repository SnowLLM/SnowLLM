# SPDX-License-Identifier: Apache-2.0
"""Actual full-model GLM graphs, GPU MTP metadata, state parity and prefix reuse."""
import gc
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams
import snowllm.engine as engine_module
from snowllm.engine.glm5_next_runner import Glm5NextRunner
from snowllm.engine.spec_decode import SpecDecoder
from snowllm.engine.forward_context import DecodeShape
from snowllm.models.glm5_next.layers import LinearAttention, SparseAttention
from snowllm.ops.glm5_next import sparse_cache_gather
from test_glm5_next_arena import NoCudaAllocation


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def state(model,runner,requests):
    fields = {}
    for i,r in enumerate(requests):
        if r.done or not r.prefilled:
            continue
        n,slot = r.num_cached,r.state_head
        fields[f"request{i}.cached"] = n
        for j,layer in enumerate(model.layers):
            a = layer.attention
            if isinstance(a,LinearAttention):
                fields[f"request{i}.layer{j}.history"] = digest(a.history[slot])
                fields[f"request{i}.layer{j}.state"] = digest(a.state[slot])
            else:
                for name,value in zip(("kv","index","carry_k","carry_g"),sparse_state(a,r,n,slot,runner.block_size)):
                    fields[f"request{i}.layer{j}.{name}"] = value
        if runner.num_spec:
            for name,value in zip(("kv","index","carry_k","carry_g"),sparse_state(model.mtp.attention,r,n,r.slot,runner.block_size)):
                fields[f"request{i}.mtp.{name}"] = value
    return fields


def sparse_state(a,r,n,slot,bs):
    cells = torch.tensor([r.blocks[p//bs]*bs+p%bs for p in range(n)],device="cuda",dtype=torch.int32)
    pools = torch.tensor([r.blocks[p//(bs//4)]*(bs//4)+p%(bs//4) for p in range(n//4)],device="cuda")
    logical = sparse_cache_gather(a.kv,cells,bs)
    return [digest(logical),digest(a.index[pools]),
            digest(a.carry_k[slot,:n%4]),digest(a.carry_g[slot,:n%4])]


def reference_runner():
    root = os.environ.get("SNOWLLM_GLM_REFERENCE")
    if not root:
        return Glm5NextRunner
    def read(name,filename):
        spec = importlib.util.spec_from_file_location("snowllm.engine."+name,Path(root)/filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    old_spec = read("_glm_spec_reference","glm5_next_spec.py").Glm5NextSpecDecoder
    old_runner = read("_glm_runner_reference","glm5_next_runner.py").Glm5NextRunner
    old_runner.spec_decoder_type = old_spec
    return old_runner


def main():
    root = Path(os.environ.get("SNOWLLM_GLM_DIR",Path.home()/"models/UD-IQ2_XXS"))
    if not root.is_dir():
        _harness.skip("no GLM checkpoint")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    ck = _harness.Checks()
    tok,_ = loader.load_tokenizer(root)
    model = loader.load(root,mtp=True,vision=False,device_map="auto",
                        kv=dict(ctx=2048,slots=2,util=.95,chunk=129,num_spec=2,reserve=1<<30))
    prompts = [tok.encode(tok.apply_chat_template([dict(role="user",content=s)],tokenize=False,
        add_generation_prompt=True,reasoning_effort="low"),add_special_tokens=False)
        for s in ("从1开始列出前20个正整数，用逗号分隔。","计算1+1，只回答结果。")]
    old_runner = reference_runner()

    def run(depth,graphs,sampled=False,reference=False):
        cls = old_runner if reference else Glm5NextRunner
        with patch.object(engine_module,"_runner_for",return_value=cls):
            e = Engine(model,max_num_seqs=2,max_model_len=2048,num_spec=depth,stop_token_ids=(),
                gpu_memory_utilization=.95,prefill_chunk=129,enforce_eager=not graphs,
                prefix_memory_ratio=0.,account=False,seed=71,preempt=False)
        r = e.runner
        if not reference and depth:
            ck("GLM shares speculative verification",type(e.spec).verify_step is SpecDecoder.verify_step)
        params = SamplingParams(temperature=.7 if sampled else 0,top_k=32,top_p=.9,
                                repetition_penalty=1.1 if sampled else 1.,max_new_tokens=24)
        qs = [e.add(p,params) for p in prompts]
        forced,logits,snapshot = set(),[],None
        forward = r.forward
        def watched(b):
            out = forward(b)
            logits.append(digest(out))
            return out
        r.forward = watched
        phase = False
        original_list = torch.Tensor.tolist
        def checked_list(tensor):
            if phase and tensor.is_cuda:
                raise AssertionError("GPU metadata read back before drafting")
            return original_list(tensor)
        if depth and not reference:
            verify,propose = e.spec.verify_step,e.spec.propose_first
            def verifying(*args):
                nonlocal phase
                phase = True
                try:
                    return verify(*args)
                finally:
                    phase = False
            def proposing(*args):
                nonlocal phase
                out = propose(*args)
                phase = False
                return out
            e.spec.verify_step,e.spec.propose_first = verifying,proposing
        def step():
            for i,q in enumerate(qs):
                if depth and q.prefilled and not q.done and i not in forced:
                    q.drafts = [0]*depth
                    forced.add(i)
            with patch.object(torch.Tensor,"tolist",checked_list):
                e.step()
        while not all(q.prefilled or q.done for q in qs):
            step()
        # Snapshots are outside the allocation guard and cover all live state fields.
        snapshot = state(model,r,qs)
        if reference:
            while not all(q.done for q in qs):
                step()
        else:
            with NoCudaAllocation():
                while not all(q.done for q in qs):
                    step()
        result = ([q.out for q in qs],logits,snapshot,list(e.spec.accept_hist) if depth else [])
        if graphs:
            ck(f"depth={depth} sampled={sampled}: real main graph replay",r.n_graph_replays>0)
            if depth:
                ck(f"sampled={sampled}: real MTP graph replay",r.n_draft_replays>0)
        ck(f"depth={depth}: slots/pages recovered",len(e.slots.free)==2 and len(e.blocks.free)==e.blocks.total)
        return result

    for depth,sampled in ((0,False),(2,False),(2,True),(3,False)):
        comparison = 'previous' if old_runner is not Glm5NextRunner else 'repeat eager'
        baseline = run(depth,False,sampled,reference=old_runner is not Glm5NextRunner)
        gc.collect();torch.cuda.empty_cache()
        eager = run(depth,False,sampled)
        ck(f"depth={depth} sampled={sampled}: {comparison} tokens",eager[0]==baseline[0])
        ck(f"depth={depth} sampled={sampled}: {comparison} main logits",eager[1]==baseline[1])
        ck(f"depth={depth} sampled={sampled}: {comparison} live state",eager[2]==baseline[2],
           str([key for key in baseline[2] if baseline[2][key]!=eager[2].get(key)])[:1500])
        ck(f"depth={depth} sampled={sampled}: {comparison} acceptance",eager[3]==baseline[3])
        del baseline
        gc.collect();torch.cuda.empty_cache()
        replay = run(depth,True,sampled)
        ck(f"depth={depth} sampled={sampled}: graph tokens",replay[0]==eager[0])
        ck(f"depth={depth} sampled={sampled}: graph full logits",replay[1]==eager[1])
        ck(f"depth={depth} sampled={sampled}: graph live state",replay[2]==eager[2])
        ck(f"depth={depth} sampled={sampled}: graph acceptance",replay[3]==eager[3])
        del eager,replay
        gc.collect();torch.cuda.empty_cache()
    e = Engine(model,max_num_seqs=2,max_model_len=2048,num_spec=2,stop_token_ids=(),
        gpu_memory_utilization=.95,prefill_chunk=129,enforce_eager=False,
        prefix_memory_ratio=.5,account=False,seed=71,preempt=False)
    prefix = tok.encode("The expedition recorded the river width in fathoms. "*150)[:768]
    long_prompts = [prefix+tok.encode(s) for s in (" What unit was used?"," Which feature was measured?")]
    def ask(prompt):
        q = e.add(prompt,SamplingParams(temperature=0,max_new_tokens=12))
        e.run()
        return q
    cache,e.cache = e.cache,None
    expected = [ask(p).out for p in long_prompts]
    e.cache = cache
    ck("graph prefix storage fits",cache is not None)
    ask(long_prompts[0])
    warm = ask(long_prompts[1])
    ck("MTP graph restores a divergent prefix",warm.cached_tokens>=512 and warm.out==expected[1])
    qs = [e.add(p,SamplingParams(temperature=0,max_new_tokens=12)) for p in long_prompts]
    e.run()
    ck("concurrent prefix graph outputs",[q.out for q in qs]==expected and all(q.cached_tokens>=512 for q in qs))
    ck("prefix generation replays both model graphs",e.runner.n_graph_replays>0 and e.runner.n_draft_replays>0)
    while cache.evict():
        pass
    ck("prefix graph releases pages and slots",len(e.blocks.free)==e.blocks.total and len(e.slots.free)==2)
    e.runner.graphs.clear()
    with patch.object(e.runner,"_graph_room",return_value=False):
        ck("graph ceiling allows eager fallback",e.runner.capture([DecodeShape(1,3,True)])==[])
    with patch.object(e.runner,"_capture",side_effect=torch.OutOfMemoryError("simulated graph OOM")):
        ck("capture OOM allows eager fallback",e.runner.capture([DecodeShape(1,3,True)])==[] and e.runner.budget.full)
    before = e.runner.n_graph_replays
    fallback = ask(long_prompts[0])
    ck("refused graphs preserve generation",fallback.out==expected[0] and e.runner.n_graph_replays==before)
    return ck.done()


if __name__=="__main__":
    sys.exit(main())
