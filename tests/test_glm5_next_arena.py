# SPDX-License-Identifier: Apache-2.0
"""Static GLM activations: exact eager parity and zero CUDA allocations from the first step."""
import gc
import hashlib
import os
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


class NoCudaAllocation(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        if name in ("aten::empty", "aten::empty_like", "aten::empty_strided",
                    "aten::clone", "aten::_to_copy", "aten::zeros", "aten::full"):
            device = kwargs.get("device")
            if device is None and args and isinstance(args[0],torch.Tensor):
                device = args[0].device
            if device is not None and torch.device(device).type == "cuda":
                raise AssertionError(f"runtime CUDA allocation: {func}")
        return func(*args,**kwargs)


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    root = Path(os.environ.get("SNOWLLM_GLM_DIR",Path.home()/"models/UD-IQ2_XXS"))
    if not root.is_dir():
        _harness.skip("no GLM checkpoint")
    torch.set_num_threads(4)
    ck = _harness.Checks()
    tok, _ = loader.load_tokenizer(root)
    model = loader.load(root,mtp=True,vision=False,device_map="auto",
                        kv=dict(ctx=1024,slots=2,util=.95,chunk=33,num_spec=2))
    prompts = [tok.encode(tok.apply_chat_template([dict(role="user",content=s)],
               tokenize=False,add_generation_prompt=True,reasoning_effort="low"),
               add_special_tokens=False)
               for s in ("计算1+1，只回答结果。","从1开始列出前20个正整数，用逗号分隔。")]
    for depth, chunk, sampled in ((0,33,False),(0,3,False),(2,33,False),(2,33,True)):
        expected = None
        for static in (False,True):
            engine = Engine(model,max_num_seqs=2,max_model_len=1024,num_spec=depth,
                stop_token_ids=(),seed=71,enforce_eager=True,prefill_chunk=chunk,
                gpu_memory_utilization=.95,plan_activations=static,
                prefix_memory_ratio=0.,account=False,batch_prefill=True)
            r = engine.runner
            logits, hidden = [], []
            hooks = [model.lm_head.register_forward_hook(lambda _m,_a,out: logits.append(digest(out))),
                     model.register_forward_hook(lambda _m,_a,_out: hidden.append(digest(_a[0].last_hidden)))]
            params = SamplingParams(temperature=.7 if sampled else 0,top_k=32,top_p=.9,
                                    repetition_penalty=1.1 if sampled else 1.,max_new_tokens=16)
            requests = [engine.add(p,params) for p in prompts]
            forced = set()

            def run():
                while not all(q.done for q in requests):
                    if depth:
                        for i,q in enumerate(requests):
                            if q.prefilled and not q.done and i not in forced:
                                q.drafts = [0]*depth
                                forced.add(i)
                    engine.step()
                    if static and depth:
                        assert digest(r.last_hidden) == hidden[-1], "MTP overwrote trunk hidden"

            tag = f"depth={depth} chunk={chunk} sampled={sampled}"
            if static:
                ws, arena = r.ws.data_ptr(), r.arena.buf.data_ptr()
                assert all(not p.overlaps() for p in r.arena.planner.plans.values())
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                with NoCudaAllocation():
                    run()
                ck(tag+" zero CUDA allocations",torch.cuda.memory_stats()["allocation.all.allocated"]==before)
                ck(tag+" fixed storage",r.ws.data_ptr()==ws and r.arena.buf.data_ptr()==arena)
                ck(tag+" every placement planned",r.arena.planner.served>0 and r.arena.planner.eager==0)
            else:
                run()
            for hook in hooks:
                hook.remove()
            result = ([list(q.out) for q in requests],logits,hidden,
                      list(engine.spec.accept_hist) if engine.spec else [])
            if static:
                ck(tag+" token parity",result[0]==expected[0])
                ck(tag+" full logits parity",result[1]==expected[1])
                ck(tag+" trunk hidden parity",result[2]==expected[2])
                ck(tag+" MTP acceptance parity",result[3]==expected[3])
                try:
                    r.workspace(r.ws.numel()+1)
                except RuntimeError:
                    ck(tag+" workspace growth rejected",True)
                else:
                    ck(tag+" workspace growth rejected",False)
            else:
                expected = result
            del engine,r,requests,hooks
            gc.collect()
            torch.cuda.empty_cache()
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
