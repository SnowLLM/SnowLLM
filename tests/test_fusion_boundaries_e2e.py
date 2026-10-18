# SPDX-License-Identifier: Apache-2.0
"""Production fusion switches removed: independent old pipelines versus native execution."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch

import _harness
import _fusion_pipeline_reference as reference
from test_sampling_fusion import legacy_sample
import snowllm.ops.sampling as sampling
from snowllm import ops
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,
                        default=Path.home() / 'models/Qwen3.6-35B-A3B-UD-Q4_K_XL')
    parser.add_argument('--draft',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--kv-int8',action='store_true')
    args = parser.parse_args()
    draft = args.draft or args.model/'dflash'
    if not args.model.is_dir() or not draft.exists():
        _harness.skip('no target/draft checkpoint')
    torch.set_num_threads(4)
    ck = _harness.Checks()
    tok,_ = loader.load_tokenizer(args.model)
    model = loader.load(args.model,mtp=False,vision=False,device_map='auto',
                        kv=dict(ctx=1024,slots=2,util=.95,chunk=128))
    d = ops.dflash
    native = ops.qk_norm_rope_cache,d.gate_up_swiglu,ops.sample
    native_sampling = sampling.sample
    short = [tok.encode(p) for p in ('The capital of France is','Explain why the sky is blue.')]
    long = tok.encode('The history of computing begins with the abacus. '*35)
    ck('long prompt exercises multiple prefill chunks',128<len(long)<900)
    result = dict(model=str(args.model),draft=str(draft),kv_int8=args.kv_int8,runs=[])

    def cache_tensors(e):
        # Both main attention pools and draft pools, including INT8 scales.
        for layer in model.layers:
            a=getattr(layer,'attn',getattr(layer,'attention',None))
            if a is None:continue
            for name in ('kv','kv_scale'):
                pair=getattr(a,name,None)
                if pair is not None:
                    for t in pair:yield t
        for pair in e.spec.draft.pools:
            yield from pair

    def snapshot(e):
        return [hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                for t in cache_tensors(e)]

    try:
        for eager in (True,False):
            expected = None
            for before in (True,False):
                ops.qk_norm_rope_cache = reference.qk_norm_rope_cache if before else native[0]
                d.gate_up_swiglu = reference.gate_up_swiglu if before else native[1]
                ops.sample = sampling.sample = legacy_sample if before else native_sampling
                e = Engine(model,num_kv_blocks=256,max_num_seqs=2,max_model_len=1024,stop_token_ids=(),seed=71,
                           enforce_eager=eager,kv_int8=args.kv_int8,gpu_memory_utilization=.95,prefill_chunk=128,
                           prefix_memory_ratio=0.,account=False,dflash_path=str(draft),dflash_block=8)
                for t in cache_tensors(e):t.zero_()
                outputs,states = [],[]
                for label,prompts,mixed,force in (
                    ('single',short[:1],False,False),
                    ('concurrent',short,False,True),
                    ('ragged/sample/reuse',[short[1],long],True,False),
                ):
                    rr = [e.add(p,SamplingParams(temperature=.8 if mixed and i else 0.,
                                                top_k=17,top_p=.9,repetition_penalty=1.1,
                                                presence_penalty=.2,max_new_tokens=24))
                          for i,p in enumerate(prompts)]
                    forced = set()
                    while not all(r.done for r in rr):
                        if force:
                            for i,r in enumerate(rr):
                                if r.prefilled and not r.done and i not in forced:
                                    r.drafts = [0]*7
                                    forced.add(i)
                        e.step()
                    outputs.append([list(r.out) for r in rr])
                    states.append(snapshot(e))
                    pools = e.blocks.values() if isinstance(e.blocks,dict) else (e.blocks,)
                    ck(f'{label} eager={eager} legacy={before} cleanup',
                       len(e.slots.free)==2 and all(len(p.free)==p.total for p in pools)
                       and len(e.spec.blocks.free)==e.spec.blocks.total)
                    if force:
                        ck(f'{label} eager={eager} legacy={before} forced rejection',
                           bool(forced) and e.spec.accept_hist[0]>0)
                row = dict(eager=eager,legacy=before,tokens=outputs,cache_sha256=states,
                           acceptance=list(e.spec.accept_hist),
                           target_replays=e.runner.n_graph_replays,
                           draft_replays=e.runner.n_draft_replays,
                           captured_draft_batches=sorted(e.spec.graphs),
                           draft_pool_blocks=e.spec.blocks.total,
                           fc_storage='KQuant' if hasattr(e.spec.draft.w['fc'],'quant') else 'BF16')
                if before:
                    expected = row
                else:
                    ck(f'eager={eager} complete token parity',outputs==expected['tokens'])
                    ck(f'eager={eager} complete main/draft KV parity',states==expected['cache_sha256'])
                    ck(f'eager={eager} acceptance parity',row['acceptance']==expected['acceptance'])
                if not eager:
                    ck(f'legacy={before} real target/draft graph replay',
                       row['target_replays']>0 and row['draft_replays']>0 and bool(e.spec.graphs))
                result['runs'].append(row)
                if args.out:
                    args.out.write_text(json.dumps(result,indent=2)+'\n')
                print('DONE',dict(eager=eager,legacy=before,fc_storage=row['fc_storage'],
                                  target_replays=row['target_replays'],draft_replays=row['draft_replays']),flush=True)
                del rr,e
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        ops.qk_norm_rope_cache,d.gate_up_swiglu,ops.sample = native
        sampling.sample = native_sampling
    result['passed'] = bool(ck.ok)
    if args.out:
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    return ck.done()


if __name__ == '__main__':
    sys.exit(main())
