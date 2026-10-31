# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import sys

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine,SamplingParams


def main():
    root=pathlib.Path(os.environ.get('SNOWLLM_LFM_DIR',pathlib.Path.home()/'models/LFM2.5-8B-A1B-Q4_K_M'))
    draft=pathlib.Path(os.environ.get('SNOWLLM_LFM_DSPARK_DIR',pathlib.Path.home()/'models/LFM2.5-8B-A1B-DSpark-Q8_0'))
    if not any(root.glob('*.gguf')) or not any(draft.glob('*.gguf')):
        _harness.skip('no complete LFM target/draft checkpoint')
    tok,stops=loader.load_tokenizer(root)
    model=loader.load(root,mtp=False,vision=False,kv=dict(ctx=4096,slots=2,util=.9,chunk=64))
    engine=Engine(model,max_num_seqs=2,max_model_len=4096,stop_token_ids=stops,
        prefill_chunk=64,dflash_path=str(draft),dflash_block=9,prefix_memory_ratio=.04,
        batch_prefill=True,dspark_adaptive=True)
    def prompt(text):
        return tok.encode(tok.apply_chat_template([dict(role='user',content=text)],
            tokenize=False,add_generation_prompt=True),add_special_tokens=False)
    params=SamplingParams(temperature=0,max_new_tokens=256)
    ck=_harness.Checks()
    p=prompt('What is the capital of France? Answer briefly.')
    q=prompt('Calculate 2+3. Answer briefly.')
    a,b=engine.add(p,params),engine.add(q,params)
    engine.run()
    ck('batched knowledge answer','Paris' in tok.decode(a.out).split('</think>')[-1])
    ck('batched arithmetic answer','5' in tok.decode(b.out).split('</think>')[-1])
    ck('both requests reach EOS',a.finish_reason==b.finish_reason=='stop')
    ck('target graph replay',engine.runner.n_graph_replays>0)
    ck('draft graph replay',engine.runner.n_draft_replays>0)
    ck('multiple candidates accepted',engine.stats().accounting['accept_len']>1)
    before=engine.stats().cache_hits
    engine.prefill_chunk=3
    a=engine.add(p,params);engine.run()
    ck('prefix snapshots reused',engine.stats().cache_hits>before and a.cached_tokens>0)
    ck('resumed convolution state gives correct answer','Paris' in tok.decode(a.out).split('</think>')[-1])
    print('DSpark stats:',engine.stats(),flush=True)
    print('Acceptance histogram:',engine.dflash.accept_hist,flush=True)
    return ck.done()


if __name__=='__main__':
    sys.exit(main())
