# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import sys

import _harness
from snowllm.checkpoint import loader
from snowllm.engine import Engine, SamplingParams


def main():
    root=pathlib.Path(os.environ.get('SNOWLLM_LFM_DIR',pathlib.Path.home()/'models/LFM2.5-8B-A1B-Q4_K_M'))
    if not any(root.glob('*.gguf')):
        _harness.skip('no complete LFM checkpoint')
    ck=_harness.Checks()
    tok,stops=loader.load_tokenizer(root)
    model=loader.load(root,mtp=False,vision=False,kv=dict(ctx=2048,slots=2,util=.9,chunk=64))
    engine=Engine(model,max_num_seqs=2,max_model_len=2048,stop_token_ids=stops,
        gpu_memory_utilization=.9,prefill_chunk=64,prefix_memory_ratio=.05,batch_prefill=True)
    def prompt(text):
        rendered=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,
            add_generation_prompt=True)
        return tok.encode(rendered,add_special_tokens=False)
    params=SamplingParams(temperature=0,max_new_tokens=256)
    p=prompt('What is the capital of France? Answer briefly.')
    a=engine.add(p,params);engine.run()
    print('LFM answer:',tok.decode(a.out),flush=True)
    ck('knowledge answer',a.done and 'Paris' in tok.decode(a.out))
    baseline=list(a.out)
    engine.prefill_chunk=3
    b=engine.add(p,params);engine.run()
    ck('prefix reuse preserves greedy output',b.out==baseline)
    # A different prefix forces short-convolution state through every prefill chunk.
    p2=prompt('Calculate 2+3. Answer only the result.')
    b=engine.add(p2,params);engine.run();chunked=list(b.out)
    engine.prefill_chunk=64
    c=engine.add(p2,params);engine.run()
    ck('chunked prefill first 32 greedy tokens',c.out[:32]==chunked[:32])
    ck('chunked arithmetic answer','5' in tok.decode(chunked).split('</think>')[-1])
    a=engine.add(p,params);b=engine.add(p2,params);engine.run()
    ck('two requests finish',a.done and b.done)
    capital=tok.decode(a.out).split('</think>')[-1]
    arithmetic=tok.decode(b.out).split('</think>')[-1]
    ck('batched capital answer','Paris' in capital)
    ck('batched arithmetic answer',arithmetic.strip().startswith('5'))
    ck('batched requests reach EOS',a.finish_reason==b.finish_reason=='stop')
    return ck.done()


if __name__=='__main__':
    sys.exit(main())
