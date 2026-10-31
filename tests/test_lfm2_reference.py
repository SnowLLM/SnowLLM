# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import types

import torch

import _harness
from snowllm import ops
from snowllm.checkpoint.gguf.source import GGUFReader
from snowllm.checkpoint.gguf.lfm2 import config
from snowllm.models.lfm2.weights import load_gguf_weights, load_weights
from snowllm.engine.lfm2_runner import Lfm2Runner
from snowllm.engine.inputs import synthetic_batch


def reference(rd, cfg):
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeModel
    hc=Lfm2MoeConfig(**dict(cfg,rope_parameters=dict(rope_theta=cfg['rope_theta'],rope_type='default')))
    hc._attn_implementation='eager'
    with torch.device('meta'):
        hf=Lfm2MoeModel(hc)
    hf=hf.to_empty(device='cuda').to(dtype=torch.bfloat16).eval()
    inv,scale=hf.pos_emb.compute_default_rope_parameters(hc,device='cuda')
    hf.pos_emb.inv_freq=inv
    hf.pos_emb.original_inv_freq=inv.clone()
    hf.pos_emb.attention_scaling=scale
    def copy(dst,key):
        dst.copy_(rd.tensor(key,dst.dtype))
    copy(hf.embed_tokens.weight,'token_embd.weight')
    copy(hf.embedding_norm.weight,'token_embd_norm.weight')
    for i,layer in enumerate(hf.layers):
        p=f'blk.{i}.'
        copy(layer.operator_norm.weight,p+'attn_norm.weight')
        copy(layer.ffn_norm.weight,p+'ffn_norm.weight')
        if cfg['layer_types'][i]=='conv':
            for name in ('in_proj','out_proj'):
                copy(getattr(layer.conv,name).weight,p+f'shortconv.{name}.weight')
            layer.conv.conv.weight.copy_(rd.tensor(p+'shortconv.conv.weight')[:,None])
        else:
            for name,key in [('q_proj','q'),('k_proj','k'),('v_proj','v'),('out_proj','output')]:
                copy(getattr(layer.self_attn,name).weight,p+f'attn_{key}.weight')
            copy(layer.self_attn.q_layernorm.weight,p+'attn_q_norm.weight')
            copy(layer.self_attn.k_layernorm.weight,p+'attn_k_norm.weight')
        ff=layer.feed_forward
        if i<cfg['num_dense_layers']:
            for name,key in [('w1','gate'),('w3','up'),('w2','down')]:
                copy(getattr(ff,name).weight,p+f'ffn_{key}.weight')
        else:
            copy(ff.gate.weight,p+'ffn_gate_inp.weight')
            ff.expert_bias=rd.tensor(p+'exp_probs_b.bias',torch.float32)
            if hasattr(ff.experts,'gate_up_proj'):
                ff.experts.gate_up_proj.copy_(torch.cat(
                    [rd.tensor(p+f'ffn_{key}_exps.weight') for key in ('gate','up')],dim=1))
                copy(ff.experts.down_proj,p+'ffn_down_exps.weight')
            else:
                for key,name in [('gate','w1'),('up','w3'),('down','w2')]:
                    weights=rd.tensor(p+f'ffn_{key}_exps.weight')
                    for e in range(cfg['num_experts']):
                        getattr(ff.experts[e],name).weight.copy_(weights[e])
    return hf


@torch.no_grad()
def main():
    root=pathlib.Path(os.environ.get('SNOWLLM_LFM_DIR',pathlib.Path.home()/'models/LFM2.5-8B-A1B-Q4_K_M'))
    paths=sorted(root.glob('*.gguf'))
    if not paths:
        _harness.skip('no complete LFM checkpoint')
    path=paths[0]
    with GGUFReader(path) as rd:
        cfg=config(rd.gguf)
        cfg['num_hidden_layers']=3
        cfg['layer_types']=cfg['layer_types'][:3]
        model=load_gguf_weights(rd,cfg,range(3))
        runner=Lfm2Runner(model,32,32,max_num_seqs=2,max_prefill_tokens=64)
        hf=reference(rd,cfg)
        captured={}
        model.layers[2].attn.register_forward_hook(
            lambda mod,inp,out:captured.__setitem__('attention',inp[2].clone()))
        hf.layers[2].self_attn.register_forward_hook(
            lambda mod,inp,out:captured.__setitem__('hf_attention',out[0].clone()))
        def remember_moe(mod,inp,out):
            captured['moe_input']=inp[1].clone()
            captured['moe_output']=inp[2].clone()
        model.layers[2].mlp.register_forward_hook(remember_moe)
        ids=[[124894,35808,1530,9,25,229,9792,25677,23944],
             [124894,11253,5899,697,10869,8167,25,229,9792]]
        expected=hf(torch.tensor(ids,device='cuda'),use_cache=True)
        total,qmap=ops.prefill_q_plan([9,9])
        batch=synthetic_batch([9,9],32,input_ids=sum(ids,[]),state_indices=[0,1],
            positions=(list(range(9))*2)*3,
            block_tables=[[0]*32,[1]*32],is_prefill=True,
            slot_mapping=list(range(9))+list(range(16,25)),last_row=[8,17],
            total_q_blocks=total,q_block_map=qmap)
        runner._forward_eager(batch)
        want=expected.last_hidden_state.reshape(18,-1).float()
        error=((runner.x[:18].float()-want).norm()/want.norm()).item()
        print('GGUF batched prefill reference relL2',error,flush=True)
        assert error<.025
        next_ids=[1530,10869]
        want=hf(torch.tensor(next_ids,device='cuda')[:,None],
            past_key_values=expected.past_key_values,use_cache=True).last_hidden_state[:,0].float()
        decode=synthetic_batch([1,1],32,input_ids=next_ids,positions=[9]*6,
            state_indices=[0,1],seq_lens=[10,10],slot_mapping=[9,25],block_tables=[[0]*32,[1]*32])
        runner._forward_eager(decode)
        actual=captured['attention'].float()
        want=captured['hf_attention'].reshape(2,-1).float()
        error=((actual-want).norm()/want.norm()).item()
        print('GGUF batched cached attention reference relL2',error,flush=True)
        assert error<.025
        def float_router(self,x,expert_bias=None):
            logits=torch.nn.functional.linear(x.float(),self.weight.float())
            prob=logits.sigmoid()
            ids=(prob+expert_bias).topk(self.top_k,dim=-1).indices
            weights=prob.gather(1,ids)
            weights=weights/(weights.sum(-1,keepdim=True)+1e-6)
            return logits,weights*self.routed_scaling_factor,ids
        # Match FP32 routing and isolate expert math from upstream near-tie route changes.
        ff=hf.layers[2].feed_forward
        original_router=ff.gate.forward
        ff.gate.forward=types.MethodType(float_router,ff.gate)
        want=ff(captured['moe_input'][None])[0].float()
        ff.gate.forward=original_router
        error=((captured['moe_output'].float()-want).norm()/want.norm()).item()
        print('GGUF cached MoE at identical inputs reference relL2',error,flush=True)
        assert error<.025
        state=hf.state_dict()
        class ReferenceWeights:
            def has(self,key):
                return key.removeprefix('model.') in state
            def read(self,key):
                return state[key.removeprefix('model.')]
        dense=load_weights(ReferenceWeights(),cfg,range(3),False,False)
        runner=Lfm2Runner(dense,32,32,max_num_seqs=2,max_prefill_tokens=64)
        runner._forward_eager(batch)
        want=hf(torch.tensor(ids,device='cuda'),use_cache=False).last_hidden_state.reshape(18,-1).float()
        error=((runner.x[:18].float()-want).norm()/want.norm()).item()
        print('BF16 loader batched reference relL2',error,flush=True)
        assert error<.025
        ff.gate.weight.zero_()
        ff.expert_bias=torch.arange(32,device='cuda',dtype=torch.float32)
        ff.gate.forward=types.MethodType(float_router,ff.gate)
        router=ops.moe_shuffle_router(ff.gate.weight,ff.expert_bias)
        moe=model.layers[2].mlp
        for rows in (1,10,20,512):
            x=torch.randn(rows,2048,device='cuda',dtype=torch.bfloat16)*.05
            want=ff(x[None])[0].float()
            ws=torch.empty(ops.inference_moe_workspace_bytes(rows),device='cuda',dtype=torch.uint8)
            out=torch.empty_like(x)
            previous=None
            for poison in (0,255,127):
                ws.fill_(poison)
                ops.inference_moe(x,router,moe.gate_up_w,moe.down_w,None,None,out,ws)
                error=((out.float()-want).norm()/want.norm()).item()
                print('last routed expert reference',rows,poison,error,flush=True)
                assert error<.025
                if previous is not None:
                    torch.testing.assert_close(out,previous,atol=0,rtol=0)
                previous=out.clone()


if __name__=='__main__':
    main()
