# SPDX-License-Identifier: Apache-2.0
import torch

from ... import ops, term
from ..._capi import GEO_LFM25_8B_A1B, SnowLLMError, build_geometry
from ...checkpoint.gguf.source import KQUANT_ID
from .. import placement
from ..geometry import ModelGeometry
from ..projection import pack_bf16, pack_kquant
from ..embedding import VocabEmbedding
from ..qwen3_5.layers import FusedMoE, LMHead
from .layers import Attention, DenseMLP, Layer, Lfm2MoeForCausalLM, Projection, ShortConv


def validate_config(c):
    ops.select_geometry(GEO_LFM25_8B_A1B)
    g = build_geometry()
    expected = dict(hidden_size=g.hidden,vocab_size=g.vocab_size,
        num_attention_heads=g.num_heads,num_key_value_heads=g.num_kv_heads,
        num_experts=g.moe_num_experts,num_experts_per_tok=g.moe_topk,
        moe_intermediate_size=g.moe_inter,intermediate_size=7168,num_dense_layers=2,
        conv_L_cache=3,conv_bias=False,norm_topk_prob=True,use_expert_bias=True,
        routed_scaling_factor=1.0,tie_word_embeddings=True)
    bad = [f"{k}={c.get(k)!r} (want {v!r})" for k,v in expected.items() if c.get(k)!=v]
    types = c.get('layer_types',[])
    if len(types)!=c['num_hidden_layers'] or not types or any(t not in ('conv','full_attention') for t in types):
        bad.append('layer_types must describe each conv or attention layer')
    if c.get('num_dense_layers',0)>len(types):
        bad.append('num_dense_layers exceeds trunk layers')
    if c.get('quantization_config'):
        bad.append('quantized safetensors are not supported; use BF16 or GGUF')
    if bad:
        raise SnowLLMError('unsupported LFM geometry: '+', '.join(bad))


def geometry(c):
    return ModelGeometry(c['hidden_size'],c['num_attention_heads'],c['num_key_value_heads'],
        c['hidden_size']//c['num_attention_heads'],c['vocab_size'],c['num_hidden_layers'],
        False,0,0,0,0,3,c['num_experts'],c['moe_intermediate_size'],c['num_experts_per_tok'])


def _finish(c,layers,embed,norm,head):
    cfg = dict(c)
    cfg['rope_theta'] = (c.get('rope_parameters') or {}).get('rope_theta',c.get('rope_theta',5000000.0))
    return Lfm2MoeForCausalLM(cfg,geometry(c),layers,embed,norm,head)


def _check_layers(cfg,want):
    if tuple(want)!=tuple(range(cfg['num_hidden_layers'])):
        raise SnowLLMError('LFM serving requires all trunk layers')


def _dense(t):
    weight,n,k,_ = pack_bf16(t.to(torch.bfloat16))
    return Projection(n,k,dense=weight)


def load_weights(w,cfg,want,fp8,with_mtp,device_map=None):
    _check_layers(cfg,want)
    if fp8:
        raise SnowLLMError('LFM safetensors requires BF16 weights')
    read = lambda key: w.read(key).to(torch.bfloat16)
    embed_weight=read('model.embed_tokens.weight')
    embed=VocabEmbedding(embed_weight)
    with placement.place(device_map,'head'):
        head=LMHead(ops.lm_head_shuffle_weight(embed_weight))
    g,layers=geometry(cfg),[]
    for i in want:
        p=f'model.layers.{i}.'
        with placement.place(device_map,'dense',i):
            proj=lambda name:_dense(read(p+name+'.weight'))
            if cfg['layer_types'][i]=='conv':
                attn=ShortConv(proj('conv.in_proj'),read(p+'conv.conv.weight').reshape(g.hidden,3),proj('conv.out_proj'))
            else:
                attn=Attention(*(proj('self_attn.'+n+'_proj') for n in ('q','k','v','out')),
                    read(p+'self_attn.q_layernorm.weight'),read(p+'self_attn.k_layernorm.weight'),g)
            mlp=(DenseMLP(*(proj('feed_forward.'+n) for n in ('w1','w3','w2')))
                 if i<cfg['num_dense_layers'] else None)
        if mlp is None:
            router=ops.moe_shuffle_router(read(p+'feed_forward.gate.weight'),
                w.read(p+'feed_forward.expert_bias').to(torch.float32))
            with placement.place(device_map,'experts',i):
                key=p+'feed_forward.experts.'
                if w.has(key+'gate_up_proj'):
                    gu=read(key+'gate_up_proj')
                    gate,up=gu.chunk(2,dim=1)
                    down=read(key+'down_proj')
                else:
                    gate=torch.stack([read(key+f'{e}.w1.weight') for e in range(g.moe_num_experts)])
                    up=torch.stack([read(key+f'{e}.w3.weight') for e in range(g.moe_num_experts)])
                    down=torch.stack([read(key+f'{e}.w2.weight') for e in range(g.moe_num_experts)])
                mlp=FusedMoE(router,ops.moe_shuffle_gate_up(gate.contiguous(),up.contiguous()),ops.moe_shuffle_down(down))
        layers.append(Layer(attn,mlp,read(p+'operator_norm.weight'),read(p+'ffn_norm.weight')))
    model=_finish(cfg,layers,embed,read('model.embedding_norm.weight'),head)
    model.device_map=device_map
    return model


def load_gguf_weights(rd,cfg,want,device_map=None,mtp=False):
    _check_layers(cfg,want)
    validate_config(cfg)
    g,layers=geometry(cfg),[]
    def proj(name):
        t=rd.gguf[name]
        fmt=KQUANT_ID.get(t.quant.name)
        if fmt is None:
            return _dense(rd.tensor(name))
        if fmt not in (4,6,8):
            raise SnowLLMError(f'{name}: LFM projections support Q4_K, Q6_K, Q8_0 and BF16')
        n,k=t.shape
        return Projection(n,k,quant=pack_kquant(rd.raw(name),fmt,n,k))
    t=rd.gguf['token_embd.weight']
    embed=VocabEmbedding(rd.raw(t.name).view(g.vocab_size,-1),4) if t.quant.name=='Q4_K' else VocabEmbedding(rd.tensor(t.name))
    with placement.place(device_map,'head'):
        name='output.weight' if 'output.weight' in rd.gguf else t.name
        fmt=KQUANT_ID.get(rd.gguf[name].quant.name)
        head=LMHead(ops.lm_head_kquant_shuffle_weight(rd.raw(name),fmt) if fmt is not None
                    else ops.lm_head_shuffle_weight(rd.tensor(name)))
    need=max((sum(rd.gguf[f'blk.{i}.ffn_{n}_exps.weight'].nbytes for n in ('gate','up'))
              for i in want if i>=cfg['num_dense_layers']),default=0)
    rd.reserve_transient(need)
    for i in want:
        p=f'blk.{i}.'
        with placement.place(device_map,'dense',i):
            if cfg['layer_types'][i]=='conv':
                attn=ShortConv(proj(p+'shortconv.in_proj.weight'),rd.tensor(p+'shortconv.conv.weight'),proj(p+'shortconv.out_proj.weight'))
            else:
                attn=Attention(*(proj(p+'attn_'+n+'.weight') for n in ('q','k','v','output')),
                    rd.tensor(p+'attn_q_norm.weight'),rd.tensor(p+'attn_k_norm.weight'),g)
            mlp=(DenseMLP(*(proj(p+'ffn_'+n+'.weight') for n in ('gate','up','down')))
                 if i<cfg['num_dense_layers'] else None)
        if mlp is None:
            router=ops.moe_shuffle_router(rd.tensor(p+'ffn_gate_inp.weight'),rd.tensor(p+'exp_probs_b.bias',torch.float32))
            with placement.place(device_map,'experts',i):
                names=[p+f'ffn_{n}_exps.weight' for n in ('gate','up','down')]
                fmts=[KQUANT_ID.get(rd.gguf[n].quant.name) for n in names]
                if fmts[0]!=fmts[1] or any(f not in (4,6,8) for f in fmts):
                    raise SnowLLMError(f'{p}: unsupported LFM expert formats {fmts}')
                gate,up=rd.transient(*names[:2])
                gu=ops.moe_kquant_shuffle_gate_up(gate,up,fmts[0],g.moe_num_experts)
                down,=rd.transient(names[2])
                dw=ops.moe_kquant_shuffle_down(down,fmts[2],g.moe_num_experts)
                mlp=FusedMoE(router,gu,dw)
        layers.append(Layer(attn,mlp,rd.tensor(p+'attn_norm.weight'),rd.tensor(p+'ffn_norm.weight')))
        if ops.drying() is None:
            torch.cuda.synchronize()
            print(f'{term.stamp()} LFM layer {i+1}/{len(want)}',flush=True)
    model=_finish(cfg,layers,embed,rd.tensor('token_embd_norm.weight'),head)
    rd.drop_transient()
    model.device_map=device_map
    return model
