# SPDX-License-Identifier: Apache-2.0
from ...errors import SnowLLMError

ARCH = 'lfm2moe'


def config(g):
    n=int(g.need('{arch}.block_count'))
    kv=g.need('{arch}.attention.head_count_kv')
    if not isinstance(kv,list) or len(kv)!=n or not any(kv):
        raise SnowLLMError('LFM GGUF needs per-layer KV head counts')
    heads={int(v) for v in kv if v}
    if len(heads)!=1:
        raise SnowLLMError('LFM GGUF needs one attention geometry')
    if int(g.need('{arch}.expert_gating_func'))!=2:
        raise SnowLLMError('LFM GGUF needs sigmoid expert routing')
    return dict(architectures=['Lfm2MoeForCausalLM'],model_type='lfm2_moe',
        hidden_size=int(g.need('{arch}.embedding_length')),
        num_hidden_layers=n,num_attention_heads=int(g.need('{arch}.attention.head_count')),
        num_key_value_heads=heads.pop(),vocab_size=len(g.need('tokenizer.ggml.tokens')),
        layer_types=['full_attention' if v else 'conv' for v in kv],
        max_position_embeddings=int(g.need('{arch}.context_length')),
        norm_eps=float(g.need('{arch}.attention.layer_norm_rms_epsilon')),
        rope_theta=float(g.need('{arch}.rope.freq_base')),
        intermediate_size=int(g.need('{arch}.feed_forward_length')),
        num_dense_layers=int(g.need('{arch}.leading_dense_block_count')),
        num_experts=int(g.need('{arch}.expert_count')),
        num_experts_per_tok=int(g.need('{arch}.expert_used_count')),
        moe_intermediate_size=int(g.need('{arch}.expert_feed_forward_length')),
        conv_L_cache=int(g.need('{arch}.shortconv.l_cache')),conv_bias=False,
        norm_topk_prob=True,use_expert_bias=True,routed_scaling_factor=1.0,
        tie_word_embeddings='output.weight' not in g)
