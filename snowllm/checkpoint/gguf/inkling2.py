# SPDX-License-Identifier: Apache-2.0
from ...errors import SnowLLMError

ARCH = "inkling2_loop"


def config(g):
    def need(key):
        return g.need("{arch}." + key)

    n = need("block_count")
    start, end, passes = (need("loop." + key) for key in ("start", "end", "count"))
    block = need("attn_res.block_size")
    if (need("loop.varying") or not 0 < start < end <= n or passes != 3
            or block <= 0 or start % block or end % block):
        raise SnowLLMError("Inkling requires a fixed three-pass, block-aligned loop core")

    def head_count(key):
        counts = need("attention." + key)
        if isinstance(counts, int) and counts > 0:
            return counts
        if isinstance(counts, list) and len(counts) == n and len(set(counts)) == 1 and counts[0] > 0:
            return counts[0]
        raise SnowLLMError(f"Inkling requires uniform {key} across {n} weight layers")

    sliding = need("attention.sliding_window_pattern")
    if not isinstance(sliding, list) or len(sliding) != n:
        raise SnowLLMError("Inkling sliding_window_pattern must describe each weight layer")
    head = need("attention.key_length")
    heads, kv_heads = head_count("head_count"), head_count("head_count_kv")
    if head != 128 or heads % kv_heads or any(need("attention." + key) != head
            for key in ("value_length", "key_length_swa", "value_length_swa")):
        raise SnowLLMError("Inkling requires grouped 128-wide query and KV heads for local and global attention")

    vocab = need("vocab_size_unpadded")
    padded = need("vocab_size")
    if vocab != len(g.need("tokenizer.ggml.tokens")) or not 0 < vocab <= padded:
        raise SnowLLMError("Inkling tokenizer size differs from the output vocabulary")
    return dict(
        architectures=["Inkling2LoopForConditionalGeneration"], model_type=ARCH,
        num_hidden_layers=n, hidden_size=need("embedding_length"), head_dim=head,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
        layer_types=["sliding_attention" if local else "full_attention" for local in sliding],
        max_position_embeddings=need("context_length"), sliding_window=need("attention.sliding_window"),
        rms_norm_eps=need("attention.layer_norm_rms_epsilon"), vocab_size=padded,
        unpadded_vocab_size=vocab, eos_token_id=g.need("tokenizer.ggml.eos_token_id"),
        loop_config=dict(core_start=start, core_end=end, num_passes=passes, gate_rank=need("loop.gate_rank")),
        n_layers_per_attn_res_block=block, sconv_kernel_size=need("shortconv.l_cache"),
        d_rel=need("attention.relative_length"), rel_extent=need("attention.relative_extent"),
        qk_l2_clip_threshold=need("attention.qk_clip"), rel_l2_clip_threshold=need("attention.relative_clip"),
        log_scaling_n_floor=need("attention.log_scale_floor"), log_scaling_alpha=need("attention.log_scale_alpha"),
        num_experts=need("expert_count"), num_experts_per_tok=need("expert_used_count"),
        moe_intermediate_size=need("expert_feed_forward_length"), route_scale=need("expert_weights_scale"),
        mlp_gate_softcap=need("activation.situ_beta"), mlp_up_softcap=need("activation.situ_linear_beta"),
        final_logit_softcapping=need("final_logit_softcapping"),
    )
