# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from ._common import (
    KV_BLOCK_SIZE,
    PREFILL_ROW_QUANTUM,
    SAMPLING_MAX_K,
    Path,
    SnowLLMError,
    empty_bytes,
    zero_bytes,
    synchronize,
)
from .attention_full import (
    PAGED_DECODE_MAX_Q_TOKENS,
    PAGED_PREFILL_MAX_TOKENS,
    attn_out_scale_oproj,
    attn_out_scale_oproj_fp8,
    attn_out_scale_oproj_shuffle_w,
    attn_out_scale_oproj_shuffle_w_fp8,
    attn_out_scale_oproj_scratch_bytes,
    paged_attn_decode,
    paged_attn_decode_int8,
    paged_attn_decode_plan,
    paged_attn_prefill,
    paged_attn_prefill_int8,
    kv_pool_bytes,
    kv_scale_bytes,
    paged_decode_num_slots,
    paged_window_view,
    prefill_q_plan,
    paged_decode_plan_elems,
    paged_decode_workspace_size,
    qk_norm,
    qk_norm_rope,
    qkv_proj,
    qkv_proj_fp8,
    qkv_proj_shuffle_w,
    qkv_proj_shuffle_w_fp8,
    qkv_proj_scratch_bytes,
)
from .attention_linear import (
    LinearAttnWeights,
    fused_linear_attn,
    fused_linear_attn_workspace_bytes,
    linear_in_proj_ba_shuffle_w,
    linear_in_proj_shuffle_w,
    linear_state_slots,
    linear_in_proj_qz_shuffle_w_fp8,
    linear_out_proj_shuffle_w,
    linear_out_proj_shuffle_w_fp8,
)
from .cache import (
    kv_blocks_for,
    resolve_slots,
    reshape_and_cache,
    reshape_and_cache_int8,
)
from .embedding import (
    gather_embedding,
    lm_head,
    lm_head_pad_bytes,
    lm_head_scratch_bytes,
    lm_head_rows_for,
    lm_head_shuffle_weight,
)
from .gemm import (
    gemm_bf16_shuffle_b,
    shuffle_bytes,
    proj_scale_shuffle_fp8,
)
from .moe import (
    fused_moe,
    fused_moe_fp8,
    moe_variant_force,
    moe_variant_name,
    moe_shuffle_down,
    moe_shuffle_down_fp8,
    moe_shuffle_gate_up,
    moe_shuffle_gate_up_fused,
    moe_shuffle_gate_up_fp8,
    moe_shuffle_router,
    moe_scale_shuffle_down_fp8,
    moe_scale_shuffle_gate_up_fp8,
    moe_workspace_bytes,
)
from .mtp import (
    mtp_fc,
    mtp_fc_shuffle_w,
    mtp_fc_scratch_bytes,
    mtp_pre_fc,
)
from .norm import (
    rmsnorm,
    rmsnorm_shuffled,
    rmsnorm_residual,
    rmsnorm_residual_shuffled,
)
from .rope import (
    rope_apply,
    rope_cos_sin,
)
from .sampling import (
    argmax,
    sample,
)
from .vision import (
    vision_attn,
    vision_gemm_bias_act,
    vision_gemm_scratch_bytes,
    vision_layernorm,
    vision_rope_qk,
)
