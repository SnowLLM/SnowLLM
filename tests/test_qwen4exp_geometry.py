# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import sys

from snowllm import _capi
from snowllm.checkpoint.gguf import GGUF
from snowllm.checkpoint.gguf.names import config
from snowllm.checkpoint.gguf.source import find_gguf
from snowllm.models.geometry import Qwen4ExpGeometry

import _harness

CKPT = _harness.checkpoint("Qwen3.8-Flash-Next-UD-Q3_K_XL")

BAKED = ("hidden", "num_heads", "num_kv_heads", "head_size", "vocab_size",
         "qkv_proj_n", "qkv_q_head_stride", "qkv_off_scale", "qkv_off_k", "qkv_off_v",
         "lin_num_k_heads", "lin_num_v_heads", "lin_head_k", "lin_head_v", "lin_key_dim",
         "lin_value_dim", "lin_conv_dim", "lin_conv_k", "lin_conv_state", "lin_in_proj_n",
         "lin_in_proj_n_pad", "lin_in_proj_ba_n_pad", "lin_off_qkv", "lin_off_z", "lin_off_b",
         "lin_off_a", "moe_num_experts", "moe_shared_expert", "moe_num_slabs", "moe_topk",
         "moe_inter")


def main() -> int:
    ck = _harness.Checks(50)
    geo = Qwen4ExpGeometry.from_config(config(GGUF(find_gguf(CKPT)))["text_config"])

    ck("the build carries a fourth geometry",
       _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT) == "qwen3.8-flash-next",
       _capi.geometry_name(_capi.GEO_QWEN38_FLASH_NEXT) or "(absent)")
    _capi.select_geometry(_capi.GEO_QWEN38_FLASH_NEXT)
    ck("and selects it", _capi.geometry_id() == _capi.GEO_QWEN38_FLASH_NEXT)
    build = _capi.build_geometry()

    for name in BAKED:
        ck(f"{name} agrees with the checkpoint", getattr(geo, name) == getattr(build, name),
           f"{getattr(geo, name)} against the build's {getattr(build, name)}")
    ck("moe_topk_all counts the shared expert", build.moe_topk_all == geo.moe_topk + 1)
    ck("the KV page is the engine's, not the model's", build.block_size == 16)

    ck("the fused qkv carries a gate per query head",
       geo.qkv_proj_n == 2 * geo.q_dim + 2 * geo.kv_dim and geo.attn_output_gate)
    ck("in_proj pads to a whole GEMM column quantum",
       geo.lin_in_proj_n_pad % 128 == 0 and geo.lin_in_proj_n_pad >= geo.lin_in_proj_n,
       f"{geo.lin_in_proj_n} -> {geo.lin_in_proj_n_pad}")
    ck("three value heads to a key head", geo.lin_num_v_heads == 3 * geo.lin_num_k_heads)
    ck("the conv spans q|k|v", geo.lin_conv_dim == 2 * geo.lin_key_dim + geo.lin_value_dim)

    ck("four residual streams of the hidden width", (geo.hc_count, geo.hc_dim) == (4, 10240))
    ck("the hyper-connection rank is below the stream width", geo.hc_lowrank < geo.hidden,
       f"{geo.hc_lowrank} of {geo.hc_dim}")
    ck("12 full-attention layers and 36 linear ones",
       (len(geo.full_layers), len(geo.linear_layers)) == (12, 36))
    ck("the indexer keeps 512 blocks of 4", (geo.index_blocks, geo.index_ratio) == (512, 4))
    ck("PLE hashes 16 heads into rows of 160",
       (geo.ple_heads, geo.ple_head_dim, geo.ple_embed_dim) == (16, 160, geo.hidden))

    ck("the hidden width takes a split-K of 4 and not 8", geo.hidden % 128 == 0
       and geo.hidden % 1024 != 0, f"{geo.hidden} = {geo.hidden // 128} * 128")
    ck("the expert intermediate does NOT divide the k-quant super-block",
       geo.moe_inter % 256 != 0, f"{geo.moe_inter} % 256 = {geo.moe_inter % 256}")
    return ck.done()


if __name__ == "__main__":
    sys.exit(main())
