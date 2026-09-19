# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ctypes
import glob
import importlib.util
import os
import sys
from ctypes import POINTER, c_char_p, c_float, c_int, c_int64, c_void_p

from ._platform import HIP_RUNTIME_GLOBS, hip_runtimes

_PROFILER_ENV = ("SNOWLLM_TRACE", "HSA_TOOLS_LIB", "ROCP_TOOL_LIBRARIES",
                 "ROCPROFILER_REGISTER_LIBRARY", "ROCPROF_KERNEL_TRACE")
if not (any(a == "--profile-dir" or a.startswith("--profile-dir=") for a in sys.argv)
        or any(os.environ.get(v) for v in _PROFILER_ENV)):
    os.environ.setdefault("HSA_TOOLS_DISABLE_REGISTER", "1")

os.environ.setdefault("ROCSHMEM_DEBUG_LEVEL", "NONE")

__all__ = ["SnowLLMError", "build_geometry", "draft_count", "draft_name", "draft_shape", "lib",
           "select_draft", "select_geometry", "geometry_id", "geometry_name", "synchronize",
           "assert_single_hip_runtime", "GEO_QWEN36_35B_A3B", "GEO_QWEN36_27B",
           "GEO_DEEPSEEK_V4_FLASH", "GEO_QWEN38_FLASH_NEXT"]

GEO_QWEN36_35B_A3B = 0
GEO_QWEN36_27B = 1
GEO_DEEPSEEK_V4_FLASH = 2
GEO_QWEN38_FLASH_NEXT = 3


class SnowLLMError(RuntimeError):
    pass


class BuildGeometry(ctypes.Structure):
    _fields_ = [
        (n, c_int64)
        for n in (
            "hidden", "num_heads", "num_kv_heads", "head_size", "block_size", "vocab_size",
            "qkv_proj_n", "qkv_q_head_stride", "qkv_off_scale", "qkv_off_k", "qkv_off_v",
            "lin_num_k_heads", "lin_num_v_heads", "lin_head_k", "lin_head_v", "lin_key_dim",
            "lin_value_dim", "lin_conv_dim", "lin_conv_k", "lin_conv_state", "lin_in_proj_n",
            "lin_in_proj_n_pad",
            "lin_in_proj_ba_n_pad", "lin_off_qkv", "lin_off_z", "lin_off_b", "lin_off_a",
            "moe_num_experts", "moe_shared_expert", "moe_num_slabs", "moe_topk", "moe_topk_all",
            "moe_inter", "sampling_max_k", "lm_head_decode_max_m", "mlp_inter",
            "mlp_gate_up_n",
        )
    ]


def build_geometry() -> BuildGeometry:
    g = BuildGeometry()
    lib.snowllm_model_config(ctypes.byref(g))
    return g


DRAFT_SHAPE_FIELDS = ("hidden", "num_layers", "num_heads", "num_kv_heads", "head_size",
                      "intermediate", "num_taps")


class DraftShape(ctypes.Structure):
    _fields_ = [(n, c_int64) for n in DRAFT_SHAPE_FIELDS]


def select_draft(shape: DraftShape) -> int:
    return int(lib.snowllm_select_draft(ctypes.byref(shape)))


def draft_count() -> int:
    return int(lib.snowllm_draft_count())


def draft_shape(i: int) -> DraftShape | None:
    s = DraftShape()
    return None if lib.snowllm_draft_shape(int(i), ctypes.byref(s)) else s


def draft_name(i: int) -> str:
    buf = ctypes.create_string_buffer(64)
    lib.snowllm_draft_name(int(i), buf, 64)
    return buf.value.decode()


def select_geometry(geo: int) -> None:
    if lib.snowllm_select_geometry(int(geo)) != 0:
        raise SnowLLMError(
            f"this build has no geometry {geo}; it carries "
            + ", ".join(f"{i} ({geometry_name(i)})"
                        for i in (GEO_QWEN36_35B_A3B, GEO_QWEN36_27B,
                                  GEO_DEEPSEEK_V4_FLASH, GEO_QWEN38_FLASH_NEXT)
                        if geometry_name(i))
        )


def geometry_id() -> int:
    return int(lib.snowllm_active_geometry())


def geometry_name(geo: int) -> str:
    buf = ctypes.create_string_buffer(64)
    lib.snowllm_geometry_name(int(geo), buf, 64)
    return buf.value.decode()


def _find_lib() -> str:
    if env := os.environ.get("SNOWLLM_LIB"):
        return env
    try:
        import snowllm_kernels
    except ModuleNotFoundError as e:
        if e.name != "snowllm_kernels":
            raise
        raise SnowLLMError(
            "the snowllm-kernels package is not installed, so there are no kernels to call. "
            "`pip install snowllm-kernels` (it carries libsnowllm.so and needs no compiler), or "
            "point SNOWLLM_LIB at a libsnowllm.so you built yourself."
        ) from e
    lib = str(snowllm_kernels.LIB)
    if not os.path.exists(lib):
        raise SnowLLMError(
            f"snowllm-kernels resolved to {os.path.dirname(snowllm_kernels.__file__)} but there is "
            f"no libsnowllm.so at {lib}. Reinstall the wheel, or point SNOWLLM_LIB at a "
            f"libsnowllm.so."
        )
    return lib


def _preload_hip_runtime() -> None:
    for pkg, pattern in HIP_RUNTIME_GLOBS:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        found = sorted(glob.glob(os.path.join(spec.submodule_search_locations[0], pattern)))
        if not found:
            continue
        try:
            ctypes.CDLL(found[0], mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return


_preload_hip_runtime()
_lib_path = _find_lib()
lib = ctypes.CDLL(_lib_path)

ABI_VERSION = 57


def _check_abi() -> None:
    try:
        lib.snowllm_abi_version.restype = c_int
        got = lib.snowllm_abi_version()
    except AttributeError:
        got = 0
    if got != ABI_VERSION:
        raise SnowLLMError(
            f"ABI mismatch: this snowllm expects {ABI_VERSION}, "
            f"{_lib_path} provides {got or 'no version at all'}. Rebuild the one that is stale."
        )


_check_abi()

class Stream:
    pass


P, S = c_void_p, Stream
_SIGS = [
    ("snowllm_model_config", None, [POINTER(BuildGeometry)]),
    ("snowllm_select_draft", c_int, [POINTER(DraftShape)]),
    ("snowllm_active_draft", c_int, []),
    ("snowllm_draft_count", c_int64, []),
    ("snowllm_draft_shape", c_int, [c_int, POINTER(DraftShape)]),
    ("snowllm_draft_name", c_int64, [c_int, c_char_p, c_int64]),
    ("snowllm_select_geometry", c_int, [c_int]),
    ("snowllm_active_geometry", c_int, []),
    ("snowllm_geometry_name", c_int64, [c_int, c_char_p, c_int64]),
    ("snowllm_kv_block_sizes", c_int64, [c_void_p, c_int64]),
    ("snowllm_shuffle_bytes", c_int64, [c_int64]),
    ("snowllm_sampling_max_k", c_int64, []),
    ("snowllm_error_string", c_char_p, [c_int]),
    ("snowllm_synchronize", c_int, [S]),

    ("snowllm_qkv_proj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_qkv_proj_index_shuffle_w", c_int, [P, P, S]),
    ("snowllm_qkv_proj_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_qkv_proj_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_attn_out_scale_oproj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_attn_out_scale_oproj_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_attn_out_scale_oproj_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_linear_in_proj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_linear_out_proj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_linear_out_proj_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_linear_in_proj_qz_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_qkv_proj_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_qkv_proj_qk_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_qkv_proj_v_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_qkv_proj_q_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_qkv_proj_kv_shuffle_w", c_int, [P, P, S]),
    ("snowllm_attn_out_scale_oproj_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_linear_in_proj_qz_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_linear_in_proj_qkv_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_linear_in_proj_z_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_linear_out_proj_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_linear_in_proj_ba_shuffle_w", c_int, [P, P, S]),
    ("snowllm_fused_linear_attn_workspace_bytes", c_int64, [c_int64, c_int]),
    ("snowllm_mtp_fc_shuffle_w", c_int, [P, P, S]),
    ("snowllm_mtp_fc_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_mtp_pre_fc", c_int, [P, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_mtp_fc", c_int, [P, P, P, P, c_int64, c_int, S]),
    ("snowllm_draft_proj_shuffle_w", c_int, [c_int, P, P, S]),
    ("snowllm_draft_proj_scratch_bytes", c_int64, [c_int, c_int64, c_int]),
    ("snowllm_draft_proj", c_int, [c_int, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_draft_swiglu", c_int, [P, c_int64, P, c_int64, S]),
    ("snowllm_draft_proj_shuffle_w_kquant", c_int, [c_int, c_int, P, P, P, S]),
    ("snowllm_draft_proj_kquant", c_int, [c_int, c_int, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_draft_dyn_conv", c_int, [P, P, P, P, c_int64, c_int64, c_int, S]),
    ("snowllm_draft_select_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_draft_select", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, S]),
    ("snowllm_draft_rope_cos_sin", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_draft_qk_norm_rope", c_int, [P, c_int64, P, P, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_draft_k_norm_rope", c_int, [P, c_int64, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_draft_reshape_and_cache", c_int,
     [P, P, c_int64, c_int64, P, P, P, c_int64, c_int64, S]),
    ("snowllm_draft_paged_attn", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, P, c_float, c_int64, c_int64, S]),
    ("snowllm_draft_paged_attn_plan", c_int, [P, P, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_draft_paged_attn_workspace_size", c_int64, [c_int64, c_int64]),
    ("snowllm_draft_paged_attn_num_slots", c_int64, [c_int64]),
    ("snowllm_draft_paged_attn_plan_elems", c_int64, [c_int64, c_int64]),
    ("snowllm_gated_delta_rule_advance", c_int,
     [P, P, P, c_int64, P, P, P, P, P, c_int64, c_int64, S]),
    ("snowllm_moe_shuffle_gate_up", c_int, [P, P, P, c_int64, S]),
    ("snowllm_moe_shuffle_down", c_int, [P, P, c_int64, S]),
    ("snowllm_moe_shuffle_gate_up_fused", c_int, [P, P, c_int64, S]),
    ("snowllm_proj_scale_shuffle_fp8", c_int, [P, P, c_int64, c_int64, S]),
    ("snowllm_moe_scale_shuffle_gate_up_fp8", c_int, [P, P, P, c_int64, S]),
    ("snowllm_moe_scale_shuffle_down_fp8", c_int, [P, P, c_int64, S]),
    ("snowllm_moe_shuffle_gate_up_fp8", c_int, [P, P, P, c_int64, S]),
    ("snowllm_moe_shuffle_down_fp8", c_int, [P, P, c_int64, S]),
    ("snowllm_moe_shuffle_router_bytes", c_int64, []),
    ("snowllm_moe_shuffle_router", c_int, [P, P, P, S]),
    ("snowllm_moe_workspace_bytes", c_int64, [c_int64]),
    ("snowllm_moe_lowbit_narrow_force_max_m", None, [c_int]),
    ("snowllm_moe_lowbit_force_split", None, [c_int]),

    ("snowllm_lm_head_kquant_gguf_bytes", c_int64, [c_int]),
    ("snowllm_lm_head_kquant_quant_bytes", c_int64, [c_int]),
    ("snowllm_lm_head_kquant_meta_bytes", c_int64, [c_int]),
    ("snowllm_lm_head_kquant_shuffle_weight", c_int, [c_int, P, P, P, S]),
    ("snowllm_lm_head_kquant", c_int, [c_int, P, P, P, P, c_int64, P, S]),

    ("snowllm_kquant_gguf_bytes", c_int64, [c_int, c_int64, c_int64]),
    ("snowllm_kquant_quant_bytes", c_int64, [c_int, c_int64, c_int64]),
    ("snowllm_kquant_meta_bytes", c_int64, [c_int, c_int64, c_int64]),
    ("snowllm_moe_kquant_gate_up_quant_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_kquant_gate_up_meta_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_kquant_down_quant_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_kquant_down_meta_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_kquant_shuffle_gate_up", c_int, [c_int, P, P, P, P, c_int64, S]),
    ("snowllm_moe_kquant_shuffle_down", c_int, [c_int, P, P, P, c_int64, S]),
    ("snowllm_fused_moe_kquant_split", c_int,
     [c_int, c_int, c_int, c_int, P, P, P, P, P, P, P, P, P, P, P, P, c_int64, S]),

    ("snowllm_moe_lowbit_gate_up_p0_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_lowbit_gate_up_p1_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_lowbit_gate_up_p2_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_lowbit_down_p0_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_lowbit_down_p1_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_lowbit_down_p2_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_moe_lowbit_gate_up_stride", c_int64, [c_int64]),
    ("snowllm_moe_lowbit_down_stride", c_int64, [c_int64]),
    ("snowllm_moe_lowbit_shuffle_gate_up", c_int, [c_int, P, P, P, P, P, c_int64, S]),
    ("snowllm_moe_lowbit_shuffle_gate_up_fused", c_int, [c_int, P, P, P, P, c_int64, S]),
    ("snowllm_moe_lowbit_shuffle_down", c_int, [c_int, P, P, P, P, c_int64, S]),
    ("snowllm_fused_moe_lowbit_split", c_int,
     [c_int, c_int, c_int, c_int, P, P, P, P, P, P, P, P, P, P, P, P, P, P, c_int64, S]),
    ("snowllm_fused_moe_lowbit_kquant_down_split", c_int,
     [c_int, c_int, c_int, c_int, P, P, P, P, P, P, P, P, P, P, P, P, P, c_int64, S]),
    ("snowllm_moe_router_tid2eid", c_int, [P, P, P, P, P, P, P, c_int64, S]),
    ("snowllm_moe_router_shuffle_bytes", c_int64, [c_int64]),
    ("snowllm_moe_experts_lowbit_split_tid2eid", c_int,
     [c_int, c_int, c_int, c_int, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, c_int64, S]),
    ("snowllm_lm_head_shuffle_weight", c_int, [P, P, S]),
    ("snowllm_lm_head_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_lm_head", c_int, [P, P, P, c_int64, P, S]),
    ("snowllm_paged_prefill_max_tokens", c_int64, []),
    ("snowllm_paged_decode_workspace_size", c_int64, [c_int64, c_int64]),
    ("snowllm_kv_pool_bytes", None, [c_int64, c_int64, c_int, P, P]),
    ("snowllm_kv_scale_bytes", None, [c_int64, c_int64, P, P]),
    ("snowllm_resolve_slots", c_int, [P, c_int64, P, P, P, c_int64, c_int64, S]),
    ("snowllm_kv_blocks_for", c_int64, [c_int64, c_int64]),
    ("snowllm_linear_state_slots", None, [c_int64, c_int64, c_int64, c_int64, P]),
    ("snowllm_prefill_q_blocks", c_int64, [P, c_int64]),
    ("snowllm_prefill_q_block_map", None, [P, c_int64, P]),
    ("snowllm_moe_variant_force", None, [c_int]),
    ("snowllm_moe_variant_name", c_char_p, [c_int]),
    ("snowllm_paged_window_blocks", c_int64, [c_int64, c_int64, c_int64]),
    ("snowllm_paged_window_view", c_int, [P, c_int64, P, c_int64, c_int64, c_int64, P,
                                          c_int64, P, c_int64, S]),
    ("snowllm_paged_decode_max_q_tokens", c_int64, []),
    ("snowllm_paged_decode_num_slots", c_int64, [c_int64]),
    ("snowllm_paged_decode_plan_elems", c_int64, [c_int64, c_int64]),

    ("snowllm_gather_embedding", c_int, [P, P, P, c_int64, S]),
    ("snowllm_rmsnorm", c_int, [P, P, P, c_int64, c_float, S]),
    ("snowllm_dsv4_rmsnorm", c_int,
     [P, P, P, c_int64, c_int64, c_float, c_int64, c_int64, S]),
    ("snowllm_rmsnorm_residual", c_int, [P, P, P, P, c_int64, c_float, S]),
    ("snowllm_qkv_proj", c_int, [P, P, P, c_float, P, P, P, c_int64, c_int, P, P, S]),
    ("snowllm_qkv_proj_fp8", c_int,
     [P, P, P, c_float, P, P, P, P, c_int64, c_int, P, P, S]),
    ("snowllm_qkv_proj_kquant", c_int,
     [c_int, P, P, P, c_float, P, P, P, P, c_int64, c_int, P, P, S]),
    ("snowllm_qkv_proj_q_kv_kquant", c_int,
     [c_int, P, P, P, c_float, P, P, P, P, P, c_int64, c_int, P, P, S]),
    ("snowllm_qkv_proj_pair_kquant", c_int,
     [c_int, c_int, P, P, P, c_float, P, P, P, P, P, P, c_int64, c_int, P, P, S]),
    ("snowllm_qk_norm", c_int, [P, c_int64, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_qk_norm_rope_split_k", c_int,
     [P, c_int64, P, c_int64, P, P, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_qk_norm_rope", c_int, [P, c_int64, P, P, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_attn_out_scale_oproj", c_int, [P, P, c_int64, P, P, P, c_int64, c_int, S]),
    ("snowllm_attn_out_scale_oproj_fp8", c_int, [P, P, c_int64, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_attn_oproj_shuffled_a", c_int, [P, P, P, c_int64, S]),
    ("snowllm_attn_oproj_shuffled_a_fp8", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_attn_oproj_shuffled_a_kquant", c_int, [c_int, P, P, P, P, c_int64, S]),
    ("snowllm_attn_out_scale_oproj_kquant", c_int,
     [c_int, P, P, c_int64, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_rope_cos_sin", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_rope_cos_sin_mrope", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_rope_apply_q", c_int, [P, P, P, c_int64, S]),
    ("snowllm_rope_apply_k", c_int, [P, P, P, c_int64, S]),
    ("snowllm_dsv4_rope_cos_sin", c_int, [P, P, c_float, P, P, c_int64, S]),
    ("snowllm_dsv4_rope_tail", c_int, [P, P, P, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_dsv4_rope_tail_group", c_int,
     [P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_reshape_and_cache", c_int,
     [P, P, c_int64, c_int64, P, P, P, c_int64, c_int64, S]),
    ("snowllm_reshape_and_cache_int8", c_int,
     [P, P, c_int64, c_int64, P, P, P, P, P, c_int64, c_int64, S]),
    ("snowllm_paged_attn_prefill_int8", c_int,
     [P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, c_int64, S, P]),
    ("snowllm_paged_attn_decode_int8", c_int,
     [P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, P, c_float, c_int64, c_int64, c_int64,
      c_int64, S]),
    ("snowllm_paged_attn_prefill", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, c_int64, S, P, P, c_int64,
      c_int64]),
    ("snowllm_dsv4_hc_split_sinkhorn", c_int,
     [P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_float, S]),
    ("snowllm_dsv4_hc_gate", c_int,
     [P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_float, S]),
    ("snowllm_dsv4_indexer_weights", c_int,
     [P, P, c_int64, c_int64, c_int64, c_int64, c_float, S]),
    ("snowllm_dsv4_hc_broadcast", c_int, [P, P, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_dsv4_hc_weighted_sum", c_int,
     [P, P, P, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_dsv4_hc_expand", c_int,
     [P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_qwen4exp_hc_norm", c_int, [P, P, P, c_int64, c_int64, c_int64, c_float, S]),
    ("snowllm_qwen4exp_hc_mix_ws_bytes", c_int64, [c_int64, c_int64, c_int64, c_int64]),
    ("snowllm_qwen4exp_hc_mix", c_int,
     [P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_float, P, P, S]),
    ("snowllm_qwen4exp_hc_mix_kquant", c_int,
     [c_int, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_float, P, P,
      S]),
    ("snowllm_qwen4exp_hc_lowrank_act", c_int,
     [P, P, c_int64, c_int64, c_int64, c_int, c_int64, S]),
    ("snowllm_qwen4exp_hc_fold", c_int,
     [P, P, P, c_int64, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_qwen4exp_hc_combine", c_int,
     [P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_qwen4exp_hc_combine_norm", c_int,
     [P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int, c_float, S]),
    ("snowllm_qwen4exp_ple_gate", c_int, [P, P, P, P, P, c_int64, c_int64, c_int64, S]),
    ("snowllm_qwen4exp_ple_conv", c_int,
     [P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_qwen4exp_ple_state_checkpoint", c_int,
     [P, P, P, P, P, P, P, c_int64, P, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_qwen4exp_indexer_pool_norm", c_int,
     [P, P, P, c_int64, c_int64, c_int64, c_float, S]),
    ("snowllm_qwen4exp_indexer_rope", c_int, [P, P, P, c_int64, c_int64, S]),
    ("snowllm_qwen4exp_qsa_produce", c_int,
     [P, c_int64, P, P, P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64,
      c_int64, c_float, S]),
    ("snowllm_qwen4exp_indexer_q", c_int,
     [P, c_int64, P, P, P, P, c_int64, c_int64, c_int64, c_float, S]),
    ("snowllm_qwen4exp_qsa_attn_prefill", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_float, P, P, c_int64, P, P, c_int64, c_int64, P,
      c_int64, c_int64, S]),
    ("snowllm_qwen4exp_qsa_q_tile", c_int64, []),
    ("snowllm_qwen4exp_qsa_tile_axis", c_int,
     [P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_qwen4exp_qsa_gather", c_int,
     [P, P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
      S]),
    ("snowllm_dsv4_indexer_topk_mask", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_dsv4_fp8_kv_quantize", c_int, [P, c_int64, c_int64, c_int64, S]),
    ("snowllm_dsv4_indexer_scores", c_int,
     [P, P, P, P, P, P, P, P, c_int, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
      c_int64, c_int64, S]),
    ("snowllm_dsv4_compressor_pool", c_int,
     [P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_float, P, P, c_int64, P, P,
      c_int, c_int64, P, P, P, c_int64, S]),
    ("snowllm_dsv4_carry_slide", c_int,
     [P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_dsv4_mla_attn_prefill", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, P, P, c_int64, c_int64, S]),
    ("snowllm_dsv4_mla_attn_prefill_block", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, P, P, c_int64, c_int64, S]),
    ("snowllm_dsv4_mla_attn_prefill_compressed", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, P, P, c_int64,
      P, P, P, P, P, c_int64, c_int64, P, P, c_int64, c_int64, S]),
    ("snowllm_dsv4_mla_attn_split_compressed", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, P, P, c_int64,
      P, P, P, P, P, c_int64, c_int64, P, P, c_int64, P, c_int64, S]),
    ("snowllm_dsv4_mla_split_worth", c_int64, [c_int64]),
    ("snowllm_dsv4_mla_split_workspace_bytes", c_int64, [c_int64]),
    ("snowllm_dsv4_mla_raw_ring_blocks", c_int64, [c_int64, c_int64, c_int64]),
    ("snowllm_paged_attn_decode", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, P, c_float, c_int64, c_int64, c_int64,
      c_int64, S]),
    ("snowllm_paged_attn_decode_plan", c_int, [P, P, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_fused_linear_attn", c_int,
     [P, P, P, c_float,
      P, P, P, P, P, P,
      P, P, P, P,
      P, P, P, P, c_int, c_int,
      P, P, c_int, P, P, c_int,
      c_int,
      P, P, P, P,
      P, P,
      P, P, c_int64, P, P,
      P, P,
      c_int64, c_int64, c_int, S,
      P, P, P]),
    ("snowllm_qwen4exp_hc_qkv_ws_bytes", c_int64, [c_int64, c_int64]),
    ("snowllm_qwen4exp_hc_linear_ws_bytes", c_int64, [c_int64, c_int64]),
    ("snowllm_qwen4exp_hc_qkv_proj_kquant", c_int,
     [P,
      P, P, P, P, P, c_int, c_int64, c_int64, c_float,
      P,
      P,
      c_int, P, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_qwen4exp_hc_linear_attn", c_int,
     [P,
      P, P, P, P, P, c_int, c_int64, c_int64, c_float,
      P,
      P,
      P, P, P, P, P, P,
      P, P, P, P,
      P, P, P, P, c_int, c_int,
      P, P, c_int, P, P, c_int,
      c_int,
      P, P, P, P,
      P, P,
      P, P, c_int64, P, P,
      P, P,
      c_int64, c_int64, c_int, S,
      P, P, P]),
    ("snowllm_linear_attn_retain_bytes", c_int64, [c_int64, c_int]),
    ("snowllm_linear_attn_advance", c_int,
     [P, P, P, P, P, P, P, P, P, c_int64, c_int64, S]),
    ("snowllm_fused_moe", c_int, [P, P, P, P, P, P, c_int64, S]),
    ("snowllm_fused_moe_fp8", c_int, [P, P, P, P, P, P, P, P, c_int64, S]),
    ("snowllm_lm_head_gemm_decode", c_int, [P, P, P, c_int64, S]),
    ("snowllm_lm_head_gemm_prefill", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_sampling_softmax", c_int, [P, P, c_int64, c_int64, P, S]),
    ("snowllm_sampling_topk_topp", c_int, [P, P, P, c_int64, c_int64, c_int64, P, P, S]),
    ("snowllm_sampling_multinomial", c_int, [P, P, P, P, c_int64, c_int64, S]),
    ("snowllm_sampling_argmax", c_int, [P, P, c_int64, c_int64, S]),

    ("snowllm_mlp_gate_up_shuffle_w", c_int, [P, P, S]),
    ("snowllm_mlp_gate_up_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_mlp_gate_up_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_mlp_down_shuffle_w", c_int, [P, P, S]),
    ("snowllm_mlp_down_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_mlp_down_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_fused_mlp_workspace_bytes", c_int64, [c_int64]),
    ("snowllm_fused_mlp", c_int, [P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_fused_mlp_fp8", c_int, [P, P, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_fused_mlp_kquant", c_int,
     [c_int, c_int, P, P, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_mlp_gate_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_mlp_up_shuffle_w_kquant", c_int, [c_int, P, P, P, S]),
    ("snowllm_fused_mlp_kquant_split", c_int,
     [c_int, c_int, c_int, P, P, P, P, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_fused_mlp_kquant_bf16_down", c_int,
     [c_int, P, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_fused_mlp_kquant_split_bf16_down", c_int,
     [c_int, c_int, P, P, P, P, P, P, P, P, c_int64, c_int, S]),

    ("snowllm_gemm_bf16_shuffle_b", c_int, [P, P, c_int64, c_int64, S]),
    ("snowllm_gemm_bf16_a_ws_bytes", c_int64, [c_int64, c_int64]),
    ("snowllm_gemm_bf16_a", c_int,
     [P, P, P, c_int64, c_int64, c_int64, c_int, P, S]),
    ("snowllm_gemm_bf16_a2", c_int,
     [P, P, P, P, P, c_int64, c_int64, c_int64, c_int64, c_int, P, S]),
    ("snowllm_gemm_kquant_shuffle_b", c_int, [c_int, P, P, P, c_int64, c_int64, S]),
    ("snowllm_gemm_kquant_a_ws_bytes", c_int64, [c_int64, c_int64]),
    ("snowllm_gemm_kquant_a", c_int,
     [c_int, P, P, P, P, c_int64, c_int64, c_int64, c_int, P, S]),
    ("snowllm_dsv4_norm_proj_scratch_bytes", c_int64, [c_int, c_int64, c_int]),
    ("snowllm_dsv4_norm_proj_bf16", c_int,
     [c_int, P, P, c_float, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_dsv4_norm_proj_kquant_scratch_bytes", c_int64, [c_int, c_int64, c_int]),
    ("snowllm_dsv4_norm_proj_kquant", c_int,
     [c_int, P, P, c_float, c_int, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_dsv4_attn_fanout_scratch_bytes", c_int64, [c_int64, c_int]),
    ("snowllm_dsv4_attn_fanout", c_int,
     [P, P, c_float, c_int, c_int, P, P, P, c_int, P, P, P, P, P, P, P, c_int64, c_int,
      S]),
    ("snowllm_dsv4_q_proj_scratch_bytes", c_int64, [c_int64, c_int]),
    ("snowllm_dsv4_q_proj_kquant", c_int,
     [P, P, c_float, c_int, P, P, P, c_int, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_dsv4_o_proj_scratch_bytes", c_int64, [c_int64, c_int]),
    ("snowllm_dsv4_o_proj_kquant", c_int,
     [P, P, P, c_int, P, P, c_int, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_dsv4_proj_scratch_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_dsv4_proj_kquant", c_int, [c_int, c_int, P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_dsv4_proj_bf16", c_int, [c_int, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_dsv4_proj_kquant_grouped", c_int, [c_int, P, P, P, P, c_int64, S]),
    ("snowllm_dsv4_narrow_proj_rows", c_int64, [c_int]),
    ("snowllm_dsv4_narrow_proj_partials_bytes", c_int64, [c_int, c_int64]),
    ("snowllm_dsv4_narrow_proj_bf16", c_int, [c_int, P, P, P, P, c_int64, S]),
    ("snowllm_vision_layernorm", c_int, [P, P, P, P, c_int64, c_int64, c_float, S]),
    ("snowllm_vision_gemm_scratch_bytes", c_int64, [c_int64, c_int64]),
    ("snowllm_vision_gemm_bias_act", c_int,
     [P, P, P, P, P, P, c_int64, c_int64, c_int64, c_int, S]),
    ("snowllm_vision_rope_qk", c_int, [P, P, P, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_vision_attn", c_int, [P, P, P, P, c_int64, c_int64, c_int64, c_int64, S]),
    ("snowllm_vision_supported_head_dims", c_int64, [P, c_int64]),
    ("snowllm_vision_supported_ln_hidden", c_int64, [P, c_int64]),

    ("snowllm_file_reader_create", P, [c_int, c_int64, c_int]),
    ("snowllm_file_reader_destroy", None, [P]),
    ("snowllm_file_reader_read", c_int, [P, c_char_p, c_int64, c_int64, P, S]),
]
_missing = []
for _name, _res, _args in _SIGS:
    try:
        _fn = getattr(lib, _name)
    except AttributeError:
        _missing.append(_name)
        continue
    _fn.restype = _res
    _fn.argtypes = [c_void_p if _a is S else _a for _a in _args]
if _missing:
    raise SnowLLMError(f"{_lib_path} is missing {len(_missing)} entry points this binding needs:\n  "
                       + "\n  ".join(_missing))


LAUNCHES = frozenset(_n for _n, _, _a in _SIGS if _a and S in _a)


def check(status: int, what: str) -> None:
    if status != 0:
        msg = lib.snowllm_error_string(status).decode()
        raise SnowLLMError(f"{what}: hip status {status}: {msg}")


def synchronize(stream: int = 0) -> None:
    check(lib.snowllm_synchronize(stream), "synchronize")


def assert_single_hip_runtime() -> None:
    paths = hip_runtimes()
    if len(paths) > 1:
        raise SnowLLMError(
            "two HIP runtimes are loaded in this process -- torch's device pointers are not valid "
            "in snowllm's kernels:\n  " + "\n  ".join(sorted(paths))
        )
