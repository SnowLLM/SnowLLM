# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import ctypes
import importlib.util
import os
from ctypes import POINTER, c_char_p, c_float, c_int, c_int64, c_void_p

__all__ = ["SnowLLMError", "build_geometry", "lib", "synchronize",
           "assert_single_hip_runtime"]


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
            "moe_inter", "sampling_max_k", "lm_head_decode_max_m",
        )
    ]


def build_geometry() -> "BuildGeometry":
    g = BuildGeometry()
    lib.snowllm_model_config(ctypes.byref(g))
    return g


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
            f"no libsnowllm.so at {lib}. If that is a source checkout, build it "
            f"(`cmake --build build-release`); if it is an install, reinstall the wheel."
        )
    return lib


def _preload_hip_runtime() -> None:
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return
    sdk = os.path.join(spec.submodule_search_locations[0], "lib", "libamdhip64.so.7")
    if os.path.exists(sdk):
        try:
            ctypes.CDLL(sdk, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_hip_runtime()
_lib_path = _find_lib()
lib = ctypes.CDLL(_lib_path)

ABI_VERSION = 7


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

P, S = c_void_p, c_void_p
_SIGS = [
    ("snowllm_model_config", None, [POINTER(BuildGeometry)]),
    ("snowllm_shuffle_bytes", c_int64, [c_int64]),
    ("snowllm_prefill_row_quantum", c_int64, []),
    ("snowllm_sampling_max_k", c_int64, []),
    ("snowllm_error_string", c_char_p, [c_int]),
    ("snowllm_synchronize", c_int, [S]),

    ("snowllm_qkv_proj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_qkv_proj_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_qkv_proj_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_attn_out_scale_oproj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_attn_out_scale_oproj_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_attn_out_scale_oproj_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_linear_in_proj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_linear_out_proj_shuffle_w", c_int, [P, P, S]),
    ("snowllm_linear_out_proj_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_linear_in_proj_qz_shuffle_w_fp8", c_int, [P, P, S]),
    ("snowllm_linear_in_proj_ba_shuffle_w", c_int, [P, P, S]),
    ("snowllm_fused_linear_attn_workspace_bytes", c_int64, [c_int64]),
    ("snowllm_mtp_fc_shuffle_w", c_int, [P, P, S]),
    ("snowllm_mtp_fc_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_mtp_pre_fc", c_int, [P, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_mtp_fc", c_int, [P, P, P, P, c_int64, c_int, S]),
    ("snowllm_moe_shuffle_gate_up", c_int, [P, P, P, c_int64, S]),
    ("snowllm_moe_shuffle_down", c_int, [P, P, c_int64, S]),
    ("snowllm_moe_shuffle_gate_up_fused", c_int, [P, P, c_int64, S]),
    ("snowllm_proj_scale_shuffle_fp8", c_int, [P, P, c_int64, c_int64, S]),
    ("snowllm_moe_scale_shuffle_gate_up_fp8", c_int, [P, P, P, c_int64, S]),
    ("snowllm_moe_scale_shuffle_down_fp8", c_int, [P, P, c_int64, S]),
    ("snowllm_moe_shuffle_gate_up_fp8", c_int, [P, P, P, c_int64, S]),
    ("snowllm_moe_shuffle_down_fp8", c_int, [P, P, c_int64, S]),
    ("snowllm_moe_shuffle_router_bytes", c_int64, []),
    ("snowllm_moe_shuffle_router", c_int, [P, P, S]),
    ("snowllm_moe_workspace_bytes", c_int64, [c_int64]),
    ("snowllm_lm_head_shuffle_weight", c_int, [P, P, S]),
    ("snowllm_lm_head_rows_for", c_int64, [c_int64]),
    ("snowllm_lm_head_pad_bytes", c_int64, [c_int64]),
    ("snowllm_lm_head_scratch_bytes", c_int64, [c_int64]),
    ("snowllm_lm_head", c_int, [P, P, P, c_int64, P, P, S]),
    ("snowllm_paged_prefill_max_tokens", c_int64, []),
    ("snowllm_paged_decode_workspace_size", c_int64, [c_int64, c_int64]),
    ("snowllm_kv_pool_bytes", None, [c_int64, c_int, P, P]),
    ("snowllm_kv_scale_bytes", None, [c_int64, P, P]),
    ("snowllm_resolve_slots", c_int, [P, c_int64, P, P, P, c_int64, S]),
    ("snowllm_kv_blocks_for", c_int64, [c_int64]),
    ("snowllm_linear_state_slots", None, [c_int64, c_int64, c_int64, c_int64, P]),
    ("snowllm_prefill_q_blocks", c_int64, [P, c_int64]),
    ("snowllm_prefill_q_block_map", None, [P, c_int64, P]),
    ("snowllm_moe_variant_force", None, [c_int]),
    ("snowllm_moe_variant_name", c_char_p, [c_int]),
    ("snowllm_paged_window_blocks", c_int64, [c_int64, c_int64]),
    ("snowllm_paged_window_view", c_int, [P, c_int64, P, c_int64, c_int64, c_int64, P,
                                          c_int64, P, S]),
    ("snowllm_paged_decode_max_q_tokens", c_int64, []),
    ("snowllm_paged_decode_num_slots", c_int64, [c_int64]),
    ("snowllm_paged_decode_plan_elems", c_int64, [c_int64, c_int64]),

    ("snowllm_gather_embedding", c_int, [P, P, P, c_int64, S]),
    ("snowllm_rmsnorm", c_int, [P, P, P, c_int64, c_float, S]),
    ("snowllm_rmsnorm_residual", c_int, [P, P, P, P, c_int64, c_float, S]),
    ("snowllm_rmsnorm_shuffled", c_int, [P, P, P, c_int64, c_float, S]),
    ("snowllm_rmsnorm_residual_shuffled", c_int, [P, P, P, P, c_int64, c_float, S]),
    ("snowllm_qkv_proj", c_int, [P, P, P, P, c_int64, c_int, S]),
    ("snowllm_qkv_proj_fp8", c_int, [P, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_qk_norm", c_int, [P, c_int64, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_qk_norm_rope", c_int, [P, c_int64, P, P, P, P, P, P, c_int64, c_float, S]),
    ("snowllm_attn_out_scale_oproj", c_int, [P, P, c_int64, P, P, P, c_int64, c_int, S]),
    ("snowllm_attn_out_scale_oproj_fp8", c_int, [P, P, c_int64, P, P, P, P, c_int64, c_int, S]),
    ("snowllm_rope_cos_sin", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_rope_cos_sin_mrope", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_rope_apply_q", c_int, [P, P, P, c_int64, S]),
    ("snowllm_rope_apply_k", c_int, [P, P, P, c_int64, S]),
    ("snowllm_reshape_and_cache", c_int, [P, P, c_int64, c_int64, P, P, P, c_int64, S]),
    ("snowllm_reshape_and_cache_int8", c_int,
     [P, P, c_int64, c_int64, P, P, P, P, P, c_int64, S]),
    ("snowllm_paged_attn_prefill_int8", c_int,
     [P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, S, P]),
    ("snowllm_paged_attn_decode_int8", c_int,
     [P, P, P, P, P, P, P, P, P, c_int64, c_int64, c_int64, P, c_float, c_int64, S]),
    ("snowllm_paged_attn_prefill", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, c_float, S, P]),
    ("snowllm_paged_attn_decode", c_int,
     [P, P, P, P, P, P, P, c_int64, c_int64, c_int64, P, c_float, c_int64, S]),
    ("snowllm_paged_attn_decode_plan", c_int, [P, P, c_int64, c_int64, c_int, S]),
    ("snowllm_fused_linear_attn", c_int,
     [P, P, P, P, P, P, P,
      P, P, P, P,
      P, P, P, P,
      P, P,
      P, P, c_int64, P, P,
      P, P,
      c_int64, c_int64, c_int, S]),
    ("snowllm_fused_moe", c_int, [P, P, P, P, P, P, c_int64, S]),
    ("snowllm_fused_moe_fp8", c_int, [P, P, P, P, P, P, P, P, c_int64, S]),
    ("snowllm_lm_head_gemm_decode", c_int, [P, P, P, c_int64, S]),
    ("snowllm_lm_head_gemm_prefill", c_int, [P, P, P, P, c_int64, S]),
    ("snowllm_sampling_softmax", c_int, [P, P, c_int64, c_int64, P, S]),
    ("snowllm_sampling_topk_topp", c_int, [P, P, P, c_int64, c_int64, c_int64, P, P, S]),
    ("snowllm_sampling_multinomial", c_int, [P, P, P, P, c_int64, c_int64, S]),
    ("snowllm_sampling_argmax", c_int, [P, P, c_int64, c_int64, S]),

    ("snowllm_gemm_bf16_shuffle_b", c_int, [P, P, c_int64, c_int64, S]),
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
    _fn.argtypes = _args
if _missing:
    raise SnowLLMError(f"{_lib_path} is missing {len(_missing)} entry points this binding needs:\n  "
                       + "\n  ".join(_missing))


def check(status: int, what: str) -> None:
    if status != 0:
        msg = lib.snowllm_error_string(status).decode()
        raise SnowLLMError(f"{what}: hip status {status}: {msg}")


def synchronize(stream: int = 0) -> None:
    check(lib.snowllm_synchronize(stream), "synchronize")


def assert_single_hip_runtime() -> None:
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            i = line.find("libamdhip64.so")
            if i != -1:
                paths.add(os.path.realpath(line.rsplit(" ", 1)[-1].strip()))
    if len(paths) > 1:
        raise SnowLLMError(
            "two HIP runtimes are loaded in this process -- torch's device pointers are not valid "
            "in snowllm's kernels:\n  " + "\n  ".join(sorted(paths))
        )
