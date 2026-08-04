# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import dataclass

GEMM_N_QUANTUM = 128


def _align(n: int, a: int = GEMM_N_QUANTUM) -> int:
    return ((n + a - 1) // a) * a


@dataclass(frozen=True)
class ModelGeometry:
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    vocab_size: int
    num_layers: int

    attn_output_gate: bool

    lin_num_k_heads: int
    lin_num_v_heads: int
    lin_head_k: int
    lin_head_v: int
    lin_conv_k: int

    moe_num_experts: int
    moe_inter: int
    moe_topk: int

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelGeometry":
        return cls(
            hidden=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_size=cfg["head_dim"],
            vocab_size=cfg["vocab_size"],
            num_layers=cfg["num_hidden_layers"],
            attn_output_gate=bool(cfg.get("attn_output_gate", False)),
            lin_num_k_heads=cfg["linear_num_key_heads"],
            lin_num_v_heads=cfg["linear_num_value_heads"],
            lin_head_k=cfg["linear_key_head_dim"],
            lin_head_v=cfg["linear_value_head_dim"],
            lin_conv_k=cfg["linear_conv_kernel_dim"],
            moe_num_experts=cfg["num_experts"],
            moe_inter=cfg["moe_intermediate_size"],
            moe_topk=cfg["num_experts_per_tok"],
        )

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_size

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_size

    @property
    def qkv_proj_n(self) -> int:
        return (2 if self.attn_output_gate else 1) * self.q_dim + 2 * self.kv_dim

    @property
    def qkv_q_head_stride(self) -> int:
        return (2 if self.attn_output_gate else 1) * self.head_size

    @property
    def qkv_off_scale(self) -> int:
        return self.head_size

    @property
    def qkv_off_k(self) -> int:
        return (2 if self.attn_output_gate else 1) * self.q_dim

    @property
    def qkv_off_v(self) -> int:
        return self.qkv_off_k + self.kv_dim

    @property
    def lin_key_dim(self) -> int:
        return self.lin_num_k_heads * self.lin_head_k

    @property
    def lin_value_dim(self) -> int:
        return self.lin_num_v_heads * self.lin_head_v

    @property
    def lin_conv_dim(self) -> int:
        return 2 * self.lin_key_dim + self.lin_value_dim

    @property
    def lin_conv_state(self) -> int:
        return self.lin_conv_k - 1

    @property
    def lin_in_proj_n(self) -> int:
        return self.lin_conv_dim + self.lin_value_dim + 2 * self.lin_num_v_heads

    @property
    def lin_in_proj_n_pad(self) -> int:
        return _align(self.lin_in_proj_n)

    @property
    def lin_in_proj_ba_n_pad(self) -> int:
        return _align(2 * self.lin_num_v_heads)

    @property
    def lin_qz_n(self) -> int:
        return self.lin_conv_dim + self.lin_value_dim

    @property
    def lin_off_qkv(self) -> int:
        return 0

    @property
    def lin_off_z(self) -> int:
        return self.lin_conv_dim

    @property
    def lin_off_b(self) -> int:
        return self.lin_off_z + self.lin_value_dim

    @property
    def lin_off_a(self) -> int:
        return self.lin_off_b + self.lin_num_v_heads

    @property
    def moe_num_slabs(self) -> int:
        return self.moe_num_experts + 1

    @property
    def moe_shared_expert(self) -> int:
        return self.moe_num_experts

    @property
    def mtp_fc_k(self) -> int:
        return 2 * self.hidden
