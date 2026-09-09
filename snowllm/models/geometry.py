# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

from dataclasses import dataclass, replace

from ..checkpoint.gguf.deepseek4 import CSA, HCA, INDEXED_RATIO, SWA

GEMM_N_QUANTUM = 128

COFF = {4: 2, 128: 1}


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

    moe_num_experts: int = 0
    moe_inter: int = 0
    moe_topk: int = 0
    mlp_inter: int = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelGeometry":
        moe = "num_experts" in cfg
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
            moe_num_experts=cfg["num_experts"] if moe else 0,
            moe_inter=cfg["moe_intermediate_size"] if moe else 0,
            moe_topk=cfg["num_experts_per_tok"] if moe else 0,
            mlp_inter=0 if moe else cfg["intermediate_size"],
        )

    @property
    def is_moe(self) -> bool:
        return self.moe_num_experts > 0

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
    def mlp_gate_up_n(self) -> int:
        return 2 * self.mlp_inter

    @property
    def mtp_fc_k(self) -> int:
        return 2 * self.hidden


@dataclass(frozen=True)
class DeepSeekV4Geometry:
    hidden: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    v_head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    vocab_size: int
    sliding_window: int
    compress_ratios: tuple
    compress_rope_theta: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    num_hash_layers: int
    moe_num_experts: int
    moe_inter: int
    moe_topk: int
    moe_num_shared: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    scoring_func: str
    swiglu_limit: float
    swiglu_limit_shexp: float
    rope_parameters: tuple
    eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_config(cls, cfg: dict) -> "DeepSeekV4Geometry":
        ratios = [int(r) for r in cfg["compress_ratios"]]
        return cls(
            hidden=cfg["hidden_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_size=cfg["head_dim"],
            v_head_dim=cfg["v_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            q_lora_rank=cfg["q_lora_rank"],
            o_lora_rank=cfg["o_lora_rank"],
            o_groups=cfg["o_groups"],
            vocab_size=cfg["vocab_size"],
            sliding_window=cfg["sliding_window"],
            compress_ratios=tuple(ratios),
            compress_rope_theta=float(cfg["compress_rope_theta"]),
            index_n_heads=cfg["index_n_heads"],
            index_head_dim=cfg["index_head_dim"],
            index_topk=cfg["index_topk"],
            hc_mult=cfg["hc_mult"],
            hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
            hc_eps=float(cfg["hc_eps"]),
            num_hash_layers=cfg.get("num_hash_layers", 0),
            moe_num_experts=cfg["n_routed_experts"],
            moe_inter=cfg["moe_intermediate_size"],
            moe_topk=cfg["num_experts_per_tok"],
            moe_num_shared=cfg.get("n_shared_experts", 0),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", False)),
            scoring_func=cfg.get("scoring_func", "softmax"),
            swiglu_limit=float(cfg.get("swiglu_limit", 0.0)),
            swiglu_limit_shexp=float(cfg.get("swiglu_limit_shexp", 0.0)),
            rope_parameters=tuple(sorted(cfg["rope_parameters"].items())),
            eps=float(cfg["rms_norm_eps"]),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )

    @property
    def is_moe(self) -> bool:
        return True

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_size

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_size

    @property
    def moe_num_slabs(self) -> int:
        return self.moe_num_experts + self.moe_num_shared

    @property
    def moe_shared_expert(self) -> int:
        return self.moe_num_experts

    @property
    def rope(self) -> dict:
        return dict(self.rope_parameters)

    @property
    def layer_types(self) -> tuple:
        return tuple(SWA if r == 0 else CSA if r == INDEXED_RATIO else HCA
                     for r in self.compress_ratios)

    def is_indexed(self, layer: int) -> bool:
        return self.compress_ratios[layer] == INDEXED_RATIO

    def is_hashed(self, layer: int) -> bool:
        return layer < self.num_hash_layers


@dataclass(frozen=True)
class DFlashGeometry:
    hidden: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    intermediate: int
    sliding_window: int
    num_sliding_layers: int
    tap_layers: tuple
    mask_token_id: int
    num_target_layers: int
    block_size: int
    rope_theta: float
    eps: float

    conv_taps: int = 0
    conv_group: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "DFlashGeometry":
        d = cfg["dflash_config"]
        types = cfg.get("layer_types") or []
        return cls(
            conv_taps=int(d.get("conv_kernel_size", 0)),
            conv_group=int(d.get("conv_group_size", 0)),
            selector_rank=int(d.get("selector_rank", 0)),
            selector_top_k=int(d.get("selector_top_k", 0)),
            hidden=cfg["hidden_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_size=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            intermediate=cfg["intermediate_size"],
            sliding_window=cfg["sliding_window"],
            num_sliding_layers=sum(t == "sliding_attention" for t in types),
            tap_layers=tuple(d["target_layer_ids"]),
            mask_token_id=d["mask_token_id"],
            num_target_layers=cfg["num_target_layers"],
            block_size=d["block_size"],
            rope_theta=float(cfg["rope_parameters"]["rope_theta"]),
            eps=float(cfg["rms_norm_eps"]),
        )

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_size

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_size

    @property
    def qkv_proj_n(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @property
    def ctx_kv_proj_n(self) -> int:
        return self.num_layers * 2 * self.kv_dim

    @property
    def gate_up_n(self) -> int:
        return 2 * self.intermediate

    @property
    def num_taps(self) -> int:
        return len(self.tap_layers)

    @property
    def fc_k(self) -> int:
        return self.num_taps * self.hidden

    @property
    def dflash2(self) -> bool:
        return self.conv_taps > 0

    @property
    def conv_groups(self) -> int:
        return self.hidden // self.conv_group if self.conv_group else 0

    @property
    def conv_proj_n(self) -> int:
        return 2 * self.conv_taps * self.conv_groups


@dataclass(frozen=True)
class DSparkGeometry:
    stack: DeepSeekV4Geometry
    block_size: int
    markov_rank: int
    mask_token_id: int
    tap_layers: tuple

    @classmethod
    def from_config(cls, cfg: dict) -> "DSparkGeometry":
        return cls(
            stack=DeepSeekV4Geometry.from_config(cfg),
            block_size=int(cfg["dspark_block_size"]),
            markov_rank=int(cfg["dspark_markov_rank"]),
            mask_token_id=int(cfg["dspark_noise_token_id"]),
            tap_layers=tuple(int(i) for i in cfg["dspark_target_layer_ids"]),
        )

    @property
    def num_taps(self) -> int:
        return len(self.tap_layers)

    @property
    def fc_k(self) -> int:
        return self.num_taps * self.stack.hidden

    @property
    def windowed_only(self) -> bool:
        return not any(self.stack.compress_ratios)

    def with_taps(self, taps: tuple) -> "DSparkGeometry":
        return replace(self, tap_layers=tuple(taps))


@dataclass(frozen=True)
class Qwen4ExpGeometry(ModelGeometry):
    layer_types: tuple = ()
    hc_count: int = 0
    hc_lowrank: int = 0
    index_n_heads: int = 0
    index_kv_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    index_ratio: int = 0
    ple_layers: tuple = ()
    ple_heads: int = 0
    ple_head_dim: int = 0
    ple_conv_k: int = 0
    ngram_size: int = 0
    ngram_offsets: tuple = ()
    ngram_vocab_sizes: tuple = ()
    ngram_multipliers: tuple = ()
    ple_eos_token_id: int = 0
    eps: float = 0.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "Qwen4ExpGeometry":
        base = ModelGeometry.from_config(cfg)
        return cls(
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            layer_types=tuple(cfg["layer_types"]),
            hc_count=cfg["hc_count"],
            hc_lowrank=cfg["hc_lowrank"],
            index_n_heads=cfg["indexer_n_heads"],
            index_kv_heads=cfg["indexer_kv_heads"],
            index_head_dim=cfg["indexer_head_dim"],
            index_topk=cfg["indexer_budget"],
            index_ratio=cfg["indexer_compress_ratio"],
            ple_layers=tuple(cfg["ple_layers"]),
            ple_heads=len(cfg["ngram_head_offsets"]),
            ple_head_dim=cfg["ngram_head_dim"],
            ple_conv_k=cfg["ple_conv_kernel_size"],
            ngram_size=cfg["ngram_size"],
            ngram_offsets=tuple(cfg["ngram_head_offsets"]),
            ngram_vocab_sizes=tuple(cfg["ngram_head_vocab_sizes"]),
            ngram_multipliers=tuple(cfg["ngram_multipliers"]),
            ple_eos_token_id=cfg["ple_eos_token_id"],
            eps=float(cfg["rms_norm_eps"]),
            tie_word_embeddings=bool(cfg["tie_word_embeddings"]),
        )

    @property
    def hc_dim(self) -> int:
        return self.hc_count * self.hidden

    @property
    def full_layers(self) -> tuple:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @property
    def linear_layers(self) -> tuple:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def index_blocks(self) -> int:
        return self.index_topk // self.index_ratio

    @property
    def ple_embed_dim(self) -> int:
        return self.ple_heads * self.ple_head_dim
