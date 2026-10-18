# SPDX-License-Identifier: Apache-2.0
import torch

from ... import ops, term
from ..._capi import GEO_GLM53_FLASH, SnowLLMError, build_geometry
from ...checkpoint.gguf.source import KQUANT_ID
from .. import placement
from ..geometry import is_gemm_n_aligned
from .layers import (DenseMLP, Glm5NextForConditionalGeneration,
                     HyperConnection, Layer, LinearAttention, MoE, Projection, SparseAttention,
                     Glm5NextMTP)


def validate_config(c):
    ops.select_geometry(GEO_GLM53_FLASH)
    build = build_geometry()
    expected = {"hidden_size": build.hidden, "vocab_size": build.vocab_size,
                "num_attention_heads": build.num_heads,
                "n_routed_experts": build.moe_num_experts,
                "num_experts_per_tok": build.moe_topk,
                "moe_intermediate_size": build.moe_inter,
                "n_shared_experts": build.moe_num_slabs-build.moe_num_experts,
                "q_lora_rank": 1536, "kv_lora_rank": 512,
                "qk_nope_head_dim": 256, "qk_rope_head_dim": 0, "v_head_dim": 256,
                "linear_head_dim": 128, "linear_conv_kernel_dim": 4, "linear_lower_bound": -5.0,
                "hc_mult": 4, "scoring_func": "sigmoid", "norm_topk_prob": True,
                "routed_scaling_factor": 2.5, "index_kpool": 4,
                "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048}
    bad = [f"{k}={c.get(k)!r} (want {v!r})" for k, v in expected.items() if c.get(k) != v]
    n = c["num_hidden_layers"]
    if n <= 0 or len(c["layer_types"]) != n or any(
            t not in ("linear_attention", "deepseek_sparse_attention") for t in c["layer_types"]):
        bad.append("layer_types must describe each trunk layer")
    if "deepseek_sparse_attention" not in c["layer_types"]:
        bad.append("the hybrid GLM path needs at least one sparse trunk layer")
    if not 0 <= c["first_k_dense_replace"] <= n:
        bad.append("first_k_dense_replace must be within the trunk")
    if len(c["swiglu_limit"]) != n or any(v != 10.0 for v in c["swiglu_limit"]):
        bad.append("the compiled SwiGLU clamp requires 10.0 for each trunk layer")
    shared = c.get("swiglu_limit_shexp", c["swiglu_limit"])
    if len(shared) != n or any(v != 10.0 for v in shared):
        bad.append("the compiled shared-expert SwiGLU clamp requires 10.0 for each trunk layer")
    if c.get("tie_word_embeddings", False):
        bad.append("tied embedding/LM-head weights are not supported")
    if not c.get("index_kpool_always_select_tail", True):
        bad.append("the compiled sparse indexer requires selecting the incomplete tail")
    indexers = c.get("indexer_types", [True] * n)
    if len(indexers) != n or any(not full for t, full in zip(c["layer_types"], indexers)
                               if t == "deepseek_sparse_attention"):
        bad.append("the sparse attention path requires a full indexer in every sparse layer")
    if bad:
        raise SnowLLMError("unsupported GLM5-Next geometry: " + ", ".join(bad))


def projection(rd, name):
    t = rd.gguf[name]
    n, k = t.shape
    fmt = KQUANT_ID.get(t.quant.name)
    if fmt is None or not is_gemm_n_aligned(n):
        raise SnowLLMError(f"{name}: GLM quantized projection needs a supported K-quant format "
                          f"and 128-aligned output rows, got {t.quant.name} {t.shape}")
    q, meta = ops.gemm_kquant_shuffle_b(fmt, rd.raw(name), n, k)
    return Projection(n, k, fmt, q, meta)


def _narrow_projection(rd, name, quant, rows, columns, dtype):
    t = rd.gguf[name]
    if t.quant.name != quant or t.shape != (rows, columns):
        raise SnowLLMError(f"{name}: GLM narrow projection needs {quant} {(rows, columns)}, "
                          f"got {t.quant.name} {t.shape}")
    return Projection(rows, columns, dense=rd.tensor(name, dtype))


def _hyper(rd, p, side, c):
    return HyperConnection(rd.tensor(p+f"hc_{side}_fn.weight", torch.float32),
                           rd.tensor(p+f"hc_{side}_scale.weight", torch.float32),
                           rd.tensor(p+f"hc_{side}_base.weight", torch.float32), c)


def _moe(rd, p, cfg, dmap, i):
    experts, shared = cfg["n_routed_experts"], cfg["n_shared_experts"]
    gate_name = rd.gguf[p+"ffn_gate_exps.weight"].quant.name
    up_name = rd.gguf[p+"ffn_up_exps.weight"].quant.name
    down_name = rd.gguf[p+"ffn_down_exps.weight"].quant.name
    if gate_name != up_name or gate_name not in (ops.LOWBIT_FORMATS | ops.KQUANT_FORMATS) or down_name not in (ops.LOWBIT_FORMATS | ops.KQUANT_FORMATS):
        raise SnowLLMError(f"{p}: unsupported GLM expert formats {gate_name}/{up_name}/{down_name}; "
                           "gate/up must share a supported quantization format")
    shared_fmt = {name: rd.gguf[p+f"ffn_{name}_shexp.weight"].quant.name
                  for name in ("gate", "up", "down")}
    if (any(fmt not in ops.KQUANT_FORMATS for fmt in shared_fmt.values()) or
            shared_fmt["gate"] != shared_fmt["up"]):
        raise SnowLLMError(f"{p}: unsupported GLM shared-expert formats {shared_fmt}; "
                          "gate/up must use the same supported K-quant format")
    router = ops.moe_shuffle_router(rd.tensor(p+"ffn_gate_inp.weight"),
                                   rd.tensor(p+"exp_probs_b.bias", torch.float32))
    with placement.place(dmap, "experts", i):
        gate, up = rd.transient(p+"ffn_gate_exps.weight", p+"ffn_up_exps.weight")
        gate_up = (ops.moe_lowbit_shuffle_gate_up_split(gate, up, ops.LOWBIT_FORMATS[gate_name], experts)
                   if gate_name in ops.LOWBIT_FORMATS else
                   ops.moe_kquant_shuffle_gate_up(gate, up, ops.KQUANT_FORMATS[gate_name], experts))
        raw, = rd.transient(p+"ffn_down_exps.weight")
        fmt = rd.gguf[p+"ffn_down_exps.weight"].quant.name
        down = (ops.moe_lowbit_shuffle_down(raw, ops.LOWBIT_FORMATS[fmt], experts)
                if fmt in ops.LOWBIT_FORMATS else
                ops.moe_kquant_shuffle_down(raw, ops.KQUANT_FORMATS[fmt], experts))
    with placement.place(dmap, "shared", i):
        gate, up = rd.transient(p+"ffn_gate_shexp.weight", p+"ffn_up_shexp.weight")
        sfmt = ops.KQUANT_FORMATS[rd.gguf[p+"ffn_gate_shexp.weight"].quant.name]
        shared_gu = ops.moe_kquant_shuffle_gate_up(gate, up, sfmt, shared)
        name = p+"ffn_down_shexp.weight"
        shared_down = ops.moe_kquant_shuffle_down(rd.raw(name), ops.KQUANT_FORMATS[rd.gguf[name].quant.name], shared)
    return MoE(router, gate_up, down, shared_gu, shared_down)


def _sparse(rd, p, cfg, dmap, layer):
    with placement.place(dmap, "dense", layer):
        qa, qb, kva, out, iq, ik, gate = [projection(rd, p+n) for n in
            ("attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight",
             "indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer_compressor_gate.weight")]
    read = lambda n, dtype=torch.bfloat16: rd.tensor(p+n,dtype)
    return SparseAttention(qa, qb, kva, read("attn_q_a_norm.weight"),
        read("attn_kv_a_norm.weight"), read("attn_k_b.weight"), read("attn_v_b.weight"),
        out, iq, ik, read("indexer.k_norm.weight"), read("indexer.k_norm.bias"),
        _narrow_projection(rd, p+"indexer.proj.weight", "F32", cfg["index_n_heads"],
                           cfg["hidden_size"], torch.float32),
        gate, read("indexer_compressor_ape.weight",torch.float32),cfg)


def _linear(rd, p, cfg, dmap, layer):
    with placement.place(dmap, "dense", layer):
        q, k, v, fa, fb, ga, gb, out = [projection(rd, p+n) for n in
            ("attn_q.weight", "attn_k.weight", "attn_v.weight", "ssm_f_a.weight", "ssm_f_b.weight",
             "ssm_g_a.weight", "ssm_g_b.weight", "attn_output.weight")]
    read = lambda n, dtype=torch.bfloat16: rd.tensor(p+n, dtype)
    return LinearAttention([q, k, v], fa, fb,
        _narrow_projection(rd, p+"ssm_beta.weight", "Q8_0", cfg["num_attention_heads"],
                           cfg["hidden_size"], torch.bfloat16), ga, gb, out,
        torch.cat([read(f"ssm_conv1d_{r}.weight", torch.float32).reshape(-1, cfg["linear_conv_kernel_dim"])
                   for r in ("q", "k", "v")]),
        # GGUF stores -exp(A_log), while KDA accepts A_log.
        torch.log(-read("ssm_a", torch.float32)), read("ssm_dt.bias", torch.float32),
        read("ssm_norm.weight"), cfg)


def _validate_mtp(cfg):
    if cfg.get("num_nextn_predict_layers", 0) != 1:
        raise SnowLLMError("GLM MTP needs exactly one appended NextN block")
    for key in ("mtp_swiglu_limit", "mtp_swiglu_limit_shexp"):
        limits = cfg.get(key, [10.0])
        if len(limits) != 1 or limits[0] != 10.0:
            raise SnowLLMError(f"the compiled {key} requires 10.0 for the NextN layer")


def load_mtp(rd, cfg, dmap, at):
    _validate_mtp(cfg)
    p = f"blk.{at}."
    independent = [p+f"nextn.{name}.weight" for name in ("embed_tokens", "shared_head_head")
                   if p+f"nextn.{name}.weight" in rd.gguf]
    if independent:
        raise SnowLLMError("GLM NextN currently shares the trunk embedding and LM head; "
                          "independent weights are not supported: " + ", ".join(independent))
    attn = _sparse(rd, p, cfg, dmap, at)
    with placement.place(dmap, "dense", at):
        fc = projection(rd, p+"nextn.eh_proj.weight")
    names = [p+n for n in ("nextn.enorm.weight", "nextn.hnorm.weight", "attn_norm.weight",
                          "ffn_norm.weight")]
    head_norm = p+"nextn.shared_head_norm.weight"
    names.append(head_norm if head_norm in rd.gguf else "output_norm.weight")
    norms = [rd.tensor(name, torch.bfloat16) for name in names]
    return Glm5NextMTP(attn, _moe(rd, p, cfg, dmap, at), fc, *norms, cfg["rms_norm_eps"])


def _transient_need(rd, layers):
    need = 0
    for i in layers:
        for kind in ("exps", "shexp"):
            names = [f"blk.{i}.ffn_{proj}_{kind}.weight" for proj in ("gate", "up", "down")]
            if names[0] in rd.gguf:
                gate, up, down = (rd.gguf[name].nbytes for name in names)
                need = max(need, gate+up, down)
    return need


def load_gguf_weights(rd, cfg, want, device_map=None, mtp=False):
    validate_config(cfg)
    n = cfg["num_hidden_layers"]
    if tuple(want) != tuple(range(n)):
        raise SnowLLMError(f"GLM serving requires all {n} trunk layers")
    with_mtp = mtp and cfg.get("num_nextn_predict_layers", 0) > 0
    if with_mtp:
        _validate_mtp(cfg)
    dmap = device_map
    rd.reserve_transient(_transient_need(rd, range(n + int(with_mtp))))
    embed = rd.tensor("token_embd.weight")
    with placement.place(dmap, "head"):
        head = projection(rd, "output.weight")
    layers = []
    for i in range(n):
        p = f"blk.{i}."
        read = lambda n, dtype=torch.bfloat16: rd.tensor(p+n, dtype)
        attn = (_linear(rd, p, cfg, dmap, i) if cfg["layer_types"][i] == "linear_attention"
                else _sparse(rd, p, cfg, dmap, i))
        with placement.place(dmap, "dense", i):
            mlp = (DenseMLP(*(projection(rd, p+n) for n in
                             ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")))
                   if i < cfg["first_k_dense_replace"] else None)
        ah, fh = _hyper(rd, p, "attn", cfg), _hyper(rd, p, "ffn", cfg)
        an, fn = read("attn_norm.weight"), read("ffn_norm.weight")
        if mlp is None:
            mlp = _moe(rd, p, cfg, dmap, i)
        layers.append(Layer(attn, mlp, ah, fh, an, fn, cfg["rms_norm_eps"]))
        if ops.drying() is None:
            torch.cuda.synchronize()
            print(f"{term.stamp()} GLM layer {i+1}/{n}", flush=True)
    model = Glm5NextForConditionalGeneration(cfg, layers, embed, rd.tensor("output_norm.weight"), head)
    if with_mtp:
        model.mtp = load_mtp(rd, cfg, dmap, n)
        if ops.drying() is None:
            print(f"{term.stamp()} GLM NextN/MTP block loaded",flush=True)
    rd.drop_transient()
    model.device_map = dmap
    return model
