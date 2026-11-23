# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ... import ops
from ..._platform import gpu_mem_info
from ...checkpoint.gguf.inkling2 import config
from ...checkpoint.gguf.source import GGUFReader
from ...errors import SnowLLMError
from ...ops.inkling import attention, loop_control, residual_mix
from .experts import InklingMoE


def rms(x, eps, weight=None):
    return F.rms_norm(x.float(), (x.shape[-1],), weight, eps)


def mix(blocks, current, query, eps, prefill):
    states = torch.stack((*blocks, current), dim=1)
    return residual_mix(states, query.view(-1, current.size(-1)), eps, prefill)


def clip(x, limit):
    return x * (limit / (x.norm(dim=-1, keepdim=True) + 1e-6)).clamp_max_(1)


def convolve(x, history, weight, dynamic=None, plan=None):
    taps = weight.size(1)
    if plan is None:
        sequence = torch.cat((history, x), dim=0)
        history.copy_(sequence[-history.size(0):])
        windows = sequence.unfold(0, taps, 1).float()
    else:
        from ...ops.inkling import short_conv
        return short_conv(x, weight, dynamic, history, *plan)
    out = (windows * weight).sum(-1) + x
    if dynamic is not None:
        taps_by_head = dynamic.view(x.size(0), -1, 1, taps).flip((-1,)).float()
        by_head = windows.reshape(x.size(0), taps_by_head.size(1), -1, taps)
        out.add_((by_head * taps_by_head).sum(-1).flatten(1))
    return out.mul_(1 / math.sqrt((2 if dynamic is not None else 1) * taps + 1)).bfloat16()


class LayerState:
    def __init__(self, cfg, pages, page_size, device):
        dim, kv = cfg["head_dim"], cfg["num_key_value_heads"]
        shape = (pages, page_size, kv, dim)
        self.k, self.v = (torch.empty(shape, device=device, dtype=torch.bfloat16) for _ in range(2))
        channels = (kv * dim, kv * dim, cfg["hidden_size"], cfg["hidden_size"])
        self.history = [torch.zeros((cfg["sconv_kernel_size"] - 1, width), dtype=torch.bfloat16,
                                    device=device) for width in channels]


class InklingCache:
    def __init__(self, model, capacity, chunk_size):
        cfg = model.config
        if not 0 < chunk_size <= capacity <= cfg["max_position_embeddings"]:
            raise ValueError("Inkling cache requires 0 < chunk_size <= capacity <= model context")
        self.model, self.capacity, self.chunk_size = model, capacity, chunk_size
        self.position = 0
        page_size = 16
        full = math.ceil(capacity / page_size)
        # The whole chunk is stored before attention reads the oldest visible key.
        local = min(full, math.ceil((cfg["sliding_window"] - 1 + chunk_size) / page_size))
        device = model.embedding.device
        self.tables = {size: torch.arange(size, dtype=torch.int32, device=device).view(1, -1)
                       for size in {local, full}}
        self.states = [LayerState(cfg, local if model.layers[i].local else full, page_size, device)
                       for i in model.visits]


class InklingLayer:
    def __init__(self, rd, cfg, index):
        self.cfg = cfg
        self.local = cfg["layer_types"][index] == "sliding_attention"
        self.block_start = index % cfg["n_layers_per_attn_res_block"] == 0
        prefix = f"blk.{index}."
        self.weights = {}
        for name, tensor in rd.gguf.tensors.items():
            if not name.startswith(prefix) or not name.endswith(".weight"):
                continue
            key = name[len(prefix):-len(".weight")]
            if key in ("ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_inp"):
                continue
            if tensor.quant.name not in ("F32", "BF16"):
                raise SnowLLMError(f"{name}: unsupported dense Inkling weight {tensor.quant.name}")
            dtype = torch.float32 if tensor.quant.name == "F32" else torch.bfloat16
            self.weights[key] = rd.tensor(name, dtype)
        self.moe = InklingMoE(rd, cfg, index)

    def linear(self, x, name):
        weight = self.weights[name]
        return F.linear(x.to(weight.dtype), weight)

    def control(self, x, msg, name, prefill):
        stem = f"loop_{name}_ctrl_"
        w = self.weights
        return loop_control(x, msg, w[stem + "down"], w[stem + "forget"], w[stem + "gain"],
                            self.cfg["rms_norm_eps"], prefill)

    def attend(self, inputs, state, table, position, sequences, indices, plan):
        cfg, w = self.cfg, self.weights
        heads, kv, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        q_in, k_in, v_in = inputs
        q = self.linear(q_in, "attn_q").view(-1, heads, dim)
        q = clip(q.float() * w["attn_q_clip"], cfg["qk_l2_clip_threshold"])
        r = self.linear(q_in, "attn_rel").view(-1, heads, cfg["d_rel"])
        r = clip(r.float() * w["attn_rel_clip"], cfg["rel_l2_clip_threshold"])
        if not self.local:
            tau = ((position.float() + 1) / cfg["log_scaling_n_floor"]).clamp_min_(1).log_()
            tau = tau.mul_(cfg["log_scaling_alpha"]).add_(1).view(-1, 1, 1)
            q, r = q * tau, r * tau
        for j, (name, x, dest) in enumerate((("k", k_in, state.k), ("v", v_in, state.v))):
            stem = "attn_" + name
            dynamic = None if self.local else self.linear(x, stem + "_conv_proj")
            projected = convolve(self.linear(x, stem), state.history[j], w[stem + "_conv"], dynamic, plan)
            rows = projected.view(-1, kv, dim).float()
            rows = clip(rows, cfg["qk_l2_clip_threshold"]) if j == 0 else rms(rows, cfg["rms_norm_eps"])
            dest.flatten(0, 1).index_copy_(0, indices, rows.bfloat16())
        relative = F.linear(r, w["attn_rel_bias"])
        sinks = self.linear(q_in, "attn_sink_proj").float()
        y = attention(q.bfloat16(), state.k, state.v, relative, sinks, sequences,
                      position, table, cfg["sliding_window"] if self.local else 0)
        gated = y.flatten(1).float() * self.linear(q_in, "attn_gate").float().sigmoid_()
        return convolve(self.linear(gated, "attn_output"), state.history[2], w["attn_post_conv"], plan=plan)

    def __call__(self, x, blocks, bank, state, table, position, sequences, indices, plan, prefill):
        w, eps = self.weights, self.cfg["rms_norm_eps"]
        query = w.get("attn_res_query")
        mixed = (x,) if query is None else mix(blocks, x, query, eps, prefill).unbind(1)
        inputs = mixed * 3 if len(mixed) == 1 else mixed
        messages = None
        if bank is not None:
            messages = mix(bank[:-1], bank[-1], w["loop_msg_query"], eps, prefill).unbind(1)
            inputs = tuple(self.control(v, m, n, prefill) for v, m, n in zip(inputs, messages, ("q", "k", "v")))
        norms = ("attn_norm",) * 3 if self.local else tuple(f"attn_{n}_inp_norm" for n in ("q", "k", "v"))
        inputs = tuple(rms(v, eps, w[n]).bfloat16() for v, n in zip(inputs, norms))
        y = self.attend(inputs, state, table, position, sequences, indices, plan).float()
        if self.block_start:
            blocks.append(x)
        else:
            y.add_(x)
        z = mix(blocks, y, w["ffn_res_query"], eps, prefill).squeeze(1)
        if messages is not None:
            z = self.control(z, messages[3], "ffn", prefill)
        z = self.moe(rms(z, eps, w["ffn_norm"]).bfloat16(), prefill)
        return (y + convolve(z, state.history[3], w["ffn_post_conv"], plan=plan)).bfloat16()


@dataclass(frozen=True)
class InklingGeometry:
    hidden: int
    vocab_size: int


class InklingTextModel:
    mtp = None

    @classmethod
    def from_gguf(cls, path):
        with GGUFReader(path) as reader:
            with ops.dry_load() as allocations:
                planned = cls(reader)
                needed = allocations.peak_device
            available, _ = gpu_mem_info()
            if needed > available:
                raise SnowLLMError(f"Inkling weights need {needed} device bytes, {available} available")
            model = cls(reader)
            model.planning_model = planned
            return model

    def __init__(self, reader):
        cfg = self.config = config(reader.gguf)
        tokens = reader.gguf.need("tokenizer.ggml.tokens")
        self.chat_channels = {tokens.index("<|message_model|>"): "role", tokens.index("<|end_message|>"): "end",
                              tokens.index("<|content_thinking|>"): "thinking", tokens.index("<|content_text|>"): "text"}
        self.geo = InklingGeometry(cfg["hidden_size"], cfg["unpadded_vocab_size"])
        self.embedding = reader.tensor("token_embd.weight")
        self.embed_norm = reader.tensor("token_embd_norm.weight", torch.float32)
        self.output = reader.tensor("output.weight")
        self.norm = reader.tensor("output_norm.weight", torch.float32)
        self.output_query = reader.tensor("output_res_query.weight", torch.float32)
        self.layers = [InklingLayer(reader, cfg, i) for i in range(cfg["num_hidden_layers"])]
        loop = cfg["loop_config"]
        self.start, self.end, self.passes = loop["core_start"], loop["core_end"], loop["num_passes"]
        self.visits = (*range(self.start), *(tuple(range(self.start, self.end)) * self.passes),
                       *range(self.end, len(self.layers)))

    def weight_tensors(self):
        yield from (self.embedding, self.embed_norm, self.output, self.norm, self.output_query)
        for layer in self.layers:
            yield from layer.weights.values()
            moe = layer.moe
            yield moe.router
            if moe.bias is not None:
                yield moe.bias
            for proj in (moe.gate_up, moe.down):
                yield proj.weight
                if proj.scale is not None:
                    yield proj.scale

    def cache(self, capacity, chunk_size):
        return InklingCache(self, capacity, chunk_size)

    @torch.inference_mode()
    def forward(self, input_ids, cache, *, prefill=None):
        size = input_ids.numel()
        if cache.model is not self or cache.position < 0:
            raise ValueError("Inkling cache belongs to another model or an interrupted forward")
        if input_ids.ndim != 1 or not 0 < size <= cache.chunk_size or cache.position + size > cache.capacity:
            raise ValueError("Inkling input exceeds the request cache or chunk size")
        if input_ids.device != self.embedding.device or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Inkling expects integer token ids on the model device")
        offset = cache.position
        position = torch.arange(offset, offset + size, device=input_ids.device, dtype=torch.int32)
        sequences = torch.zeros_like(position)
        cache.position = -1
        def visit_inputs(visit):
            state = cache.states[visit]
            slots = position.long() % (state.k.size(0) * state.k.size(1))
            return state, cache.tables[state.k.size(0)], slots
        if prefill is None:
            prefill = offset == 0 or size > 1
        x = self.hidden(input_ids, position, sequences, visit_inputs, None, prefill)
        logits = self.logits(x)
        cache.position = offset + size
        return logits

    def hidden(self, input_ids, position, sequences, visit_inputs, plan, prefill):
        eps = self.config["rms_norm_eps"]
        x = rms(F.embedding(input_ids, self.embedding), eps, self.embed_norm).bfloat16()
        blocks, bank = [], None
        for visit, i in enumerate(self.visits):
            if i == self.start:
                if visit == self.start:
                    initial, initial_blocks = x, tuple(blocks)
                else:
                    bank = (*blocks[len(initial_blocks) + 1:], x)
                    x, blocks = initial, list(initial_blocks)
            if i == self.end:
                bank = None
            state, table, slots = visit_inputs(visit)
            x = self.layers[i](x, blocks, bank, state, table, position, sequences, slots, plan, prefill)
        x = mix(blocks, x, self.output_query, eps, prefill).squeeze(1)
        return rms(x, eps, self.norm).bfloat16()

    def logits(self, x):
        output_weight = self.output[:self.config["unpadded_vocab_size"]]
        logits = torch.mm(x, output_weight.T, out_dtype=torch.float32)
        cap = self.config["final_logit_softcapping"]
        logits.div_(cap).tanh_().mul_(cap)
        return logits
