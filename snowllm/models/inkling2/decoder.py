# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass

import torch

from ... import ops
from ..._platform import gpu_mem_info
from ...checkpoint.gguf.inkling2 import config
from ...checkpoint.gguf.source import GGUFReader
from ...errors import SnowLLMError
from ...ops.inkling import (attention, attention_output_supported, capped_logits, controller_tanh_table, converted,
                           dense, input_projections, input_projections_supported, normalize_qkv, normalize_rows,
                           pack_compact, pack_controller, pack_search, packed_logits, q8_projections_available, sigmoid_gate,
                           packed_loop_control, qkv_loop_control, prepare_attention_heads, relative_projection, residual_mix_blocks, short_conv, unpacked_linear)
from ..geometry import pad_gemm_n
from ..projection import pack_bf16, pack_kquant
from .experts import InklingMoE


def mix(blocks, current, query, eps, prefill):
    return residual_mix_blocks(blocks, current, query.view(-1, current.size(-1)), eps, prefill)


class LayerState:
    def __init__(self, cfg, pages, page_size, device):
        dim, kv = cfg["head_dim"], cfg["num_key_value_heads"]
        shape = (pages, page_size, kv, dim)
        self.k, self.v = (torch.empty(shape, device=device, dtype=torch.bfloat16) for _ in range(2))
        channels = (kv * dim, kv * dim, cfg["hidden_size"], cfg["hidden_size"])
        self.history = [torch.zeros((1, cfg["sconv_kernel_size"] - 1, width), dtype=torch.bfloat16,
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
    def __init__(self, rd, cfg, index, controller_tanh, device_map=None, compact=False):
        self.controller_tanh = controller_tanh
        self.cfg = cfg
        self.local = cfg["layer_types"][index] == "sliding_attention"
        self.block_start = index % cfg["n_layers_per_attn_res_block"] == 0
        prefix = f"blk.{index}."
        self.input_names = ("attn_q", "attn_gate", "attn_k", "attn_v", "attn_rel", "attn_sink_proj")
        if not self.local:
            self.input_names += ("attn_k_conv_proj", "attn_v_conv_proj")
        inputs = [rd.gguf.tensors.get(prefix + key + ".weight") for key in self.input_names]
        hidden = cfg["hidden_size"]
        self.input_widths = tuple(w.shape[0] for w in inputs if w is not None)
        self.group_inputs = (all(w is not None and len(w.shape) == 2 and w.shape[1] == hidden
                                 and w.quant.name in ("Q8_0", "BF16") for w in inputs)
                             and input_projections_supported(hidden, self.input_widths))
        self.weights = {}
        self.projections = {}
        for name, tensor in rd.gguf.tensors.items():
            if not name.startswith(prefix) or not name.endswith(".weight"):
                continue
            key = name[len(prefix):-len(".weight")]
            if key in ("ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_inp"):
                continue
            if tensor.quant.name not in ("F32", "BF16", "Q8_0"):
                raise SnowLLMError(f"{name}: unsupported dense Inkling weight {tensor.quant.name}")
            if (self.group_inputs and key in self.input_names and tensor.quant.name == "Q8_0"
                    and tensor.shape[0] % 128 == 0 and q8_projections_available()):
                n, k = tensor.shape
                self.weights[key] = pack_kquant(rd.raw(name), 8, n, k)
                self.projections[key] = (n, k, n)
                continue
            dtype = torch.float32 if tensor.quant.name == "F32" and "_ctrl_" not in key else torch.bfloat16
            weight = rd.tensor(name, dtype)
            if dtype == torch.bfloat16 and weight.size(-1) % 32 == 0 and key in (
                    'attn_q', 'attn_k', 'attn_v', 'attn_rel', 'attn_sink_proj', 'attn_output',
                    'attn_gate', 'attn_k_conv_proj', 'attn_v_conv_proj'):
                packed, n, k, rows = pack_bf16(weight)
                self.weights[key] = packed
                self.projections[key] = (n, k, rows)
            else:
                self.weights[key] = weight.bfloat16() if key == "attn_rel_bias" else weight
        for key in tuple(self.weights):
            if key.startswith("loop_") and key.endswith("_ctrl_down"):
                stem = key[:-4]
                down, gates = pack_controller(self.weights.pop(key), self.weights.pop(stem + "forget"),
                                              self.weights.pop(stem + "gain"))
                self.weights[key], self.weights[stem + "gates"] = down, gates
        self.compact_inputs = None
        if compact and self.group_inputs and all(isinstance(self.weights[key], torch.Tensor) for key in self.input_names):
            self.compact_inputs = tuple(pack_compact(self.weights[key], self.projections[key][0], hidden)
                                        for key in self.input_names)
        self.project_attention = self.group_inputs and attention_output_supported(self.projections.get("attn_output"))
        self.moe = InklingMoE(rd, cfg, index, device_map, compact=compact)

    def linear(self, x, name, prefill):
        weight = self.weights[name]
        if name in self.projections:
            n, k, rows = self.projections[name]
            return dense(converted(x, torch.bfloat16), weight, n, k, rows, prefill)
        return unpacked_linear(x, weight, prefill)

    def control(self, x, msg, name, prefill):
        stem = f"loop_{name}_ctrl_"
        w = self.weights
        return packed_loop_control(x, msg, w[stem + "down"], w[stem + "gates"],
                                   self.controller_tanh, self.cfg["rms_norm_eps"], prefill)

    def attend(self, inputs, state, table, position, sequences, indices, plan, prefill, residual):
        cfg, w = self.cfg, self.weights
        heads, kv, dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        if self.group_inputs:
            weights = tuple(w[n] for n in self.input_names)
            q, gate, k, v, r, sinks, *dynamic = input_projections(inputs, weights, self.input_widths, prefill, self.compact_inputs)
        else:
            sources = (0, 0, 1, 2, 0, 0, 1, 2)
            q, gate, k, v, r, sinks, *dynamic = (
                self.linear(inputs[i], name, prefill) for i, name in zip(sources, self.input_names))
            sinks = converted(sinks, torch.float32)
        retained = getattr(state, "retain", None)
        if retained is not None:
            ops.copy_batch(retained[:2], (k,v))
        q, r = prepare_attention_heads(
            q.view(-1, heads, dim), r.view(-1, heads, cfg["d_rel"]), k, v, w["attn_q_clip"], w["attn_rel_clip"],
            (w["attn_k_conv"], w["attn_v_conv"]), (None, None) if self.local else dynamic,
            state.history[:2], (state.k, state.v), plan, indices, cfg["qk_l2_clip_threshold"],
            cfg["rel_l2_clip_threshold"], cfg["rms_norm_eps"], cfg["log_scaling_n_floor"],
            cfg["log_scaling_alpha"], not self.local, prefill)
        relative = relative_projection(r, w["attn_rel_bias"])
        project = self.project_attention
        y = attention(q, state.k, state.v, relative, sinks, sequences, position, table,
                      cfg["sliding_window"] if self.local else 0,
                      prefill=prefill, output_projection=(gate, w["attn_output"], prefill) if project else None)
        if not project:
            y = self.linear(sigmoid_gate(y.flatten(1), gate), "attn_output", prefill)
        if retained is not None:
            ops.copy_rows(retained[2], y)
        return short_conv(y, w["attn_post_conv"], None,
                          state.history[2], *plan, residual=residual, out_dtype=torch.float32)

    def __call__(self, x, blocks, bank, state, table, position, sequences, indices, plan, prefill, bank_messages=None):
        w, eps = self.weights, self.cfg["rms_norm_eps"]
        query = w.get("attn_res_query")
        mixed = (x,) if query is None else mix(blocks, x, query, eps, prefill).unbind(1)
        inputs = mixed * 3 if len(mixed) == 1 else mixed
        messages = None
        if bank is not None:
            if bank_messages is None:
                bank_messages = mix(bank[:-1], bank[-1], w["loop_msg_query"], eps, prefill)
            messages = bank_messages.unbind(1)
            downs = tuple(w[f"loop_{n}_ctrl_down"] for n in ("q", "k", "v"))
            gates = tuple(w[f"loop_{n}_ctrl_gates"] for n in ("q", "k", "v"))
            inputs = qkv_loop_control(inputs, messages[:3], downs, gates, self.controller_tanh, eps, prefill)
        norms = ("attn_norm",) * 3 if self.local else tuple(f"attn_{n}_inp_norm" for n in ("q", "k", "v"))
        inputs = normalize_qkv(inputs, tuple(w[n] for n in norms), eps)
        y = self.attend(inputs, state, table, position, sequences, indices, plan, prefill,
                        None if self.block_start else x)
        if self.block_start:
            blocks.append(x)
        z = mix(blocks, y, w["ffn_res_query"], eps, prefill).squeeze(1)
        if messages is not None:
            z = self.control(z, messages[3], "ffn", prefill)
        z = normalize_rows(z, eps, w["ffn_norm"])
        retained = getattr(state, "retain", None)
        if retained is not None:
            raw = self.moe(z, prefill)
            ops.copy_rows(retained[3], raw)
            return short_conv(raw, w["ffn_post_conv"], None, state.history[3], *plan,
                              residual=y, out_dtype=torch.bfloat16)
        return self.moe(z, prefill,
                        post_conv=(w["ffn_post_conv"], state.history[3], plan, y))


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

    def __init__(self, reader, device_map=None, fast_draft=False):
        cfg = self.config = config(reader.gguf)
        tokens = reader.gguf.need("tokenizer.ggml.tokens")
        self.chat_channels = {tokens.index("<|message_model|>"): "role", tokens.index("<|end_message|>"): "end",
                              tokens.index("<|content_thinking|>"): "thinking", tokens.index("<|content_text|>"): "text"}
        self.tool_invoke_id = tokens.index("<|content_invoke_tool_json|>")
        self.geo = InklingGeometry(cfg["hidden_size"], cfg["unpadded_vocab_size"])
        self.embedding = reader.tensor("token_embd.weight")
        self.embed_norm = reader.tensor("token_embd_norm.weight", torch.float32)
        head = reader.gguf["output.weight"]
        self.output_columns = pad_gemm_n(self.geo.vocab_size)
        self.search_head = self.compact_head = None
        compact = fast_draft and head.quant.name == "BF16"
        if (q8_projections_available() and head.quant.name == "Q8_0" and head.shape[0] >= self.output_columns
                and (self.output_columns, self.geo.hidden) == (200064, 2304)):
            raw = reader.raw(head.name)
            self.output = pack_kquant(raw, 8, self.output_columns, self.geo.hidden)
            if fast_draft:
                from ...checkpoint.gguf.source import dequantize
                rows, k = self.output_columns, self.geo.hidden
                dense_head = dequantize(raw[:rows * (k // 32) * 34], "Q8_0", rows * k, torch.bfloat16)
                self.search_head = pack_search(dense_head.view(rows, k), rows)
                del dense_head
            del raw
        else:
            dense_head = reader.tensor(head.name)
            self.output, self.output_columns, _, _ = pack_bf16(dense_head[:self.geo.vocab_size])
            if fast_draft:
                self.search_head = pack_search(dense_head, self.output_columns)
            if compact:
                self.compact_head = pack_compact(self.output, self.output_columns, self.geo.hidden)
            del dense_head
        self.norm = reader.tensor("output_norm.weight", torch.float32)
        self.output_query = reader.tensor("output_res_query.weight", torch.float32)
        self.controller_tanh = controller_tanh_table(self.embedding.device)
        self.layers = [InklingLayer(reader, cfg, i, self.controller_tanh, device_map, compact)
                       for i in range(cfg["num_hidden_layers"])]
        loop = cfg["loop_config"]
        self.start, self.end, self.passes = loop["core_start"], loop["core_end"], loop["num_passes"]
        self.visits = (*range(self.start), *(tuple(range(self.start, self.end)) * self.passes),
                       *range(self.end, len(self.layers)))

        self.loop_queries = None
        if self.passes > 1 and self.start < self.end:
            core = self.layers[self.start:self.end]
            self.loop_queries = torch.stack([layer.weights["loop_msg_query"].view(4, -1) for layer in core])
            for layer, query in zip(core, self.loop_queries.unbind(0)):
                layer.weights["loop_msg_query"] = query.view_as(layer.weights["loop_msg_query"])

    def weight_tensors(self):
        def buffers(weights):
            for weight in weights:
                if isinstance(weight, ops.KQuantProjWeight):
                    yield weight.quant
                    yield weight.meta
                else:
                    yield weight
        yield from buffers((self.embedding, self.embed_norm, self.output, self.norm, self.output_query, self.controller_tanh))
        yield from self.search_head or ()
        yield from self.compact_head or ()
        for layer in self.layers:
            for pair in layer.compact_inputs or ():
                yield from pair
            for projection in (layer.moe.gate_up, layer.moe.down):
                yield from projection.compact or ()
            yield from buffers(layer.weights.values())
            moe = layer.moe
            yield moe.router
            if moe.packed_router is not None:
                yield moe.packed_router
            if moe.bias is not None:
                yield moe.bias
            for proj in (moe.gate_up, moe.down):
                yield proj.weight
                if proj.scale is not None:
                    yield proj.scale
                if proj.meta is not None:
                    yield proj.meta

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
        def i32(*ranges):
            return torch.int32, ranges
        position, sequences, cu, state_slot = ops.metadata_ranges([
            i32((offset, 1, size)), i32((0, 0, size)), i32((0, size, 2)), i32((0, 0, 1))])
        plan = sequences, position, cu, state_slot
        mappings = {}
        for pages in cache.tables:
            capacity = pages * 16
            start = offset % capacity
            first = min(size, capacity - start)
            mappings[pages], = ops.metadata_ranges([i32((start, 1, first), (0, 1, size-first))])
        cache.position = -1
        def visit_inputs(visit):
            state = cache.states[visit]
            pages = state.k.size(0)
            return state, cache.tables[pages], mappings[pages]
        if prefill is None:
            prefill = offset == 0 or size > 1
        x = self.hidden(input_ids, position, sequences, visit_inputs, plan, prefill)
        logits = self.logits(x)
        cache.position = offset + size
        return logits

    def hidden(self, input_ids, position, sequences, visit_inputs, plan, prefill, *, taps=None, tap_at=None,
               embeds=None, embed_rows=None):
        eps = self.config["rms_norm_eps"]
        x = normalize_rows(ops.gather_rows(self.embedding, input_ids), eps, self.embed_norm)
        if embeds is not None:
            ops.copy_rows(x, embeds, dst_index=embed_rows)
        blocks, bank, pass_messages = [], None, None
        for visit, i in enumerate(self.visits):
            if i == self.start:
                if visit == self.start:
                    initial, initial_blocks = x, tuple(blocks)
                else:
                    bank = (*blocks[len(initial_blocks) + 1:], x)
                    if not prefill:
                        all_messages = mix(bank[:-1], bank[-1], self.loop_queries, eps, prefill)
                        pass_messages = all_messages.unflatten(1, (self.end - self.start, 4)).unbind(1)
                    x, blocks = initial, list(initial_blocks)
            if i == self.end:
                bank, pass_messages = None, None
            if taps is not None and visit in tap_at:
                ops.copy_rows(taps[:, tap_at[visit]], converted(x, torch.bfloat16))
            state, table, slots = visit_inputs(visit)
            x = self.layers[i](x, blocks, bank, state, table, position, sequences, slots, plan, prefill,
                               None if pass_messages is None else pass_messages[i - self.start])
        x = mix(blocks, x, self.output_query, eps, prefill).squeeze(1)
        x = normalize_rows(x, eps, self.norm)
        if taps is not None and len(self.visits) in tap_at:
            ops.copy_rows(taps[:, tap_at[len(self.visits)]], x)
        return x

    def logits(self, x):
        logits = packed_logits(x, self.output, self.output_columns, self.compact_head)
        return capped_logits(logits, self.geo.vocab_size, self.config["final_logit_softcapping"])
