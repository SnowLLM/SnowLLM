#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import sys
import tempfile

import gguf
from safetensors import safe_open
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snowllm.checkpoint.gguf.dflash import translate
from snowllm.models.geometry import DFlashGeometry


def convert(checkpoint, output):
    cfg = json.loads((checkpoint / 'config.json').read_text())
    geo = DFlashGeometry.from_config(cfg)
    if geo.hidden != 2560 or geo.num_target_layers != 48 or not (geo.dflash2 or geo.dspark):
        raise ValueError("expected a Qwen3.8 Flash-Next DFlash2 or DSpark checkpoint")
    writer = gguf.GGUFWriter(output, 'dflash', use_temp_file=True)
    fields = {
        'embedding_length': geo.hidden, 'block_count': geo.num_layers,
        'attention.head_count': geo.num_heads, 'attention.head_count_kv': geo.num_kv_heads,
        'attention.key_length': geo.head_size, 'feed_forward_length': geo.intermediate,
        'block_size': geo.block_size, 'attention.sliding_window': geo.sliding_window,
        'target_layer_count': geo.num_target_layers,
    }
    if geo.dflash2:
        fields.update(conv_kernel_size=geo.conv_taps, conv_group_size=geo.conv_group,
                      selector_rank=geo.selector_rank, selector_top_k=geo.selector_top_k)
    for key, value in fields.items():
        writer.add_uint32('dflash.' + key, value)
    if geo.is_causal is not None:
        writer.add_bool('dflash.attention.causal', geo.is_causal)
    if geo.sliding_window:
        writer.add_array('dflash.attention.sliding_window_pattern',
                         [geo.sliding(i) for i in range(geo.num_layers)])
    writer.add_bool('dflash.sample_from_anchor', geo.dspark)
    writer.add_array('dflash.target_layers', [i + 1 for i in geo.tap_layers])
    writer.add_uint32('tokenizer.ggml.mask_token_id', geo.mask_token_id)
    writer.add_float32('dflash.rope.freq_base', geo.rope_theta)
    writer.add_float32('dflash.attention.layer_norm_rms_epsilon', geo.eps)
    qtype = gguf.GGMLQuantizationType
    try:
        with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as source:
            for key in source.keys():
                dest = translate(key)
                tensor = source.get_tensor(key)
                if dest is None or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"unsupported draft tensor {key}, dtype {tensor.dtype}")
                projection = (key in ('fc.weight', 'candidate_selector.hidden_projection.weight')
                              or key.startswith('layers.') and key.endswith(('_proj.weight', 'kernel_projection.weight')))
                if projection:
                    data = gguf.quants.quantize(tensor.float().numpy(), qtype.Q8_0)
                    writer.add_tensor(dest, data, raw_dtype=qtype.Q8_0)
                else:
                    writer.add_tensor(dest, tensor.view(torch.uint8).numpy(), raw_dtype=qtype.BF16)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Export the Qwen3.8 Flash-Next draft projections as Q8_0 GGUF.")
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.gguf':
        parser.error("output must be a .gguf file")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        pending = Path(tmp) / args.output.name
        convert(args.checkpoint, pending)
        pending.replace(args.output)


if __name__ == '__main__':
    main()
