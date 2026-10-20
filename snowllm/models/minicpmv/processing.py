# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

from ..._capi import SnowLLMError
from ..multimodal import image_rows


SPECIAL_TOKENS = dict(image_token="<|image_pad|>", video_token="<|video_pad|>",
                      image_start_token="<image>", image_end_token="</image>",
                      slice_start_token="<slice>", slice_end_token="</slice>",
                      image_id_start_token="<image_id>", image_id_end_token="</image_id>")


def processor(tok, cfg):
    try:
        from transformers import (MiniCPMV4_6ImageProcessor, MiniCPMV4_6VideoProcessor,
                                  MiniCPMV4_7Processor)
    except ImportError as exc:
        raise SnowLLMError("MiniCPM-V 4.7 image inputs need transformers>=5.18") from exc
    options = dict(patch_size=cfg["patch_size"], scale_resolution=cfg["scale_resolution"],
                   max_slice_nums=cfg["max_slice_nums"], downsample_mode=cfg["downsample_mode"])
    return MiniCPMV4_7Processor(image_processor=MiniCPMV4_6ImageProcessor(**options),
                                video_processor=MiniCPMV4_6VideoProcessor(**options),
                                tokenizer=tok, chat_template=tok.chat_template)


def canvas_positions(raw, enc, mode):
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7Model
    helper = SimpleNamespace(config=SimpleNamespace(**raw))
    helper._frame_end_idx = MiniCPMV4_7Model._frame_end_idx
    for name in ("_group_visual_frames", "get_vision_position_ids"):
        setattr(helper, name, MethodType(getattr(MiniCPMV4_7Model, name), helper))
    # Position construction uses the HF CPU algorithm without instantiating any model weights.
    positions, delta = MiniCPMV4_7Model.get_rope_index(
        helper, enc["input_ids"], mm_token_type_ids=enc["mm_token_type_ids"],
        target_sizes=enc["target_sizes"], downsample_mode=mode,
        attention_mask=enc.get("attention_mask"))
    return positions[:, 0].contiguous(), int(delta[0, 0])


def prepare(model, processor, messages, images, **kwargs):
    mode = kwargs.pop("downsample_mode", model.visual.downsample_mode)
    max_slices = kwargs.pop("max_slice_nums", model.vision_config["max_slice_nums"])
    if mode not in ("4x", "16x"):
        raise ValueError(f"unknown MiniCPM downsample_mode {mode!r}; use '4x' or '16x'")
    if isinstance(max_slices, bool) or not isinstance(max_slices, int) or max_slices < 1:
        raise ValueError("MiniCPM max_slice_nums must be a positive integer")
    text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
    enc = processor(text=[text], images=images, return_tensors="pt",
                    downsample_mode=mode, max_slice_nums=max_slices)
    ids = enc["input_ids"][0].tolist()
    positions, delta = canvas_positions(model.multimodal_config, enc, mode)
    rows = image_rows(ids, model.image_token_id)
    embeds = model.visual.forward(enc["pixel_values"], enc["target_sizes"], mode)
    if embeds.shape[0] != len(rows):
        raise SnowLLMError("MiniCPM processor and projector disagree about image token count")
    return dict(prompt=ids, mrope=positions.cuda(), pos_delta=delta, embeds=embeds, embed_rows=rows)
