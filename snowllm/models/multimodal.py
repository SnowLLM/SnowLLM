# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project
# SPDX-FileCopyrightText: Copyright contributors to the HuggingFace transformers project

import itertools

import torch


def mrope_positions(input_ids: list[int], grid_thw: torch.Tensor, image_token_id: int,
                    spatial_merge_size: int) -> tuple[torch.Tensor, int]:
    grids = iter(grid_thw.tolist())
    rows, pos = [], 0
    for is_image, group in itertools.groupby(input_ids, lambda t: t == image_token_id):
        n = len(list(group))
        if not is_image:
            rows.append(torch.arange(n).view(1, -1).expand(3, -1) + pos)
            pos += n
            continue
        t, h, w = next(grids)
        gh, gw = h // spatial_merge_size, w // spatial_merge_size
        grid = torch.meshgrid(torch.arange(t), torch.arange(gh) + pos, torch.arange(gw) + pos,
                              indexing="ij")
        block = torch.stack(grid, dim=0).reshape(3, -1)
        block[0] += pos
        rows.append(block)
        pos += max(h, w) // spatial_merge_size
    out = torch.cat(rows, dim=1).to(torch.int64)
    return out, int(out.max()) + 1 - len(input_ids)


def image_rows(input_ids: list[int], image_token_id: int) -> list[int]:
    return [i for i, t in enumerate(input_ids) if t == image_token_id]


def prepare(model: object, processor: object, messages: list[dict], images: list,
            **template_kwargs: object) -> dict:
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         **template_kwargs)
    enc = processor(text=[text], images=images or None, return_tensors="pt")
    ids = [int(t) for t in enc["input_ids"][0]]
    if not images:
        return {"prompt": ids}

    if model.visual is None:
        raise ValueError("this checkpoint carries no vision tower (model.visual.*), so it cannot "
                         "answer about an image")
    grid = enc["image_grid_thw"]
    embeds = model.visual.forward(enc["pixel_values"], grid).to(torch.bfloat16)
    rows = image_rows(ids, model.image_token_id)
    if len(rows) != embeds.shape[0]:
        raise ValueError(f"{len(rows)} image placeholders but {embeds.shape[0]} vision rows -- the "
                         f"processor and the tower disagree about this image")
    merge = model.vision_config["spatial_merge_size"]
    pos, delta = mrope_positions(ids, grid, model.image_token_id, merge)
    return {"prompt": ids, "mrope": pos.cuda(), "pos_delta": delta,
            "embeds": embeds, "embed_rows": rows}
