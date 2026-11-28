# SPDX-License-Identifier: Apache-2.0
import torch
from PIL import Image


def patches(image, rows, cols, patch, merge):
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        y = torch.arange(image.height)[:, None] // 8
        x = torch.arange(image.width)[None, :] // 8
        board = ((x + y) % 2 * -51 + 204).byte().numpy()
        rgb = Image.composite(rgba.convert("RGB"), Image.fromarray(board).convert("RGB"), rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(image.height, image.width, 3)
    canvas = torch.full((rows * merge * patch, cols * merge * patch, 3), -1., dtype=torch.float32)
    canvas[:image.height, :image.width].copy_(pixels)
    canvas.mul_(2 / 255).sub_(1)
    return (canvas.view(rows, merge, patch, cols, merge, patch, 3)
            .permute(0, 3, 1, 4, 6, 2, 5).reshape(-1, 3 * patch * patch).bfloat16())


def prepare(model, tokenizer, messages, images, **kwargs):
    tower = model.visual
    unit = tower.patch_size * tower.merge
    grids = [((im.height + unit - 1) // unit, (im.width + unit - 1) // unit) for im in images]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=True, **kwargs)["input_ids"]
    marker = model.image_token_id
    if marker is None or ids.count(marker) != len(images):
        raise ValueError("Inkling requires one image placeholder per supplied image")
    count = sum(h * w for h, w in grids)
    if len(ids) + count - len(images) >= tokenizer.model_max_length:
        raise ValueError("the text and image tokens fill the model context")

    prompt, embed_rows = [], []
    sizes = iter(h * w for h, w in grids)
    for token in ids:
        if token == marker:
            n = next(sizes)
            embed_rows.extend(range(len(prompt), len(prompt) + n))
            prompt.extend([marker] * n)
        else:
            prompt.append(token)
    embeds = tower.allocate_output(count)
    if getattr(tower, 'streamed', False):
        tower.forward_images([(patches(im, h, w, tower.patch_size, tower.merge), h, w)
                              for im, (h, w) in zip(images, grids)], embeds)
        return dict(prompt=prompt, embeds=embeds, embed_rows=embed_rows)
    offset = 0
    for image, (h, w) in zip(images, grids):
        features = patches(image, h, w, tower.patch_size, tower.merge)
        tower.forward(features, h, w, embeds[offset:offset + h * w])
        offset += h * w
    return dict(prompt=prompt, embeds=embeds, embed_rows=embed_rows)
