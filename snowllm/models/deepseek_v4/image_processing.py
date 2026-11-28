# SPDX-License-Identifier: Apache-2.0
import math

import torch
from PIL import ImageOps


def image_token_count(rows, columns):
    return rows * (columns + 1) + 2


def image_grid(width, height, patch, merge, min_pixels, max_tokens):
    if width * height < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    h, w = math.ceil(height / patch), math.ceil(width / patch)
    if image_token_count(math.ceil(h / merge), math.ceil(w / merge)) > max_tokens:
        ratio = height / width
        cols = math.sqrt((max_tokens - 2) / ratio + 0.25) - 0.5
        rows = cols * ratio
        if cols < 1:
            return ((max_tokens - 2) // 2) * merge, merge
        if rows < 1:
            return merge, (max_tokens - 3) * merge
        scale = min(int(cols) * merge / width, int(rows) * merge / height)
        h, w = max(1, int(height * scale)), max(1, int(width * scale))
    return h, w


def patches(image, h, w, patch):
    rgb = ImageOps.pad(
        image if image.mode == 'RGB' else image.convert('RGB'),
        (w * patch, h * patch),
        color=(127, 127, 127),
    )
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(h, patch, w, patch, 3)
    del rgb
    return pixels.permute(0, 2, 4, 1, 3).reshape(h * w, -1).float().div_(255).sub_(0.5).div_(0.5).bfloat16()


def image_ranges(rows):
    ranges = []
    for row in rows:
        if ranges and ranges[-1][1] == row:
            ranges[-1] = (ranges[-1][0], row + 1)
        else:
            ranges.append((row, row + 1))
    return tuple(ranges)


def modality_runs(ranges, first, end):
    runs = []
    at = first
    for left, right in ranges:
        lo, hi = max(left, first), min(right, end)
        if lo >= hi:
            continue
        if at < lo:
            runs.append((at - first, lo - first, False))
        runs.append((lo - first, hi - first, True))
        at = hi
    if not runs:
        return ()
    if at < end:
        runs.append((at - first, end - first, False))
    return tuple(runs)
