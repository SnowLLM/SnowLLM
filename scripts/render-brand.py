#!/usr/bin/env python3
# Render the raster brand assets from the same geometry the SVGs use.
#
# SVG is the source for anything a browser draws, but three places need pixels: link previews, the
# GitHub social preview, and the organisation avatar. None of them accept SVG. Keeping the geometry
# here rather than in a drawing file means the mark cannot drift between the site and the avatar.
#
#   web/og.png                   1200x630   og:image, the size the meta tags declare
#   brand/social-preview.png     1280x640   repo Settings -> General -> Social preview
#   brand/avatar.png              512x512   organisation Settings -> Profile -> Picture
#
# brand/ is not tracked. Those two are uploaded by hand and read by nothing here, so the script
# is the artefact worth keeping; web/og.png stays tracked because the site serves it.
#
# The avatar is fitted to the hexagon's own bounding box rather than the 128 unit drawing
# box, which carries margin of its own; AVATAR_FILL is that fit as a fraction of the square.
#
# Usage: scripts/render-brand.py

import math
import pathlib
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("render-brand.py: needs pillow -- pip install pillow")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SS = 4

GROUND = (237, 241, 243, 255)
INK = (16, 24, 32, 255)
MUTED = (75, 88, 99, 255)
FAINT = (94, 109, 120, 255)
CORE = (47, 211, 232, 255)
LINE = (204, 214, 220, 255)
WARM = (237, 162, 63, 255)
HOT = (226, 85, 31, 255)
ACCENT = (11, 114, 133, 255)

MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono{}.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono{}.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono{}.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def font(size, bold=False):
    for pattern in MONO_CANDIDATES:
        path = pathlib.Path(pattern.format("-Bold" if bold else ""))
        if path.exists():
            return ImageFont.truetype(str(path), size)
    sys.exit("render-brand.py: no monospace font found -- install fonts-dejavu-core")


def mark(px):
    n = px * SS
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = n / 128.0
    off = 0.0

    def P(x, y):
        return (off + x * k, off + y * k)

    d.polygon([P(64 + 58 * math.cos(math.radians(90 + 60 * i)),
                 64 - 58 * math.sin(math.radians(90 + 60 * i))) for i in range(6)], fill=INK)

    segs = []
    for i in range(6):
        a = math.radians(90 + 60 * i)
        segs.append(((64, 64), (64 + 42 * math.cos(a), 64 - 42 * math.sin(a))))
        root = (64 + 26 * math.cos(a), 64 - 26 * math.sin(a))
        for turn in (60, -60):
            b = math.radians(90 + 60 * i + turn)
            segs.append((root, (root[0] + 16 * math.cos(b), root[1] - 16 * math.sin(b))))

    for p, q in segs:
        d.line([P(*p), P(*q)], fill=(0, 0, 0, 0), width=int(round(11 * k)))
        for t in (p, q):
            cx, cy = P(*t)
            r = 11 * k / 2
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 0))

    d.polygon([P(55, 55), P(73, 55), P(73, 73), P(55, 73)], fill=CORE)
    return img.resize((px, px), Image.LANCZOS)


HEX_R = 58.0
AVATAR_FILL = 0.88


def avatar(px):
    n = px * SS
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = (n * AVATAR_FILL) / (2 * HEX_R)
    c = n / 2

    def P(x, y):
        return (c + (x - 64) * k, c + (y - 64) * k)

    d.polygon([P(64 + HEX_R * math.cos(math.radians(90 + 60 * i)),
                 64 - HEX_R * math.sin(math.radians(90 + 60 * i))) for i in range(6)], fill=INK)

    segs = []
    for i in range(6):
        a = math.radians(90 + 60 * i)
        segs.append(((64, 64), (64 + 42 * math.cos(a), 64 - 42 * math.sin(a))))
        root = (64 + 26 * math.cos(a), 64 - 26 * math.sin(a))
        for turn in (60, -60):
            b = math.radians(90 + 60 * i + turn)
            segs.append((root, (root[0] + 16 * math.cos(b), root[1] - 16 * math.sin(b))))

    for p, q in segs:
        d.line([P(*p), P(*q)], fill=(0, 0, 0, 0), width=int(round(11 * k)))
        for t in (p, q):
            cx, cy = P(*t)
            r = 11 * k / 2
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 0))

    d.polygon([P(55, 55), P(73, 55), P(73, 73), P(55, 73)], fill=CORE)

    out = Image.new("RGBA", (n, n), GROUND)
    out.alpha_composite(img)
    return out.resize((px, px), Image.LANCZOS)


STROKE = 10.0
CHAMFER = 9.0
DIAG = math.tan(math.radians(60))
KERN = {("S", "N"): -2, ("N", "O"): -7, ("O", "W"): -7,
        ("W", "L"): 1, ("L", "L"): -1, ("L", "M"): -2}


def glyphs():
    leg = 60.0 / DIAG
    hexo = [(30 * math.sqrt(3) / 2 + 30 * math.sin(math.radians(60 * i)),
             30 - 30 * math.cos(math.radians(60 * i))) for i in range(6)]
    return {
        "S": (42, [[(42, 0), (CHAMFER, 0), (0, CHAMFER), (0, 20), (CHAMFER, 29), (42 - CHAMFER, 29),
                    (42, 29 + CHAMFER), (42, 60 - CHAMFER), (42 - CHAMFER, 60), (0, 60)]]),
        "N": (leg, [[(0, 60), (0, 0), (leg, 60), (leg, 0)]]),
        "O": (30 * math.sqrt(3), [hexo + [hexo[0]]]),
        "W": (leg * 2, [[(0, 0), (leg / 2, 60), (leg, 0), (leg * 1.5, 60), (leg * 2, 0)]]),
        "L": (34, [[(0, 0), (0, 60), (34, 60)]]),
        "M": (leg * 2, [[(0, 60), (0, 0), (leg, 42), (leg * 2, 0), (leg * 2, 60)]]),
    }


def wordmark(cap):
    g = glyphs()
    placed, x = [], 0.0
    word = "SNOWLLM"
    for i, ch in enumerate(word):
        w, strokes = g[ch]
        placed.append((ch, x, w, strokes))
        if i + 1 < len(word):
            x += w + 15 + KERN.get((ch, word[i + 1]), 0)
    total = x + g[word[-1]][0]

    pad = STROKE
    k = (cap * SS) / 60.0
    img = Image.new("RGBA", (int((total + 2 * pad) * k), int((60 + 2 * pad) * k)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = int(round(STROKE * k))
    for i, (ch, x0, w, strokes) in enumerate(placed):
        colour = INK if i < 4 else MUTED
        for poly in strokes:
            pts = [((x0 + px + pad) * k, (py + pad) * k) for px, py in poly]
            d.line(pts, fill=colour, width=lw, joint="curve")
            for p in (pts[0], pts[-1]):
                r = lw / 2
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=colour)
        if ch == "O":
            cx, cy = (x0 + w / 2 + pad) * k, (30 + pad) * k
            h = 8 * k
            d.rectangle([cx - h, cy - h, cx + h, cy + h], fill=CORE)
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


DECAY = [("1K", 86.9), ("8K", 76.9), ("32K", 72.2), ("128K", 56.3), ("220K", 49.3)]


def card():
    img = Image.new("RGBA", (1200, 630), GROUND)
    d = ImageDraw.Draw(img)

    size = 104
    img.alpha_composite(mark(size), (80, 66))
    wm = wordmark(52)
    img.alpha_composite(wm, (80 + size + 34, 66 + (size - wm.height) // 2))
    d.line([80, 232, 1120, 232], fill=LINE, width=2)

    big, unit = font(168, True), font(34, True)
    d.text((80, 286), "76.7", font=big, fill=INK)
    d.text((80 + d.textlength("76.7", font=big) + 22, 320), "TOK/S", font=unit, fill=MUTED)

    d.text((80, 472), "Local LLM inference for Ryzen AI Max+ 395", font=font(31), fill=INK)
    d.text((80, 516), "Strix Halo / gfx1151  ·  Qwen3.6-35B-A3B-FP8  ·  8K context",
           font=font(24), fill=MUTED)
    d.text((80, 556), "snowllm.dev", font=font(26, True), fill=ACCENT)

    x0, x1, y0, y1 = 700, 1120, 272, 392
    tick, val = font(19), font(21, True)
    for row in range(4):
        y = y0 + (y1 - y0) * row / 3
        d.line([x0, y, x1, y], fill=LINE, width=1)
    xs = [x0 + (x1 - x0) * i / (len(DECAY) - 1) for i in range(len(DECAY))]
    lo, hi = DECAY[-1][1], DECAY[0][1]
    ys = [y1 - (y1 - y0) * (v - lo) / (hi - lo) for _, v in DECAY]
    d.line([c for p in zip(xs, ys) for c in p], fill=WARM, width=4, joint="curve")
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i < len(DECAY) - 1:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=GROUND, outline=WARM, width=4)
        else:
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=HOT)
    for x, (label, _) in zip(xs, DECAY):
        d.text((x - d.textlength(label, font=tick) / 2, y1 + 14), label, font=tick, fill=FAINT)
    d.text((x0, y0 - 32), f"{DECAY[0][1]}", font=val, fill=MUTED)
    d.text((x1 - d.textlength(f"{DECAY[-1][1]}", font=val), y1 - 44), f"{DECAY[-1][1]}",
           font=val, fill=MUTED)
    d.text((x0, y1 + 42), "OUTPUT TOK/S BY INPUT CONTEXT LENGTH", font=tick, fill=FAINT)
    return img


def write(img, relative):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "PNG", optimize=True)
    print(f"  {relative}  {img.width}x{img.height}  {path.stat().st_size:,} B")


og = card()
write(og, "web/og.png")

social = Image.new("RGBA", (1280, 640), GROUND)
social.alpha_composite(og, ((1280 - og.width) // 2, (640 - og.height) // 2))
write(social, "brand/social-preview.png")

write(avatar(512), "brand/avatar.png")
