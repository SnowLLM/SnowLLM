#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import bisect
import json
import os
import re
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BG = (13, 17, 23)
PANE_BG = (22, 27, 34)
CARD_BG = (17, 22, 30)
BORDER = (48, 54, 61)
TITLE = (230, 237, 243)
TEXT = (201, 209, 217)
HEAD = (210, 220, 235)
TABLE = (170, 205, 235)
THINK = (120, 130, 142)
DONE = (63, 185, 80)
DIM = (110, 118, 129)
BAR = (33, 41, 54)
TOOL = (57, 197, 187)
TOOL_DONE = (126, 231, 135)
LEFT_C = (240, 136, 62)
RIGHT_C = (88, 166, 255)
PROMPT_C = (210, 168, 255)
ACCENT = (255, 214, 102)
PHASE_COL = {"startup": (70, 78, 92), "read": TOOL, "prefill": PROMPT_C,
             "think": (150, 160, 175), "answer": DONE, "tail": (70, 78, 92)}


def tool_arg(name, args):
    if name == "read":
        return str(args.get("path", ""))
    if name == "bash":
        return str(args.get("command", ""))
    return json.dumps(args, ensure_ascii=False)


def tool_suffix(name, summary):
    if summary.get("isError"):
        return "  (error)"
    if name == "read":
        return f"  {summary.get('chars', 0) / 1024:.1f} KB, {summary.get('lines', 0)} lines"
    if name == "bash":
        return f"  exit 0, {summary.get('lines', 0)} lines"
    return f"  {summary.get('chars', 0)} chars"


def wrap(text, width):
    lines = []
    for para in text.split("\n"):
        if para == "":
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else cur + " " + word
            if len(cand) <= width:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                while len(word) > width:
                    lines.append(word[:width])
                    word = word[width:]
                cur = word
        lines.append(cur)
    return lines


def _table(rows):
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if any("-" in c for c in cells) and all(set(c) <= set("-: ") for c in cells):
            continue
        parsed.append(cells)
    if not parsed:
        return []
    n = max(len(r) for r in parsed)
    widths = [0] * n
    for r in parsed:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    return [("TABLE", "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))) for r in parsed]


def prettify(text):
    src = text.split("\n")
    out, i = [], 0
    while i < len(src):
        ln = src[i]
        if ln.strip().startswith("|") and i + 1 < len(src) and src[i + 1].strip().startswith("|"):
            rows = []
            while i < len(src) and src[i].strip().startswith("|"):
                rows.append(src[i])
                i += 1
            out += _table(rows)
            continue
        m = re.match(r"^(#{1,6})\s*(.*)$", ln)
        b = re.match(r"^\s*\*\*(.+?)\*\*\s*$", ln)
        if m:
            out.append(("HEAD", m.group(2).strip()))
        elif b:
            out.append(("HEAD", b.group(1).strip()))
        else:
            ln = ln.replace("**", "").replace("`", "")
            out.append(("TEXT", re.sub(r"^\s*[-*]\s+", "\u2022 ", ln)))
        i += 1
    return out


def display_lines(text, cols):
    lines = []
    for style, ln in prettify(text):
        if style == "TABLE":
            lines.append((ln[:cols], TABLE))
            continue
        for w in wrap(ln, cols):
            lines.append((w, HEAD if style == "HEAD" else TEXT))
    return lines


class Transcript:
    def __init__(self, path):
        self.meta = {}
        self.entries = []
        self.tool_by_id = {}
        self.text_times = []
        self.times = []
        self.first_t = None
        self.exit_t = None
        self.read_start = None
        self.read_end = None
        self.bash_start = None
        self.answer_start = None
        self.answer_end = None
        self.prefill_end = None
        delta_times = []
        for line in open(path):
            e = json.loads(line)
            t = e.get("t")
            if e["type"] == "meta":
                self.meta = e
                continue
            self.first_t = t if self.first_t is None else self.first_t
            self.times.append(t)
            if e["type"] == "delta":
                delta_times.append(t)
                if e["kind"] == "text":
                    self.text_times.append(t)
                last = self.entries[-1] if self.entries else None
                if last and last["kind"] == e["kind"] and last.get("open", True):
                    last["deltas"].append((t, e["text"]))
                else:
                    if last:
                        last["open"] = False
                    self.entries.append({"kind": e["kind"], "t0": t,
                                         "deltas": [(t, e["text"])], "open": True})
            elif e["type"] == "tool_start":
                if self.entries:
                    self.entries[-1]["open"] = False
                entry = {"kind": "tool", "t0": t, "name": e.get("name"),
                         "args": e.get("args"), "end": None, "open": True}
                self.entries.append(entry)
                self.tool_by_id[e.get("id")] = entry
                if e.get("name") == "read" and self.read_start is None:
                    self.read_start = t
                if e.get("name") == "bash" and self.bash_start is None:
                    self.bash_start = t
            elif e["type"] == "tool_end":
                entry = self.tool_by_id.get(e.get("id"))
                if entry is not None:
                    entry["end"] = e
                    entry["open"] = False
                if e.get("name") == "read" and self.read_end is None:
                    self.read_end = t
            elif e["type"] == "exit":
                self.exit_t = t
        if self.read_end is not None:
            after = [t for t in delta_times if t > self.read_end]
            if after:
                self.prefill_end = after[0]
        if self.bash_start is not None:
            after = [t for t in self.text_times if t > self.bash_start]
            if after:
                self.answer_start, self.answer_end = after[0], after[-1]

    def running(self, t):
        return t < (self.exit_t or 0)

    def last_event_at(self, t):
        i = bisect.bisect_right(self.times, t) - 1
        return self.times[i] if i >= 0 else None

    def content(self, entry, t):
        return "".join(text for tt, text in entry["deltas"] if tt <= t)

    def prefill(self):
        if self.read_end is None or self.bash_start is None:
            return 0.0
        return self.bash_start - self.read_end

    def phases(self):
        out = []
        ts = sorted(self.times)
        if not ts:
            return out
        start = ts[0]
        if start > 0:
            out.append(("startup", 0.0, start))
        if self.read_start is not None and self.read_end is not None:
            out.append(("read", start, self.read_end))
            pe = self.prefill_end or self.answer_start or self.read_end
            out.append(("prefill", self.read_end, pe))
            end_think = self.answer_start if self.answer_start is not None else pe
            if end_think > pe:
                out.append(("think", pe, end_think))
        if self.answer_start is not None:
            out.append(("answer", self.answer_start, self.answer_end))
        tail = self.answer_end or start
        if self.exit_t is not None and self.exit_t > tail:
            out.append(("tail", tail, self.exit_t))
        return out

    def tokens(self, t):
        return sum(1 for tt in self.text_times if tt <= t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Same task")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--gif-width", type=int, default=1000)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gifsicle", default=os.environ.get("GIFSICLE"))
    ap.add_argument("--intro", type=float, default=2.6)
    ap.add_argument("--outro", type=float, default=5.0)
    a = ap.parse_args()

    L, R = Transcript(a.left), Transcript(a.right)
    total = max(L.exit_t or 0, R.exit_t or 0)
    duration = a.intro + total + a.outro
    fps = a.fps

    fs = lambda n: max(8, int(n * a.scale))
    f_text = ImageFont.truetype(FONT, fs(19))
    f_bold = ImageFont.truetype(FONT_B, fs(21))
    f_small = ImageFont.truetype(FONT, fs(15))
    f_big = ImageFont.truetype(FONT_B, fs(34))
    f_mid = ImageFont.truetype(FONT_B, fs(24))
    char_w = f_text.getlength("M")
    line_h = int(fs(19) * 1.32)

    margin, gap, header_h = 20, 20, fs(96)
    pane_w, pane_h = fs(840), fs(600)
    W = margin * 2 + pane_w * 2 + gap
    H = header_h + pane_h + fs(104) + margin
    W -= W % 2
    H -= H % 2
    cols = max(10, int((pane_w - fs(32)) / char_w))
    rows = max(4, int((pane_h - fs(34) - fs(48)) / line_h))
    x_left, x_right = margin, margin + pane_w + gap

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)

    spinner = "|/-\\"

    def phase(t, s):
        if s.first_t is None or t < s.first_t:
            return "starting\u2026", DIM
        if not s.running(t):
            return "done", DONE
        last = s.last_event_at(t)
        if last is None or t - last > 0.25:
            return f"prefilling {spinner[int(t * 8) % 4]}", PROMPT_C
        return "streaming", DONE

    def draw_pane(d, x, s, t, color, label):
        fin = not s.running(t)
        d.rectangle([x, header_h, x + pane_w, header_h + pane_h],
                    fill=PANE_BG, outline=DONE if fin else BORDER, width=3 if fin else 2)
        d.text((x + 14, header_h + 8), label, font=f_bold, fill=color)
        st, stc = phase(t, s)
        d.text((x + pane_w - f_bold.getlength(st) - 16, header_h + 8), st, font=f_bold, fill=stc)
        n = s.tokens(t)
        rate = 0.0
        if n > 1:
            seen = [tt for tt in s.text_times if tt <= t]
            rate = (n - 1) / max(seen[-1] - seen[0], 1e-6)
        et = t if s.running(t) else (s.exit_t or t)
        d.text((x + 16, header_h + 32),
               f"{et:6.2f}s      {n:5d} tok      {rate:5.1f} tok/s",
               font=f_small, fill=DIM)

        bar_y = header_h + 54
        d.rectangle([x + 16, bar_y, x + pane_w - 16, bar_y + 6], fill=BAR)
        last = s.last_event_at(t)
        if s.running(t) and last is not None and t - last > 0.25:
            seg = (pane_w - 32) * 0.22
            pos = ((t * 0.55) % 1.6) / 1.6 * ((pane_w - 32) + seg) - seg
            x0 = x + 16 + max(0.0, pos)
            d.rectangle([x0, bar_y, min(x + pane_w - 16, x0 + seg), bar_y + 6], fill=PROMPT_C)
        elif n > 1:
            d.rectangle([x + 16, bar_y, x + pane_w - 16, bar_y + 6], fill=DONE)

        lines = []
        for e in s.entries:
            if e["t0"] > t:
                break
            if e["kind"] == "tool":
                arg = tool_arg(e["name"], e["args"])
                if e["end"] is None:
                    body, col = f"\u25b6 {e['name']} {arg}", TOOL
                else:
                    body = f"\u2713 {e['name']} {arg}{tool_suffix(e['name'], e['end'].get('summary'))}"
                    col = TOOL_DONE
                lines += [(ln, col) for ln in wrap(body, cols)]
            elif e["kind"] == "text":
                lines += display_lines(s.content(e, t), cols)
            else:
                lines += [("\u00b7" + ln, THINK) for ln in wrap(s.content(e, t), cols)]
        if len(lines) > rows:
            lines = lines[-rows:]
        y = bar_y + 16
        for text, col in lines:
            d.text((x + 16, y), text, font=f_text, fill=col)
            y += line_h
        if s.running(t) and y + line_h <= header_h + pane_h - 4:
            d.rectangle([x + 16, y + 2, x + 16 + char_w * 0.7, y + line_h], fill=color)
        if fin:
            txt = f"\u2713 DONE in {s.exit_t:.1f}s"
            tw = f_bold.getlength(txt)
            bx0 = x + (pane_w - tw) / 2 - 14
            by0 = header_h + pane_h - fs(46)
            d.rectangle([bx0, by0, bx0 + tw + 28, by0 + fs(34)], fill=(9, 32, 16),
                        outline=DONE, width=2)
            d.text((bx0 + 14, by0 + fs(4)), txt, font=f_bold, fill=DONE)

    def center(d, y, text, font, fill):
        d.text(((W - font.getlength(text)) / 2, y), text, font=font, fill=fill)

    def draw_intro(d):
        d.rectangle([margin, margin, W - margin, H - margin], fill=CARD_BG, outline=BORDER, width=2)
        center(d, int(H * 0.26), "One model, one GPU, two engines.", f_big, TITLE)
        center(d, int(H * 0.37), a.title, f_mid, PROMPT_C)
        center(d, int(H * 0.47), a.subtitle, f_text, TEXT)
        center(d, int(H * 0.61), "both agents run the same step:", f_text, DIM)
        center(d, int(H * 0.67), "read access.log", f_text, TOOL)
        center(d, int(H * 0.81),
               f"left: {L.meta.get('label', 'left')}        right: {R.meta.get('label', 'right')}",
               f_bold, ACCENT)

    def draw_timeline(d, t):
        y0 = header_h + pane_h + fs(14)
        rh = fs(28)
        bx = x_left + fs(132)
        bw = (x_left + pane_w * 2 + gap) - bx
        for i, (s, col) in enumerate(((L, LEFT_C), (R, RIGHT_C))):
            yy = y0 + i * (rh + fs(10))
            d.text((x_left, yy + fs(2)), s.meta.get("label", ""), font=f_bold, fill=col)
            d.rectangle([bx, yy, bx + bw, yy + rh], fill=BAR, outline=BORDER)
            for name, pa, pb in s.phases():
                x0 = bx + bw * min(pa, t) / total
                x1 = bx + bw * min(pb, t) / total
                if x1 - x0 >= 1:
                    d.rectangle([x0, yy + 1, x1, yy + rh - 1], fill=PHASE_COL[name])
                    if x1 - x0 > f_small.getlength(name) + 12:
                        d.text(((x0 + x1 - f_small.getlength(name)) / 2, yy + 5), name,
                               font=f_small, fill=BG)
        px = bx + bw * min(t, total) / total
        d.line([px, y0, px, y0 + 2 * rh], fill=ACCENT, width=2)

    def draw_outro(d):
        total_l, total_r = L.exit_t or 0, R.exit_t or 0
        box_w, box_h = int(W * 0.62), int(H * 0.44)
        bx, by = (W - box_w) // 2, int(H * 0.30)
        d.rectangle([bx, by, bx + box_w, by + box_h], fill=CARD_BG, outline=ACCENT, width=2)
        d.text((bx + 30, by + 24), "Result", font=f_mid, fill=ACCENT)
        y = by + 76
        for label, tot, pre, col in ((L.meta.get("label", "left"), total_l, L.prefill(), LEFT_C),
                                     (R.meta.get("label", "right"), total_r, R.prefill(), RIGHT_C)):
            d.text((bx + 30, y), f"{label:<12}", font=f_bold, fill=col)
            d.text((bx + 30 + fs(200), y), f"prefill {pre:5.1f}s", font=f_text, fill=PROMPT_C)
            d.text((bx + 30 + fs(400), y), f"total {tot:5.1f}s", font=f_text, fill=TITLE)
            y += fs(38)
        center(d, by + box_h - fs(70), f"{total_l / total_r:.1f}\u00d7 faster end to end", f_big, DONE)
        center(d, by + box_h - fs(26), "identical tool calls  \u00b7  identical answer", f_small, DIM)

    n_frames = int(duration * fps) + 1
    for i in range(n_frames):
        vt = i / fps
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        if vt < a.intro:
            draw_intro(d)
        else:
            t = min(vt - a.intro, total)
            done = vt - a.intro >= total
            ld, rd = not L.running(t), not R.running(t)
            if (ld or rd) and not (ld and rd):
                win, lose = (L, R) if ld else (R, L)
                msg = (f"{win.meta.get('label')} finished in {win.exit_t:.1f}s"
                       f"  \u2014  {lose.meta.get('label')} still running")
                bw = f_bold.getlength(msg) + 40
                d.rectangle([(W - bw) / 2, fs(40), (W + bw) / 2, fs(40) + fs(34)],
                            fill=(9, 32, 16), outline=DONE, width=2)
                d.text(((W - f_bold.getlength(msg)) / 2, fs(46)), msg, font=f_bold, fill=DONE)
            if not done:
                d.text((margin, margin - 6), a.title, font=f_bold, fill=PROMPT_C)
            draw_pane(d, x_left, L, t, LEFT_C, L.meta.get("label", "left"))
            draw_pane(d, x_right, R, t, RIGHT_C, R.meta.get("label", "right"))
            draw_timeline(d, t)
            if done:
                draw_outro(d)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    rc = proc.wait()

    if a.gif and rc == 0:
        pal = a.out + ".pal.png"
        gv = f"fps={a.gif_fps},scale={a.gif_width}:-1:flags=lanczos"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", a.out,
                        "-vf", gv + ",palettegen=max_colors=64:stats_mode=diff", pal])
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", a.out, "-i", pal, "-lavfi",
                        gv + "[x];[x][1:v]paletteuse=dither=none", a.gif])
        os.remove(pal)
        if a.gifsicle and os.path.exists(a.gifsicle):
            subprocess.run([a.gifsicle, "-O3", a.gif, "-o", a.gif + ".opt"])
            os.replace(a.gif + ".opt", a.gif)

    print(f"wrote {a.out}  {W}x{H} {n_frames} frames @ {fps}fps  dur={duration:.2f}s rc={rc}"
          + (f"  gif={a.gif}" if a.gif else ""), file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
