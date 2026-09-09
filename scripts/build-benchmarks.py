import argparse
import json
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "benchmarks" / "runs"
OUT = ROOT / "web" / "benchmarks.json"
PAGE = ROOT / "web" / "index.html"


def merged_rows() -> list[dict]:
    rows: dict[tuple, dict] = {}
    for f in sorted(RUNS.glob("*.json")):
        for r in json.loads(f.read_text()):
            rows[(r["label"], r["isl_target"])] = r
    return list(rows.values())

COMPARISONS = [
    {
        "model": "Qwen3.6-35B-A3B UD-Q4_K_XL",
        "snow_label": "qwen3_6_35b_a3b_q4_k_xl_dflash_snowllm",
        "llama_label": "qwen3_6_35b_a3b_q4_k_xl_dflash_llama",
        "snow_arm": "DFlash",
        "llama_arm": "DFlash",
        "rungs": [(1024, "1K"), (8192, "8K"), (32768, "32K"), (131072, "131K"), (225280, "225K")],
    },
    {
        "model": "Qwen3.6-35B-A3B FP8",
        "snow_label": "qwen3_6_35b_a3b_fp8_dflash_snowllm",
        "llama_label": None,
        "snow_arm": "DFlash",
        "rungs": [(1024, "1K"), (8192, "8K"), (32768, "32K"), (131072, "131K"), (225280, "225K")],
    },
]


ARMS = {
    "qwen3_6_35b_a3b_q4_k_xl_snowllm":
        ("SnowLLM", "Qwen3.6-35B-A3B UD-Q4_K_XL", "no spec"),
    "qwen3_6_35b_a3b_q4_k_xl_mtp_snowllm":
        ("SnowLLM", "Qwen3.6-35B-A3B UD-Q4_K_XL", "MTP k=2"),
    "qwen3_6_35b_a3b_q4_k_xl_dflash_snowllm":
        ("SnowLLM", "Qwen3.6-35B-A3B UD-Q4_K_XL", "DFlash"),
    "qwen3_6_35b_a3b_q4_k_xl_dflash_llama":
        ("llama.cpp", "Qwen3.6-35B-A3B UD-Q4_K_XL", "DFlash", 7),
    "qwen3_6_35b_a3b_fp8_dflash_snowllm":
        ("SnowLLM", "Qwen3.6-35B-A3B FP8", "DFlash"),
    "qwen3_6_35b_a3b_fp8_mtp_snowllm":
        ("SnowLLM", "Qwen3.6-35B-A3B FP8", "MTP k=2"),
    "qwen3_8_27b_q4_k_xl_snowllm":
        ("SnowLLM", "Qwen3.8-27B UD-Q4_K_XL", "MTP k=2"),
    "qwen3_8_27b_q4_k_xl_llama":
        ("llama.cpp", "Qwen3.8-27B UD-Q4_K_XL", "MTP k=2", 2),
    "qwen3_8_27b_q4_k_xl_dflash2_snowllm":
        ("SnowLLM", "Qwen3.8-27B UD-Q4_K_XL", "DFlash2"),
    "qwen3_8_flash_next_q3_k_xl_snowllm":
        ("SnowLLM", "Qwen3.8-Flash-Next UD-Q3_K_XL", "MTP k=2"),
    "deepseek_v4_flash_iq2_xxs_snowllm":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "DSpark, ctx 227328"),
    "deepseek_v4_flash_iq2_xxs_snowllm_128k":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "DSpark, ctx 132096"),
    "deepseek_v4_flash_iq2_xxs_nospec_snowllm_128k":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "no spec, ctx 132096"),
    "deepseek_v4_flash_iq2_xxs_nospec_llama":
        ("llama.cpp", "DeepSeek-V4-Flash IQ2_XXS", "no spec, ctx 132096", 0),
}


def al_of(row: dict, n_max: int | None) -> tuple[float | None, bool]:
    if row.get("accept_unit") == "AL" and row.get("accept_mean") is not None:
        return row["accept_mean"], False
    if n_max == 0:
        return 1.0, False
    if n_max and row.get("accept_unit") == "accept%" and row.get("accept_mean") is not None:
        return 1.0 + n_max * row["accept_mean"], True
    return None, False


def short_isl(isl_target: int) -> str:
    return f"{isl_target // 1024}K" if isl_target >= 1024 else str(isl_target)


def featured_labels() -> set:
    out = set()
    for cmp in COMPARISONS:
        out.add(cmp["snow_label"])
        if cmp.get("llama_label"):
            out.add(cmp["llama_label"])
    return out


def point_of(r: dict, n_max: int | None) -> dict:
    al, derived = al_of(r, n_max)
    step = step_ms_of(r)
    if step is None and al and r.get("decode_tok_s"):
        step = round(1e3 * al / r["decode_tok_s"], 2)
    return {
        "isl_target": r["isl_target"],
        "isl_short": short_isl(r["isl_target"]),
        "isl": r["isl"],
        "decode": round(r["decode_tok_s"], 2),
        "prefill": round(r["prefill_tok_s"], 1),
        "ttft": round(r["ttft_s"], 3),
        "step_ms": step,
        "al": round(al, 3) if al is not None else None,
        "accept_pct": (round(r["accept_mean"], 4)
                       if r.get("accept_unit") == "accept%" else None),
        "derived": derived,
    }


def build_series(rows: list[dict]) -> tuple[list[dict], list[str]]:
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r.get("label"), []).append(r)
    skipped = sorted(l for l in by_label if l not in ARMS)

    featured = featured_labels()
    out = []
    for label, spec in ARMS.items():
        engine, model, arm = spec[0], spec[1], spec[2]
        n_max = spec[3] if len(spec) > 3 else None
        got = sorted(by_label.get(label, []), key=lambda r: r["isl_target"])
        if not got:
            continue
        out.append({
            "key": label,
            "engine": engine,
            "model": model,
            "arm": arm,
            "label": f"{engine} · {model} · {arm}",
            "featured": label in featured,
            "points": [point_of(r, n_max) for r in got],
        })
    out.sort(key=lambda s: (s["model"], s["arm"], s["engine"]))
    return out, skipped


def rows_by_isl(rows: list[dict], label: str) -> dict:
    return {r["isl_target"]: r for r in rows if r["label"] == label}


def accept_of(row: dict) -> dict | None:
    return ({"unit": row["accept_unit"], "mean": round(row["accept_mean"], 3)}
            if row.get("accept_mean") is not None else None)


def step_ms_of(row: dict) -> float | None:
    return round(row["step_s"] * 1e3, 2) if row.get("step_s") else None


def build_rows(rows: list[dict]) -> list[dict]:
    out = []
    for cmp in COMPARISONS:
        snow = rows_by_isl(rows, cmp["snow_label"])
        llama = rows_by_isl(rows, cmp["llama_label"]) if cmp.get("llama_label") else {}
        for isl, short in cmp["rungs"]:
            s, l = snow.get(isl), llama.get(isl)
            if not s or (cmp.get("llama_label") and not l):
                missing = cmp["snow_label"] if not s else cmp["llama_label"]
                print(f"  {cmp['model']}: no {missing!r} row at isl_target={isl}, skipping",
                     file=sys.stderr)
                continue
            row = {
                "model": cmp["model"],
                "isl_target": isl,
                "isl_short": short,
                "isl": s["isl"],
                "snow_decode": round(s["decode_tok_s"], 1),
                "snow_prefill": round(s["prefill_tok_s"], 1),
                "snow_ttft": round(s["ttft_s"], 2),
                "snow_accept": accept_of(s),
                "snow_arm": cmp.get("snow_arm"),
            }
            if step_ms_of(s) is not None:
                row["snow_step_ms"] = step_ms_of(s)
            if l:
                row["llama_decode"] = round(l["decode_tok_s"], 1)
                row["ratio"] = round(s["decode_tok_s"] / l["decode_tok_s"], 2)
                row["llama_prefill"] = round(l["prefill_tok_s"], 1)
                row["llama_ttft"] = round(l["ttft_s"], 2)
                row["ttft_ratio"] = round(l["ttft_s"] / s["ttft_s"], 2)
                row["llama_accept"] = accept_of(l)
                row["llama_arm"] = cmp.get("llama_arm")
                if step_ms_of(l) is not None:
                    row["llama_step_ms"] = step_ms_of(l)
            out.append(row)
    if not out:
        raise SystemExit("no comparison had data -- nothing to publish")
    return out


HEADLINE = {
    "label": "qwen3_6_35b_a3b_fp8_mtp_snowllm_hero",
    "decay_label": "qwen3_6_35b_a3b_fp8_mtp_snowllm",
    "model": "Qwen3.6-35B-A3B-FP8",
    "machine": "Ryzen AI Max+ 395",
    "spec": "MTP depth 2",
    "isl": 8192,
    "users": 4,
    "aime": "91.7% avg@4, &plusmn;4.2 &mdash; Qwen reports 92.7 avg@8",
}


def between(text: str, name: str) -> tuple[int, int]:
    a = text.index(f"<!-- {name}:begin -->") + len(f"<!-- {name}:begin -->\n")
    return a, text.index(f"      <!-- {name}:end -->", a)


def headline_html(rows: list[dict]) -> str:
    by = {(r["label"], r["isl_target"]): r for r in rows}
    one = by.get((HEADLINE["label"], HEADLINE["isl"]))
    if not one:
        raise SystemExit(f"no {HEADLINE['label']} row at ISL {HEADLINE['isl']} to headline")
    many = by.get((f"{HEADLINE['label']}_c{HEADLINE['users']}", HEADLINE["isl"]))
    al = one.get("accept_mean")
    spec = HEADLINE["spec"] + (f", AL {al:.3f}" if al else "")
    rows_html = [
        ("Model", HEADLINE["model"]),
        ("Machine", HEADLINE["machine"]),
        ("Shape", f"{short_isl(one['isl_target'])} in / {short_isl(one['osl'])} out, one user"),
        ("Spec", spec),
    ]
    if many:
        rows_html.append((f"{HEADLINE['users']} users",
                          f"{many['decode_tok_s']:.1f} tok/s total"))
    rows_html.append(("AIME 2026", HEADLINE["aime"]))
    dl = "\n".join(f"          <dt>{k}</dt><dd>{v}</dd>" for k, v in rows_html)
    return (f'      <span class="figure">{one["decode_tok_s"]:.1f}<sup>tok/s</sup></span>\n'
            f'      <div class="caliper">\n        <dl>\n{dl}\n        </dl>\n      </div>\n')


def decay_html(rows: list[dict]) -> str:
    got = sorted((r for r in rows if r["label"] == HEADLINE["decay_label"]),
                 key=lambda r: r["isl_target"])
    if len(got) < 2:
        raise SystemExit(f"{HEADLINE['decay_label']} has {len(got)} rung(s), need a ladder")
    top = max(r["decode_tok_s"] for r in got)
    step = 404 / (len(got) - 1)
    pts = [(round(8 + i * step), round(130 - r["decode_tok_s"] / top * 105))
           for i, r in enumerate(got)]
    first, last = got[0]["decode_tok_s"], got[-1]["decode_tok_s"]
    label = (f"Output throughput falls from {first:.1f} tokens per second at a "
             f"{short_isl(got[0]['isl_target'])}-token input context to {last:.1f} at "
             f"{short_isl(got[-1]['isl_target'])} tokens.")
    poly = " ".join(f"{x},{y}" for x, y in pts)
    dots = "\n".join(f'          <circle cx="{x}" cy="{y}" r="3.5" />' for x, y in pts[:-1])
    ticks = "\n".join(f'          <text x="{max(0, x - 28 if i == len(pts) - 1 else x)}" '
                       f'y="148">{short_isl(r["isl_target"])}</text>'
                       for i, (r, (x, _)) in enumerate(zip(got, pts)))
    return (f'      <svg viewBox="0 0 420 150" role="img" aria-label="{label}">\n'
            f'        <g stroke="var(--hair)" stroke-width="1">\n'
            f'          <line x1="0" y1="10" x2="420" y2="10" />\n'
            f'          <line x1="0" y1="50" x2="420" y2="50" />\n'
            f'          <line x1="0" y1="90" x2="420" y2="90" />\n'
            f'          <line x1="0" y1="130" x2="420" y2="130" />\n'
            f'        </g>\n'
            f'        <polyline points="{poly}"\n'
            f'                  fill="none" stroke="var(--c2)" stroke-width="2"\n'
            f'                  stroke-linejoin="round" stroke-linecap="round" />\n'
            f'        <g fill="var(--ground)" stroke="var(--c2)" stroke-width="2">\n{dots}\n'
            f'        </g>\n'
            f'        <circle cx="{pts[-1][0]}" cy="{pts[-1][1]}" r="4.5" fill="var(--c3)" '
            f'stroke="none" />\n'
            f'        <g font-family="ui-monospace, monospace" font-size="11" '
            f'fill="var(--faint)">\n{ticks}\n        </g>\n'
            f'        <g font-family="ui-monospace, monospace" font-size="11" font-weight="600" '
            f'fill="var(--muted)">\n'
            f'          <text x="8" y="16">{first:.1f}</text>\n'
            f'          <text x="380" y="{pts[-1][1] - 7}">{last:.1f}</text>\n'
            f'        </g>\n'
            f'      </svg>\n')


def render_page(rows: list[dict]) -> None:
    text = PAGE.read_text()
    for name, html in (("decay", decay_html(rows)), ("headline", headline_html(rows))):
        a, b = between(text, name)
        text = text[:a] + html + text[b:]
    one = {(r["label"], r["isl_target"]): r for r in rows}[HEADLINE["label"], HEADLINE["isl"]]
    figure = f"{one['decode_tok_s']:.1f}"
    model = HEADLINE["model"]
    text, n = re.subn(rf"\d+\.\d+( tok/s on {re.escape(model)})", rf"{figure}\1", text)
    text, m = re.subn(rf"\d+\.\d+( tokens per second on {re.escape(model)})", rf"{figure}\1",
                      text)
    if not n or not m:
        raise SystemExit(f"web/index.html has {n} meta description(s) and {m} image alt(s) naming "
                         f"{model} -- the head no longer matches what this writes")
    PAGE.write_text(text)


def main() -> int:
    argparse.ArgumentParser().parse_args()

    merged = merged_rows()
    rows = build_rows(merged)
    series, skipped = build_series(merged)

    OUT.write_text(json.dumps({"schema": 2, "rows": rows, "series": series}, indent=1) + "\n")
    render_page(merged)
    print(f"{len(merged)} row(s) from {RUNS.relative_to(ROOT)}/ -> "
          f"{len(rows)} comparison row(s) and {len(series)} series in {OUT.relative_to(ROOT)}")
    if skipped:
        print(f"  not an arm in ARMS, so not drawn ({len(skipped)}): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
