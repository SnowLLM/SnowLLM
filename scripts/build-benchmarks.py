#!/usr/bin/env python3
import argparse
import html
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "benchmarks" / "runs"
OUT = ROOT / "web" / "benchmarks.json"
PAGE = ROOT / "web" / "benchmark" / "index.html"
START = "<!-- BEGIN GENERATED BENCHMARK DATA -->"
END = "<!-- END GENERATED BENCHMARK DATA -->"
GFX1103_LABELS = {
    "minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_static_bf16_4k_gfx1103_http",
    "minicpm_mix_q3gud_dspark8_llama_rocm_bf16_ub512_nocache_http",
}


def platform_of(label: str) -> str:
    return "gfx1103" if label in GFX1103_LABELS else "gfx1151"


def merged_rows() -> list[dict]:
    rows: dict[tuple, dict] = {}
    for f in sorted(RUNS.glob("*.json")):
        for r in json.loads(f.read_text()):
            rows[(r["label"], r["isl_target"])] = r
    return list(rows.values())

COMPARISONS = [
    {
        "model": "MiniCPM-V 4.7 35B-A3B mix_q3gud_im",
        "snow_label": "minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_static_bf16_4k_gfx1103_http",
        "llama_label": "minicpm_mix_q3gud_dspark8_llama_rocm_bf16_ub512_nocache_http",
        "snow_arm": "DSpark block 8, static eager, HTTP",
        "llama_arm": "DSpark block 8, ROCm, HTTP",
        "rungs": [(1024, "1K"), (8192, "8K"), (32768, "32K"), (122880, "120K")],
    },
    {
        "model": "GLM-5.3-Flash UD-IQ2_XXS",
        "snow_label": "glm5_3_flash_iq2_xxs_mtp_snowllm",
        "llama_label": "glm5_3_flash_iq2_xxs_mtp_llama",
        "snow_arm": "MTP k=2",
        "llama_arm": "MTP k=2",
        "rungs": [(1024, "1K"), (8192, "8K"), (32768, "32K"), (131072, "128K"), (225280, "220K")],
    },
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
    "minicpm_v4_7_35b_a3b_mix_q3gud_im_dspark8_static_bf16_4k_gfx1103_http":
        ("SnowLLM", "MiniCPM-V 4.7 35B-A3B mix_q3gud_im", "DSpark block 8, static eager, HTTP"),
    "minicpm_mix_q3gud_dspark8_llama_rocm_bf16_ub512_nocache_http":
        ("llama.cpp", "MiniCPM-V 4.7 35B-A3B mix_q3gud_im", "DSpark block 8, ROCm, HTTP", None),
    "glm5_3_flash_iq2_xxs_mtp_snowllm":
        ("SnowLLM", "GLM-5.3-Flash UD-IQ2_XXS", "MTP k=2"),
    "glm5_3_flash_iq2_xxs_mtp_llama":
        ("llama.cpp", "GLM-5.3-Flash UD-IQ2_XXS", "MTP k=2", 2),
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
        ("SnowLLM", "Qwen3.8-Flash-Next UD-Q3_K_XL", "MTP k=2, HTTP"),
    "qwen3_8_flash_next_q3_k_xl_mtp_engine_snowllm":
        ("SnowLLM", "Qwen3.8-Flash-Next UD-Q3_K_XL", "MTP k=2, Engine"),
    "qwen3_8_flash_next_q3_k_xl_nospec_engine_snowllm":
        ("SnowLLM", "Qwen3.8-Flash-Next UD-Q3_K_XL", "no spec, Engine"),
    "deepseek_v4_flash_iq2_xxs_snowllm":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "DSpark, ctx 227328"),
    "deepseek_v4_flash_iq2_xxs_400k_adaptive_snowllm_http":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "DSpark adaptive, block 5, ctx 409600, HTTP"),
    "deepseek_v4_flash_iq2_xxs_snowllm_128k":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "DSpark, ctx 132096"),
    "deepseek_v4_flash_iq2_xxs_nospec_snowllm_128k":
        ("SnowLLM", "DeepSeek-V4-Flash IQ2_XXS", "no spec, ctx 132096"),
    "deepseek_v4_flash_iq2_xxs_nospec_llama":
        ("llama.cpp", "DeepSeek-V4-Flash IQ2_XXS", "no spec, ctx 132096", 0),
    "minicpm_v4_7_35b_a3b_q4_k_m_dspark_block8_fixed_snowllm_http":
        ("SnowLLM", "MiniCPM-V 4.7 35B-A3B Q4_K_M", "DSpark block 8, fixed, HTTP"),
    "minicpm_v4_7_35b_a3b_q4_k_m_dspark_block8_adaptive_snowllm_http":
        ("SnowLLM", "MiniCPM-V 4.7 35B-A3B Q4_K_M", "DSpark block 8, adaptive, HTTP"),
}


def al_of(row: dict, n_max: int | None) -> tuple[float | None, bool]:
    if row.get("accept_length_mean") is not None:
        return row["accept_length_mean"], False
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
    return {"qwen3_8_flash_next_q3_k_xl_mtp_engine_snowllm"}


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
            "platform": got[0].get("platform", platform_of(label)),
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


def accept_of(row: dict, n_max: int | None = None) -> dict | None:
    al, derived = al_of(row, n_max)
    if al is None:
        return None
    return {"unit": "AL", "mean": round(al, 3), "derived": derived}


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
                "platform": s.get("platform", platform_of(cmp["snow_label"])),
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
                row["llama_accept"] = accept_of(l, ARMS[cmp["llama_label"]][3])
                row["llama_arm"] = cmp.get("llama_arm")
                if step_ms_of(l) is not None:
                    row["llama_step_ms"] = step_ms_of(l)
            out.append(row)
    if not out:
        raise SystemExit("no comparison had data -- nothing to publish")
    return out


def render_results(series_list: list[dict], messages: dict, result_id: str,
                   note_key: str = "benchmark.resultsNote") -> str:
    def text(key: str) -> str:
        return f'data-i18n="{key}">{html.escape(messages[key])}'

    columns = [
        ("engine", "benchmark.engine"), ("arm", "benchmark.variant"),
        ("isl_target", "benchmark.targetInput"), ("isl", "benchmark.actualInput"),
        ("decode", "metric.decode.label"), ("prefill", "metric.prefill.label"),
        ("ttft", "metric.ttft.label"), ("step_ms", "metric.ms.label"),
        ("al", "metric.al.label"),
    ]
    parts = [
        f'<details id="{result_id}" class="benchmark-results" open>',
        f'<summary {text("benchmark.allResults")}</summary>',
        f'<p class="note" {text(note_key)}</p>',
    ]
    models = dict.fromkeys(series["model"] for series in series_list)
    for model in models:
        parts.extend([
            '<div class="scroll"><table>',
            f'<caption>{html.escape(model)}</caption>',
            '<thead><tr>' + ''.join(f'<th scope="col" {text(key)}</th>' for _, key in columns) + '</tr></thead>',
            '<tbody>',
        ])
        for series in series_list:
            if series["model"] != model:
                continue
            for point in series["points"]:
                values = {**point, "engine": series["engine"], "arm": series["arm"]}
                cells = []
                for column, _ in columns:
                    value = values.get(column)
                    if value is None:
                        content = "—"
                    elif column in ("isl_target", "isl"):
                        content = f"{value:,}"
                    elif isinstance(value, (int, float)):
                        content = str(value)
                    elif column == "arm" and "no spec" in value:
                        content = html.escape(value).replace("no spec", f'<span {text("benchmark.noSpec")}</span>')
                    else:
                        content = html.escape(str(value))
                    tag = 'th scope="row"' if column == "engine" else "td"
                    cells.append(f'<{tag}>{content}</{tag.split()[0]}>')
                parts.append(f'<tr data-series-key="{html.escape(series["key"], quote=True)}" data-isl-target="{point["isl_target"]}">' + ''.join(cells) + '</tr>')
        parts.extend(['</tbody></table></div>'])
    if any(point.get("derived") for series in series_list for point in series["points"]):
        parts.append(f'<p class="note" {text("benchmark.derived")}</p>')
    parts.append('</details>')
    return '\n'.join(parts)


def render_benchmark_content(data: dict, messages: dict) -> str:
    def text(key: str) -> str:
        return f'data-i18n="{key}">{html.escape(messages[key])}'

    primary = [s for s in data["series"] if s.get("platform", "gfx1151") == "gfx1151"]
    secondary = [s for s in data["series"] if s.get("platform") == "gfx1103"]
    parts = [render_results(primary, messages, "benchmark-results")]
    if secondary:
        parts.extend([
            '<section id="benchmarks-780m" class="benchmark-platform" aria-labelledby="benchmarks-780m-heading">',
            '<h2 id="benchmarks-780m-heading" class="benchmark-hardware-heading">Radeon 780M / Ryzen 9 7940HS</h2>',
            f'<p class="benchmark-hardware-spec" {text("benchmark.gfx1103.hardware")}</p>',
            '<div id="benchmarks-780m-picker" class="benchmark-picker" data-src="/benchmarks.json" data-inline="benchmarks-data" data-platform="gfx1103" data-charts="benchmarks-780m-charts" data-results="benchmark-results-780m"></div>',
            '<div id="benchmarks-780m-charts"></div>',
            render_results(secondary, messages, "benchmark-results-780m", "benchmark.gfx1103.resultsNote"),
            f'<p class="note"><a href="/docs/development/benchmarks#radeon-780m" {text("benchmark.gfx1103.details")}</a></p>',
            '</section>',
        ])
    parts.extend([
        '<script type="application/json" id="benchmarks-data">',
        json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c'),
        '</script>',
    ])
    return '\n'.join(parts)


def render_benchmark_page(template: str, data: dict, messages: dict) -> str:
    if template.count(START) != 1 or template.count(END) != 1:
        raise ValueError("benchmark page must contain one generated data region")
    before, rest = template.split(START)
    _, after = rest.split(END)
    return before + START + '\n' + render_benchmark_content(data, messages) + '\n' + END + after


def main() -> int:
    argparse.ArgumentParser().parse_args()

    merged = merged_rows()
    rows = build_rows(merged)
    series, skipped = build_series(merged)

    data = {"schema": 2, "rows": rows, "series": series}
    messages = json.loads((ROOT / "web" / "locales" / "en.json").read_text())
    page = render_benchmark_page(PAGE.read_text(), data, messages)
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    PAGE.write_text(page)
    print(f"{len(merged)} row(s) from {RUNS.relative_to(ROOT)}/ -> "
          f"{len(rows)} comparison row(s) and {len(series)} series in {OUT.relative_to(ROOT)}")
    if skipped:
        print(f"  not an arm in ARMS, so not drawn ({len(skipped)}): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
