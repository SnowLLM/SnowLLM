# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import json
import os
import pathlib
import re
import statistics as st
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_CORPUS_PATH = pathlib.Path(os.environ.get("SNOWLLM_BENCH_CORPUS")
                            or pathlib.Path(__file__).parent / "data"
                            / "pg19_context_bench.txt")
_CORPUS_SPLIT = "\n<<<SNOWLLM-BENCH-SPLIT>>>\n"
_corpus_cache: dict = {}


def _corpus() -> tuple[str, list[str]]:
    if "pad" not in _corpus_cache:
        pad, *tails = _CORPUS_PATH.read_text().split(_CORPUS_SPLIT)
        _corpus_cache["pad"], _corpus_cache["tails"] = pad, tails
    return _corpus_cache["pad"], _corpus_cache["tails"]


def _tokenizer(path: str) -> "PreTrainedTokenizerBase":
    if "tok" not in _corpus_cache:
        from transformers import AutoTokenizer
        try:
            _corpus_cache["tok"] = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            import pathlib as _pl

            from snowllm.checkpoint.loader import load_tokenizer
            print(f"  transformers cannot read {path!r} ({type(e).__name__}); reading the "
                  f"tokenizer out of the GGUF instead", flush=True)
            _corpus_cache["tok"] = load_tokenizer(_pl.Path(path).expanduser())[0]
    return _corpus_cache["tok"]


def _tokenized_corpus(path: str) -> tuple[list[int], list[list[int]]]:
    if "pad_ids" not in _corpus_cache:
        tok = _tokenizer(path)
        pad, tails = _corpus()
        _corpus_cache["pad_ids"] = tok.encode(pad, add_special_tokens=False)
        _corpus_cache["tail_ids"] = [tok.encode(t, add_special_tokens=False) for t in tails]
    return _corpus_cache["pad_ids"], _corpus_cache["tail_ids"]


_WORD_SNAP = 8


def _cut_at_word(tok: "PreTrainedTokenizerBase", ids: list[int], end: int) -> int:
    for _ in range(_WORD_SNAP):
        if end <= 1 or end >= len(ids) or tok.decode(ids[end:end + 1])[:1].isspace():
            break
        end -= 1
    return end


def prompt_of(target_tokens: int, rep: int, tokenizer: str, warm: bool = False,
              lane: int = 0) -> str:
    tok = _tokenizer(tokenizer)
    pad_ids, tail_ids_all = _tokenized_corpus(tokenizer)
    tail_ids = tail_ids_all[rep % len(tail_ids_all)][:max(0, target_tokens - 2)]
    pad_want = max(0, target_tokens - len(tail_ids) - 2)
    start = lane * pad_want
    if start + pad_want > len(pad_ids):
        raise ValueError(f"lane {lane} of {target_tokens} tokens wants padding to "
                         f"{start + pad_want} but the corpus only tokenizes to {len(pad_ids)} "
                         f"-- extend the PAD section, or sweep this rung at a lower --conc")
    ids = (pad_ids[len(pad_ids) - pad_want:] if warm
           else pad_ids[_cut_at_word(tok, pad_ids, start):_cut_at_word(tok, pad_ids,
                                                                      start + pad_want)])
    return tok.decode(ids) + "\n\n" + tok.decode(tail_ids)


def loop_period(text: str, min_reps: int = 3, min_span: int = 24, max_period: int = 400) -> int:
    for p in range(1, min(max_period, len(text) // min_reps) + 1):
        unit = text[-p:]
        if not unit.strip():
            continue
        k = 1
        while (k + 1) * p <= len(text) and text[-(k + 1) * p:-k * p] == unit:
            k += 1
        if k >= min_reps and k * p >= min_span:
            return p
    return 0


def one(url: str, model: str, text: str, osl: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text + "\n\nSummarise the passage above."}],
        "max_tokens": osl, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    out = 0
    got: list[str] = []
    usage = None
    timings = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            ch = ev.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if not (delta.get("content") or delta.get("reasoning_content")):
                continue
            if first is None:
                first = time.perf_counter()
            got.append(delta.get("content") or delta.get("reasoning_content"))
            out += 1
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("the server streamed no content")
    if not (usage or {}).get("prompt_tokens"):
        raise RuntimeError("the server streamed no usage block; this client needs "
                           "stream_options.include_usage to know the achieved input length")
    return {
        "ttft_s": first - t0,
        "decode_tok_s": (out - 1) / (end - first) if out > 1 and end > first else 0.0,
        "total_s": end - t0,
        "isl": (usage or {}).get("prompt_tokens"),
        "osl": (usage or {}).get("completion_tokens", out),
        "timings": timings,
        "loop_period": loop_period("".join(got)),
        "text_tail": "".join(got)[-160:],
    }


def edge() -> int | None:
    for f in sorted(pathlib.Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/temp1_input")):
        try:
            return int(f.read_text()) // 1000
        except OSError:
            continue
    return None


def draft_accept_len(base_url: str) -> tuple[int, int] | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/").removesuffix("/v1") + "/health",
                                    timeout=3) as r:
            a = json.load(r)["accounting"]
        return a["decode_tokens"], a["decode_rows"]
    except Exception:
        return None


_CTX_OVERFLOW_RE = re.compile(r"prompt \((\d+) tokens\) fills context (\d+)")


def _overflow(url: str, model: str, text: str) -> tuple[int, int] | None:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": 1, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60)
        return None
    except urllib.error.HTTPError as e:
        m = _CTX_OVERFLOW_RE.search(e.read().decode())
        return (int(m.group(1)), int(m.group(2))) if m else None


def one_at(url: str, model: str, isl: int, rep: int, osl: int, warm: bool,
          tokenizer: str, lane: int = 0) -> dict:
    target = isl
    for attempt in range(3):
        try:
            return one(url, model, prompt_of(target, rep, tokenizer, warm=warm, lane=lane), osl)
        except RuntimeError:
            if attempt == 2:
                raise
            overflow = _overflow(url, model,
                                 prompt_of(target, rep, tokenizer, warm=warm, lane=lane))
            if overflow:
                actual, ctx = overflow
                target = int(target * (ctx - 64) / actual)
                print(f"  isl {isl} rep {rep}: {actual} tokens against a {ctx} ceiling, "
                     f"retrying at ~{target}", flush=True)
            else:
                target = int(target * 0.97)
                print(f"  isl {isl} rep {rep}: server rejected the prompt with no ctx-overflow "
                     f"message to measure from, retrying at ~{target} tokens", flush=True)


def one_rep(url: str, model: str, isl: int, rep: int, osl: int, warm: bool,
           tokenizer: str, conc: int) -> dict:
    if conc == 1:
        return one_at(url, model, isl, rep, osl, warm, tokenizer)
    out: list = [None] * conc
    err: list = []

    def lane(i: int) -> None:
        try:
            out[i] = one_at(url, model, isl, rep, osl, warm, tokenizer, lane=i)
        except BaseException as e:                                                # noqa: BLE001
            err.append(e)

    threads = [threading.Thread(target=lane, args=(i,)) for i in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if err:
        raise err[0]
    drafted = sum((r["timings"] or {}).get("draft_n", 0) for r in out)
    return {
        "ttft_s": max(r["ttft_s"] for r in out),
        "decode_tok_s": sum(r["decode_tok_s"] for r in out),
        "decode_tok_s_user": st.median(r["decode_tok_s"] for r in out),
        "total_s": max(r["total_s"] for r in out),
        "isl": out[0]["isl"],
        "isl_total": sum(r["isl"] for r in out),
        "osl": sum(r["osl"] for r in out),
        "timings": ({"draft_n": drafted,
                     "draft_n_accepted": sum((r["timings"] or {}).get("draft_n_accepted", 0)
                                             for r in out)} if drafted else None),
        "loop_period": max(r["loop_period"] for r in out),
        "text_tail": out[0]["text_tail"],
    }


def accept_stat(result: dict, health_before: tuple[int, int] | None,
                health_after: tuple[int, int] | None
                ) -> tuple[float, str] | tuple[None, None]:
    t = result.get("timings") or {}
    if t.get("draft_n"):
        return t["draft_n_accepted"] / t["draft_n"], "accept%"
    if health_before and health_after and health_after[1] != health_before[1]:
        return (health_after[0] - health_before[0]) / (health_after[1] - health_before[1]), "AL"
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    ap.add_argument("--isl", default="1024,32768", help="comma list of target input lengths")
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warm", type=int, default=1, help="untimed requests before the timed ones")
    ap.add_argument("--conc", default="1", help="comma list of concurrencies; each request "
                                                "gets its own slice of the padding corpus, and "
                                                "a rung above 1 is labelled <label>_c<conc>")
    ap.add_argument("--label", default="", help="which configuration these rows describe")
    ap.add_argument("--version", default="", help="the serving engine's own version string")
    ap.add_argument("--commit", default="", help="the source revision these rows were served from")
    ap.add_argument("--out", default="benchmarks/results/results_dsv4_context.json")
    ap.add_argument("--tokenizer", required=True,
                    help="local HF tokenizer dir/name; prompt_of() slices the corpus by exact "
                         "token count against it")
    a = ap.parse_args()

    url = a.base_url.rstrip("/") + "/chat/completions"
    print(f"  using local tokenizer {a.tokenizer!r} for exact prompt lengths", flush=True)
    rows = []

    def save() -> None:
        out = pathlib.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(out.read_text()) if out.exists() else []
        mine = {(r["label"], r["isl_target"]) for r in rows}
        old = [r for r in old if (r.get("label"), r.get("isl_target")) not in mine]
        out.write_text(json.dumps(old + rows, indent=1) + "\n")

    concs = [int(v) for v in a.conc.split(",")]
    for conc in concs:
        for isl in [int(v) for v in a.isl.split(",")]:
            edge_before = edge()
            got, accepts, steps_s, unit = [], [], [], None
            for i in range(a.reps + a.warm):
                is_warm = i < a.warm
                h0 = draft_accept_len(a.base_url)
                r = one_rep(url, a.model, isl, i, a.osl, is_warm, a.tokenizer, conc)
                h1 = draft_accept_len(a.base_url)
                if is_warm:
                    continue
                got.append(r)
                val, u = accept_stat(r, h0, h1)
                if val is not None:
                    accepts.append(val)
                    unit = u
                    if u == "AL" and r["decode_tok_s"] > 0:
                        steps_s.append(val / r["decode_tok_s"])
            edge_after = edge()
            ttft = got[0]["ttft_s"]
            accept_mean = st.mean(accepts) if accepts else None
            accept_stdev = st.stdev(accepts) if len(accepts) > 1 else None
            row = {
                "label": a.label if conc == 1 else f"{a.label}_c{conc}",
                "isl_target": isl,
                "isl": got[0]["isl"],
                "osl": got[0]["osl"],
                "conc": conc,
                "reps": a.reps,
                "warm": a.warm,
                "ttft_s": ttft,
                "ttft_s_reps": [r["ttft_s"] for r in got],
                "decode_tok_s": st.median(r["decode_tok_s"] for r in got),
                "decode_tok_s_user": (st.median(r["decode_tok_s_user"] for r in got)
                                      if conc > 1 else None),
                "prefill_tok_s": (got[0].get("isl_total") or got[0]["isl"]) / ttft if ttft else 0.0,
                "edge_before_c": edge_before,
                "edge_after_c": edge_after,
                "accept_unit": unit,
                "accept_mean": accept_mean,
                "accept_stdev": accept_stdev,
                "accept_reps": accepts,
                "step_s": st.median(steps_s) if steps_s else None,
                "step_s_reps": steps_s,
                "loop_period_reps": [r["loop_period"] for r in got],
                "text_tails": [r["text_tail"] for r in got],
            }
            if a.version:
                row["version"] = a.version
            if a.commit:
                row["commit"] = a.commit
            rows.append(row)
            edge_str = f"{edge_before}->{edge_after}degC" if edge_before is not None else "?"
            if accept_mean is None:
                acc_str = "accept ?"
            elif accept_stdev is None:
                acc_str = f"{unit} {accept_mean:.3f}"
            else:
                acc_str = f"{unit} {accept_mean:.3f}+-{accept_stdev:.3f}"
            loops = [p for p in row["loop_period_reps"] if p]
            loop_str = f"  LOOPED {loops}" if loops else ""
            step_str = "step ?" if row["step_s"] is None else f"step {row['step_s'] * 1e3:6.2f}ms"
            conc_str = "" if conc == 1 else f"  x{conc} ({row['decode_tok_s_user']:.2f}/user)"
            print(f"  ISL {row['isl']:>7d}{conc_str}  TTFT {row['ttft_s']:7.2f}s  "
                  f"prefill {row['prefill_tok_s']:7.0f} tok/s  "
                  f"decode {row['decode_tok_s']:6.2f} tok/s  {step_str}  edge {edge_str}  "
                  f"{acc_str}  "
                  f"[{' '.join(f'{t:.2f}' for t in row['ttft_s_reps'])}]{loop_str}", flush=True)
            save()

    save()


if __name__ == "__main__":
    main()
