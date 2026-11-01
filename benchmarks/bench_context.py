# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics as st
import threading
import time
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
    tokenizers = _corpus_cache.setdefault("tokenizers", {})
    if path not in tokenizers:
        from transformers import AutoTokenizer
        try:
            tokenizers[path] = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            import pathlib as _pl

            from snowllm.checkpoint.loader import load_tokenizer
            print(f"  transformers cannot read {path!r} ({type(e).__name__}); reading the "
                  f"tokenizer out of the GGUF instead", flush=True)
            tokenizers[path] = load_tokenizer(_pl.Path(path).expanduser())[0]
    return tokenizers[path]


def _tokenized_corpus(path: str) -> tuple[list[int], list[list[int]]]:
    tokenized = _corpus_cache.setdefault("tokenized", {})
    if path not in tokenized:
        tok = _tokenizer(path)
        pad, tails = _corpus()
        tokenized[path] = (tok.encode(pad, add_special_tokens=False),
                           [tok.encode(t, add_special_tokens=False) for t in tails])
    return tokenized[path]


_WORD_SNAP = 8


def _cut_at_word(tok: "PreTrainedTokenizerBase", ids: list[int], end: int) -> int:
    for _ in range(_WORD_SNAP):
        if end <= 1 or end >= len(ids) or tok.decode(ids[end:end + 1])[:1].isspace():
            break
        end -= 1
    return end


def _passage_of(target_tokens: int, rep: int, tokenizer: str, warm: bool,
                lane: int, lane_width: int) -> str:
    tok = _tokenizer(tokenizer)
    pad_ids, tail_ids_all = _tokenized_corpus(tokenizer)
    tail_ids = tail_ids_all[rep % len(tail_ids_all)][:max(0, target_tokens - 2)]
    pad_want = max(0, target_tokens - len(tail_ids) - 2)
    start = _cut_at_word(tok, pad_ids, lane * lane_width)
    if start + pad_want > len(pad_ids):
        raise ValueError(f"lane {lane} of {target_tokens} tokens wants padding to "
                         f"{start + pad_want} but the corpus only tokenizes to {len(pad_ids)} "
                         f"-- extend the PAD section, or sweep this rung at a lower --conc")
    ids = (pad_ids[len(pad_ids) - pad_want:] if warm
           else pad_ids[start:start + pad_want])
    return tok.decode(ids) + "\n\n" + tok.decode(tail_ids)


def _messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text + "\n\nSummarise the passage above."}]


def _thinking_options(thinking: bool | None) -> dict:
    return {} if thinking is None else {"thinking": thinking, "enable_thinking": thinking}


def resolve_thinking(mode: str, capabilities: dict) -> bool | None:
    if capabilities.get("required"):
        if mode == "off":
            raise ValueError("this model requires thinking; use --thinking auto or on")
        return True
    if mode == "auto":
        return False if capabilities.get("switch") else None
    if not capabilities.get("switch"):
        raise ValueError("this model does not advertise a thinking switch; use --thinking auto")
    return mode == "on"


def model_info(base_url: str, model: str) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:
        models = json.load(response)["data"]
    selected = next((item for item in models if item.get("id") == model), None)
    if selected is None:
        raise ValueError(f"model {model!r} is not advertised by {base_url}/models")
    return selected


def server_runtime(base_url: str) -> dict:
    url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            runtime = json.load(response).get("runtime_config") or {}
            return {k: v for k, v in runtime.items()
                    if k not in ("model_path", "dflash_path", "profile_dir")}
    except (OSError, ValueError):
        return {}


def check_platform(expected: str, runtime: dict) -> None:
    actual = runtime.get("gpu_arch")
    if expected and actual != expected:
        raise ValueError(f"expected {expected}, but server GPU architecture is {actual!r}; "
                         "benchmark hardware must match before measuring")


def _prompt_tokens(tok: "PreTrainedTokenizerBase", text: str,
                   thinking: bool | None = False) -> int:
    encoded = tok.apply_chat_template(
        _messages(text), add_generation_prompt=True, tokenize=True, return_dict=True,
        tools=None, **_thinking_options(thinking))
    ids = encoded["input_ids"]
    return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)


def prompt_of(target_tokens: int, rep: int, tokenizer: str, warm: bool = False,
              lane: int = 0, thinking: bool | None = False) -> str:
    tok = _tokenizer(tokenizer)
    minimum = _prompt_tokens(tok, "", thinking)
    if target_tokens < minimum:
        raise ValueError(f"input length {target_tokens} is below the {minimum} tokens needed "
                         "for the chat template and instruction")
    if target_tokens == minimum:
        return ""
    budget, seen, over = target_tokens, set(), None
    for _ in range(32):
        if budget in seen:
            break
        seen.add(budget)
        text = _passage_of(budget, rep, tokenizer, warm, lane, target_tokens)
        actual = _prompt_tokens(tok, text, thinking)
        if actual == target_tokens:
            return text
        if actual > target_tokens and (over is None or actual < over[0]):
            over = actual, text
        budget = max(0, budget + target_tokens - actual)
    # BPE decode/encode can skip a token length at a corpus boundary.
    if over is not None:
        for cut in range(1, min(128, len(over[1])) + 1):
            text = over[1][:-cut]
            if _prompt_tokens(tok, text, thinking) == target_tokens:
                return text
    raise ValueError(f"cannot construct an exact {target_tokens}-token chat prompt "
                     f"for rep {rep}, lane {lane}; last length was {actual}")


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


def one(url: str, model: str, text: str, osl: int, thinking: bool | None = False) -> dict:
    body = json.dumps({
        "model": model,
        "messages": _messages(text),
        "max_tokens": osl, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        **_thinking_options(thinking),
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    out = 0
    got: list[str] = []
    usage = None
    timings = None
    finish_reason = None
    done = False
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            ev = json.loads(payload)
            if ev.get("error"):
                raise RuntimeError(f"server stream error: {ev['error']}")
            if ev.get("usage"):
                usage = ev["usage"]
                timings = usage.get("timings") or timings
            if ev.get("timings"):
                timings = ev["timings"]
            ch = ev.get("choices") or []
            if ch and ch[0].get("finish_reason"):
                finish_reason = ch[0]["finish_reason"]
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
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 0 < tokens <= osl:
        raise RuntimeError(f"invalid completion_tokens={tokens!r} for output cap {osl}")
    if not done or finish_reason not in ("stop", "length"):
        raise RuntimeError(f"incomplete or failed stream: finish_reason={finish_reason!r}, done={done}")
    return {
        "ttft_s": first - t0,
        "decode_tok_s": (tokens - 1) / (end - first) if tokens > 1 and end > first else 0.0,
        "decode_seconds": end - first,
        "stream_chunks": out,
        "finish_reason": finish_reason,
        "total_s": end - t0,
        "isl": (usage or {}).get("prompt_tokens"),
        "osl": tokens,
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


def one_at(url: str, model: str, isl: int, rep: int, osl: int, warm: bool,
          tokenizer: str, lane: int = 0, thinking: bool | None = False) -> dict:
    text = prompt_of(isl, rep, tokenizer, warm=warm, lane=lane, thinking=thinking)
    result = one(url, model, text, osl, thinking=thinking)
    if result["isl"] != isl:
        raise ValueError(f"requested {isl} input tokens but server reported {result['isl']}; "
                         "check that the local tokenizer and chat template match the server")
    return result


def one_rep(url: str, model: str, isl: int, rep: int, osl: int, warm: bool,
           tokenizer: str, conc: int, thinking: bool | None = False) -> dict:
    if conc == 1:
        return one_at(url, model, isl, rep, osl, warm, tokenizer, thinking=thinking)
    out: list = [None] * conc
    err: list = []

    def lane(i: int) -> None:
        try:
            out[i] = one_at(url, model, isl, rep, osl, warm, tokenizer, lane=i, thinking=thinking)
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
        "lanes": out,
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
    ap.add_argument("--isl", default="1024,32768",
                    help="comma list of exact input lengths, including the chat template")
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warm", type=int, default=1, help="untimed requests before the timed ones")
    ap.add_argument("--conc", default="1", help="comma list of concurrencies; each request "
                                                "gets its own slice of the padding corpus, and "
                                                "a rung above 1 is labelled <label>_c<conc>")
    ap.add_argument("--label", default="", help="which configuration these rows describe")
    ap.add_argument("--version", default="", help="the serving engine's own version string")
    ap.add_argument("--commit", default="", help="the source revision these rows were served from")
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                    help="auto enables required thinking, disables switchable thinking, otherwise uses the model default")
    ap.add_argument("--platform", default="", help="expected GPU architecture, checked against server metadata")
    ap.add_argument("--hardware", default="", help="machine description recorded with these measurements")
    ap.add_argument("--out", default="benchmarks/results/results_dsv4_context.json")
    ap.add_argument("--tokenizer", required=True,
                    help="local tokenizer matching the server, including its chat template")
    a = ap.parse_args()
    try:
        concs = [int(v) for v in a.conc.split(",")]
        lengths = [int(v) for v in a.isl.split(",")]
    except ValueError:
        ap.error("--isl and --conc must be comma-separated integers")
    if a.reps < 1 or a.warm < 0 or a.osl < 1 or not all(v > 0 for v in concs+lengths):
        ap.error("lengths, concurrency, output cap and repetitions must be positive; warm-up may be zero")
    info = model_info(a.base_url, a.model)
    runtime = server_runtime(a.base_url)
    try:
        check_platform(a.platform, runtime)
    except ValueError as exc:
        ap.error(str(exc))
    capabilities = info.get("thinking")
    if not isinstance(capabilities, dict):
        from snowllm.serve.thinking import thinking_capabilities
        capabilities = thinking_capabilities(_tokenizer(a.tokenizer))
    try:
        thinking = resolve_thinking(a.thinking, capabilities)
    except ValueError as exc:
        ap.error(str(exc))
    if info.get("context_length") and max(lengths)+a.osl > info["context_length"]:
        ap.error(f"input + output needs {max(lengths)+a.osl} tokens, but model context is "
                 f"{info['context_length']}; increase server context or reduce --isl/--osl")

    url = a.base_url.rstrip("/") + "/chat/completions"
    print(f"  using local tokenizer {a.tokenizer!r} for exact prompt lengths", flush=True)
    rows = []
    corpus_sha256 = hashlib.sha256(_CORPUS_PATH.read_bytes()).hexdigest()

    def save() -> None:
        out = pathlib.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(out.read_text()) if out.exists() else []
        mine = {(r["label"], r["isl_target"]) for r in rows}
        old = [r for r in old if (r.get("label"), r.get("isl_target")) not in mine]
        out.write_text(json.dumps(old + rows, indent=1) + "\n")

    for conc in concs:
        for isl in lengths:
            edge_before = edge()
            got, accepts, steps_s, unit = [], [], [], None
            for i in range(a.reps + a.warm):
                is_warm = i < a.warm
                h0 = draft_accept_len(a.base_url)
                r = one_rep(url, a.model, isl, i, a.osl, is_warm, a.tokenizer, conc, thinking=thinking)
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
                "input_length_mode": "chat_template_exact",
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
                "measurement": "http_stream_usage_tokens",
                "decode_token_count": "usage.completion_tokens_minus_first",
                "aggregation": "decode median; TTFT and prefill first retained request",
                "thinking": "default" if thinking is None else "on" if thinking else "off",
                "thinking_requested": a.thinking,
                "thinking_capabilities": capabilities,
                "output_cap": a.osl,
                "ttft_s_median": st.median(r["ttft_s"] for r in got),
                "ttft_s_p95": sorted(r["ttft_s"] for r in got)[math.ceil(.95*len(got))-1],
                "decode_tok_s_reps": [r["decode_tok_s"] for r in got],
                "osl_reps": [r["osl"] for r in got],
                "samples": got,
                "runtime_config": runtime,
                "corpus_sha256": corpus_sha256,
            }
            if runtime.get("gpu_arch"):
                row["platform"] = runtime["gpu_arch"]
            if a.hardware:
                row["hardware"] = a.hardware
            if a.version or runtime.get("version"):
                row["version"] = a.version or runtime["version"]
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
