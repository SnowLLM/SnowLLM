# Benchmarks

One Ryzen AI Max+ 395 (gfx1151), `Qwen3.6-35B-A3B-FP8`, ROCm torch 2.12.0+rocm7.14.0, kernel ABI 1,
KV block 16. Decode graphs captured at B=1..4, speculation off (`k=2` rows are eager).

```sh
benchmarks/run-concurrency.sh        # 8K/1K, 1-4 users      (~20 min)
benchmarks/run-context-sweep.sh      # 1K..220K context      (~35 min, splittable)
benchmarks/run-aime.sh               # AIME 2026 accuracy    (~1.8 h; --n for more)
```

Each writes a JSON under `benchmarks/results/` with every per-repetition sample.

| `run-aime.sh` | samples | output tokens | wall |
|---|---|---|---|
| `--n 1` (default) | 30 | 0.9M | ~1.8 h |
| `--n 2` | 60 | 1.9M | ~3.6 h |
| `--n 4` (measured) | 120 | 3.92M | 8.54 h |
| `--n 8` | 240 | 7.4M | ~14 h |

## Metrics

| | |
|---|---|
| `prefill tok/s` | prompt tokens ÷ time to last first-token (`S_PP`) |
| `output tok/s` | generated tokens ÷ remaining time (`S_TG`) |
| `per user` | `output tok/s ÷ concurrency` |
| `TTFT` | submit → first token, median |
| `AL` | accepted tokens/decode step. 1.000 with speculation off, ceiling `num_spec + 1` |
| `k` | `--num-spec` (MTP depth). `k=0` floor, `k=2` shipped default |

Prefill/decode reported separately (split = last request's first token). Blended tok/s kept in JSON
as `blended_tok_s`, not quoted here.

## 8K in, 1K out

Median of 3 reps, fresh prompts each time.

| C | k | TTFT ms | prefill tok/s | output tok/s | per user | AL |
|---|---|---|---|---|---|---|
| 1 | 0 | 2091 | 3511 | 49.6 | 49.6 | 1.000 |
| 1 | 2 | 2117 | 3467 | **76.7** | 76.7 | 2.983 |
| 2 | 0 | 3194 | 3430 | 77.1 | 38.5 | 1.000 |
| 2 | 2 | 3176 | 3461 | 130.0 | 65.0 | 2.983 |
| 3 | 0 | 4269 | 3444 | 95.8 | 31.9 | 1.000 |
| 3 | 2 | 4250 | 3454 | 143.5 | 47.8 | 2.974 |
| 4 | 0 | 5371 | 3390 | 113.7 | 28.4 | 1.000 |
| 4 | 2 | 5313 | 3430 | **176.9** | 44.2 | 2.978 |

## Context, 1K out, one user

2 reps at top two rows, 3 below.

| ISL | achieved | k | TTFT s | prefill tok/s | output tok/s | AL |
|---|---|---|---|---|---|---|
| 1024 | 917 | 0 | 0.4 | 2409 | 51.4 | 1.000 |
| 1024 | 917 | 2 | 0.4 | 2315 | 86.9 | 2.991 |
| 8192 | 7341 | 0 | 2.0 | 3596 | 49.6 | 1.000 |
| 8192 | 7341 | 2 | 2.1 | 3473 | 76.9 | 2.983 |
| 32768 | 29369 | 0 | 9.4 | 3133 | 45.1 | 1.000 |
| 32768 | 29369 | 2 | 9.8 | 2994 | 72.2 | 2.974 |
| 131072 | 118059 | 0 | 65.2 | 1810 | 32.9 | 1.000 |
| 131072 | 118059 | 2 | 69.0 | 1711 | 56.3 | 2.805 |
| 225280 | 206628 | 0 | 157.0 | 1316 | 26.3 | 1.000 |
| 225280 | 206628 | 2 | 168.4 | 1227 | 49.3 | 2.957 |

Prefill k=2 vs k=0: -1.3% at 8K, -4.4% at 32K, -5.5% at 128K, -6.8% at 220K (draft layer also scans
prefill). `Engine(mtp_window=...)` bounds this, off by default, unmeasured here.

## Prompt source: synthetic vs real text

ISL 8192, **OSL 256** (not 1024 above), C=1, 24 prompts/row (`--prompts speedbench` for
`high_entropy`/`low_entropy`; `mixed` tier ships placeholders, unusable).

| prompts | k | AL | ms/decode step | per user |
|---|---|---|---|---|
| SPEED-Bench high_entropy | 0 | 1.000 | 20.15 | 49.63 |
| SPEED-Bench low_entropy | 0 | 1.000 | 20.12 | 49.71 |
| random | 0 | 1.000 | 20.17 | 49.58 |
| SPEED-Bench high_entropy | 2 | 2.318 | 37.45 | 61.90 |
| SPEED-Bench low_entropy | 2 | 2.615 | 37.74 | 69.31 |
| random | 2 | **2.898** | 42.83 | 67.67 |

Random-token AL rises with output length: 2.898 at OSL 256, 2.983 at OSL 1024 (ceiling 3.000). The
8K/1K table's `k=2` rows sit at 99% draft acceptance — expect ~10-20% below on real text.

## Accuracy

**AIME 2026, avg@4 over 30 problems: 91.7%** (110/120, ±4.2). Model card: 92.7 avg@8 for
`Qwen3.6-35B-A3B` (twice this depth).

Settings: `temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5`, 81920-token cap.
`MathArena/aime_2026`, graded on last `\boxed{}` as integer. ABI 4, `ctx 84000`,
`--max-num-seqs 16`, run 2026-08-03 11:47-20:20 (8.54 h). Not comparable to ABI 1 tables above.

Misses (10, over 5 problems):

| | |
|---|---|
| p15 | 0/4 — 3 samples cap-out, 1 answered 81 for 83 |
| p29 | 1/4 — 1 cap-out, 2 reasoning errors |
| p9, p10, p30 | 3/4 each — 1 reasoning error apiece |

p15 has never converged on this checkpoint (0/4 here, ABI 2 run's one sample also capped). Alone:
320K output tokens, 8% of run, largest single term in the score. 28/30 problems ≥3/4.

Engine step accounting, 511 one-minute windows (432 at full B=16):

| | |
|---|---|
| decode | 138.9 tok/s mean at B=16, 135.1 median, 109.4-286.5 |
| `AL` | 2.605 over 113046 steps, 2.497-2.945 per window |
| prefill | 520.3 tok/s over 95 windows containing any prefill |
| KV pool | 1.4% → 35.6% |


## Caveats

- One machine, one thermal state, TDP unpinned. Spread: 0.1-4.9% (`k=0`), 6.8-11.1% (`k=2`).
- Concurrency stops at C=4. MoE expert-set effects at wide batch untested above C=4.
- No cross-hardware comparison implied — shared method, not a scoreboard.
- `max_model_len` sized to the test (`ISL + OSL + 64`); pass `--max-model-len` for real deployments.
- Accuracy is avg@4 vs card's avg@8 (±4.2, gap sits inside error bar — not evidence of a difference).
  Speed and accuracy are separate runs; no row pairs throughput with a quality metric.