# SPDX-License-Identifier: Apache-2.0
import hashlib
from functools import lru_cache
from pathlib import Path


FILE = '''import csv
from pathlib import Path


def load_measurements(path: Path) -> list[dict]:
    samples = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                raise ValueError(f"line {line_number}: missing measurement name")
            try:
                elapsed = float(row["elapsed_seconds"])
                token_count = int(row["tokens"])
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"line {line_number}: invalid measurement") from error
            if elapsed <= 0 or token_count < 0:
                raise ValueError(f"line {line_number}: invalid duration or token count")
            samples.append({
                "name": name,
                "elapsed_seconds": elapsed,
                "tokens": token_count,
                "tokens_per_second": token_count / elapsed,
            })
    return samples


def group_by_name(measurements: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for measurement in measurements:
        name = measurement["name"]
        grouped.setdefault(name, []).append(measurement)
    return grouped
'''


def messages(excerpt):
    return [{"role": "user", "content": (
        f"Repository context from earlier file reads:\n{excerpt}\n\n"
        f"read_file('metrics.py') returned:\n```python\n{FILE}```\n\n"
        "Edit metrics.py: rename the local samples list in load_measurements to records "
        "at every use. Return the complete edited file as plain Python."
    )}]


@lru_cache(maxsize=1)
def source():
    root = Path(__file__).resolve().parents[1]
    paths = ("serve/api.py", "serve/generation.py", "serve/messages.py", "serve/responses.py")
    return "\n\n".join(f"# {p}\n{(root / p).read_text(encoding='utf-8')}" for p in paths)


@lru_cache(maxsize=4)
def repository_tokens(tokenizer):
    from .context import _tokenizer
    return _tokenizer(tokenizer).encode(source(), add_special_tokens=False)


def request_at(url, model, isl, rep, osl, warm, tokenizer, lane=0, thinking=False):
    from .context import _chat_tokens, _fit_excerpt, _tokenizer, stream_chat
    tok = _tokenizer(tokenizer)
    ids = repository_tokens(tokenizer)
    offset = ((1 + rep + 3 * lane + int(warm) * 17) * 997) % len(ids)

    def excerpt(length):
        repeated = ids * ((offset + length) // len(ids) + 1)
        return tok.decode(repeated[offset:offset + length])

    text = _fit_excerpt(isl, excerpt, lambda s: _chat_tokens(tok, messages(s), thinking))
    prompt = messages(text)
    row = stream_chat(url, model, prompt, osl, thinking)
    if row["isl"] != isl:
        raise ValueError(f"file edit requested {isl} input tokens; server reported {row['isl']}")
    row["prompt_sha256"] = hashlib.sha256(prompt[0]["content"].encode()).hexdigest()
    return row
