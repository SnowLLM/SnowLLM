# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ACCESS_LOG = """GET /health 200
POST /orders 503
GET /missing 404
GET /catalog 500
POST /orders 201
POST /orders 502
GET /catalog 304
"""
PROMPT = (
    "Read access.log in the current directory. Each line is METHOD PATH STATUS. "
    "Count the requests with HTTP status 500 through 599, grouped by path. "
    "Write report.json with two keys: total (the number of such requests) and "
    "by_path (an object mapping only paths with server errors to their counts). "
    "Leave access.log unchanged. Tell me when you have finished."
)


def check(work, output):
    completed_tools = []
    last_assistant = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "tool_execution_end" and event.get("isError") is False:
            completed_tools.append(event["toolName"])
        if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
            last_assistant = event["message"]
    if last_assistant.get("stopReason") != "stop":
        raise ValueError(f"Pi did not finish normally: {last_assistant.get('errorMessage') or last_assistant.get('stopReason')}")
    if "read" not in completed_tools or "write" not in completed_tools[completed_tools.index("read") + 1:]:
        raise ValueError(f"Expected a successful read followed by a write, got {completed_tools}")
    if (work / "access.log").read_bytes() != ACCESS_LOG.encode():
        raise ValueError("Pi modified access.log")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    expected = {"total": 3, "by_path": {"/orders": 2, "/catalog": 1}}
    # JSON booleans compare equal to Python integers.
    if json.dumps(report, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(f"Incorrect report.json: {report!r}")


def main():
    parser = argparse.ArgumentParser(description="Manually test tool use with Pi and a running SnowLLM model.")
    parser.add_argument("--model", required=True, help="Served model ID in Pi's snowllm provider")
    parser.add_argument("--timeout", default=180, type=int, help="Maximum runtime in seconds (default: 180)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    pi = shutil.which("pi")
    if not pi:
        parser.exit(2, "Pi CLI was not found on PATH. Install Pi and configure its snowllm provider first.\n")
    try:
        with tempfile.TemporaryDirectory(prefix="snowllm-pi-task-") as folder:
            work = Path(folder)
            (work / "access.log").write_bytes(ACCESS_LOG.encode())
            result = subprocess.run(
                [pi, "--provider", "snowllm", "--model", args.model, "--mode", "json",
                 "--no-session", "--tools", "read,write", "-p", PROMPT],
                cwd=work, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8",
                timeout=args.timeout,
            )
            if result.returncode:
                raise ValueError(f"Pi exited with status {result.returncode}: {result.stderr.strip() or result.stdout[-1000:]}")
            check(work, result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {args.model} used Pi to read access.log and write the correct report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
