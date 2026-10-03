#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

import argparse
import json
import os
import subprocess
import time


def result_summary(result):
    text = "\n".join(c.get("text", "") for c in result.get("content", []))
    return {"chars": len(text), "lines": text.count("\n") + (1 if text else 0),
            "isError": bool(result.get("isError"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-dir", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--system-file", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cmd = ["pi", "--provider", a.provider, "--model", a.model,
           "--mode", "json", "--no-session",
           "--no-context-files", "--no-skills", "--no-extensions",
           "--tools", "read,bash",
           "--system-prompt", os.path.abspath(a.system_file),
           "--thinking", "off", open(a.prompt_file).read()]
    env = dict(os.environ, PI_CODING_AGENT_DIR=os.path.abspath(a.agent_dir),
               PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1")

    events = [{"type": "meta", "label": a.label}]
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)
    for raw in p.stdout:
        t = time.monotonic() - t0
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        typ = ev.get("type")
        if typ == "message_update":
            ame = ev.get("assistantMessageEvent") or {}
            if ame.get("type") in ("text_delta", "thinking_delta"):
                kind = "text" if ame["type"] == "text_delta" else "thinking"
                events.append({"type": "delta", "kind": kind, "t": round(t, 4),
                               "text": ame.get("delta", "")})
        elif typ == "tool_execution_start":
            events.append({"type": "tool_start", "t": round(t, 4),
                           "id": ev.get("toolCallId"), "name": ev.get("toolName"),
                           "args": ev.get("args")})
        elif typ == "tool_execution_end":
            events.append({"type": "tool_end", "t": round(t, 4),
                           "id": ev.get("toolCallId"), "name": ev.get("toolName"),
                           "summary": result_summary(ev.get("result"))})
    rc = p.wait()
    events.append({"type": "exit", "t": round(time.monotonic() - t0, 4)})
    json.dump(events, open(a.out, "w"), ensure_ascii=False)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
