#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Supervise one development job and its process group (POSIX, CPU only)."""
import argparse
import math
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('a command is required after --')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be positive and finite')
    received = 0

    def interrupted(signum, _frame):
        nonlocal received
        received = received or signum

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        child = subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        print(f'run-job: {exc}', file=sys.stderr)
        return 127
    deadline = None if args.timeout is None else time.monotonic() + args.timeout

    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    try:
        while True:
            if received:
                result = 128 + received
                break
            if child.poll() is not None:
                result = child.returncode
                break
            if deadline is not None and time.monotonic() >= deadline:
                print(f'run-job: timed out after {args.timeout:g}s',
                      file=sys.stderr)
                result = 124
                break
            time.sleep(.05)
    finally:
        # Also collect background children when their immediate parent exits.
        # Kill only this newly created group.
        send(signal.SIGTERM)
        until = time.monotonic() + 2
        while time.monotonic() < until:
            child.poll()
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.05)
        send(signal.SIGKILL)
        child.wait()
    return 128 - result if result < 0 else result


if __name__ == '__main__':
    sys.exit(main())
