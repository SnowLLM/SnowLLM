# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snowllm.benchmark import context

if __name__ == "__main__":
    context.main()
else:
    sys.modules[__name__] = context
