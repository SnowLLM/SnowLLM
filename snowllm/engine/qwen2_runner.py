# SPDX-License-Identifier: Apache-2.0
from .qwen_runner import QwenRunner


class Qwen2Runner(QwenRunner):
    tunable_rope = False
