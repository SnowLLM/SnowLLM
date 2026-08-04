# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SnowLLM project

"""`\\boxed{}` extraction and judging, for benchmarks/eval_aime.py.

Extraction is the one part of a benchmark with nothing to check it against: a model that answers
correctly and an extractor that cannot see the answer produce the same 0.0, and the run costs three
hours either way. It gets a test with no GPU in it so it can be wrong loudly instead of quietly.

The case that motivated this file: `\\boxed{277}` scored as no-answer, because the scan skipped the
opening brace the search string had consumed without opening the depth counter to match, so the
first `}` took depth to -1 and nothing ever closed.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from eval_aime import boxed, scored  # noqa: E402

ok = True


def check(name, got, want):
    global ok
    hit = got == want
    ok &= hit
    print(f"  {name:<52} {'PASS' if hit else 'FAIL'}  got {got!r}, want {want!r}")


print("boxed()")
check("plain integer", boxed(r"so $m+n = \boxed{277}$."), "277")
check("the whole payload, not one character",
      boxed(r"\boxed{1000}"), "1000")
check("braces inside survive", boxed(r"$\boxed{\frac{252}{25}}$"), r"\frac{252}{25}")
check("nested two deep", boxed(r"\boxed{\frac{a}{\sqrt{b}}}"), r"\frac{a}{\sqrt{b}}")
check("the LAST box wins", boxed(r"first \boxed{1} then \boxed{2}"), "2")
check("no box at all", boxed("the answer is 277"), None)
check("cut off inside the box", boxed(r"\boxed{27"), None)
check("empty box", boxed(r"\boxed{}"), "")
check("bare \\boxed without a brace", boxed(r"\boxed 277"), None)
check("trailing text after the box is dropped",
      boxed(r"\boxed{277}$. That is the answer."), "277")

print("\nscored()")
check("exact integer", scored("277", 277), True)
check("wrong integer", scored("278", 277), False)
check("None never scores", scored(None, 277), False)
check("surrounding whitespace", scored("  277\n", 277), True)
check("thousands separator", scored("1,000", 1000), True)
check("latex thin space", scored(r"1\!000", 1000), True)
check("a fraction is not an AIME answer", scored(r"\frac{252}{25}", 277), False)
check("prose is not an answer", scored("two hundred seventy-seven", 277), False)
check("zero is a legal answer", scored("0", 0), True)

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
