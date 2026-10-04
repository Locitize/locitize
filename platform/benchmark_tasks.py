"""Fixed local deterministic task set for the LOCITIZE benchmark scorer (M5.3).

This module is PURE DATA: a fixed set of quality / reasoning / coding tasks,
each carrying a prompt and a machine-checkable expected-output rule. There
is no network, no model weights, and no download here.

It is the single source of what the benchmark "score" means:

    N of M deterministic local checks passed.

The score is NOT an LLM-judged rating and NOT a subjective quality number. Each
task is sent to the model with temperature 0 and a fixed seed, and its returned
text is fed to a deterministic checker (exact_match / keyword / numeric) that is a
pure function of (response_text, expected). A task passes or fails; the category
score is 100 * passed / total. This honest framing is repeated in the results
file header (docs/benchmark_results.md) so a reader is never misled into thinking
the number is a quality verdict.

Checker kinds (evaluated in benchmark.py):
- exact_match : the normalized (trim + casefold) response must CONTAIN the
                normalized expected string. Containment (not full equality) is
                used because instruct models wrap the answer in a sentence.
- keyword     : the response must contain every string in expected["all"]
                (case-insensitive) and none in expected.get("none", []).
- numeric     : the final number parsed from the response must equal
                expected["value"] within expected.get("tol", 0).

The set was 6 checks through M15.3 and is 24 as of M15.4 (8 per category). The
expansion exists because 6 checks could not discriminate: 0.9 GB fine-tunes
outscored 8B models, and a single miss moved a category by 50 points. 24 is
still NOT a comprehensive capability benchmark (that would need a human or an
LLM judge, explicitly out of scope, Architecture M5.3) - it is a wider health
signal with 12.5-point granularity per category, and its number must always be
read WITH its sample size.

Task-authoring rules, learned the hard way and kept here so additions follow
them:
- numeric checks are the most robust: the checker parses the FINAL number in
  the response, so "Reply with only the number" prompts double as
  instruction-following checks (a model that answers "2 hours and 35 minutes"
  to a minutes question parses as 35 and fails - signal, not noise);
- containment is case-folded, so never use a short or embeddable expected
  string ("no" matches "know", "Au" matches "because") and never ask about
  letter case at all;
- for boolean answers use TRUE/FALSE tokens, which are not substrings of each
  other, with the wrong one in the "none" list.

Scorer caveat (Reviewer F-4, accepted Low, documented here honestly): the
deterministic checkers match by CONTAINMENT, not full equality, so a model that
merely MENTIONS the expected token incidentally -- e.g. writes "Paris" inside an
unrelated sentence, or names "len" while discussing something else -- can pass a
task it did not really "answer". This is a deliberate, honest trade: containment is
what lets a terse instruct model wrap the answer in a sentence and still score,
and the prompts steer hard toward a bare answer to minimize the incidental-match
window. The number this produces is therefore "N of M deterministic containment
checks passed", NOT a semantic-correctness verdict; it is a stable machinery/health
signal across runs and models, and is never presented as a quality grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    """One deterministic task: a prompt plus a machine-checkable expected rule.

    category is one of "quality", "reasoning", "coding" (the three scored
    buckets). check names the deterministic checker; expected carries that
    checker's parameters (a string, a keyword spec, or a numeric spec).
    """

    id: str
    category: str
    prompt: str
    check: str  # "exact_match" | "keyword" | "numeric"
    expected: Any


# The fixed task set. Unambiguous and machine-checkable. Prompts steer the
# model toward a terse answer so the deterministic checkers are reliable.
TASKS: tuple[BenchmarkTask, ...] = (
    # ---- quality: factual recall (8) --------------------------------------- #
    BenchmarkTask(
        id="q-capital-france", category="quality",
        prompt="What is the capital of France? Answer with only the city name.",
        check="exact_match", expected="Paris",
    ),
    BenchmarkTask(
        id="q-largest-planet", category="quality",
        prompt="What is the largest planet in our solar system? "
        "Answer with only the planet name.",
        check="exact_match", expected="Jupiter",
    ),
    BenchmarkTask(
        id="q-author-1984", category="quality",
        prompt="Who wrote the novel Nineteen Eighty-Four? "
        "Answer with only the author's surname.",
        check="exact_match", expected="Orwell",
    ),
    BenchmarkTask(
        id="q-element-8", category="quality",
        prompt="Which chemical element has atomic number 8? "
        "Answer with only the element name.",
        check="exact_match", expected="oxygen",
    ),
    BenchmarkTask(
        id="q-continent-sahara", category="quality",
        prompt="On which continent is the Sahara Desert? "
        "Answer with only the continent name.",
        check="exact_match", expected="Africa",
    ),
    BenchmarkTask(
        id="q-year-moon", category="quality",
        prompt="In which year did humans first walk on the Moon? "
        "Reply with only the year.",
        check="numeric", expected={"value": 1969, "tol": 0},
    ),
    BenchmarkTask(
        id="q-feb-leap", category="quality",
        prompt="How many days does February have in a leap year? "
        "Reply with only the number.",
        check="numeric", expected={"value": 29, "tol": 0},
    ),
    BenchmarkTask(
        id="q-hex-ff", category="quality",
        prompt="What is the hexadecimal number FF in decimal? "
        "Reply with only the number.",
        check="numeric", expected={"value": 255, "tol": 0},
    ),
    # ---- reasoning: multi-step arithmetic and logic (8) -------------------- #
    BenchmarkTask(
        id="r-arithmetic", category="reasoning",
        prompt="Compute 17 + 26. Reply with only the number.",
        check="numeric", expected={"value": 43, "tol": 0},
    ),
    BenchmarkTask(
        id="r-word-problem", category="reasoning",
        prompt="A basket has 12 apples. You remove 5 and then add 3. "
        "How many apples are in the basket now? Reply with only the number.",
        check="numeric", expected={"value": 10, "tol": 0},
    ),
    BenchmarkTask(
        id="r-order-ops", category="reasoning",
        prompt="Compute 2 + 3 * 4 - 6 / 2. Reply with only the number.",
        check="numeric", expected={"value": 11, "tol": 0},
    ),
    BenchmarkTask(
        id="r-train-minutes", category="reasoning",
        prompt="A train departs at 09:40 and arrives at 12:15 on the same day. "
        "How many minutes does the journey take? Reply with only the number.",
        check="numeric", expected={"value": 155, "tol": 0},
    ),
    BenchmarkTask(
        id="r-remainder", category="reasoning",
        prompt="What is the remainder when 1000 is divided by 7? "
        "Reply with only the number.",
        check="numeric", expected={"value": 6, "tol": 0},
    ),
    BenchmarkTask(
        id="r-sequence", category="reasoning",
        prompt="What number comes next in this sequence: 2, 6, 12, 20, 30? "
        "Reply with only the number.",
        check="numeric", expected={"value": 42, "tol": 0},
    ),
    BenchmarkTask(
        id="r-ages", category="reasoning",
        prompt="Anna is twice as old as Ben, and their ages add up to 36. "
        "How old is Ben? Reply with only the number.",
        check="numeric", expected={"value": 12, "tol": 0},
    ),
    BenchmarkTask(
        id="r-letter-count", category="reasoning",
        prompt="How many times does the letter E appear in the word BEEKEEPER? "
        "Reply with only the number.",
        check="numeric", expected={"value": 5, "tol": 0},
    ),
    # ---- coding: constructs and output prediction (8) ---------------------- #
    BenchmarkTask(
        id="c-python-reverse", category="coding",
        prompt="In Python, write one line that reverses the string s using "
        "slicing. Show the slice expression.",
        check="keyword", expected={"all": ["[::-1]"], "none": []},
    ),
    BenchmarkTask(
        id="c-python-length", category="coding",
        prompt="In Python, which built-in function returns the number of items "
        "in a list? Answer with only the function name.",
        check="keyword", expected={"all": ["len"], "none": []},
    ),
    BenchmarkTask(
        id="c-print-len", category="coding",
        prompt='What does this Python code print: print(len("hello") * 2) '
        "Reply with only the printed output.",
        check="numeric", expected={"value": 10, "tol": 0},
    ),
    BenchmarkTask(
        id="c-negative-index", category="coding",
        prompt="What does this Python expression evaluate to: [1, 2, 3][-1] "
        "Reply with only the value.",
        check="numeric", expected={"value": 3, "tol": 0},
    ),
    BenchmarkTask(
        id="c-modulo", category="coding",
        prompt="What does 17 % 5 evaluate to in Python? "
        "Reply with only the number.",
        check="numeric", expected={"value": 2, "tol": 0},
    ),
    BenchmarkTask(
        id="c-range-len", category="coding",
        prompt="What does len(range(3, 10)) evaluate to in Python? "
        "Reply with only the number.",
        check="numeric", expected={"value": 7, "tol": 0},
    ),
    BenchmarkTask(
        id="c-empty-bool", category="coding",
        prompt="In Python, what does bool([]) evaluate to? "
        "Answer with exactly TRUE or FALSE.",
        check="keyword", expected={"all": ["false"], "none": ["true"]},
    ),
    BenchmarkTask(
        id="c-find-index", category="coding",
        prompt='What does "python".find("h") return in Python? '
        "Reply with only the number.",
        check="numeric", expected={"value": 3, "tol": 0},
    ),
)


# The three scored categories, single-sourced so the scorer and the results
# header stay in agreement.
CATEGORIES: tuple[str, ...] = ("quality", "reasoning", "coding")


def tasks_for(category: str) -> list[BenchmarkTask]:
    """Return the fixed tasks in one category (quality/reasoning/coding)."""
    return [t for t in TASKS if t.category == category]
