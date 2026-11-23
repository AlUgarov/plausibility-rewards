"""Judge-format data for Countdown: gold examples (for SFT) and chosen/rejected pairs (for DPO).

The judge format is what the plausibility judge scores: a neutral system prompt + the bare puzzle, then the solution
steps one per line and "Answer: <expression>". Gold solutions come from the solver route of traces.py. Rejected ones
are whole routes of their own (not the gold route edited), wrong in exactly one way, at any step:
    swap   one given number is replaced by a random unavailable one (1-99) and the puzzle is solved with it: every
           step's arithmetic is right and the route reaches the target, but over the real numbers it would not; the
           unavailable number is used wherever the solver happens to use it
    slip   a random natural route over the given numbers that misses the target, with one step's result (a random
           step) changed so that the later steps, done correctly from it, reach the target
    miss   a random natural route over the given numbers that misses the target, every step true (an honest wrong
           answer: the GRPO pass arm on the unfiltered sample drifted into these, which the swap/slip judge passed,
           2026-10-17); not in KINDS by default, so the swap/slip pairs are unchanged
    extra  one number too many: a given number used twice, or an unavailable one (1-99) added, and the puzzle solved
           with it: every step true, reaches the target (the SFT policy's main error on 3-number puzzles, which it
           solves in 4-number style, 2026-10-21); not in KINDS by default
The answer is the route's own expression, so solution and answer agree; it fails the checker (an unavailable number /
a value other than the target). The GRPO policy cheated in all three ways (Consistency_GRPO_Countdown, 2026-10-16/17);
the first pairs (2026-10-16) broke only the last step of the gold route, which a judge can learn by position.
Pairs: kinds in turn per puzzle, or quotas per kind (dpo_pairs, shares). Works for 3- and 4-number puzzles.
Plain Python (imports countdown.py and traces.py).
"""
from __future__ import annotations

import random
from fractions import Fraction

from countdown import check_answer
from prompts import NEUTRAL_SYSTEM_PROMPT, countdown_question, judge_messages  # re-exported: older imports
from traces import _ops, _step_text, solution_route

KINDS = ("swap", "slip")
U_RANGE = (1, 99)                 # unavailable numbers for swap (the TinyZero number range)


def judge_text(steps, answer: str) -> str:
    """Solution lines + "Answer: ..." exactly as the judge scores them."""
    return "\n".join(_step_text(*s) for s in steps) + f"\nAnswer: {answer}"


def gold(numbers: list[int], target: int, rng: random.Random):
    """(steps, answer) of a correct route, or None."""
    route = solution_route(numbers, target, rng)
    if route is None:
        return None
    steps, expr = route
    assert check_answer(expr, numbers, target), (expr, numbers, target)
    return steps, expr


def _strip(e: str) -> str:
    return e[1:-1] if e.startswith("(") and e.endswith(")") else e


def swap_solution(numbers: list[int], target: int, rng: random.Random, tries: int = 40):
    """(steps, answer, step index where the unavailable number is first used, the number), or None."""
    for _ in range(tries):
        k, u = rng.randrange(len(numbers)), rng.randint(*U_RANGE)
        if u in numbers:
            continue
        route = solution_route(numbers[:k] + [u] + numbers[k + 1:], target, rng)
        if route is None:
            continue
        steps, expr = route
        pos = next(i for i, (_, x, y, _r) in enumerate(steps) if Fraction(u) in (x, y))
        return steps, expr, pos, u
    return None


def _natural(op: str, a: Fraction, b: Fraction):
    """(left, right, result) of a natural move on a and b (+, *, larger - smaller, whole division), None otherwise."""
    hi, lo = (a, b) if a >= b else (b, a)
    if op == "+":
        return a, b, a + b
    if op == "*":
        return a, b, a * b
    if op == "-":
        return hi, lo, hi - lo
    return (hi, lo, hi / lo) if lo != 0 and (hi / lo).denominator == 1 else None


def _replay(numbers, plan, k=None, claim=None):
    """Steps (op, left, right, result) and expressions of a plan [(op, ref, ref)] (ref: ("n", i) = given number i,
    ("s", j) = result of step j); step k's result is replaced by claim. None if a move isn't natural."""
    vals, exprs = {}, {}
    get = lambda r: (Fraction(numbers[r[1]]), str(numbers[r[1]])) if r[0] == "n" else (vals[r[1]], exprs[r[1]])
    steps = []
    for j, (op, ra, rb) in enumerate(plan):
        (a, ea), (b, eb) = get(ra), get(rb)
        m = _natural(op, a, b)
        if m is None:
            return None
        x, y, r = m
        ex, ey = (ea, eb) if (x, y) == (a, b) else (eb, ea)
        if j == k:
            r = claim
        vals[j], exprs[j] = r, f"({ex} {op} {ey})"
        steps.append((op, x, y, r))
    return steps, _strip(exprs[len(plan) - 1])


def _random_route(numbers: list[int], rng: random.Random):
    """A random plan of natural moves over the given numbers (see _replay)."""
    refs, plan = [("n", i) for i in range(len(numbers))], []
    while len(refs) > 1:
        i, j = rng.sample(range(len(refs)), 2)
        rest = [r for q, r in enumerate(refs) if q not in (i, j)]
        plan = next(c for op in rng.sample(["+", "*", "-", "/"], 4)    # "/" only when whole; "+" always works
                    if _replay(numbers, c := plan + [(op, refs[i], refs[j])]) is not None)
        refs = rest + [("s", len(plan) - 1)]
    return plan


def miss_solution(numbers: list[int], target: int, rng: random.Random, draws: int = 8):
    """(steps, answer) of a natural route whose every step is true and which ends off the target: of `draws` random
    routes, the one ending nearest the target (a random route usually ends far off, e.g. 1233 for 56, which a judge
    can reject by size alone), or None."""
    t = Fraction(target)
    routes = [_replay(numbers, _random_route(numbers, rng)) for _ in range(draws)]
    routes = [r for r in routes if r[0][-1][3] != t]
    return min(routes, key=lambda r: abs(r[0][-1][3] - t)) if routes else None


def slip_solution(numbers: list[int], target: int, rng: random.Random, tries: int = 60, max_claim: int = 999):
    """(steps, answer, index of the false step), or None. The step is drawn first and kept (an early false result
    often can't be carried to the target, so drawing it per try would favor the last step); any step after `tries`."""
    t = Fraction(target)
    k0 = rng.randrange(len(numbers) - 1)
    for n in range(2 * tries):
        plan = _random_route(numbers, rng)
        true_steps, true_expr = _replay(numbers, plan)
        if true_steps[-1][3] == t:
            continue                      # this route solves the puzzle
        k = k0 if n < tries else rng.randrange(len(plan))
        claims = [c for c in range(1, max_claim + 1) if Fraction(c) != true_steps[k][3]
                  and (got := _replay(numbers, plan, k, Fraction(c))) is not None and got[0][-1][3] == t]
        if not claims:
            continue
        steps, _ = _replay(numbers, plan, k, Fraction(rng.choice(claims)))
        return steps, true_expr, k
    return None


def extra_solution(numbers: list[int], target: int, rng: random.Random, tries: int = 40):
    """(steps, answer) of a route over the given numbers plus one more (half the time a given number again, else an
    unavailable one), every step true, reaching the target; or None."""
    for _ in range(tries):
        u = rng.choice(numbers) if rng.random() < 0.5 else rng.randint(*U_RANGE)
        route = solution_route(numbers + [u], target, rng)
        if route is not None:
            return route
    return None


def wrong_solution(numbers: list[int], target: int, kind: str, rng: random.Random):
    """(steps, answer) of a wrong solution of this kind, or None if none was found."""
    got = {"swap": swap_solution, "slip": slip_solution, "miss": miss_solution,
           "extra": extra_solution}[kind](numbers, target, rng)
    if got is None or check_answer(got[1], numbers, target):
        return None
    return got[0], got[1]


def judge_examples(puzzles, rng: random.Random) -> list[dict]:
    """Gold judge-format SFT examples: {"prompt": judge messages, "completion": [assistant: solution + answer]}."""
    out = []
    for p in puzzles:
        g = gold(p["numbers"], p["target"], rng)
        if g:
            out.append({"prompt": judge_messages(p["numbers"], p["target"]),
                        "completion": [{"role": "assistant", "content": judge_text(*g)}]})
    return out


def _pair(p, g, bad, kind) -> dict:
    return {"prompt": judge_messages(p["numbers"], p["target"]),
            "chosen": [{"role": "assistant", "content": judge_text(*g)}],
            "rejected": [{"role": "assistant", "content": judge_text(*bad)}], "kind": kind,
            "n_numbers": len(p["numbers"])}


def dpo_pairs(puzzles, rng: random.Random, kinds=KINDS, n: int | None = None, shares: dict | None = None) -> list[dict]:
    """One pair per puzzle: {"prompt", "chosen", "rejected", "kind", "n_numbers"} in TRL's conversational format.
    Without shares: kinds in turn (a puzzle with no solution of its kind takes the next). With shares ({kind: share})
    and n: quotas round(n * share); each puzzle tries the kind furthest below its quota first (a 3-number puzzle often
    has no swap: the next puzzle tries swap again), until every quota is filled or the puzzles run out."""
    out = []
    if shares:
        quota = {k: round(n * s) for k, s in shares.items()}
        got = dict.fromkeys(quota, 0)
        for p in puzzles:
            if all(got[k] >= quota[k] for k in quota):
                break
            g = gold(p["numbers"], p["target"], rng)
            if not g:
                continue
            for kind in sorted((k for k in quota if got[k] < quota[k]), key=lambda k: got[k] / quota[k]):
                bad = wrong_solution(p["numbers"], p["target"], kind, rng)
                if bad:
                    out.append(_pair(p, g, bad, kind))
                    got[kind] += 1
                    break
        return out
    for i, p in enumerate(puzzles):
        g = gold(p["numbers"], p["target"], rng)
        if not g:
            continue
        for k in range(len(kinds)):       # this puzzle's kind, or the next one that applies
            kind = kinds[(i + k) % len(kinds)]
            bad = wrong_solution(p["numbers"], p["target"], kind, rng)
            if bad:
                out.append(_pair(p, g, bad, kind))
                break
    return out
