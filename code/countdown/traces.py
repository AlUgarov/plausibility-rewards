"""Program-written SFT traces for Countdown, in the <reasoning>/<solution>/<answer> format.

The reasoning shows a search: 0-3 dead ends (real attempts, every number computed by the program), each followed by
backing up, then the route that works and a check. Two kinds of dead end:
    miss       a complete route whose last step misses the target ("... = 108, off by 10")
    exhausted  two steps, then the natural ways to combine the last two numbers, none of which gives the target
Dead ends use only natural moves (+, *, larger - smaller, whole-number division); the solving route prefers them too and
uses negative or fractional intermediates only when the puzzle needs them.
The solution lists the steps one per line ("a op b = c"); the answer is the expression.
"""
from __future__ import annotations

import random
from fractions import Fraction

from countdown import check_answer, solve
from prompts import COUNTDOWN_SUFFIX, countdown_prompt, countdown_question  # re-exported: older imports


def fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _ops(a: Fraction, b: Fraction, natural: bool = True):
    """(text op, left, right, result) for combining a and b. natural: +, *, larger - smaller, and division only when it
    comes out whole (the moves a person would try). natural=False adds negative differences and fractional divisions."""
    hi, lo = (a, b) if a >= b else (b, a)
    yield "+", a, b, a + b
    yield "*", a, b, a * b
    yield "-", hi, lo, hi - lo
    if lo != 0 and (not natural or (hi / lo).denominator == 1):
        yield "/", hi, lo, hi / lo
    if not natural:
        if hi != 0 and lo != 0 and hi != lo:
            yield "/", lo, hi, lo / hi
        if hi != lo:
            yield "-", lo, hi, lo - hi


def _step_text(op, x, y, r):
    par = lambda v: f"({fmt(v)})" if v.denominator != 1 or v < 0 else fmt(v)
    return f"{par(x)} {op} {par(y)} = {fmt(r)}"


def solution_route(numbers: list[int], target: int, rng: random.Random):
    """A random solving route: list of (op, x, y, result) and the expression, by DFS over steps in random order."""
    target = Fraction(target)

    def dfs(items, natural):  # items: list of (value, expression)
        if len(items) == 1:
            return ([], items[0][1]) if items[0][0] == target else None
        pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
        rng.shuffle(pairs)
        for i, j in pairs:
            (a, ea), (b, eb) = items[i], items[j]
            rest = [t for k, t in enumerate(items) if k not in (i, j)]
            options = list(_ops(a, b, natural))
            rng.shuffle(options)
            for op, x, y, r in options:
                ex, ey = (ea, eb) if x == a and (y == b) else (eb, ea)
                got = dfs(rest + [(r, f"({ex} {op} {ey})")], natural)
                if got is not None:
                    steps, expr = got
                    return [(op, x, y, r)] + steps, expr
        return None

    start = [(Fraction(n), str(n)) for n in numbers]
    found = dfs(start, True) or dfs(start, False)     # whole-number routes first, fractions/negatives only if needed
    if found is None:
        return None
    steps, expr = found
    return steps, expr[1:-1] if expr.startswith("(") and expr.endswith(")") else expr


def _random_step(values: list[Fraction], rng: random.Random, target: Fraction, greedy: bool):
    """Pick two values and an operation; greedy = the result closest to the target (a plausible move)."""
    i, j = rng.sample(range(len(values)), 2)
    options = list(_ops(values[i], values[j], natural=True))
    op = min(options, key=lambda o: abs(o[3] - target)) if greedy else rng.choice(options)
    rest = [v for k, v in enumerate(values) if k not in (i, j)]
    return op, rest + [op[3]]


def dead_end(numbers: list[int], target: int, rng: random.Random, kind: str) -> list[str] | None:
    """Lines describing one failed attempt (None if this attempt happens to solve the puzzle)."""
    target = Fraction(target)
    values = [Fraction(n) for n in numbers]
    lines = []
    op1, values = _random_step(values, rng, target, greedy=False)
    lines.append(f"Try {_step_text(*op1)}. Left: {', '.join(fmt(v) for v in values)}.")
    op2, values = _random_step(values, rng, target, greedy=True)
    lines.append(f"Then {_step_text(*op2)}. Left: {', '.join(fmt(v) for v in values)}.")
    a, b = values
    finals = list(_ops(a, b, natural=True))
    if any(r == target for *_, r in list(_ops(a, b, natural=False))):
        return None                       # this attempt actually works (possibly via a fraction): not a dead end
    if kind == "exhausted":
        shown = ", ".join(_step_text(*o) for o in finals)
        lines.append(f"Check the ways to combine {fmt(a)} and {fmt(b)}: {shown}. None gives {fmt(target)}.")
    else:
        op, x, y, r = min(finals, key=lambda o: abs(o[3] - target))
        lines.append(f"Closest: {_step_text(op, x, y, r)}, off by {fmt(abs(r - target))}.")
    lines.append("That doesn't work. Back up.")
    return lines


def make_trace(numbers: list[int], target: int, rng: random.Random, n_dead_ends: int | None = None,
               version: int = 1, **v3) -> dict:
    """One SFT example. n_dead_ends None: 0 with probability 0.25, else 1-3 uniformly. version 2: make_trace_v2;
    version 3 / 4: make_trace_v3 / _v4 (v3: their keyword arguments)."""
    if version == 2:
        return make_trace_v2(numbers, target, rng, n_dead_ends)
    if version == 3:
        return make_trace_v3(numbers, target, rng, n_dead_ends, **v3)
    if version == 4:
        return make_trace_v4(numbers, target, rng, **v3)
    assert version == 1, version
    if n_dead_ends is None:
        n_dead_ends = 0 if rng.random() < 0.25 else rng.randint(1, 3)
    route = solution_route(numbers, target, rng)
    assert route is not None, (numbers, target)
    steps, expr = route
    lines = [f"Numbers: {', '.join(map(str, numbers))}. Target: {target}."]
    made = 0
    for _ in range(10 * n_dead_ends):
        if made == n_dead_ends:
            break
        d = dead_end(numbers, target, rng, kind=rng.choice(["miss", "exhausted"]))
        if d:
            lines += d
            made += 1
    lines.append("Try " + _step_text(*steps[0]) + "." if made else "Start with " + _step_text(*steps[0]) + ".")
    for s in steps[1:]:
        lines.append(f"Then {_step_text(*s)}.")
    lines.append(f"That reaches {target}.")
    lines.append(f"Check: {expr} uses {', '.join(map(str, sorted(numbers)))} once each and equals {target}.")
    solution = "\n".join(_step_text(*s) for s in steps)
    assert check_answer(expr, numbers, target), (expr, numbers, target)
    text = (f"<reasoning>\n" + "\n".join(lines) + f"\n</reasoning>\n<solution>\n{solution}\n</solution>\n"
            f"<answer>\n{expr}\n</answer>")
    return {"numbers": numbers, "target": target, "n_dead_ends": made, "completion_text": text,
            "solution": solution, "answer": expr}


# ---- version 2 (2026-10-15): fixes for what the first Countdown SFT showed (repeated dead ends until truncation; an
# "off by d" gap added as if d were a number, then a "Check:" that always says yes).
#   * "Left: ..." after every step, the winning route included (it ends at "Left: <target>"), so the available numbers
#     are always written down and a number that isn't there stands out;
#   * false finishes: in a share of traces one dead end ends in an apparent success that the check rejects, either
#     "gap" (the missing amount used as a number) or "slip" (the target claimed with a wrong result);
#   * "Tried: ..." (first steps so far) after every "Back up."; dead ends never repeat a first step, and the winning
#     route starts with a step not tried before.

def _left(values: list[Fraction]) -> str:
    return ", ".join(fmt(v) for v in values)


def dead_end_v2(numbers: list[int], target: int, rng: random.Random, kind: str) -> dict | None:
    """One failed attempt: {"lines", "first" (its first step)}; None if this attempt happens to solve the puzzle.
    Steps until two numbers are left (4 numbers: two steps, 3 numbers: one), then the final combination.
    kind: "miss" (closest result, off by d), "exhausted" (all natural ways to combine the last two numbers),
    "gap" or "slip" (false finishes, caught by the check)."""
    target = Fraction(target)
    values = [Fraction(n) for n in numbers]
    op1, values = _random_step(values, rng, target, greedy=False)
    first = _step_text(*op1)
    lines = [f"Try {first}. Left: {_left(values)}."]
    while len(values) > 2:                # 4 numbers: one more (greedy) step; 3 numbers: none
        op2, values = _random_step(values, rng, target, greedy=True)
        lines.append(f"Then {_step_text(*op2)}. Left: {_left(values)}.")
    a, b = values
    if any(r == target for *_, r in _ops(a, b, natural=False)):
        return None                       # this attempt actually works (possibly via a fraction): not a dead end
    finals = list(_ops(a, b, natural=True))
    op, x, y, r = min(finals, key=lambda o: abs(o[3] - target))
    d = abs(r - target)
    if kind == "exhausted":
        shown = ", ".join(_step_text(*o) for o in finals)
        lines.append(f"Check the ways to combine {fmt(a)} and {fmt(b)}: {shown}. None gives {fmt(target)}.")
    elif kind == "miss":
        lines.append(f"Closest: {_step_text(op, x, y, r)}, off by {fmt(d)}.")
    elif kind == "gap":
        lines.append(f"Then {_step_text(op, x, y, r)}. Left: {fmt(r)}.")
        lines.append(f"Then {fmt(r)} {'+' if r < target else '-'} {fmt(d)} = {fmt(target)}.")
        lines.append(f"Check: {fmt(d)} is not one of the numbers left (only {fmt(r)} is).")
    elif kind == "slip":
        lines.append(f"Then {_step_text(op, x, y, target)}.")
        lines.append(f"Check: {_step_text(op, x, y, r)}, not {fmt(target)}.")
    else:
        raise ValueError(kind)
    lines.append("That doesn't work. Back up.")
    return {"lines": lines, "first": first}


def make_trace_v2(numbers: list[int], target: int, rng: random.Random, n_dead_ends: int | None = None,
                  false_finish_p: float = 0.3) -> dict:
    """One SFT example (version 2). n_dead_ends None: 0 with probability 0.25, else 1-3 uniformly; with probability
    false_finish_p one of the dead ends is a false finish (at least one dead end then)."""
    if n_dead_ends is None:
        n_dead_ends = 0 if rng.random() < 0.25 else rng.randint(1, 3)
    want_false = rng.random() < false_finish_p
    if want_false and n_dead_ends == 0:
        n_dead_ends = 1
    route = solution_route(numbers, target, rng)
    assert route is not None, (numbers, target)
    steps, expr = route
    win_first = _step_text(*steps[0])
    false_at = rng.randrange(n_dead_ends) if want_false else -1
    lines = [f"Numbers: {', '.join(map(str, numbers))}. Target: {target}."]
    tried, made, false_kind = [], 0, None
    for _ in range(20 * n_dead_ends):
        if made == n_dead_ends:
            break
        kind = rng.choice(["gap", "slip"]) if made == false_at else rng.choice(["miss", "exhausted"])
        d = dead_end_v2(numbers, target, rng, kind)
        if d is None or d["first"] in tried or d["first"] == win_first:
            continue
        tried.append(d["first"])
        lines += d["lines"]
        lines[-1] += f" Tried: {'; '.join(tried)}."
        false_kind = kind if kind in ("gap", "slip") else false_kind
        made += 1
    values = [Fraction(n) for n in numbers]
    for i, (op, x, y, r) in enumerate(steps):
        values.remove(x)
        values.remove(y)
        values.append(r)
        word = ("Try" if made else "Start with") if i == 0 else "Then"
        lines.append(f"{word} {_step_text(op, x, y, r)}. Left: {_left(values)}.")
    assert values == [Fraction(target)], (values, target)
    lines.append(f"That reaches {target}.")
    lines.append(f"Check: {expr} uses {', '.join(map(str, sorted(numbers)))} once each and equals {target}.")
    solution = "\n".join(_step_text(*s) for s in steps)
    assert check_answer(expr, numbers, target), (expr, numbers, target)
    text = (f"<reasoning>\n" + "\n".join(lines) + f"\n</reasoning>\n<solution>\n{solution}\n</solution>\n"
            f"<answer>\n{expr}\n</answer>")
    return {"numbers": numbers, "target": target, "n_dead_ends": made, "false_finish": false_kind,
            "completion_text": text, "solution": solution, "answer": expr}


# ---- version 3 (2026-10-24): the check decides instead of following the branch. In v2 the check on the winning route
# only restated "uses ... once each and equals T", and only planned false finishes recomputed (and always backed up),
# so the policy learned to copy a false last step ("83 * 3 = 85. That reaches 85.") into a passing check. Now:
#   * every claimed success (true or false) ends with "That reaches T." and the same check: each step recomputed from
#     the numbers left (a number that isn't there fails it), then every given number used once. The check text is the
#     same procedure for both; only the computation tells them apart;
#   * solvable puzzles: 0, 1 or 2 false finishes (slip or gap) caught by the check, then the true route that passes;
#   * unsolvable puzzles (impossible=True; see unreachable_target): 3-4 false finishes caught, then an honest stop
#     ("No solution found.", answer "none"). Only there, so the traces never give up on a puzzle they can solve.

_APPLY = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}


def check_route(numbers: list[int], steps: list[tuple], target) -> tuple[str, bool]:
    """("Check: ...", passed) for a claimed route [(op, x, y, claimed result)]: each step recomputed in order (its
    operands must be among the numbers left), stopping at the first failure; then every number used once."""
    target = Fraction(target)
    values = [Fraction(n) for n in numbers]
    parts = []
    for op, x, y, c in steps:
        left = list(values)
        for v in (x, y):
            if v not in left:
                parts.append(f"{fmt(v)} is not one of the numbers left ({_left(values)}) ✗")
                return "Check: " + "; ".join(parts) + ".", False
            left.remove(v)
        true = _APPLY[op](x, y)
        if true != c:
            parts.append(f"{_step_text(op, x, y, true)}, not {fmt(c)} ✗")
            return "Check: " + "; ".join(parts) + ".", False
        parts.append(f"{_step_text(op, x, y, true)} ✓")
        values = left + [true]
    ok = values == [target]
    parts.append(f"{', '.join(map(str, sorted(numbers)))} used once each ✓" if ok else f"ends at {_left(values)} ✗")
    return "Check: " + "; ".join(parts) + ".", ok


def false_finish_v3(numbers: list[int], target: int, rng: random.Random, kind: str) -> dict | None:
    """A route that claims the target and the check that rejects it: {"lines", "first"}; None if the attempt's last two
    numbers can actually reach the target. kind: "slip" (the last step's result written as the target) or "gap" (the
    missing amount used as a number)."""
    target = Fraction(target)
    values = [Fraction(n) for n in numbers]
    op1, values = _random_step(values, rng, target, greedy=False)
    steps, first = [op1], _step_text(*op1)
    lines = [f"Try {first}. Left: {_left(values)}."]
    while len(values) > 2:
        op2, values = _random_step(values, rng, target, greedy=True)
        steps.append(op2)
        lines.append(f"Then {_step_text(*op2)}. Left: {_left(values)}.")
    a, b = values
    if any(r == target for *_, r in _ops(a, b, natural=False)):
        return None
    op, x, y, r = min(_ops(a, b, natural=True), key=lambda o: abs(o[3] - target))
    if kind == "slip":
        steps.append((op, x, y, target))
        lines.append(f"Then {_step_text(op, x, y, target)}. Left: {fmt(target)}.")
    elif kind == "gap":
        d, gop = abs(r - target), "+" if r < target else "-"
        steps += [(op, x, y, r), (gop, r, d, target)]
        lines.append(f"Then {_step_text(op, x, y, r)}. Left: {fmt(r)}.")
        lines.append(f"Then {_step_text(gop, r, d, target)}. Left: {fmt(target)}.")
    else:
        raise ValueError(kind)
    check, ok = check_route(numbers, steps, target)
    assert not ok, (numbers, target, steps)
    lines += [f"That reaches {fmt(target)}.", check, "That doesn't work. Back up."]
    return {"lines": lines, "first": first}


def _first_key(step: str) -> tuple:
    """A first step up to the order of + and * operands ("6 * 23" and "23 * 6" are the same move)."""
    a, op, b = step.split(" ")[:3]
    return (op, *sorted((a, b))) if op in "+*" else (op, a, b)


def unreachable_target(numbers: list[int], target: int, rng: random.Random, radius: int = 20,
                       lo: int = 10) -> int | None:
    """A target near the given one (within radius, at least lo) that no expression over the numbers reaches (fractional
    intermediates allowed, as in solve); None if every nearby target is reachable."""
    cands = [t for t in range(max(lo, target - radius), target + radius + 1) if t != target]
    rng.shuffle(cands)
    return next((t for t in cands if not solve(numbers, t)), None)


def make_trace_v3(numbers: list[int], target: int, rng: random.Random, n_dead_ends: int | None = None,
                  n_check_fails: int | None = None, fail_shares: tuple = (0.7, 0.2, 0.1),
                  impossible: bool = False, slip_p: float = 0.75) -> dict | None:
    """One SFT example (version 3). Solvable: n_dead_ends (miss / exhausted) as in v2 (None: 0 w.p. 0.25, else 1-3) and
    n_check_fails false finishes (None: 0 / 1 / 2 with probabilities fail_shares), in random order, then the true route
    and its passing check. impossible=True (the puzzle must have no solution): 3-4 false finishes and 0-1 dead ends,
    then "No solution found." Returns None when an impossible puzzle doesn't allow enough distinct attempts."""
    if impossible:
        assert not solve(numbers, target), (numbers, target)
        n_check_fails = rng.choice([3, 4]) if n_check_fails is None else n_check_fails
        n_dead_ends = rng.randint(0, 1) if n_dead_ends is None else n_dead_ends
        win_first, steps = None, None
    else:
        if n_dead_ends is None:
            n_dead_ends = 0 if rng.random() < 0.25 else rng.randint(1, 3)
        if n_check_fails is None:
            n_check_fails = rng.choices(range(len(fail_shares)), weights=fail_shares)[0]
        route = solution_route(numbers, target, rng)
        assert route is not None, (numbers, target)
        steps, expr = route
        win_first = _first_key(_step_text(*steps[0]))
    plan = ["fail"] * n_check_fails + ["dead"] * n_dead_ends
    rng.shuffle(plan)
    lines = [f"Numbers: {', '.join(map(str, numbers))}. Target: {target}."]
    tried, keys, kinds, i = [], set(), [], 0
    for _ in range(30 * len(plan)):
        if i == len(plan):
            break
        if plan[i] == "fail":
            kind = "slip" if rng.random() < slip_p else "gap"
            d = false_finish_v3(numbers, target, rng, kind)
        else:
            kind = rng.choice(["miss", "exhausted"])
            d = dead_end_v2(numbers, target, rng, kind)
        if d is None or _first_key(d["first"]) in keys or _first_key(d["first"]) == win_first:
            continue
        tried.append(d["first"])
        keys.add(_first_key(d["first"]))
        lines += d["lines"]
        lines[-1] += f" Tried: {'; '.join(tried)}."
        kinds.append(kind)
        i += 1
    n_fails = sum(k in ("slip", "gap") for k in kinds)
    if impossible:
        if n_fails < 3:
            return None
        lines.append(f"None of these reaches {target}, and I can't find another way. No solution found.")
        solution, answer = "No solution found.", "none"
    else:
        values = [Fraction(n) for n in numbers]
        for j, (op, x, y, r) in enumerate(steps):
            values.remove(x)
            values.remove(y)
            values.append(r)
            word = ("Try" if tried else "Start with") if j == 0 else "Then"
            lines.append(f"{word} {_step_text(op, x, y, r)}. Left: {_left(values)}.")
        assert values == [Fraction(target)], (values, target)
        check, ok = check_route(numbers, steps, target)
        assert ok, check
        lines += [f"That reaches {target}.", check]
        solution, answer = "\n".join(_step_text(*s) for s in steps), expr
        assert check_answer(expr, numbers, target), (expr, numbers, target)
    text = (f"<reasoning>\n" + "\n".join(lines) + f"\n</reasoning>\n<solution>\n{solution}\n</solution>\n"
            f"<answer>\n{answer}\n</answer>")
    return {"numbers": numbers, "target": target, "n_dead_ends": len(kinds) - n_fails, "n_check_fails": n_fails,
            "false_finish": ",".join(k for k in kinds if k in ("slip", "gap")) or None, "impossible": impossible,
            "completion_text": text, "solution": solution, "answer": answer}


# ---- version 4 (2026-10-24): version 3's check, with the search length capped and no cue for giving up. Version 3
# (run -tr3-f30-20-imp50x50) taught the model to give up after its third caught claim: unsolvable traces had 3-4 caught
# claims, solvable ones at most 2, so a third one meant "stop" (60% stops on solvable 4-number validation puzzles).
#   * at most 4 attempts: a solvable trace wins on attempt 1-4 (fails_before_win: 0-3 failed attempts first), an
#     unsolvable one stops after 3 or 4 failed attempts. Only "4 attempts failed" always means stop;
#   * the caught claims per trace follow fail_shares (0 / 1 / 2) for solvable and unsolvable alike, the other failed
#     attempts are honest dead ends (miss / exhausted), so the number of caught claims says nothing about solvability;
#   * every attempt after the first opens with "Start again from <the numbers>." (each attempt is a new route from the
#     given numbers, with a first move not tried before).

def make_trace_v4(numbers: list[int], target: int, rng: random.Random, fail_shares: tuple = (0.76, 0.16, 0.08),
                  fails_before_win: tuple = (0.35, 0.3, 0.25, 0.1), impossible: bool = False,
                  impossible_attempts: tuple = (3, 4), slip_p: float = 0.75) -> dict | None:
    """One SFT example (version 4). Caught claims: 0 / 1 / 2 with probabilities fail_shares. Solvable: 0-3 failed
    attempts before the winning one (probabilities fails_before_win; at least the caught claims). impossible=True (no
    solution): rng.choice(impossible_attempts) failed attempts, then "No solution found." None when the puzzle doesn't
    allow enough distinct attempts."""
    n_check_fails = rng.choices(range(len(fail_shares)), weights=fail_shares)[0]
    if impossible:
        assert not solve(numbers, target), (numbers, target)
        n_failed = rng.choice(impossible_attempts)
        win_first = steps = None
    else:
        n_failed = rng.choices(range(len(fails_before_win)), weights=fails_before_win)[0]
        route = solution_route(numbers, target, rng)
        assert route is not None, (numbers, target)
        steps, expr = route
        win_first = _first_key(_step_text(*steps[0]))
    n_check_fails = min(n_check_fails, n_failed) if impossible else n_check_fails
    n_failed = max(n_failed, n_check_fails)
    plan = ["fail"] * n_check_fails + ["dead"] * (n_failed - n_check_fails)
    rng.shuffle(plan)
    restart = f"Start again from {', '.join(map(str, numbers))}."
    lines = [f"Numbers: {', '.join(map(str, numbers))}. Target: {target}."]
    tried, keys, kinds, i = [], set(), [], 0
    for _ in range(30 * len(plan)):
        if i == len(plan):
            break
        if plan[i] == "fail":
            kind = "slip" if rng.random() < slip_p else "gap"
            d = false_finish_v3(numbers, target, rng, kind)
        else:
            kind = rng.choice(["miss", "exhausted"])
            d = dead_end_v2(numbers, target, rng, kind)
        if d is None or _first_key(d["first"]) in keys or _first_key(d["first"]) == win_first:
            continue
        if tried:
            lines.append(restart)
        tried.append(d["first"])
        keys.add(_first_key(d["first"]))
        lines += d["lines"]
        lines[-1] += f" Tried: {'; '.join(tried)}."
        kinds.append(kind)
        i += 1
    if impossible and i < len(plan):
        return None                       # an unsolvable trace needs its 3-4 attempts; a solvable one wins earlier
    n_fails = sum(k in ("slip", "gap") for k in kinds)
    if impossible:
        lines.append(f"None of these reaches {target}, and I can't find another way. No solution found.")
        solution, answer = "No solution found.", "none"
    else:
        if tried:
            lines.append(restart)
        values = [Fraction(n) for n in numbers]
        for j, (op, x, y, r) in enumerate(steps):
            values.remove(x)
            values.remove(y)
            values.append(r)
            word = ("Try" if tried else "Start with") if j == 0 else "Then"
            lines.append(f"{word} {_step_text(op, x, y, r)}. Left: {_left(values)}.")
        assert values == [Fraction(target)], (values, target)
        check, ok = check_route(numbers, steps, target)
        assert ok, check
        lines += [f"That reaches {target}.", check]
        solution, answer = "\n".join(_step_text(*s) for s in steps), expr
        assert check_answer(expr, numbers, target), (expr, numbers, target)
    text = (f"<reasoning>\n" + "\n".join(lines) + f"\n</reasoning>\n<solution>\n{solution}\n</solution>\n"
            f"<answer>\n{answer}\n</answer>")
    return {"numbers": numbers, "target": target, "n_dead_ends": len(kinds) - n_fails, "n_check_fails": n_fails,
            "n_attempts": len(kinds) + (not impossible), "false_finish": ",".join(k for k in kinds if k in ("slip", "gap")) or None,
            "impossible": impossible, "completion_text": text, "solution": solution, "answer": answer}
