"""Countdown: reach a target by combining the given numbers with + - * /, each number used exactly once.

Plain Python (no dependencies), so notebooks can import it or paste it in.
    solve(numbers, target)         all distinct solution expressions (exact fractions; fractional intermediates allowed)
    generate(rng, ...)             a puzzle that is solvable by construction, with its number of solutions
    check_answer(expr, numbers, target)   is `expr` a valid solution? (safe parse, no eval)
"""
from __future__ import annotations

import ast
import random
from fractions import Fraction
from functools import lru_cache

OPS = ("+", "-", "*", "/")


def _combine(a: tuple[Fraction, str], b: tuple[Fraction, str]):
    """All values reachable from two (value, expression) items with one operation (both orders for - and /)."""
    (x, ex), (y, ey) = a, b
    yield x + y, f"({ex} + {ey})"
    yield x * y, f"({ex} * {ey})"
    yield x - y, f"({ex} - {ey})"
    yield y - x, f"({ey} - {ex})"
    if y != 0:
        yield x / y, f"({ex} / {ey})"
    if x != 0:
        yield y / x, f"({ey} / {ex})"


def _strip(expr: str) -> str:
    """Drop the outermost parentheses the recursion adds."""
    return expr[1:-1] if expr.startswith("(") and expr.endswith(")") else expr


@lru_cache(maxsize=200_000)
def _reachable(items: tuple[tuple[Fraction, str], ...]) -> tuple[tuple[Fraction, str], ...]:
    """Every (value, expression) obtainable by combining all items (each used exactly once)."""
    if len(items) == 1:
        return items
    out = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            rest = tuple(items[k] for k in range(n) if k not in (i, j))
            for v, e in _combine(items[i], items[j]):
                out.extend(_reachable(tuple(sorted(rest + ((v, e),), key=lambda t: (t[0], t[1])))))
    return tuple(out)


def solve(numbers: list[int], target: int | Fraction) -> list[str]:
    """All distinct solution expressions (as strings). Empty list = unsolvable."""
    items = tuple(sorted(((Fraction(x), str(x)) for x in numbers), key=lambda t: (t[0], t[1])))
    target = Fraction(target)
    return sorted({_strip(e) for v, e in _reachable(items) if v == target})


def n_solutions(numbers: list[int], target: int) -> int:
    return len(solve(numbers, target))


def generate(rng: random.Random, n_numbers: int = 4, lo: int = 1, hi: int = 99, target_lo: int = 10,
             target_hi: int = 100, max_tries: int = 1000, count_solutions: bool = True) -> dict:
    """A random puzzle that is solvable by construction: random numbers, a random expression over all of them, its
    value as the target (kept if it is an integer in [target_lo, target_hi]). Returns numbers, target, one solution
    (the generating expression) and the number of distinct solutions (difficulty: fewer = harder; None without
    count_solutions: slow for 5 numbers)."""
    for _ in range(max_tries):
        nums = [rng.randint(lo, hi) for _ in range(n_numbers)]
        items = [(Fraction(x), str(x)) for x in nums]
        while len(items) > 1:
            i, j = rng.sample(range(len(items)), 2)
            a, b = items[i], items[j]
            items = [t for k, t in enumerate(items) if k not in (i, j)]
            options = list(_combine(a, b))
            items.append(rng.choice(options))
        value, expr = items[0]
        if value.denominator == 1 and target_lo <= value <= target_hi:
            target = int(value)
            return {"numbers": nums, "target": target, "solution": _strip(expr),
                    "n_solutions": n_solutions(nums, target) if count_solutions else None}
    raise RuntimeError("no puzzle found; widen the target range")


_ALLOWED_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _eval_node(node, used: list[int]) -> Fraction:
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _eval_node(node.left, used), _eval_node(node.right, used)
        op = _ALLOWED_BINOPS[type(node.op)]
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        used.append(node.value)
        return Fraction(node.value)
    raise ValueError(f"not allowed: {ast.dump(node)[:60]}")


def check_answer(expr: str, numbers: list[int], target: int) -> bool:
    """True iff `expr` uses exactly the given numbers (each once), only + - * / and parentheses, and equals target.
    Accepts '×', '÷' and '=' followed by the target (e.g. '(25-5)*4+3 = 83'). No eval."""
    if not isinstance(expr, str):
        return False
    expr = expr.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    if "=" in expr:                       # allow "expression = target"
        expr = expr.split("=")[0]
    if not expr or len(expr) > 200:
        return False
    try:
        tree = ast.parse(expr, mode="eval")
        used: list[int] = []
        value = _eval_node(tree.body, used)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False
    return sorted(used) == sorted(numbers) and value == Fraction(target)
