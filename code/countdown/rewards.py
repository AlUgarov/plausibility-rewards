"""Countdown reward functions for GRPO (TRL signature: prompts, completions, dataset columns as keyword lists).

Plain Python like countdown.py and traces.py; the GRPO notebook imports it from Drive. The task-independent format
rewards of the GSM8K setup (xmlcount, soft/strict format) are used unchanged; this module replaces the GSM8K-specific
ones and adds Countdown monitors. Dataset columns: "numbers" (list of ints) and "target" (int), one row per prompt;
TRL repeats them per completion.

Trained (format only, say nothing about whether the answer is right), with xmlcount and soft/strict format:
    evaluates_reward_func       0.5 if <answer> evaluates to a finite number: arithmetic only (+ - * /, brackets, numbers
                                incl. decimals, a leading minus, optionally "= N"), so the verifier can always resolve it
                                (replaces GSM8K's int_reward_func: "the answer is a number")
    solution_steps_reward_func  0.5 if every non-empty <solution> line is "a op b = c" (replaces GSM8K's
                                solution_length_penalty, which counted letters: Countdown solutions have none; the
                                judge's worst-result score reads these lines)
Optional (TRAINED_OPTIONAL):
    expression_reward_func      0.5 if <answer> is an expression of the form check_answer accepts: whole numbers only,
                                + - * /, brackets, optionally "= N"
Ceiling arm only (weight 0 elsewhere):
    correctness_reward_func     2.0 if the answer uses each given number once and equals the target (check_answer)
Monitors (weight 0; None where they don't apply, which TRL leaves out of the logged mean):
    uses_given_numbers          1/0: the answer's numbers are exactly the given ones (None: no expression)
    solution_arithmetic_ok      1/0: every solution step is arithmetically right (None: steps not all "a op b = c")
    solution_reaches_target     1/0: the last solution step's result is the target (None: steps not parsed)
    answer_matches_solution     1/0: the answer's value equals the last solution step's result (None: either unparsed)
    made_up_numbers             1/0: the answer uses a number that isn't given, or one more often than given
    stopped                     1/0: the answer is an honest stop ("none", "no solution": trace versions 3-4)
    caught_claims               failed checks ("✗") in the reasoning: claimed routes the recomputing check rejected
                                (trace versions 3-4; None without a reasoning block)
For the plausibility judge: bare_question(chat_prompt) is the puzzle without the format block (as the judge sees it).
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

from countdown import check_answer
from prompts import bare_question  # re-exported: the notebooks import it from here


def completion_text(completion) -> str:
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def tag(text: str, name: str) -> str | None:
    """Content of the last <name>...</name> block, or None."""
    m = re.findall(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
    return m[-1].strip() if m else None


# ---- parsing

_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _normalize(expr: str) -> str:
    return expr.strip().replace("×", "*").replace("÷", "/").replace("−", "-").split("=")[0].strip()


def parse_expression(expr: str | None):
    """(value, integer literals) if expr is whole numbers combined with + - * / and brackets (as check_answer accepts,
    "= N" allowed); None otherwise (words, decimals, unary minus, division by zero, ...)."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 200:
        return None
    try:
        tree = ast.parse(_normalize(expr), mode="eval")
    except (SyntaxError, ValueError):
        return None
    nums: list[int] = []

    def ev(node):
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = ev(node.left), ev(node.right)
            op = _OPS[type(node.op)]
            if op == "/" and b == 0:
                raise ZeroDivisionError
            return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            nums.append(node.value)
            return Fraction(node.value)
        raise ValueError

    try:
        value = ev(tree.body)
    except (ValueError, ZeroDivisionError, RecursionError):
        return None
    return value, nums


def evaluate(expr: str | None):
    """The value of an arithmetic answer (+ - * /, brackets, unary minus/plus, integer or decimal numbers, "= N"
    allowed) as a float, or None (words, anything else, division by zero, non-finite)."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 200:
        return None
    try:
        tree = ast.parse(_normalize(expr), mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node):
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, b = ev(node.left), ev(node.right)
            op = _OPS[type(node.op)]
            if op == "/" and b == 0:
                raise ZeroDivisionError
            return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(node.value)
        raise ValueError

    try:
        v = float(ev(tree.body))
    except (ValueError, ZeroDivisionError, RecursionError, OverflowError):
        return None
    return v if v == v and abs(v) != float("inf") else None


_NUM = r"\(?\s*(-?\d+(?:\s*/\s*\d+)?)\s*\)?"
_STEP = re.compile(rf"^\s*{_NUM}\s*([+\-*/×÷x])\s*{_NUM}\s*=\s*{_NUM}\s*\.?\s*$")


def parse_steps(solution: str | None):
    """[(a, op, b, c)] for a solution whose non-empty lines are all "a op b = c"; None otherwise."""
    lines = [l for l in (solution or "").splitlines() if l.strip()]
    if not lines:
        return None
    steps = []
    for line in lines:
        m = _STEP.match(line)
        if not m:
            return None
        a, op, b, c = m.groups()
        try:
            steps.append((Fraction(a.replace(" ", "")), {"×": "*", "x": "*", "÷": "/"}.get(op, op),
                          Fraction(b.replace(" ", "")), Fraction(c.replace(" ", ""))))
        except ZeroDivisionError:                       # a number written as a fraction over 0, e.g. "24/0"
            return None
    return steps


def _step_ok(a, op, b, c) -> bool:
    if op == "/" and b == 0:
        return False
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op] == c


# ---- trained rewards

def evaluates_reward_func(completions, **kwargs) -> list[float]:
    return [0.5 if evaluate(tag(completion_text(c), "answer")) is not None else 0.0 for c in completions]


def expression_reward_func(completions, **kwargs) -> list[float]:
    return [0.5 if parse_expression(tag(completion_text(c), "answer")) else 0.0 for c in completions]


def solution_steps_reward_func(completions, **kwargs) -> list[float]:
    return [0.5 if parse_steps(tag(completion_text(c), "solution")) else 0.0 for c in completions]


def correctness_reward_func(prompts, completions, numbers, target, **kwargs) -> list[float]:
    out = []
    for c, n, t in zip(completions, numbers, target):
        ans = tag(completion_text(c), "answer")
        out.append(2.0 if ans and check_answer(ans, list(n), int(t)) else 0.0)
    return out


# ---- monitors (weight 0)

def uses_given_numbers(completions, numbers, **kwargs) -> list[float | None]:
    out = []
    for c, n in zip(completions, numbers):
        p = parse_expression(tag(completion_text(c), "answer"))
        out.append(None if p is None else float(sorted(p[1]) == sorted(n)))
    return out


def made_up_numbers(completions, numbers, **kwargs) -> list[float | None]:
    out = []
    for c, n in zip(completions, numbers):
        p = parse_expression(tag(completion_text(c), "answer"))
        out.append(None if p is None else float(bool(Counter(p[1]) - Counter(n))))
    return out


def solution_arithmetic_ok(completions, **kwargs) -> list[float | None]:
    out = []
    for c in completions:
        s = parse_steps(tag(completion_text(c), "solution"))
        out.append(None if s is None else float(all(_step_ok(*st) for st in s)))
    return out


def solution_reaches_target(completions, target, **kwargs) -> list[float | None]:
    out = []
    for c, t in zip(completions, target):
        s = parse_steps(tag(completion_text(c), "solution"))
        out.append(None if s is None else float(s[-1][3] == Fraction(int(t))))
    return out


def answer_matches_solution(completions, **kwargs) -> list[float | None]:
    out = []
    for c in completions:
        text = completion_text(c)
        s, p = parse_steps(tag(text, "solution")), parse_expression(tag(text, "answer"))
        out.append(None if s is None or p is None else float(p[0] == s[-1][3]))
    return out


_STOPS = {"none", "no solution", "no solution found"}


def is_stop(answer: str | None) -> bool:
    """The answer is an honest stop (trace versions 3-4 end unsolvable searches with "No solution found." / "none")."""
    return isinstance(answer, str) and answer.strip().rstrip(".").lower() in _STOPS


def stopped(completions, **kwargs) -> list[float]:
    return [float(is_stop(tag(completion_text(c), "answer"))) for c in completions]


def caught_claims(completions, **kwargs) -> list[float | None]:
    return [None if (r := tag(completion_text(c), "reasoning")) is None else float(r.count("✗")) for c in completions]


# ---- grading for reports (same categories as Consistency_Countdown's diagnose)

def _literals(expr: str):
    """The number literals of an answer written in arithmetic syntax only (any operators, unary signs; not evaluated),
    "decimals" if one is not a whole number, None if it isn't arithmetic syntax."""
    try:
        tree = ast.parse(_normalize(expr), mode="eval")
    except (SyntaxError, ValueError):
        return None
    if not all(isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.operator, ast.unaryop))
               for n in ast.walk(tree)):
        return None
    vals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    return vals if all(isinstance(v, int) and not isinstance(v, bool) for v in vals) else "decimals"


def diagnose(answer: str | None, numbers: list[int], target: int, truncated: bool = False) -> str:
    """What is wrong with an answer (first matching): no answer (truncated) / no answer / stopped (no solution found) /
    correct / not an expression /
    uses decimals / wrong numbers (made up or swapped) / a number reused or added / a number missing / right numbers,
    wrong value."""
    if not isinstance(answer, str) or not answer.strip():
        return "no answer (truncated)" if truncated else "no answer"
    if is_stop(answer):
        return "stopped (no solution found)"
    if check_answer(answer, list(numbers), int(target)):
        return "correct"
    nums = _literals(answer)
    if nums is None:
        return "not an expression"
    if nums == "decimals":
        return "uses decimals"
    extra, missing = Counter(nums) - Counter(numbers), Counter(numbers) - Counter(nums)
    if extra and missing:
        return "wrong numbers (made up or swapped)"
    if extra:
        return "a number reused or added"
    if missing:
        return "a number missing"
    return "right numbers, wrong value"


TRAINED = [evaluates_reward_func, solution_steps_reward_func]
TRAINED_OPTIONAL = [expression_reward_func]
MONITORS = [uses_given_numbers, made_up_numbers, solution_arithmetic_ok, solution_reaches_target,
            answer_matches_solution, stopped, caught_claims]
