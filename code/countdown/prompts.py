"""Every prompt of the project, in one place: Countdown and math (GSM8K).

Policy: the bare problem + the task's format block. Countdown: one user turn, puzzle + COUNTDOWN_SUFFIX. Math:
math_prompt(question, style), "system" (MATH_SYSTEM_PROMPT as a system turn), "one_shot" (the muffin example first) or
anything else (question + MATH_FORMAT_SUFFIX, the default).
Judge (plausibility reward, judge SFT/DPO data, discrimination tests): NEUTRAL_SYSTEM_PROMPT + the bare problem, no
format block (bare_question strips either task's block); the judge then scores the solution text.

The strings are frozen: the SFT adapters, the DPO judges and every GRPO run were trained on them (test_prompts.py pins
them byte for byte). traces.py, judge_data.py and rewards.py re-export these names, so older imports keep working.
Plain Python.
"""
from __future__ import annotations

# ---- math (GSM8K; verbatim from Consistency_GRPO / Consistency_SFT [common])

MATH_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
{all the intermediate reasoning steps}
</reasoning>
<solution>
{clean and rigorous step-by-step solution}
</solution>
<answer>
{final numeric answer}
</answer>
"""
MATH_FORMAT_SUFFIX = "\n\n" + MATH_SYSTEM_PROMPT.strip()
MATH_ONE_SHOT_QUESTION = ("A baker makes 24 muffins. She sells 3/4 of them in the morning and gives 2 of the rest "
                          "to a neighbor. How many muffins does she have left?")
MATH_ONE_SHOT_ANSWER = """<reasoning>
She sells 3/4 of 24, which is 18, so 24 - 18 = 6 muffins are left after the morning.
She gives 2 of those away, so 6 - 2 = 4 remain.
</reasoning>
<solution>
Muffins sold in the morning: 3/4 * 24 = 18.
Muffins left after the morning: 24 - 18 = 6.
Muffins left after giving 2 to the neighbor: 6 - 2 = 4.
</solution>
<answer>
4
</answer>"""


def math_prompt(question: str, style: str) -> list[dict]:
    """The math policy's prompt (the notebooks' build_prompt, PROMPT_STYLE as an argument)."""
    if style == "system":
        return [{'role': 'system', 'content': MATH_SYSTEM_PROMPT}, {'role': 'user', 'content': question}]
    if style == "one_shot":
        return [{'role': 'user', 'content': MATH_ONE_SHOT_QUESTION + MATH_FORMAT_SUFFIX},
                {'role': 'assistant', 'content': MATH_ONE_SHOT_ANSWER},
                {'role': 'user', 'content': question + MATH_FORMAT_SUFFIX}]
    return [{'role': 'user', 'content': question + MATH_FORMAT_SUFFIX}]


# ---- Countdown

COUNTDOWN_SUFFIX = """

Respond in the following format:
<reasoning>
{your search for a way to reach the target}
</reasoning>
<solution>
{the steps, one per line, e.g. 25 * 4 = 100}
</solution>
<answer>
{one expression that uses each number exactly once and equals the target}
</answer>"""



def countdown_question(numbers: list[int], target: int) -> str:
    """The bare puzzle (what the plausibility judge sees)."""
    return (f"Using the numbers {', '.join(map(str, numbers))}, create an expression that equals {target}. "
            f"You can use +, -, * and /, and each number must be used exactly once.")


def countdown_prompt(numbers: list[int], target: int) -> list[dict]:
    """The policy's prompt: the puzzle + the format block, as one user turn."""
    return [{"role": "user", "content": countdown_question(numbers, target) + COUNTDOWN_SUFFIX}]


# ---- judge (both tasks)

NEUTRAL_SYSTEM_PROMPT = "Please solve the following problem step-by-step, making sure that each step is well-justified."


def strip_format_instructions(text: str) -> str:
    """The bare problem: the text without the task's format block (Countdown's or math's) if it ends with one."""
    for suffix in (COUNTDOWN_SUFFIX, MATH_FORMAT_SUFFIX):
        if text.endswith(suffix):
            return text[:-len(suffix)]
    return text


def bare_question(chat_prompt: list[dict]) -> str:
    """The problem the judge sees, whatever the prompt style: the last user turn without the format block."""
    return strip_format_instructions([m for m in chat_prompt if m.get("role") == "user"][-1]["content"])


def judge_messages_for_question(question: str) -> list[dict]:
    """The judge's prompt: the neutral system prompt + the bare problem (math judge-format SFT examples use this)."""
    return [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}, {"role": "user", "content": question}]


def judge_messages(numbers: list[int], target: int) -> list[dict]:
    """The judge's prompt for a Countdown puzzle."""
    return judge_messages_for_question(countdown_question(numbers, target))


def judge_messages_for(chat_prompt: list[dict]) -> list[dict]:
    """The judge's prompt for a policy prompt of either task (what the GRPO plausibility reward scores against)."""
    return judge_messages_for_question(bare_question(chat_prompt))
