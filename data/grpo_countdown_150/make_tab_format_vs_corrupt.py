"""GRPO on Countdown, 150 steps, two tables from per_puzzle.csv:
  ../../tables/tab_format_vs_corrupt.tex   pass@1: start, format only, format + judge, judge run minus start (paired
                                           bootstrap over puzzles, 10,000 resamples, seed 0)
  ../../tables/tab_grpo_pass8_distinct.tex pass@8 and distinct correct answers: start and judge run (appendix)
per size (3-number, 4-number) and both. Both models: 200 three-number + 200 four-number test puzzles x 8 samples, T 0.9,
cap 1,024. per_puzzle.csv (model, arm, n_numbers, puzzle, correct = correct samples of 8, distinct = distinct correct
answers, whitespace removed) is rebuilt from Drive when it is mounted (DRIVE below), else read as is:
  Llama-3.2-3B: countdown/grpo/<Llama mixed SFT>-judge_dpo-start_sft/ test_pass8_before_n200+200_max1024/ (start) and
    runs/<run>/eval_test_pass8_step150/ (GRPO_Countdown_Llama [test pass@k]): eval_test{3,4}_pass8_samples.csv
  Qwen3-0.6B: countdown/sft/<Qwen SFT>/grade_n200x8_t0.9_cap1024_seed0/gen_sft.csv (start) and
    countdown/grpo/<run>/grade/gen_<run>[-step150].csv (GRPO_Countdown_Qwen [compare]): one row per sample."""
import os
import re
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "..", "..", "tables")
DRIVE = os.path.expanduser("~/Library/CloudStorage/GoogleDrive-alugarov@gmail.com/My Drive/Consistency/countdown")
L_DIR = f"{DRIVE}/grpo/Llama-3.2-3B-Instruct-countdown3000-ep1-lr0.0001-seed0-tr2-judge3000-mix3x3000-judge_dpo-start_sft"
LLAMA = {"start": f"{L_DIR}/test_pass8_before_n200+200_max1024",
         "format": f"{L_DIR}/runs/grpo-format_only-mix3_0.5-expr-kl0.02-34-fmtw1-istt2-il-const/eval_test_pass8_step150",
         "judge": f"{L_DIR}/runs/grpo-sum-missmed-mix3_0.5-expr-kl0.02-corrupt_empty-cut-34-fmtw1-istt2-il-const/eval_test_pass8_step150"}
QWEN = {"start": f"{DRIVE}/sft/Qwen3-0.6B-countdown500-ep1-lr0.0001-seed0-tr4-f16-8-w35-30-25-10-imp0x0-mix3x500/"
                 "grade_n200x8_t0.9_cap1024_seed0/gen_sft.csv",
        "format": f"{DRIVE}/grpo/qwen3-0.6b-sft-format-n34-steps150-cap1024-istt2-seed0/grade/"
                  "gen_qwen3-0.6b-sft-format-n34-steps150-cap1024-istt2-seed0.csv",
        "judge": f"{DRIVE}/grpo/qwen3-0.6b-sft-judge-w1-median/grade/gen_qwen3-0.6b-sft-judge-w1-median-step150.csv"}
PER_PUZZLE = os.path.join(HERE, "per_puzzle.csv")
MODELS = ("Llama-3.2-3B", "Qwen3-0.6B")
SETS = ((3, "3-number"), (4, "4-number"), ("both", "Both"))

if os.path.isdir(DRIVE):
    rows = []
    for arm, d in LLAMA.items():
        for n in (3, 4):
            s = pd.read_csv(f"{d}/eval_test{n}_pass8_samples.csv")
            rows.append(pd.DataFrame({"model": MODELS[0], "arm": arm, "n_numbers": n, "puzzle": s.pid,
                                      "correct": s.correct, "distinct": s.distinct_correct, "samples": 8}))
    for arm, f in QWEN.items():
        g = pd.read_csv(f, usecols=["n_numbers", "puzzle", "sample", "answer", "correct"], keep_default_na=False,
                        dtype={"answer": str})
        g["correct"] = g.correct.astype(str) == "True"
        g["key"] = [re.sub(r"\s+", "", a) if c else None for a, c in zip(g.answer, g.correct)]   # as pass_at_k
        a = g.groupby(["n_numbers", "puzzle"]).agg(correct=("correct", "sum"), samples=("correct", "size"),
                                                   distinct=("key", "nunique")).reset_index()
        rows.append(a.assign(model=MODELS[1], arm=arm))
    pp = pd.concat(rows, ignore_index=True)[["model", "arm", "n_numbers", "puzzle", "correct", "distinct", "samples"]]
    pp.to_csv(PER_PUZZLE, index=False)
pp = pd.read_csv(PER_PUZZLE)
assert (pp.samples == 8).all() and (pp.groupby(["model", "arm", "n_numbers"]).size() == 200).all()
assert (pp.distinct <= pp.correct).all()
pp["pass1"], pp["pass8"] = pp.correct / 8, (pp.correct > 0).astype(float)
pp["key"] = pp.n_numbers.astype(str) + "-" + pp.puzzle.astype(str)

res = []
for model in MODELS:
    for n, lab in SETS:
        d = pp[(pp.model == model) & ((pp.n_numbers == n) if n != "both" else True)]
        w = d.pivot(index="key", columns="arm", values=["pass1", "pass8", "distinct"])
        B = np.random.default_rng(0).integers(0, len(w), (10_000, len(w)))
        diff = (w[("pass1", "judge")] - w[("pass1", "start")]).values
        bs = diff[B].mean(1)
        res.append({"model": model, "set": lab, "puzzles": len(w),
                    **{f"{m}_{a}": w[(m, a)].mean() for m in ("pass1", "pass8", "distinct") for a in ("start", "format", "judge")},
                    "pass1_diff": diff.mean(), "pass1_lo": np.percentile(bs, 2.5), "pass1_hi": np.percentile(bs, 97.5)})
res = pd.DataFrame(res)
res.to_csv(os.path.join(HERE, "summary.csv"), index=False)
print(res.round(3).to_string(index=False))

def num(x, d):
    """Half-up rounding (0.3675 -> 0.368), LaTeX minus."""
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP)).replace("-", "$-$")

def body(cells, sets=SETS):
    """Rows per model (multirow) and set; cells(r) -> the row's cells after the set."""
    out = []
    for model in MODELS:
        out.append(rf"\multirow{{{len(sets)}}}{{*}}{{{model}}}")
        for k, (_, lab) in enumerate(sets):
            r = res[(res.model == model) & (res.set == lab)].iloc[0]
            if lab == "Both":
                out.append(r"\cmidrule(l){2-%d}" % (2 + len(cells(r))))
            out.append(f" & {lab} & " + " & ".join(cells(r)) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    return "\n".join(out)

HEAD = r"""% Generated by data/grpo_countdown_150/make_tab_format_vs_corrupt.py (do not edit by hand: rerun it) -> per_puzzle.csv,
% summary.csv. 150 steps, constant LR 2e-5 after 20 warmup steps, KL 0.02, T 0.9; 200 + 200 test puzzles x 8, cap 1,024.
% Llama-3.2-3B runs (countdown/grpo/<Llama mixed SFT>-judge_dpo-start_sft/runs/): format only
%   grpo-format_only-mix3_0.5-expr-kl0.02-34-fmtw1-istt2-il-const; judge
%   grpo-sum-missmed-mix3_0.5-expr-kl0.02-corrupt_empty-cut-34-fmtw1-istt2-il-const ([test pass@k], checkpoint-150).
% Qwen3-0.6B runs: qwen3-0.6b-sft-format-n34-steps150-cap1024-istt2-seed0, qwen3-0.6b-sft-judge-w1-median (step 150).
"""
main = HEAD + r"""% Previous versions: tables/archive/tab_format_vs_corrupt_*.tex.
\begin{table}[H]
\centering
\caption{GRPO on Countdown, 150 steps: pass@1 of the format-only control and of format + the label-free corrupted-reasoning judge.}
\label{tab:format-vs-corrupt}
\begin{threeparttable}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
Model & Puzzles & Start & \makecell{Format\\ only} & \makecell{Format +\\ label-free judge} & \makecell{Judge run\\ $-$ start} \\
\midrule
""" + body(lambda r: [num(r.pass1_start, 3), num(r.pass1_format, 3), num(r.pass1_judge, 3),
                      f"{num(r.pass1_diff, 3)} {{\\footnotesize [{num(r.pass1_lo, 3)}, {num(r.pass1_hi, 3)}]}}"]) + r"""
\end{tabular}
\begin{tablenotes}\footnotesize
\item pass@1: share of correct samples, 200 test puzzles per size $\times$ 8 samples at temperature 0.9; start = the SFT model. Judge run $-$ start: paired over puzzles, 95\% confidence interval in brackets.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
appendix = HEAD + r"""\begin{table}[H]
\centering
\caption{GRPO on Countdown, 150 steps: pass@8 and distinct correct answers of the start and the judge run.}
\label{tab:grpo-pass8-distinct}
\begin{threeparttable}
\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{2}{c}{pass@8} & \multicolumn{2}{c}{Distinct correct} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
Model & Puzzles & Start & Judge run & Start & Judge run \\
\midrule
""" + body(lambda r: [num(r.pass8_start, 3), num(r.pass8_judge, 3), num(r.distinct_start, 2), num(r.distinct_judge, 2)],
                      sets=SETS[:2]) + r"""
\end{tabular}
\begin{tablenotes}\footnotesize
\item The runs of Table~\ref{tab:format-vs-corrupt}: 200 test puzzles per size $\times$ 8 samples at temperature 0.9; start = the SFT model, judge run = format + label-free judge. pass@8: share of puzzles with at least one correct sample. Distinct correct: distinct correct answers per puzzle (whitespace removed).
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
open(os.path.join(TABLES, "tab_format_vs_corrupt.tex"), "w").write(main)
open(os.path.join(TABLES, "tab_grpo_pass8_distinct.tex"), "w").write(appendix)
print("wrote tab_format_vs_corrupt.tex, tab_grpo_pass8_distinct.tex")
