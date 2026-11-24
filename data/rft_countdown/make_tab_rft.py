"""per_puzzle.csv and ../../tables/tab_rft_countdown.tex (appendix): rejection-sampling fine-tuning (RFT) with the
label-free judge as the filter, pass@1 and pass@8 of the start and RFT rounds 4 and 8, per size and both. Round 4 uses the
rollouts of 150 GRPO steps (4 rounds x 300 puzzles x 16 = 19,200). Both models: 200 three-number + 200 four-number test
puzzles x 8 samples, T 0.9, cap 1,024 (the puzzles of tab_format_vs_corrupt). Per round: 300 new training puzzles x 16,
per puzzle the top 2 finished completions scored above the group mean, the bottom 20% of the round's picks dropped, SFT
1 epoch at LR 2e-5.
  Llama-3.2-3B: countdown/grpo/<Llama mixed SFT>-judge_dpo-start_sft/runs/<RFT run>/eval_test_pass8_round{4,8}/
    (Consistency_RFT_Countdown34 [test pass@k]); start: test_pass8_before_n200+200_max1024/ (GRPO_Countdown_Llama [test pass@k])
  Qwen3-0.6B: countdown/grpo/qwen3-0.6b-sft-rft-judge-w3-p5-cut20/grade/gen_<run>-round{4,8}.csv; start: the SFT's
    grade_n200x8_t0.9_cap1024_seed0/gen_sft.csv
per_puzzle.csv is rebuilt from Drive when it is mounted (DRIVE below), else read as is."""
import os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "..", "tables", "tab_rft_countdown.tex")
DRIVE = os.path.expanduser("~/Library/CloudStorage/GoogleDrive-alugarov@gmail.com/My Drive/Consistency/countdown")
L_DIR = f"{DRIVE}/grpo/Llama-3.2-3B-Instruct-countdown3000-ep1-lr0.0001-seed0-tr2-judge3000-mix3x3000-judge_dpo-start_sft"
L_RFT = f"{L_DIR}/runs/rft-judge-sum-missmed-corrupt_empty-cut-r8x300k2-skip0.2-lr2e-05-mix3_0.5-rft1-cut20"
LLAMA = {"start": f"{L_DIR}/test_pass8_before_n200+200_max1024",
         "round4": f"{L_RFT}/eval_test_pass8_round4", "round8": f"{L_RFT}/eval_test_pass8_round8"}
Q_RFT = f"{DRIVE}/grpo/qwen3-0.6b-sft-rft-judge-w3-p5-cut20/grade/gen_qwen3-0.6b-sft-rft-judge-w3-p5-cut20"
QWEN = {"start": f"{DRIVE}/sft/Qwen3-0.6B-countdown500-ep1-lr0.0001-seed0-tr4-f16-8-w35-30-25-10-imp0x0-mix3x500/"
                 "grade_n200x8_t0.9_cap1024_seed0/gen_sft.csv",
        "round4": f"{Q_RFT}-round4.csv", "round8": f"{Q_RFT}-round8.csv"}
PER_PUZZLE = os.path.join(HERE, "per_puzzle.csv")
MODELS, ARMS = ("Llama-3.2-3B", "Qwen3-0.6B"), ("start", "round4", "round8")
SETS = ((3, "3-number"), (4, "4-number"), ("both", "Both"))

if os.path.isdir(DRIVE):
    rows = []
    for arm, d in LLAMA.items():
        for n in (3, 4):
            f = f"{d}/eval_test{n}_pass8_samples.csv"
            if os.path.exists(f):                  # the RFT rounds: after the RFT notebook's [test pass@k]
                s = pd.read_csv(f)
                rows.append(pd.DataFrame({"model": MODELS[0], "arm": arm, "n_numbers": n, "puzzle": s.pid,
                                          "correct": s.correct, "samples": 8}))
    for arm, f in QWEN.items():
        g = pd.read_csv(f, usecols=["n_numbers", "puzzle", "correct"])
        g["correct"] = g.correct.astype(str) == "True"
        a = g.groupby(["n_numbers", "puzzle"]).correct.agg(correct="sum", samples="size").reset_index()
        rows.append(a.assign(model=MODELS[1], arm=arm))
    pd.concat(rows, ignore_index=True)[["model", "arm", "n_numbers", "puzzle", "correct", "samples"]].to_csv(PER_PUZZLE, index=False)
pp = pd.read_csv(PER_PUZZLE)
assert (pp.samples == 8).all() and (pp.groupby(["model", "arm", "n_numbers"]).size() == 200).all()
missing = [(m, a) for m in MODELS for a in ARMS if not ((pp.model == m) & (pp.arm == a)).any()]
pp["pass1"], pp["pass8"] = pp.correct / 8, (pp.correct > 0).astype(float)

res = []
for (model, arm), d in pp.groupby(["model", "arm"]):
    for n, lab in SETS:
        x = d if n == "both" else d[d.n_numbers == n]
        res.append({"model": model, "arm": arm, "set": lab, "puzzles": len(x), "pass1": x.pass1.mean(), "pass8": x.pass8.mean()})
res = pd.DataFrame(res)
pp["key"] = pp.n_numbers.astype(str) + "-" + pp.puzzle.astype(str)
diffs = []                                 # round 8 - start: paired bootstrap over puzzles, 10,000 resamples, seed 0
for model in MODELS:
    for n, lab in SETS:
        d = pp[(pp.model == model) & ((pp.n_numbers == n) if n != "both" else True)]
        w = d.pivot(index="key", columns="arm", values=["pass1", "pass8"])
        B = np.random.default_rng(0).integers(0, len(w), (10_000, len(w)))
        for m in ("pass1", "pass8"):
            x = (w[(m, "round8")] - w[(m, "start")]).values
            bs = x[B].mean(1)
            diffs.append({"model": model, "set": lab, "metric": m, "diff": x.mean(),
                          "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
diffs = pd.DataFrame(diffs)
diffs.to_csv(os.path.join(HERE, "diff_round8_start.csv"), index=False)
print(diffs.round(3).to_string(index=False))
res.to_csv(os.path.join(HERE, "summary.csv"), index=False)
print(res.pivot_table(index=["model", "set"], columns="arm", values=["pass1", "pass8"]).round(3).to_string())
if missing:
    raise SystemExit(f"not written: no results for {missing} (run Consistency_RFT_Countdown34 [test pass@k])")

num = lambda x, d: str(Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP))   # half up
sgn = lambda x: num(x, 3).replace("-", "$-$")   # LaTeX minus
def ci(model, lab, m):
    r = diffs[(diffs.model == model) & (diffs.set == lab) & (diffs.metric == m)].iloc[0]
    return f"{sgn(r['diff'])} [{sgn(r.lo)}, {sgn(r.hi)}]"
body = []
for model in MODELS:
    body.append(rf"\multirow{{6}}{{*}}{{{model}}}")
    for k, (m, mlab) in enumerate((("pass1", "pass@1"), ("pass8", "pass@8"))):
        if k:
            body.append(r"\cmidrule(l){2-7}")
        for j, (_, lab) in enumerate(SETS):
            v = lambda a: res[(res.model == model) & (res.arm == a) & (res.set == lab)][m].iloc[0]
            body.append(f" & {mlab if j == 0 else ''} & {lab} & " + " & ".join([num(v(a), 3) for a in ARMS] + [ci(model, lab, m)]) + r" \\")
    body.append(r"\midrule")
body[-1] = r"\bottomrule"
tex = r"""% Generated by data/rft_countdown/make_tab_rft.py (do not edit by hand: rerun it) -> per_puzzle.csv, summary.csv.
% Llama-3.2-3B RFT run: countdown/grpo/<Llama mixed SFT>-judge_dpo-start_sft/runs/
%   rft-judge-sum-missmed-corrupt_empty-cut-r8x300k2-skip0.2-lr2e-05-mix3_0.5-rft1-cut20 (judge sum, missing = median wrong)
% Qwen3-0.6B RFT run: countdown/grpo/qwen3-0.6b-sft-rft-judge-w3-p5-cut20 (judge sum, missing = 5th percentile)
% Both: 300 puzzles x 16 per round, top 2 finished above the group mean, bottom 20% of picks dropped, 1 epoch at 2e-5.
\begin{table}[H]
\centering
\caption{Rejection-sampling fine-tuning on Countdown with the label-free judge as the filter: pass@1 and pass@8.}
\label{tab:rft-countdown}
\begin{threeparttable}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllcccc}
\toprule
Model & Metric & Puzzles & Start & Round 4 & Round 8 & Round 8 $-$ start \\
\midrule
""" + "\n".join(body) + r"""
\end{tabular}
\begin{tablenotes}\footnotesize
\item 200 test puzzles per size $\times$ 8 samples at temperature 0.9 (the puzzles of Table~\ref{tab:format-vs-corrupt}); start = the SFT model. Each round samples 300 new training puzzles $\times$ 16 completions, keeps per puzzle the two highest-scored finished completions above the group mean (dropping the lowest-scored 20\% of the round's picks), and fine-tunes on them for one epoch. Round 4 uses as many rollouts as 150 GRPO steps. Round 8 $-$ start: paired over puzzles, 95\% confidence interval in brackets.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
open(TEX, "w").write(tex)
print("wrote", os.path.normpath(TEX))
