"""../../tables/tab_gaming_countdown.tex: what the reported judge (plausibility) runs produce at steps 0 / 50 / 100 / 150,
by kind of completion, on the test puzzles (200 four- + 200 three-number, the same puzzles for both models) x 8 samples at
T 0.9, cap 1,024. Each completion falls in exactly one kind; the solution block is parsed step by step with the classifier
of analysis/rft_wrong_picks.py.
Sources (Drive My Drive/Consistency/countdown/):
  Llama-3.2-3B: grpo/<Llama mixed SFT>-judge_dpo-start_sft/test_pass8_before_n200+200_max1024/ (step 0) and
    runs/grpo-sum-missmed-mix3_0.5-expr-kl0.02-corrupt_empty-cut-34-fmtw1-istt2-il-const/eval_test_pass8_step{50,100,150}/
    (GRPO_Countdown_Llama [test pass@k]): eval_test{4,3}_pass8_completions.csv; puzzles from that run's
    eval_test_after_samples.csv / eval_test3_after_samples.csv.
  Qwen3-0.6B: sft/<Qwen SFT>/grade_n200x8_t0.9_cap1024_seed0/gen_sft.csv (step 0),
    grpo/qwen3-0.6b-sft-judge-w1-median/grade/gen_qwen3-0.6b-sft-judge-w1-median-step{50,100,150}.csv.
A missing checkpoint is left blank in the table. Writes kinds.csv (per model x step x size x kind) and the table.
Usage: python3 make_tab_gaming.py"""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "../../../analysis/rft_wrong_picks.py")).read().split("R=")[0])  # classify()
CD = os.path.expanduser("~/Library/CloudStorage/GoogleDrive-alugarov@gmail.com/My Drive/Consistency/countdown")
L_DIR = f"{CD}/grpo/Llama-3.2-3B-Instruct-countdown3000-ep1-lr0.0001-seed0-tr2-judge3000-mix3x3000-judge_dpo-start_sft"
L_RUN = f"{L_DIR}/runs/grpo-sum-missmed-mix3_0.5-expr-kl0.02-corrupt_empty-cut-34-fmtw1-istt2-il-const"
Q_SFT = f"{CD}/sft/Qwen3-0.6B-countdown500-ep1-lr0.0001-seed0-tr4-f16-8-w35-30-25-10-imp0x0-mix3x500"
Q_RUN = f"{CD}/grpo/qwen3-0.6b-sft-judge-w1-median"
STEPS = (0, 50, 100, 150)
OUT_TEX = os.path.join(HERE, "../../tables/tab_gaming_countdown.tex")

KINDS = [  # key, table label; the two fakes first, then the other wrong kinds
    ("correct", "Correct"),
    ("fake", "False last step equality"),
    ("false_earlier", "False earlier step equality"),
    ("miss", "Wrong result claimed as the target"),
    ("bad_numbers", "Numbers reused, invented or unused"),
    ("other_wrong", "No solution block or unparsed"),
    ("unfinished", "Unfinished (cap reached)"),
]
GAP_AFTER = {"correct", "miss"}  # \addlinespace after these rows


def kind(text, nums, target, correct, truncated):
    if correct:
        return "correct"
    if truncated:
        return "unfinished"
    k, _ = classify(text, nums, Fraction(int(target)))
    return {"fake: only last step false": "fake", "false earlier step": "false_earlier",
            "doesn't reach target": "miss", "wrong numbers (reused/invented)": "bad_numbers",
            "numbers left unused": "bad_numbers"}.get(k, "other_wrong")


def llama(step):
    d = (f"{L_DIR}/test_pass8_before_n200+200_max1024" if step == 0 else f"{L_RUN}/eval_test_pass8_step{step}")
    out = []
    for size, s, pz in ((4, "test4", "eval_test_after_samples.csv"), (3, "test3", "eval_test3_after_samples.csv")):
        f = f"{d}/eval_{s}_pass8_completions.csv"
        if not os.path.exists(f):
            return None
        c = pd.read_csv(f, keep_default_na=False, dtype={"completion": str})
        p = pd.read_csv(f"{L_RUN}/{pz}")[["pid", "numbers", "target"]]
        c = c.merge(p, on="pid", validate="many_to_one")
        out.append(pd.DataFrame({"n_numbers": size, "k": [
            kind(t.completion, json.loads(t.numbers), t.target, str(t.correct) in ("1", "True", "1.0"),
                 str(t.truncated) in ("1", "True", "1.0")) for t in c.itertuples()]}))
    return pd.concat(out)


def qwen(step):
    f = (f"{Q_SFT}/grade_n200x8_t0.9_cap1024_seed0/gen_sft.csv" if step == 0
         else f"{Q_RUN}/grade/gen_qwen3-0.6b-sft-judge-w1-median-step{step}.csv")
    if not os.path.exists(f):
        return None
    g = pd.read_csv(f, keep_default_na=False, dtype={"completion": str})
    return pd.DataFrame({"n_numbers": g.n_numbers.astype(int), "k": [
        kind(t.completion, json.loads(t.numbers), t.target, str(t.correct) in ("1", "True", "1.0"),
             str(t.truncated) in ("1", "True", "1.0")) for t in g.itertuples()]})


rows = []
for model, load in (("Llama-3.2-3B", llama), ("Qwen3-0.6B", qwen)):
    for step in STEPS:
        d = load(step)
        if d is None:
            print(f"{model} step {step}: no completions, left blank")
            continue
        assert len(d) == 3200, (model, step, len(d))
        for size, h in list(d.groupby("n_numbers")) + [("both", d)]:
            sh = h.k.value_counts(normalize=True)
            rows += [{"model": model, "step": step, "size": size, "kind": k, "share": sh.get(k, 0.0), "n": len(h)}
                     for k, _ in KINDS]
r = pd.DataFrame(rows)
r.to_csv(os.path.join(HERE, "kinds.csv"), index=False)

both = r[r["size"] == "both"].pivot_table(index="kind", columns=["model", "step"], values="share")
# Half-up rounding as in make_tab_format_vs_corrupt.py (0.5275 -> 0.528; f"{:.3f}" gives 0.527 from the float).
fmt = lambda v: "" if pd.isna(v) else str(Decimal(repr(round(float(v), 9))).quantize(Decimal("0.001"), ROUND_HALF_UP))
models = ["Llama-3.2-3B", "Qwen3-0.6B"]
lines = [
    "% Generated by data/grpo_gaming_150/make_tab_gaming.py (do not edit by hand: rerun it) -> kinds.csv.",
    "% Reported judge runs: Llama grpo-sum-missmed-mix3_0.5-expr-kl0.02-corrupt_empty-cut-34-fmtw1-istt2-il-const,",
    "% Qwen qwen3-0.6b-sft-judge-w1-median. Step 0 = the SFT start.",
    "\\begin{table}[H]", "\\centering", "\\captionsetup{singlelinecheck=false}",
    "\\caption{Kinds of completions of the judge runs over training.}", "\\label{tab:gaming}",
    "\\begin{threeparttable}", "\\small",
    "\\begin{tabular}{l" + "r" * (2 * len(STEPS)) + "}", "\\toprule",
    " & " + " & ".join(f"\\multicolumn{{{len(STEPS)}}}{{c}}{{{m}}}" for m in models) + " \\\\",
    f"\\cmidrule(lr){{2-{1 + len(STEPS)}}} \\cmidrule(lr){{{2 + len(STEPS)}-{1 + 2 * len(STEPS)}}}",
    "Step & " + " & ".join(str(s) for _ in models for s in STEPS) + " \\\\", "\\midrule",
]
for k, label in KINDS:
    vals = [both.loc[k].get((m, s)) if (m, s) in both.columns else None for m in models for s in STEPS]
    lines.append(f"{label} & " + " & ".join(fmt(v) for v in vals) + " \\\\" + (" \\addlinespace" if k in GAP_AFTER else ""))
lines += ["\\bottomrule", "\\end{tabular}", "\\begin{tablenotes}\\footnotesize",
          "\\item Share of completions, 400 test puzzles (200 three- and 200 four-number) $\\times$ 8 samples at "
          "temperature 0.9; each completion counted in one row. Step 0 = the SFT start. False equality rows: the solution's final line "
          "equals the target but that step (or an earlier one) is false. Wrong result: all numbers used once, the final "
          "line is not the target; in 84--94\\% of Qwen's the reasoning states that the target is reached.",
          "\\end{tablenotes}", "\\end{threeparttable}", "\\end{table}"]
open(OUT_TEX, "w").write("\n".join(lines) + "\n")
pd.set_option("display.width", 200)
print(r[r["size"] != "both"].pivot_table(index=["kind"], columns=["model", "size", "step"], values="share").round(3).to_string())
print(both.round(3).to_string())
print("wrote", OUT_TEX)
