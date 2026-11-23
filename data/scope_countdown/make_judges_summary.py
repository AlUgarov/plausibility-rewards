"""judges_summary.csv for tab_judges_countdown: every judge on the same 2,560 GRPO training completions.
Inputs (rows aligned): judge_dump_scores.csv (mean of solution + answer under base / SFT / solver DPO / corrupt DPO
judges; from Consistency_Corrupt_Judge), scores.csv (SFT policy on the whole completion, solver DPO low10; from
Consistency_Judge_Score_Probe), zero_shot_scores.csv (base model low10 and P(Yes)), context_scores.csv (only the base model on the whole completion
under the policy prompt, gen_full_base_mean; from Consistency_Context_Score_Countdown, a later scoring pass whose
gen_full_mean reproduces full_mean's row: 0.693 / 0.160, 2,895 pairs)."""
import pandas as pd
s, d, z = pd.read_csv("scores.csv"), pd.read_csv("judge_dump_scores.csv"), pd.read_csv("zero_shot_scores.csv")
assert (s.step.values == d.step.values).all() and (s.pid.values == d.pid.values).all()
assert (s.step.values == z.step.values).all() and (s.pid.values == z.pid.values).all()
assert (s.correct.values == d.correct.values).all()
cx = pd.read_csv("context_scores.csv")
assert (s.step.values == cx.step.values).all() and (s.pid.values == cx.pid.values).all()
assert (s.correct.values == cx.correct.values).all()
df = pd.concat([s, d[["base", "sft", "solver", "corrupt", "corrupt_all"]], z.filter(like="zs_llama3b"),
                cx[["gen_full_base_mean"]].rename(columns={"gen_full_base_mean": "full_base_mean"})], axis=1)
df["group"] = df.step.astype(str) + "/" + df.pid.astype(str)

def rates(col):
    """Within-group win rate of correct over wrong (ties 1/2) and share of mixed groups fully separated."""
    w = t = sep = mixed = 0
    for _, g in df.dropna(subset=[col]).groupby("group"):
        c, x = g.loc[g.correct, col].values, g.loc[~g.correct, col].values
        if len(c) and len(x):
            diff = c[:, None] - x[None, :]
            w, t = w + (diff > 0).sum() + 0.5 * (diff == 0).sum(), t + diff.size
            mixed, sep = mixed + 1, sep + (c.min() > x.max())
    return w / t, sep / mixed, mixed, t

ROWS = [("base", "whole completion", "mean", "full_base_mean"), ("base", "solution + answer", "mean", "base"), ("base", "solution + answer", "low10", "zs_llama3b_low10"),
        ("base", "solution + answer", "P(Yes)", "zs_llama3b_pyes"),
        ("sft_policy", "whole completion", "mean", "full_mean"), ("sft_policy", "whole completion", "low10", "full_low10"),
        ("sft_policy", "whole completion", "SC", "full_sc"),
        ("sft_judge", "solution + answer", "mean", "sft"),
        ("dpo_solver", "solution + answer", "mean", "solver"), ("dpo_solver", "solution + answer", "low10", "sol_low10"),
        ("dpo_corrupt_empty_cut", "solution + answer", "mean", "corrupt"),
        ("dpo_corrupt_empty_swap_cut", "solution + answer", "mean", "corrupt_all")]
out = pd.DataFrame([dict(judge=j, span=sp, score=sc, column=c, **dict(zip(("win_rate", "groups_separated",
                    "mixed_groups", "pairs"), rates(c)))) for j, sp, sc, c in ROWS])
out.to_csv("judges_summary.csv", index=False)
print(out.round(3).to_string(index=False))
