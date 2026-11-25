"""../../figures/fig_training_150.pdf: the reported GRPO runs (tab_format_vs_corrupt) over their first 150 steps.
Columns: Llama-3.2-3B (left), Qwen3-0.6B (right). Rows: accuracy on the training batches (correctness is logged, not
rewarded; Llama logs 2 per correct completion, so accuracy = mean / 2; Qwen logs 0/1), gradient norm before clipping
(clipped at 0.2, never reached in these bins), mean completion length in tokens. Logged every 5 steps, plotted as non-overlapping 10-step means.
Both arms of a model see the same puzzles in the same order (8 per step, 4 three- and 4 four-number, x 16 completions).
Sources (copied 2026-11-24 from Drive My Drive/Consistency/countdown/grpo/):
  llama_format.csv, llama_judge.csv: <Llama mixed SFT>-judge_dpo-start_sft/runs/grpo-{format_only,sum-missmed}-
    mix3_0.5-expr-kl0.02[-corrupt_empty-cut]-34-fmtw1-istt2-il-const/metrics.csv
  qwen_format.csv, qwen_judge.csv: qwen3-0.6b-sft-format-n34-steps150-cap1024-istt2-seed0/metrics.csv,
    qwen3-0.6b-sft-judge-w1-median/metrics.csv (continued past 150; only steps <= 150 are used)
Writes the figure (pdf, png) and fig_training_150.csv (the plotted values). Needs matplotlib and pandas."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "../../figures/fig_training_150")
LAST, BIN, MAX_GRAD_NORM = 150, 10, 0.2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("format", "Format only", "#2a78d6"), ("judge", "Format + label-free judge", "#c8102e")]
MODELS = [("llama", "Llama-3.2-3B", "rewards/correctness_reward_func/mean", 2.0),
          ("qwen", "Qwen3-0.6B", "rewards/correct/mean", 1.0)]
ROWS = [("acc", "Accuracy on training batches", 0), ("grad_norm", "Gradient norm (before clipping)", 0),
        ("length", "Completion length (tokens)", 200)]


def load(model, arm, acc_col, acc_scale):
    m = pd.read_csv(f"{HERE}/{model}_{arm}.csv").dropna(subset=[acc_col]).query("step <= @LAST")
    m = m.drop_duplicates("step", keep="last")
    m = m.assign(bin=(m.step - 1) // BIN, acc=m[acc_col] / acc_scale, length=m["completions/mean_length"])
    return m.groupby("bin").agg(step=("step", "max"), acc=("acc", "mean"), grad_norm=("grad_norm", "mean"),
                                length=("length", "mean")).reset_index()


plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(3, 2, figsize=(6.5, 6.6), sharex=True)
out = []
for j, (model, name, acc_col, scale) in enumerate(MODELS):
    for arm, label, color in ARMS:
        m = load(model, arm, acc_col, scale)
        for i, (k, _, _) in enumerate(ROWS):
            axs[i, j].plot(m.step, m[k], color=color, lw=1.6)
        out += [{"model": model, "arm": arm, "kind": k, "step": s, "value": v}
                for k, _, _ in ROWS for s, v in zip(m.step, m[k])]
    axs[0, j].set_title(name, fontsize=10, color=INK)
for i, (k, title, bottom) in enumerate(ROWS):
    for j in range(2):
        a = axs[i, j]
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_xlim(0, LAST + 3)
        a.set_xticks(range(0, LAST + 1, 25))
        a.set_ylim(bottom=bottom)
        if j == 0:
            a.set_ylabel(title, fontsize=8)
for j in range(2):
    axs[-1, j].set_xlabel("Training step")
# one shared top for each row so the two models read on the same scale
for i in range(len(ROWS)):
    top = max(axs[i, 0].get_ylim()[1], axs[i, 1].get_ylim()[1])
    for j in range(2):
        axs[i, j].set_ylim(top=top)
for j in range(2):  # gradient norm: two-digit ticks so the row's axis title lines up with the others
    axs[1, j].set_yticks([0, 0.05, 0.1, 0.15], ["0", "0.05", "0.1", "0.15"])
    axs[1, j].set_ylim(0, 0.15)
fig.legend(handles=[Line2D([], [], color=c, lw=1.6, label=l) for _, l, c in ARMS], frameon=False, fontsize=8,
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.align_ylabels(axs[:, 0])     # same x for the three row titles whatever the tick label widths
fig.savefig(FIG + ".pdf")
fig.savefig(FIG + ".png", dpi=200)
pd.DataFrame(out).to_csv(f"{HERE}/fig_training_150.csv", index=False)
print("wrote", FIG + ".pdf")
