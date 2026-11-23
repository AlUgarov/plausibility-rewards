"""../../figures/fig_training_corrupt.pdf (and training_figure.csv): the two GRPO runs of tab_format_vs_corrupt over
training. Left: accuracy on the training batches (correctness is logged, not rewarded, in both arms; the logged
correctness reward is 2 per correct completion, so accuracy = mean / 2; logged every 5 steps, plotted as non-overlapping
10-step means: 80 puzzles x 16 completions, about half of them 3-number puzzles; both runs see the same puzzles in the
same order). Right: mean token entropy of the policy's completions as logged by TRL, same bins. Greedy held-out
accuracy (heldout_curve.csv) is left out: see tab_format_vs_corrupt.
The judge arm runs to step 300: steps 1-150 from corrupt_judge_pass/, steps 151-300 from corrupt_judge_pass_cont150/
(resumed from checkpoint-150 with optimizer state, but a constant learning rate of 2e-5 instead of the cosine schedule,
which had decayed to ~0 by step 150; dotted line). The format-only control was not continued and ends at step 150.
../../figures/fig_training_diagnostics.pdf: training reward (TRL's "reward", the weighted sum the arm trains on, divided
by the arm's first bin), mean completion length (tokens), share of groups whose 16 rewards are all equal (no gradient:
"frac_reward_zero_std"), gradient norm before clipping (clipped at MAX_GRAD_NORM). Needs matplotlib and pandas."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS = [("format_only_noscale", "Format only", "#2a78d6"),
        ("corrupt_judge_pass", "Format + label-free judge", "#c8102e")]
CONTINUED = {"corrupt_judge_pass": "corrupt_judge_pass_cont150"}   # arm -> continuation run (steps after RESTART)
RESTART = 150                    # last step of the original runs; the continuation's learning rate restarts here
LAST = 300
BIN = 10                         # steps per plotted point (non-overlapping means of the 5-step logs)
MAX_GRAD_NORM = 0.2              # the runs' gradient clipping
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLS = {"acc": "rewards/correctness_reward_func/mean", "entropy": "entropy", "reward": "reward",
        "length": "completions/mean_length", "zero_std": "frac_reward_zero_std", "grad_norm": "grad_norm"}

def load(arm):
    """10-step means of the logged metrics (accuracy = correctness reward / 2)."""
    m = pd.read_csv(f"{arm}/metrics.csv").query("step <= @RESTART")
    if arm in CONTINUED:
        m = pd.concat([m, pd.read_csv(f"{CONTINUED[arm]}/metrics.csv").query("step > @RESTART")])
    m = m.dropna(subset=[COLS["acc"]])
    m = m.assign(bin=(m.step - 1) // BIN, **{k: m[c] for k, c in COLS.items()}).assign(acc=lambda d: d.acc / 2)
    return m.groupby("bin").agg(step=("step", "max"), **{k: (k, "mean") for k in COLS}).reset_index()

def style(a):
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Training step")
    a.set_xlim(0, LAST + 5)
    a.set_xticks(range(0, LAST + 1, 50))
    a.axvline(RESTART, color=MUTED, lw=0.8, ls=":")

def legend(fig, bottom):
    fig.legend(handles=[Line2D([], [], color=c, lw=1.6, label=l) for _, l, c in ARMS], frameon=False, fontsize=8,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, bottom, 1, 1))


plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
rows = []
for arm, label, color in ARMS:
    m = load(arm)
    ax.plot(m.step, m.acc, color=color, lw=1.6)
    ax2.plot(m.step, m.entropy, color=color, lw=1.6)
    rows += [{"arm": arm, "kind": "train_batch_accuracy", "step": s, "value": v} for s, v in zip(m.step, m.acc)]
    rows += [{"arm": arm, "kind": "entropy", "step": s, "value": v} for s, v in zip(m.step, m.entropy)]

for a in (ax, ax2):
    style(a)
ax2.set_ylim(bottom=0.025)
ax.set_ylim(0.3, 0.7)
ax.set_title(f"Accuracy on training batches ({BIN}-step mean)", fontsize=9, loc="left", color=INK)
ax2.set_title("Policy entropy (mean per token)", fontsize=9, loc="left", color=INK)
legend(fig, 0.08)
fig.savefig("../../figures/fig_training_corrupt.pdf")
fig.savefig("../../figures/fig_training_corrupt.png", dpi=200)
pd.DataFrame(rows).to_csv("training_figure.csv", index=False)

# Diagnostics: 2 x 2
fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.0))
PANELS = [("reward", "Training reward (first bin = 1)"), ("length", "Completion length (tokens)"),
          ("zero_std", "Groups with no gradient (all rewards equal)"), ("grad_norm", "Gradient norm (before clipping)")]
diag = []
for arm, label, color in ARMS:
    m = load(arm)
    m["reward"] = m.reward / m.reward.iloc[0]
    for a, (k, _) in zip(axs.flat, PANELS):
        a.plot(m.step, m[k], color=color, lw=1.6)
    diag += [{"arm": arm, "kind": k, "step": s, "value": v} for k, _ in PANELS for s, v in zip(m.step, m[k])]
for a, (k, title) in zip(axs.flat, PANELS):
    style(a)
    a.set_title(title, fontsize=9, loc="left", color=INK)
axs[1, 0].set_ylim(0, 1)
axs[1, 1].axhline(MAX_GRAD_NORM, color=MUTED, lw=0.9, ls="--")
axs[1, 1].text(LAST + 2, MAX_GRAD_NORM, "clip", color=MUTED, fontsize=7, va="bottom", ha="right")
axs[1, 1].set_ylim(bottom=0)
legend(fig, 0.05)
fig.savefig("../../figures/fig_training_diagnostics.pdf")
fig.savefig("../../figures/fig_training_diagnostics.png", dpi=200)
pd.DataFrame(diag).to_csv("training_diagnostics.csv", index=False)
print("wrote ../../figures/fig_training_corrupt.pdf, fig_training_diagnostics.pdf, training_figure.csv, training_diagnostics.csv")
