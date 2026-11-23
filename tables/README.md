# Tables for the paper

LaTeX (booktabs, threeparttable, float `[H]`, as in `../Note.tex`); include with `\input{tables/<file>}`.
Each file starts with a comment naming its source (a narrative.md section or the Drive run folders).

| file | content |
|---|---|
| `tab_discrimination_gsm8k.tex` | self-judging on GSM8K by model (Llama 1B/3B/8B, Qwen2.5 0.5B-7B), metrics mean, mean+answer, sum, SC, P(True) |
| `tab_discrimination_countdown.tex` | Countdown, Llama-3.2-3B base judge vs policy as judge, metrics mean, sum, worst, low5, min, length (not in Note.tex: replaced by tab_judges_countdown) |
| `tab_scope_solution_vs_completion.tex` | Countdown: solution + answer vs whole completion, trained and untrained judges (not in Note.tex: replaced by tab_judges_countdown) |
| `tab_judges_countdown.tex` | Countdown: all judges (base, SFT policy, DPO solver-made errors, DPO corrupted reasoning) and scores on the same 2,560 GRPO training completions |
| `tab_context_countdown.tex` | Countdown, SFT policy: solution + answer (judge context) vs whole completion (generation context); win rate and regressions of correctness, alone and with length, SEs |
| `archive/tab_context_gsm8k.tex` | GSM8K 8B: judge context vs generation context, incl. whole-completion mean (regression with length control) (not in Note.tex: replaced by tab_context_countdown) |
| `tab_format_vs_solver.tex` | GRPO, mixed SFT: format-only control vs solver-pair judge, 150 steps |
| `tab_format_vs_corrupt.tex` | GRPO, mixed SFT, no SD scaling: format-only control vs the label-free corrupted-reasoning judge (pass arm), 150 steps |
