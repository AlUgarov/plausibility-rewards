# Consistency: self-judged plausibility as a GRPO reward


## Goal

This experiment tests whether it is possible to improve LLM reasoning ability from internal feedback with no gold labels. The reward is the log probability of the solution block in response as scored by a judge model. It uses label-free training for the judge. 
The question is whether this reward improves reasoning quality or accuracy beyond what format rewards alone do,
and how it gets gamed.

The work lives in Colab notebooks contained in this repo.

## Paper

Read the paper in Note.pdf to understand the method and the results. The work is ongoing and should not be cited.

## Notebooks

All the training code for the experiment is in Jupyter notebooks designed originally for Google Colab. The notebooks contain:

- `SFT_Countdown_Llama.ipynb`: SFT of Llama-3.2-3B on Countdown reasoning traces.
- `Corrupt_Judge_Generate_Llama.ipynb`: the SFT model's normal and corrupted-reasoning completions (judge data).
- `Corrupt_Judge_Llama.ipynb`: DPO judge on those pairs, plus its offline evaluation.
- `GRPO_Countdown_Llama.ipynb`: GRPO from the SFT; `ARM` = format_only / sum / mean / correctness.
- `SFT_Countdown_Qwen.ipynb`: SFT of Qwen3-0.6B.
- `Judge_Countdown_Qwen.ipynb`: Qwen corrupted-reasoning data and DPO judge in one notebook.
- `GRPO_Countdown_Qwen.ipynb`: GRPO from the Qwen SFT; arms format and judge (sum).
- `RFT_Countdown_Llama.ipynb`: rejection-sampling fine-tuning (RFT) from the Llama SFT, the judge as the filter.
- `RFT_Countdown_Qwen.ipynb`: RFT from the Qwen SFT, the Qwen judge as the filter.
- `Context_Score_Countdown_Llama.ipynb` (analysis): scores dumped Llama GRPO completions with the judges in several
  contexts.
- `Context_Score_Countdown_Qwen.ipynb` (analysis): whole-completion scores of the Qwen SFT policy's grade completions.

Run order:
- Llama: `SFT_Countdown_Llama` → `Corrupt_Judge_Generate_Llama` → `Corrupt_Judge_Llama` → `GRPO_Countdown_Llama` → `RFT_Countdown_Llama`
  (RFT reuses the GRPO notebook's calibration and start evals).
- Qwen: `SFT_Countdown_Qwen` → `Judge_Countdown_Qwen` → `GRPO_Countdown_Qwen` / `RFT_Countdown_Qwen`.


