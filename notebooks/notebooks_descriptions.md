# Notebooks behind the write-up

Copies of the Colab notebooks (Drive `My Drive/Colab Notebooks/`). Run outputs live on Drive under `My Drive/Consistency/`;
the data each table/figure uses is in `../data/` (see `../data/README.md`).

| notebook | what it produced | copied from |
|---|---|---|
| `SFT_Countdown_Llama.ipynb` SFT to seed reasoning behavior to Llama
| `Corrupt_Judge_Generate_Llama.ipynb` generating reasoning traces from SFT Llama to train the judge 
| `Corrupt_Judge_Llama.ipynb` train Llama-based judge on aforementioned traces
| `SFT_Countdown_Qwen.ipynb` SFT to seed reasoning behavior to Qwen3-0.6B
| `Judge_Countdown_Qwen.ipynb` generating corrupted traces from SFT Qwen and training the Qwen-based judge on them
| `GRPO_Countdown_Llama.ipynb` trains SFT Llama model with GRPO with plausibility and format training to be better on Countdown tasks
| `GRPO_Countdown_Qwen.ipynb` trains Qwen model with GRPO with plausibility and format reward using Countdown tasks
| `RFT_Countdown_Llama.ipynb` rejection-sampling fine-tuning of SFT Llama: rounds of sampling, judge-selected completions, SFT; evaluates the rounds on the test puzzles ([test pass@k])
| `RFT_Countdown_Qwen.ipynb` rejection-sampling fine-tuning of SFT Qwen with the Qwen judge as the filter, graded per round on the test puzzles
| `Context_Score_Countdown_Llama.ipynb` scores dumped Llama GRPO completions with the judges in several contexts (judge vs generation prompt, solution vs whole completion)
| `Context_Score_Countdown_Qwen.ipynb` scores the Qwen SFT policy's grade completions over the whole completion (SFT policy and base model, policy prompt), next to the solution + answer scores from the GRPO calibration
