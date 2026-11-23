"""Intuitor's reward (Zhao et al., "Learning to Reason without External Rewards", ICLR 2026) for the Countdown GRPO
notebook: the current policy's self-certainty on its own completion, given the same prompt as at generation.

    self-certainty = mean over completion tokens of KL(uniform || p) = logsumexp(z) - mean(z) - log|V|

with z the logits at the position predicting that token (no softmax is stored). The paper's form,
-(1/(|o||V|)) sum_i sum_j log(|V| p_ij), is the same quantity. Computed online: the model's active adapter is the policy
being trained. Plain PyTorch; the notebook binds it to the trainer.
"""
from __future__ import annotations

import math

import torch


def self_certainty_scores(model, prompt_ids: list[list[int]], completion_ids: list[list[int]], pad_id: int,
                          chunk_size: int = 2) -> list[float | None]:
    """Self-certainty per (prompt, completion) pair of token ids; None for an empty completion."""
    fwd = getattr(model, "_original_forward", model.forward)   # skip accelerate's autocast + fp32 logits copy
    items = list(zip(prompt_ids, completion_ids))
    out: list[float | None] = []
    with torch.no_grad():
        for start in range(0, len(items), chunk_size):
            chunk = items[start:start + chunk_size]
            length = max(len(p) + len(c) for p, c in chunk)
            ids = torch.full((len(chunk), length), pad_id, dtype=torch.long)
            attn = torch.zeros_like(ids)
            for j, (p, c) in enumerate(chunk):
                ids[j, :len(p) + len(c)] = torch.tensor(list(p) + list(c))
                attn[j, :len(p) + len(c)] = 1
            device = next(model.parameters()).device
            logits = fwd(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
            for j, (p, c) in enumerate(chunk):
                if not c:
                    out.append(None)
                    continue
                z = logits[j, len(p) - 1:len(p) + len(c) - 1].float()   # positions predicting the completion
                out.append(float((torch.logsumexp(z, -1) - z.mean(-1) - math.log(z.shape[-1])).mean()))
            del logits
    return out


def make_self_certainty(owner, chat_prompt_ids, chunk_size: int = 2):
    """A TRL reward function named "self_certainty", bound to the trainer `owner` (its model and tokenizer).
    chat_prompt_ids(tokenizer, messages) -> the templated prompt's token ids, as used for generation. Uses TRL's
    completion_ids: exactly the sampled tokens."""
    def self_certainty(prompts, completions, completion_ids=None, **kwargs):
        tok = owner.processing_class
        return self_certainty_scores(owner.model, [chat_prompt_ids(tok, p) for p in prompts],
                                     [list(c) for c in completion_ids], tok.pad_token_id, chunk_size)
    self_certainty.__name__ = "self_certainty"
    return self_certainty
