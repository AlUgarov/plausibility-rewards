"""Pass@k Training advantages (Chen et al. 2025, "Pass@k Training for Adaptively Balancing Exploration and
Exploitation of Large Reasoning Models", arXiv 2508.10751, Eq. 11-15) for the Countdown GRPO notebook.

A group of N samples with binary rewards is scored as if k of them were drawn at random and the draw counted as a
success when any one is correct. Each response gets the mean advantage of the k-subsets that contain it, in closed form:

    R = 1 - C(N_neg, k) / C(N, k)                     mean reward of the k-subsets
    s = sqrt(R (1 - R))                               their SD
    A_pos = (1 - R) / s
    A_neg = (1 - R - C(N_neg - 1, k - 1) / C(N - 1, k - 1)) / s

Groups with s = 0 (no correct sample, or fewer than k wrong ones: every k-subset passes) get advantage 0.
"""
from __future__ import annotations

from math import comb, sqrt


def passk_advantages(correct: list[bool], k: int) -> list[float]:
    """Pass@k advantage per response of one group (correct: one bool per sample)."""
    n, n_neg = len(correct), sum(not c for c in correct)
    assert 1 <= k < n, (k, n)
    r = 1 - comb(n_neg, k) / comb(n, k)
    s = sqrt(r * (1 - r))
    if s == 0:
        return [0.0] * n
    a_pos = (1 - r) / s
    a_neg = (1 - r - comb(n_neg - 1, k - 1) / comb(n - 1, k - 1)) / s if n_neg else 0.0
    return [a_pos if c else a_neg for c in correct]


def passk_group_reward(correct: list[bool], k: int) -> float:
    """Expected pass@k of the group (the mean k-subset reward R above)."""
    n, n_neg = len(correct), sum(not c for c in correct)
    return 1 - comb(n_neg, k) / comb(n, k)
