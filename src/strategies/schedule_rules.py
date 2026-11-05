"""One definition of when a tier is due, shared by every planned schedule.

Rounds are 1-indexed, r = 1..R. The first T_w rounds synchronise every shared
tier. After warm-up, tier g is due in round r when

        (r - T_w - phi_g) mod K_g == 0,                                   (*)

with phase offset phi_g (0 unless the schedule is staggered). Both the
aligned fixed period and the privacy planner use this rule, and the DP
accountant charges exactly the rounds it returns, so the release count in the
paper, in the scheduler and in the accountant is the same number.

The legacy fixed period of the v3-v12 runs used a different rule (round 1,
then every r with r mod K_g == 0, no warm-up unless requested); it is kept
behind FixedSchedule(aligned=False) so that those logs reproduce.
"""
from __future__ import annotations

from math import gcd
from typing import Dict, List, Mapping, Optional


def stagger_phases(K: Mapping[str, int]) -> Dict[str, int]:
    """Phase offsets that keep two tiers from ever coinciding when possible.

    For two tiers with intervals K_a, K_b the sync sets {phi_a + n K_a} and
    {phi_b + m K_b} are disjoint iff phi_a - phi_b is not divisible by
    g = gcd(K_a, K_b) (Chinese remainder theorem). With g >= 2 the offset
    floor(g / 2) separates them by the largest possible margin; with g = 1
    every offset coincides somewhere, and no phase removes the overlap.
    """
    tiers = sorted(K)
    if len(tiers) < 2:
        return {g: 0 for g in tiers}
    a, b = tiers[0], tiers[1]
    g = gcd(int(K[a]), int(K[b]))
    phases = {t: 0 for t in tiers}
    phases[b] = g // 2 if g >= 2 else 0
    return phases


def due_tiers(r: int, K: Mapping[str, int], warmup: int,
              phases: Optional[Mapping[str, int]] = None) -> List[str]:
    tiers = sorted(K)
    if r <= warmup:
        return tiers
    phases = phases or {}
    return [g for g in tiers if (r - warmup - phases.get(g, 0)) % int(K[g]) == 0]


def release_rounds(K: Mapping[str, int], rounds: int, warmup: int,
                   phases: Optional[Mapping[str, int]] = None) -> List[int]:
    """All rounds in 1..R in which at least one shared tier is sent."""
    return [r for r in range(1, rounds + 1) if due_tiers(r, K, warmup, phases)]


def peak_fraction(K: Mapping[str, int], w: Mapping[str, float], rounds: int, warmup: int,
                  phases: Optional[Mapping[str, int]] = None) -> float:
    """Largest post-warm-up payload fraction of the shared model in one round."""
    peaks = [sum(w[g] for g in due_tiers(r, K, warmup, phases))
             for r in range(warmup + 1, rounds + 1)]
    return max(peaks) if peaks else 0.0
