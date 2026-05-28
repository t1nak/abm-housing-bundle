"""Extreme-share activation regime and leakage-attenuated housing-only instruments.

This module implements the extreme-share activation regime that governs the
housing-only mainstream response. The regime is named POPULIST at the code
level for backwards compatibility with prior prompts; conceptually it
represents the policy regime that activates when local_extreme_share crosses
the activation threshold. See paper/framing_v2.md for the full terminology
hierarchy.

The block is deterministic given simulation state. No new RNG draws are
introduced here; the prompt 1 RNG discipline is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import PolicyConfig


class PolicyRegime(str, Enum):
    MAINSTREAM = "mainstream"
    POPULIST = "populist"


@dataclass(frozen=True)
class EffectiveIntensities:
    """Housing-only mainstream response instruments after leakage attrition.

    `redistribution` is the active capital-tax rate inside the integrated
    material-security intervention. The intervention has no leakage parameter:
    the tax authority collects and redistributes deterministically. The lever
    is the tax rate itself.
    """
    rent_cap: float
    supply_restriction: float
    transaction_friction: float
    redistribution: float = 0.0

    @classmethod
    def from_config(cls, cfg: PolicyConfig, regime: PolicyRegime) -> "EffectiveIntensities":
        if regime is PolicyRegime.MAINSTREAM:
            return cls(0.0, 0.0, 0.0, 0.0)
        return cls(
            rent_cap=cfg.rent_cap_intensity * (1.0 - cfg.rent_cap_leakage),
            supply_restriction=cfg.supply_restriction_intensity * (1.0 - cfg.supply_leakage),
            transaction_friction=cfg.transaction_friction * (1.0 - cfg.friction_leakage),
            redistribution=(cfg.capital_tax_rate if cfg.redistribution_active else 0.0),
        )


def smoothed_vote(vote_history: np.ndarray, t: int, window: int) -> float:
    """Equal-weight moving average of the last `window` periods up to and
    including index t. For t < window-1, averages whatever history is
    available (no warm-up bias in either direction)."""
    if t < 0:
        return 0.0
    lo = max(0, t - window + 1)
    return float(np.mean(vote_history[lo : t + 1]))


def update_regime(
    prev_regime: PolicyRegime,
    smoothed_share: float,
    cfg: PolicyConfig,
) -> PolicyRegime:
    """Hysteresis switching rule for the extreme-share activation regime.
    MAINSTREAM transitions to the activated regime (enum value POPULIST) when
    the smoothed share crosses the incumbency threshold; the activated regime
    transitions back to MAINSTREAM when the smoothed share falls below the
    deactivation threshold. The smoothing window enforces the 'stays below
    for k periods' requirement."""
    if cfg.force_regime is not None:
        return PolicyRegime(cfg.force_regime)
    if prev_regime is PolicyRegime.MAINSTREAM:
        if smoothed_share >= cfg.incumbency_threshold:
            return PolicyRegime.POPULIST
        return PolicyRegime.MAINSTREAM
    # Extreme-share activation regime currently in power.
    if smoothed_share <= cfg.deactivation_threshold:
        return PolicyRegime.MAINSTREAM
    return PolicyRegime.POPULIST
