"""ABM of housing wealth, reference-dependent dissatisfaction, and the demand for political extremes.

The model produces local_extreme_share (the code-level variable), interpreted
in the German calibration as the right-exit share, theoretically as mainstream
exit, and empirically proxied by AfD vote share. See paper/framing_v2.md.
"""
from .config import (
    Config,
    RegionalConfig,
    BehavioralConfig,
    DemographicConfig,
    VotingConfig,
    PolicyConfig,
    SkillConfig,
)
from .simulation import simulate, History
from .household import HouseholdState
from .metrics import gini, top_share, bottom_share
from .policy import EffectiveIntensities, PolicyRegime

__all__ = [
    "Config",
    "RegionalConfig",
    "BehavioralConfig",
    "DemographicConfig",
    "VotingConfig",
    "PolicyConfig",
    "SkillConfig",
    "simulate",
    "History",
    "HouseholdState",
    "PolicyRegime",
    "EffectiveIntensities",
    "gini",
    "top_share",
    "bottom_share",
]
