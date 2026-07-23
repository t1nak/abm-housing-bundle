"""Path A: the calibrated three-margin (Version A) decomposition.

The three housing-pressure margin weights are CALIBRATED, not structurally
estimated. The available aggregate moments identify the overall housing
cleavage (the 2025 AfD level and the renter-owner gap) more reliably than the
separate weights of its component margins; separately estimating gamma_rent,
gamma_asset and gamma_access from macro moments alone is under-identified
(see project memory). The weights below are therefore calibrated to match the
observed vote-level anchor and the selected renter-owner gap scenario while
keeping all three housing-pressure gradients positive (non-circular sign
restrictions). Counterfactuals run on
this configuration are interpreted comparatively within the calibrated model,
not as estimates of separately identified structural parameters. A fully
structural version (Path B) would require micro-data moments (SOEP / BES:
tenure x local price growth x asset position x ownership expectations x vote).

Baseline specification (paper round 3): the rent-stress margin is GATED on
renters (rent_margin_renters_only=True), so all three margins are zero for
owners by construction and each margin is housing-specific. The ungated
variant (owners' d_rent = pure income-aspiration shortfall) is retained as a
robustness variant in the sweep. Relative to the ungated calibration
(beta_0=-2.2, aggregate 0.200 / gap +0.148), gating removes the owner-side
rent-margin contribution; re-solving the intercept alone restores the level
anchor.

Gap-scenario framing (paper round 9): the renter-owner political gap is
NOT an observed SOEP magnitude (the SOEP extraction has not been run; it
is the pre-specified design of scripts/soep_anchor_pipeline.py, executed
in the follow-up paper). The paper instead evaluates a literature-informed
scenario range for the gap, {+0.05 conservative, +0.10 central, +0.15
upper bound}, and this module pins the CENTRAL scenario as the headline
baseline. Scenario configurations for the other two scenarios are produced
by scripts/adjusted_anchor_band.py (proportional gamma rescaling with
beta_0 re-solved).

Central-scenario calibration (seeds 73-78, n_periods=15, gated,
repaired tenure block of paper round 11 -- regional rental-market depth
+ parental transfers at first-ownership entry):
    aggregate AfD       = ~0.209  (level anchor 0.208, tolerance +/-0.01)
    renter-owner gap    = ~+0.105 (central scenario +0.10)
    gradients (sign checks with one-sided floor 0.15): reproduced by
    scripts/baseline_diagnostics.py
"""
from __future__ import annotations

from dataclasses import replace

from .config import Config

# beta_renter is held at 0: the three margins explain the renter-owner cleavage
# without a tenure dummy. The split (rent : asset : access) is a disciplined
# calibration, not an estimate.
# Central gap scenario (+0.10): gamma = 0.61 * (0.48, 0.88, 0.80), beta_0
# re-solved for the level anchor (scripts/adjusted_anchor_band.py).
MARGIN_CALIBRATION: dict = dict(
    margin_decomposition=True,
    beta_renter=0.0,
    asset_gain_window=5,
    rent_margin_renters_only=True,
    gamma_rent=0.2928,
    gamma_asset=0.5368,
    gamma_access=0.4880,
    beta_0=-1.82,
)


def apply_margin_calibration(cfg: Config) -> Config:
    """Apply the calibrated three-margin Version-A configuration to a Config."""
    return replace(cfg, voting=replace(cfg.voting, **MARGIN_CALIBRATION))
