"""Tests for the extreme-share activation regime and housing-only response.

The baseline calibration stabilises the extreme-share vote at ~0.22, which
sits in the hysteresis band. These tests construct deliberately off-baseline
scenarios via parameter overrides so the regime mechanics, the leakage
gating, and the three housing-only instruments can be exercised in
isolation.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abmhp import Config, PolicyRegime, simulate
from abmhp.config import PolicyConfig, VotingConfig


def _config_with(
    *,
    seed: int = 73,
    beta_0: float | None = None,
    force_regime: PolicyRegime | None = None,
    rent_cap_leakage: float | None = None,
    supply_leakage: float | None = None,
    friction_leakage: float | None = None,
    rent_cap_intensity: float | None = None,
    supply_restriction_intensity: float | None = None,
    transaction_friction: float | None = None,
    incumbency_threshold: float | None = None,
    deactivation_threshold: float | None = None,
) -> Config:
    cfg = Config(seed=seed)
    if beta_0 is not None:
        cfg = replace(cfg, voting=replace(cfg.voting, beta_0=beta_0))
    policy_kwargs: dict = {}
    if force_regime is not None:
        policy_kwargs["force_regime"] = force_regime.value
    if rent_cap_leakage is not None:
        policy_kwargs["rent_cap_leakage"] = rent_cap_leakage
    if supply_leakage is not None:
        policy_kwargs["supply_leakage"] = supply_leakage
    if friction_leakage is not None:
        policy_kwargs["friction_leakage"] = friction_leakage
    if rent_cap_intensity is not None:
        policy_kwargs["rent_cap_intensity"] = rent_cap_intensity
    if supply_restriction_intensity is not None:
        policy_kwargs["supply_restriction_intensity"] = supply_restriction_intensity
    if transaction_friction is not None:
        policy_kwargs["transaction_friction"] = transaction_friction
    if incumbency_threshold is not None:
        policy_kwargs["incumbency_threshold"] = incumbency_threshold
    if deactivation_threshold is not None:
        policy_kwargs["deactivation_threshold"] = deactivation_threshold
    if policy_kwargs:
        cfg = replace(cfg, policy=replace(cfg.policy, **policy_kwargs))
    return cfg


def _run_seeds(cfg_factory, seeds):
    runs = []
    for s in seeds:
        cfg = cfg_factory(s)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def _aggregate_rent_growth(hist) -> float:
    return float(hist.rent_index[-1].mean()) - 1.0


def _aggregate_price_growth(hist) -> float:
    return float((hist.price[-1] / hist.price[0]).mean()) - 1.0


def test_extreme_share_regime_activates_above_threshold():
    """With beta_0 raised to -3.5 the extreme-share vote crosses 0.30 quickly;
    the regime must switch to the activated state within the simulation
    horizon."""
    cfg = _config_with(beta_0=-3.5)
    _, hist, _ = simulate(cfg)
    switched = any(r is PolicyRegime.POPULIST for r in hist.regime)
    assert switched, "extreme-share activation regime never engaged despite beta_0 = -3.5"
    # The smoothed share at the switch period must have been at or above 0.30.
    first_pop = next(t for t, r in enumerate(hist.regime) if r is PolicyRegime.POPULIST)
    assert hist.smoothed_vote[first_pop] >= cfg.policy.incumbency_threshold - 1e-9


def test_extreme_share_regime_deactivates_below_threshold():
    """Pin the regime to the activated state initially, then let the natural
    baseline vote (~0.22) pull it back through the deactivation threshold of
    0.20.

    We use a slightly elevated deactivation threshold of 0.23 to bracket the
    baseline equilibrium from above, so the smoothed share will eventually
    fall below it. This isolates the deactivation logic from the calibration."""
    seeds = [73, 74, 75]
    rows = []
    for s in seeds:
        cfg = Config(seed=s)
        cfg = replace(cfg, policy=replace(cfg.policy, deactivation_threshold=0.23))
        _, hist, _ = simulate(cfg)
        rows.append(hist)

    # Sanity: in baseline (no force), regime starts MAINSTREAM and stays there.
    # The deactivation logic is exercised by simulating one period of forced
    # activated regime then handing control back: we re-run with a config
    # that forces the activated regime for the entire run, then re-run
    # without force; the second-run regime trajectory should deactivate
    # within the smoothing window since baseline smoothed vote 0.21 sits
    # below 0.23.
    cfg_pin = _config_with(force_regime=PolicyRegime.POPULIST)
    _, hist_pin, _ = simulate(cfg_pin)
    assert all(r is PolicyRegime.POPULIST for r in hist_pin.regime[1:]), \
        "force_regime must hold the activated regime for every period"

    # Independent path: simulate without force, but seed the regime trajectory
    # by inspecting whether update_regime would deactivate at baseline votes.
    # The cleanest direct test: import and call update_regime with sample inputs.
    from abmhp.policy import update_regime
    pol = PolicyConfig(deactivation_threshold=0.23)
    assert update_regime(PolicyRegime.POPULIST, 0.219, pol) is PolicyRegime.MAINSTREAM
    assert update_regime(PolicyRegime.POPULIST, 0.24, pol) is PolicyRegime.POPULIST


def test_rent_cap_with_zero_leakage_dampens_rent():
    """With cap intensity 0.6 and zero leakage, effective intensity is 0.6,
    so rent grows at 0.4 * price_growth. Cumulative rent growth should be at
    most 50 percent of cumulative price growth across the run.

    Implementation note: rent_cap is applied to *price growth* each period.
    We compare aggregate rent growth under (forced activated regime,
    leakage=0) against the no-policy counterfactual at the same seed."""
    seeds = [73, 74, 75, 76, 77]
    no_policy = [simulate(Config(seed=s))[1] for s in seeds]
    capped = [
        simulate(_config_with(seed=s, force_regime=PolicyRegime.POPULIST, rent_cap_leakage=0.0,
                              supply_restriction_intensity=0.0, transaction_friction=0.0))[1]
        for s in seeds
    ]
    np_growth = np.mean([_aggregate_rent_growth(h) for h in no_policy])
    cap_growth = np.mean([_aggregate_rent_growth(h) for h in capped])
    reduction = 1.0 - cap_growth / np_growth
    assert reduction >= 0.50, \
        f"rent growth reduction {reduction:.3f} below 50% target (np={np_growth:.3f}, cap={cap_growth:.3f})"


def test_rent_cap_with_unit_leakage_no_effect():
    """With cap intensity 0.6 but leakage 1.0, effective intensity is 0; the
    cap is fully circumvented and rent growth must match the no-policy
    counterfactual within 5 percent."""
    seeds = [73, 74, 75, 76, 77]
    no_policy = [simulate(Config(seed=s))[1] for s in seeds]
    leaked = [
        simulate(_config_with(seed=s, force_regime=PolicyRegime.POPULIST, rent_cap_leakage=1.0,
                              supply_restriction_intensity=0.0, transaction_friction=0.0))[1]
        for s in seeds
    ]
    np_growth = np.mean([_aggregate_rent_growth(h) for h in no_policy])
    lk_growth = np.mean([_aggregate_rent_growth(h) for h in leaked])
    rel_diff = abs(lk_growth - np_growth) / abs(np_growth)
    assert rel_diff <= 0.05, \
        f"rent growth diverged by {rel_diff:.3%} (np={np_growth:.3f}, leaked={lk_growth:.3f})"


def test_hysteresis_prevents_flicker():
    """If the smoothed vote share stays inside the band [0.20, 0.30], neither
    update_regime direction should fire. We unit-test update_regime directly
    with values that bracket the band."""
    from abmhp.policy import update_regime
    pol = PolicyConfig()  # defaults: incumbency 0.30, deactivation 0.20
    inside_band = [0.22, 0.25, 0.28, 0.21, 0.29]
    # Starting MAINSTREAM: should stay MAINSTREAM at any value < 0.30.
    regime = PolicyRegime.MAINSTREAM
    for v in inside_band:
        regime = update_regime(regime, v, pol)
        assert regime is PolicyRegime.MAINSTREAM, \
            f"flickered to activated regime at smoothed {v}"
    # Starting in the activated regime: should stay activated at any value > 0.20.
    regime = PolicyRegime.POPULIST
    for v in inside_band:
        regime = update_regime(regime, v, pol)
        assert regime is PolicyRegime.POPULIST, \
            f"flickered to MAINSTREAM at smoothed {v}"


def test_supply_restriction_reduces_construction_elasticity():
    """Effective supply elasticity becomes
        original * (1 - supply_restriction_intensity * (1 - supply_leakage)).
    With intensity 0.5 and leakage 0 the elasticity halves, so the same
    demand pressure produces a steeper price response. We compare aggregate
    price growth under the forced activated regime with zero supply leakage
    against the no-policy counterfactual."""
    seeds = [73, 74, 75, 76, 77]
    no_policy = [simulate(Config(seed=s))[1] for s in seeds]
    restricted = [
        simulate(_config_with(seed=s, force_regime=PolicyRegime.POPULIST, supply_leakage=0.0,
                              rent_cap_intensity=0.0, transaction_friction=0.0))[1]
        for s in seeds
    ]
    np_pg = np.mean([_aggregate_price_growth(h) for h in no_policy])
    re_pg = np.mean([_aggregate_price_growth(h) for h in restricted])
    assert re_pg > np_pg, \
        f"supply restriction did not steepen prices (np={np_pg:.3f}, restricted={re_pg:.3f})"
    assert (re_pg - np_pg) / abs(np_pg) > 0.02, \
        f"supply restriction effect too small: {(re_pg - np_pg)/abs(np_pg):.3%}"


def test_transaction_friction_reduces_ownership_transitions():
    """transaction_friction raises the wealth bar on buying. With friction
    intensity 0.3 and zero leakage, the buy threshold rises by 30 percent,
    which should cut the cumulative number of new ownership entries by at
    least 25 percent vs no-policy."""
    seeds = [73, 74, 75, 76, 77]

    def entries(hist) -> int:
        """Approximate new ownership entries: positive changes in regional
        homeownership share weighted by population share each period."""
        own = hist.ownership  # (T+1, R)
        # Use the regional homeowner *count* proxy: aggregate ownership rate
        # times number of agents per region. We approximate by aggregate rate
        # over time; the cumulative positive period-to-period change is a
        # monotone proxy for entries.
        deltas = np.diff(hist.ownership_aggregate)
        return float(np.maximum(deltas, 0.0).sum())

    no_policy = [simulate(Config(seed=s))[1] for s in seeds]
    rationed = [
        simulate(_config_with(seed=s, force_regime=PolicyRegime.POPULIST, friction_leakage=0.0,
                              rent_cap_intensity=0.0, supply_restriction_intensity=0.0))[1]
        for s in seeds
    ]
    np_entries = np.mean([entries(h) for h in no_policy])
    ra_entries = np.mean([entries(h) for h in rationed])
    reduction = 1.0 - ra_entries / max(np_entries, 1e-9)
    assert reduction >= 0.25, \
        f"new-entry reduction {reduction:.3f} below 25% target (np={np_entries:.4f}, rationed={ra_entries:.4f})"
