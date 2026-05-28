"""Regression tests for the VotingConfig.reference_rule flag.

The flag was added so that scripts/run_kr_robustness.py can evaluate the
Koszegi-Rabin (K-R) variant claimed in manuscript Section 9.1 against a
fresh run. The headline pipeline must remain bit-identical to the
pre-flag state. These tests pin that contract:

  1. Default reference_rule is "akerlof_yellen" so callers that never
     touch the new field get the headline behaviour automatically.
  2. simulate() under explicit reference_rule="akerlof_yellen" is
     per-field array_equal to the default-config simulation. Proves the
     branch logic doesn't introduce floating-point drift in the AY path.
  3. simulate() under reference_rule="koszegi_rabin" produces a
     materially different vote trajectory (otherwise the flag does
     nothing and the K-R robustness exercise is meaningless).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abmhp import Config, simulate
from abmhp.config import VotingConfig
from abmhp.estimation.smm import apply_smm_optimum


def _simulate_history_arrays(cfg: Config) -> dict[str, np.ndarray]:
    _, hist, _ = simulate(cfg)
    return {f: getattr(hist, f) for f in dir(hist) if not f.startswith("_")}


def test_default_reference_rule_is_akerlof_yellen():
    assert VotingConfig().reference_rule == "akerlof_yellen"
    assert Config().voting.reference_rule == "akerlof_yellen"


def test_ay_explicit_matches_ay_default_bit_for_bit():
    """An explicit reference_rule='akerlof_yellen' must match the default
    config to machine precision, per field. Proves the branch logic does
    not introduce floating-point drift in the AY path."""
    cfg_default = apply_smm_optimum(Config(seed=73, n_periods=15))
    cfg_explicit = replace(
        cfg_default,
        voting=replace(cfg_default.voting, reference_rule="akerlof_yellen"),
    )
    h_default = _simulate_history_arrays(cfg_default)
    h_explicit = _simulate_history_arrays(cfg_explicit)
    assert set(h_default.keys()) == set(h_explicit.keys())
    mismatches = []
    for f in h_default:
        a, b = h_default[f], h_explicit[f]
        if isinstance(a, np.ndarray) and a.dtype.kind in "fc":
            if not np.array_equal(a, b, equal_nan=True):
                mismatches.append(f)
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                mismatches.append(f)
        else:
            if a != b:
                mismatches.append(f)
    assert not mismatches, f"AY-explicit diverged from AY-default in: {mismatches}"


def test_kr_changes_political_block_only():
    """K-R must change the political-economy outputs (otherwise the flag
    is a no-op and run_kr_robustness.py reports nothing useful) but must
    leave the distributional and housing blocks bit-identical (they are
    upstream of voting.py)."""
    cfg_ay = apply_smm_optimum(Config(seed=73, n_periods=15))
    cfg_kr = replace(
        cfg_ay, voting=replace(cfg_ay.voting, reference_rule="koszegi_rabin"),
    )
    h_ay = _simulate_history_arrays(cfg_ay)
    h_kr = _simulate_history_arrays(cfg_kr)

    political = {"vote", "vote_aggregate", "vote_by_tenure", "dissat", "smoothed_vote"}
    upstream = {"gini", "top1", "top10", "bottom50", "ownership", "ownership_aggregate",
                "price", "rent_index", "mean_wealth", "mean_income", "mean_age"}

    for f in political & set(h_ay):
        if not np.array_equal(h_ay[f], h_kr[f], equal_nan=True):
            return  # at least one political field differs; passing
    pytest.fail(
        "K-R produced no change in any political field; the reference_rule "
        "flag is effectively a no-op."
    )


def test_kr_upstream_fields_bit_identical_to_ay():
    """Distributional and housing-market fields run upstream of voting.py
    and must not be affected by the reference rule."""
    cfg_ay = apply_smm_optimum(Config(seed=73, n_periods=15))
    cfg_kr = replace(
        cfg_ay, voting=replace(cfg_ay.voting, reference_rule="koszegi_rabin"),
    )
    h_ay = _simulate_history_arrays(cfg_ay)
    h_kr = _simulate_history_arrays(cfg_kr)
    upstream = ["gini", "top1", "top10", "bottom50", "ownership",
                "ownership_aggregate", "price", "rent_index", "mean_wealth",
                "mean_income", "mean_age"]
    for f in upstream:
        assert np.array_equal(h_ay[f], h_kr[f], equal_nan=True), \
            f"{f} differs under K-R but should be upstream of voting.py"


def test_unknown_reference_rule_raises():
    cfg = apply_smm_optimum(Config(seed=73, n_periods=15))
    cfg_bad = replace(cfg, voting=replace(cfg.voting, reference_rule="invalid"))
    with pytest.raises(ValueError, match="unknown reference_rule"):
        simulate(cfg_bad)
