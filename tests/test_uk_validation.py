"""Tests for the UK out-of-sample validation.

The strict no-refit principle: the eight SMM-identified behavioural
parameters from the German calibration must not be modified when the
model is applied to the UK. If a test here fails because parameters
have drifted, the validation is destroyed and the entire exercise has
to be redone from a clean checkpoint.

The validation result itself is allowed to fail acceptance criteria;
the model not travelling is a paper-relevant finding. What is NOT
allowed is silently retuning the parameters to make the UK validation
pass.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abmhp import Config, simulate
from abmhp.validation.uk import (
    BREXIT_PERIOD,
    GERMAN_SMM_PARAMS,
    UK_REGION_NAMES,
    UKRegionalConfig,
    assert_smm_parameters_intact,
    make_uk_config,
    run_uk_validation,
    score_primary_targets,
)


# ---------------------------------------------------------------------------
# Behavioural-parameter invariance tests
# ---------------------------------------------------------------------------


def test_uk_config_pins_all_eight_smm_parameters_to_german_values():
    """make_uk_config must produce a Config in which every SMM-identified
    parameter equals its German value within float tolerance. This is the
    strict no-refit check; any drift here destroys the out-of-sample test."""
    cfg = make_uk_config()
    assert_smm_parameters_intact(cfg)
    # Explicit per-parameter check for double-safety.
    assert cfg.voting.beta_dissat == pytest.approx(GERMAN_SMM_PARAMS["beta_dissat"], abs=1e-9)
    assert cfg.voting.beta_renter == pytest.approx(GERMAN_SMM_PARAMS["beta_renter"], abs=1e-9)
    assert cfg.voting.rho_aspiration == pytest.approx(GERMAN_SMM_PARAMS["rho_aspiration"], abs=1e-9)
    assert cfg.voting.alpha_local == pytest.approx(GERMAN_SMM_PARAMS["alpha_local"], abs=1e-9)
    assert cfg.regional.price_slope == pytest.approx(GERMAN_SMM_PARAMS["price_slope"], abs=1e-9)
    assert cfg.voting.beta_0 == pytest.approx(GERMAN_SMM_PARAMS["beta_0"], abs=1e-9)
    assert cfg.demographic.assortative_exponent == pytest.approx(
        GERMAN_SMM_PARAMS["assortative_exponent"], abs=1e-9
    )
    assert cfg.demographic.intergenerational_skill_corr == pytest.approx(
        GERMAN_SMM_PARAMS["intergenerational_skill_corr"], abs=1e-9
    )


def test_uk_simulation_does_not_mutate_smm_parameters():
    """Run the UK simulation and verify that no SMM parameter changes
    on the Config object (Config is frozen, but the check is the
    defensive guard against future code refactoring)."""
    cfg = make_uk_config(seed=73)
    before = {k: _read_param(cfg, k) for k in GERMAN_SMM_PARAMS}
    _, _, _ = simulate(cfg)
    after = {k: _read_param(cfg, k) for k in GERMAN_SMM_PARAMS}
    for k in GERMAN_SMM_PARAMS:
        assert before[k] == after[k], (
            f"SMM parameter {k} mutated during simulate(): "
            f"{before[k]} -> {after[k]}"
        )
        assert before[k] == pytest.approx(GERMAN_SMM_PARAMS[k], abs=1e-9), (
            f"SMM parameter {k} differs from German value: "
            f"{before[k]} vs {GERMAN_SMM_PARAMS[k]}"
        )


def test_uk_regional_config_is_12_nuts1_regions():
    """The UK regional config carries 12 NUTS-1 regions with the right
    structural pattern (one super, eight average, three declining)."""
    cfg = make_uk_config()
    assert cfg.regional.n_regions == 12
    assert len(UK_REGION_NAMES) == 12
    types = cfg.regional.region_type
    assert (types == "super").sum() == 1, "expected exactly one superstar (Greater London)"
    assert (types == "avg").sum() == 8
    assert (types == "decl").sum() == 3
    # The population shares must sum to 1 (after normalisation).
    assert cfg.regional.pop_share.sum() == pytest.approx(1.0, abs=1e-6)
    # London must be the most productive region.
    assert cfg.regional.productivity.argmax() == 0
    # London must have the lowest supply elasticity.
    assert cfg.regional.supply_elasticity.argmin() == 0


# ---------------------------------------------------------------------------
# Validation result-shape tests
# ---------------------------------------------------------------------------


def test_run_uk_validation_returns_required_keys():
    """The validation runner must return all keys downstream artifacts
    depend on (script, markdown, figures)."""
    result = run_uk_validation(seeds=(73, 74))
    required = {
        "aggregate_leave_share",
        "cross_regional_leave_price_correlation",
        "cross_regional_leave_price_partial_correlation",
        "within_region_renter_owner_gap",
        "aggregate_price_growth_to_brexit",
        "cross_regional_dispersion",
        "regional_leave_model",
        "regional_leave_empirical",
        "brexit_period",
    }
    assert required <= set(result.keys()), f"Missing keys: {required - set(result.keys())}"
    assert result["brexit_period"] == BREXIT_PERIOD == 11
    # Regional Leave dict must have all 12 NUTS-1 regions.
    assert set(result["regional_leave_model"].keys()) == set(UK_REGION_NAMES)


def test_score_primary_targets_produces_pass_fail_for_each():
    """score_primary_targets must return a dict with pass/fail for each
    of the three primary acceptance criteria."""
    result = run_uk_validation(seeds=(73, 74))
    scored = score_primary_targets(result)
    assert set(scored.keys()) == {
        "aggregate_leave_share",
        "cross_regional_leave_price_correlation",
        "within_region_renter_owner_gap",
    }
    for name, info in scored.items():
        assert "target" in info
        assert "model" in info
        assert "acceptance_range" in info
        assert "passed" in info
        assert isinstance(info["passed"], (bool, np.bool_)), \
            f"{name} did not return a bool for 'passed'"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_param(cfg: Config, name: str) -> float:
    if name == "beta_dissat":
        return cfg.voting.beta_dissat
    if name == "beta_renter":
        return cfg.voting.beta_renter
    if name == "rho_aspiration":
        return cfg.voting.rho_aspiration
    if name == "alpha_local":
        return cfg.voting.alpha_local
    if name == "price_slope":
        return cfg.regional.price_slope
    if name == "beta_0":
        return cfg.voting.beta_0
    if name == "assortative_exponent":
        return cfg.demographic.assortative_exponent
    if name == "intergenerational_skill_corr":
        return cfg.demographic.intergenerational_skill_corr
    raise KeyError(name)
