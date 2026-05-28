"""Tests for the HANK methodological benchmark.

Two classes of test:

  1. Calibration: the three aggregate German moments (wealth Gini,
     homeownership rate, aggregate extreme-share vote) are matched within
     numerical tolerance after the closed-form calibration routine runs.

  2. Structural-zero: the three quantities the paper relies on the ABM
     to deliver are mechanically zero in HANK by construction. This is
     the methodological argument: HANK is the wrong tool for the
     paper's findings because the relevant heterogeneity is structurally
     absent.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from abmhp.hank_benchmark import (
    HankConfig,
    HankPolicyScenario,
    aggregate_material_relief,
    aggregate_wealth_gini,
    baseline_extreme_share,
    calibrate,
    cross_regional_extreme_share_dispersion,
    homeownership_rate,
    incomplete_material_repair_effect,
    make_paired_scenarios,
    vote_under_scenario,
    within_region_tenure_vote_gap,
)


@pytest.fixture(scope="module")
def calibrated() -> HankConfig:
    cfg, _ = calibrate(HankConfig())
    return cfg


def test_hank_matches_wealth_gini(calibrated):
    """Aggregate wealth Gini matches Bundesbank PHF target 0,81."""
    g = aggregate_wealth_gini(calibrated)
    assert abs(g - 0.81) <= 0.01, f"HANK Gini {g:.4f} off target 0.81"


def test_hank_matches_homeownership(calibrated):
    """Aggregate homeownership rate matches Mikrozensus target 0,50."""
    h = homeownership_rate(calibrated)
    assert abs(h - 0.50) <= 0.01, f"HANK homeownership {h:.4f} off target 0.50"


def test_hank_matches_extreme_share(calibrated):
    """Aggregate extreme-share vote matches 23 February 2025 AfD share 0,208."""
    v = baseline_extreme_share(calibrated)
    assert abs(v - 0.208) <= 0.005, f"HANK extreme-share {v:.4f} off target 0.208"


def test_cross_regional_dispersion_is_structurally_zero(calibrated):
    """HANK has no regional structure. Cross-regional dispersion is zero
    by construction, not as a numerical artefact."""
    d = cross_regional_extreme_share_dispersion(calibrated)
    assert d == 0.0, f"HANK reported nonzero cross-regional dispersion {d}"


def test_within_region_tenure_gap_is_structurally_zero(calibrated):
    """HANK has aggregate-level voting. Within-region renter-owner gap is
    zero by construction; HANK has no individual heterogeneity in voting."""
    g = within_region_tenure_vote_gap(calibrated)
    assert g == 0.0, f"HANK reported nonzero within-region tenure gap {g}"


def test_incomplete_material_repair_effect_is_structurally_zero(calibrated):
    """P3 effect: comparing two scenarios with the same total aggregate
    relief but different mixes (housing-only vs housing-plus-redistribution),
    HANK produces zero vote-share difference. The ABM's P3 = -0,113 is not
    representable in HANK because dissatisfaction is single-channel."""
    for total_relief in [0.02, 0.05, 0.10]:
        effect = incomplete_material_repair_effect(calibrated, total_relief)
        assert abs(effect) < 1e-9, \
            f"HANK reported nonzero P3 effect at total_relief={total_relief}: {effect:+.2e}"


def test_paired_scenarios_with_equal_total_relief_give_equal_votes(calibrated):
    """Directly verify the structural property: any two HANK scenarios
    delivering the same aggregate material relief produce identical vote
    shares. This is the single-channel collapse."""
    cfg = calibrated
    # Scenario one: all relief through rent cap.
    sc1 = HankPolicyScenario(
        name="all-rent",
        rent_cap_intensity=1.20,  # relief = 1.20 * 0.083 = 0.0996
        rent_cap_leakage=0.0,
        redistribution_intensity=0.0,
    )
    relief1 = aggregate_material_relief(sc1, cfg)
    # Scenario two: split between rent cap and redistribution, same total.
    half = relief1 / 2.0
    sc2 = HankPolicyScenario(
        name="split",
        rent_cap_intensity=(half / cfg.dissat_response_to_rent_relief),
        rent_cap_leakage=0.0,
        redistribution_intensity=(half / cfg.dissat_response_to_transfer),
    )
    relief2 = aggregate_material_relief(sc2, cfg)
    assert abs(relief1 - relief2) < 1e-9
    v1 = vote_under_scenario(sc1, cfg)
    v2 = vote_under_scenario(sc2, cfg)
    assert abs(v1 - v2) < 1e-9, \
        f"HANK paired scenarios diverged: housing-only {v1:.6f}, split {v2:.6f}"


def test_hank_paired_scenarios_in_practice_show_small_marginal_difference(calibrated):
    """Diagnostic side-test: the prompt's HANK paired scenarios (housing-
    only vs housing-plus-redistribution at fixed rent-cap intensity) DO
    differ in HANK, because the multichannel scenario delivers MORE total
    relief through the same single channel. This is reported in the
    comparison writeup as the side-metric that explains why P3 must be
    tested at fixed total relief to expose HANK's structural blindness."""
    housing_only, multichannel = make_paired_scenarios(
        calibrated,
        rent_cap_intensity=0.6,
        rent_cap_leakage=0.4,
        redistribution_intensity=0.027,
    )
    v_h = vote_under_scenario(housing_only, calibrated)
    v_m = vote_under_scenario(multichannel, calibrated)
    diff = v_m - v_h
    # Vote should drop modestly because the multichannel scenario adds
    # relief; the drop is the marginal effect of extra single-channel
    # relief, NOT the P3 channel-decomposition effect.
    assert diff < 0.0, "multichannel should not raise the extreme-share vote"
    assert abs(diff) < 0.05, \
        f"HANK marginal vote drop {diff:+.4f} larger than expected"
