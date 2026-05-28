"""HANK-style methodological benchmark.

Stripped representative-agent two-state model used to demonstrate categorical
limits of standard heterogeneous-agent macro for the paper's bundled-mechanism
findings. The benchmark is deliberately not state-of-the-art. It exists to
show that HANK cannot mechanically produce three quantities the ABM produces:

  1. Cross-regional dispersion in the extreme/anti-system vote
     (HANK has no regional structure).
  2. Within-region tenure-conditioned extreme-share gap
     (HANK has aggregate-level voting; no individual heterogeneity).
  3. Incomplete-material-repair effect (P3): different responses to
     housing-only versus integrated material-security intervention
     (HANK has single-channel dissatisfaction; the channel decomposition
     the ABM relies on is structurally absent).

Implementation. Closed-form steady state, no transition dynamics:
  - Two productivity states (high/low) with persistent transition matrix;
    stationary distribution is 50/50 by symmetry.
  - Two housing tenure states (owner/renter); ownership probability
    differs by productivity state.
  - Four cells (HO, HR, LO, LR); within each cell, wealth is lognormal
    around a cell-specific median with a common within-cell dispersion
    parameter sigma_w.
  - Vote share is a sigmoid of aggregate dissatisfaction; dissatisfaction
    is the single channel through which all policy enters.

Calibration. Three aggregate German moments:
  - Wealth Gini = 0,81 (Bundesbank PHF wave 4).
  - Homeownership rate = 0,50 (Mikrozensus).
  - Aggregate right-exit vote share = 0,208 (Bundeswahlleiter,
    23 February 2025 second votes).

The HANK aggregate vote is matched to the AfD second-vote share at the
2025 calibration point; this is the empirical proxy for mainstream exit
in the German calibration. The model is not a theory of AfD voting.

The methodological argument is: HANK cannot mechanically distinguish
single-instrument from multi-instrument intervention because
dissatisfaction is single-channel by construction. The ABM's
incomplete-material-repair finding (P3) requires individual-level
dual-channel dissatisfaction structure that HANK lacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
from scipy import optimize


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HankConfig:
    """HANK calibration parameters. Defaults match the German aggregate
    moments after the closed-form calibration routine below has run."""

    # Productivity states. Stationary distribution is 50/50 by symmetry
    # (productivity_persistence equal for both states).
    productivity_persistence: float = 0.92
    income_high: float = 60_000.0
    income_low: float = 20_000.0

    # Ownership probabilities by productivity state. Calibrated so that
    # aggregate homeownership matches 0,50.
    p_own_high: float = 0.78
    p_own_low: float = 0.22

    # Cell-level wealth medians (EUR). Tightly stylised. Calibrated jointly
    # with the within-cell sigma to match wealth Gini.
    median_wealth_HO: float = 600_000.0
    median_wealth_HR: float = 80_000.0
    median_wealth_LO: float = 180_000.0
    median_wealth_LR: float = 4_000.0

    # Within-cell lognormal dispersion. Single parameter controls overall
    # wealth Gini. Calibrated to match 0,81.
    within_cell_sigma: float = 1.50

    # Aggregate baseline dissatisfaction. Single scalar; this is the
    # single-channel limitation that defines HANK in the comparison.
    baseline_dissatisfaction: float = 0.20

    # Logit vote function:
    #     extreme_share = 1 / (1 + exp(-(alpha + beta * dissat))).
    # Calibrated so the baseline matches the 2025 right-exit share 0,208.
    vote_alpha: float = -3.0
    vote_beta: float = 8.45

    # Policy-to-dissatisfaction mapping. HANK reads housing-only relief
    # and redistribution as the SAME aggregate-relief channel. The two
    # response coefficients below convert nominal policy intensity into
    # dissatisfaction reduction. The paper's central claim is that the
    # ABM has TWO separate channels with different coefficients per
    # household type; HANK has one channel for the whole population.
    dissat_response_to_rent_relief: float = 0.083
    dissat_response_to_transfer: float = 0.50

    # Population size for sampling the steady-state distribution.
    sample_n: int = 40_000
    seed: int = 73


# ---------------------------------------------------------------------------
# Steady-state structure
# ---------------------------------------------------------------------------


def stationary_distribution(persistence: float) -> tuple[float, float]:
    """Symmetric 2-state Markov chain. Stationary is 50/50."""
    # The persistence parameter is preserved for transparency, but the
    # stationary distribution is invariant under the symmetry assumption.
    _ = persistence
    return 0.5, 0.5


def cell_populations(cfg: HankConfig) -> dict[str, float]:
    """Population share of each (productivity, ownership) cell."""
    pi_H, pi_L = stationary_distribution(cfg.productivity_persistence)
    return {
        "HO": pi_H * cfg.p_own_high,
        "HR": pi_H * (1.0 - cfg.p_own_high),
        "LO": pi_L * cfg.p_own_low,
        "LR": pi_L * (1.0 - cfg.p_own_low),
    }


def cell_medians(cfg: HankConfig) -> dict[str, float]:
    return {
        "HO": cfg.median_wealth_HO,
        "HR": cfg.median_wealth_HR,
        "LO": cfg.median_wealth_LO,
        "LR": cfg.median_wealth_LR,
    }


def sample_steady_state(cfg: HankConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (wealth, productivity_state, owner_flag) at the steady state.

    Returns three arrays of length cfg.sample_n. Wealth is drawn from the
    cell-specific lognormal; productivity and owner status are deterministic
    per cell assignment."""
    rng = np.random.default_rng(cfg.seed)
    pops = cell_populations(cfg)
    medians = cell_medians(cfg)
    cells = list(pops.keys())
    probs = np.array([pops[c] for c in cells])
    probs = probs / probs.sum()

    assignments = rng.choice(len(cells), size=cfg.sample_n, p=probs)
    wealth = np.empty(cfg.sample_n)
    prod = np.empty(cfg.sample_n, dtype="U1")
    owner = np.zeros(cfg.sample_n, dtype=bool)
    for i, c in enumerate(cells):
        mask = assignments == i
        if not mask.any():
            continue
        med = max(medians[c], 1.0)
        wealth[mask] = rng.lognormal(np.log(med), cfg.within_cell_sigma, size=mask.sum())
        prod[mask] = c[0]
        owner[mask] = c[1] == "O"
    return wealth, prod, owner


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    if n == 0:
        return 0.0
    cum = np.cumsum(x)
    total = cum[-1]
    if total <= 0:
        return 0.0
    return float((n + 1 - 2 * cum.sum() / total) / n)


def homeownership_rate(cfg: HankConfig) -> float:
    pops = cell_populations(cfg)
    return pops["HO"] + pops["LO"]


def aggregate_wealth_gini(cfg: HankConfig) -> float:
    wealth, _, _ = sample_steady_state(cfg)
    return gini_coefficient(wealth)


def baseline_extreme_share(cfg: HankConfig) -> float:
    """Aggregate vote share for extreme/anti-system parties at baseline
    dissatisfaction (no policy)."""
    logit = cfg.vote_alpha + cfg.vote_beta * cfg.baseline_dissatisfaction
    return float(1.0 / (1.0 + np.exp(-logit)))


# ---------------------------------------------------------------------------
# Policy scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HankPolicyScenario:
    """A HANK policy comprises a rent-cap intensity (with leakage) and a
    redistribution intensity. Both translate into the single aggregate
    relief channel; the mix is structurally invisible to HANK."""
    name: str
    rent_cap_intensity: float = 0.0
    rent_cap_leakage: float = 0.0
    redistribution_intensity: float = 0.0


def aggregate_material_relief(scenario: HankPolicyScenario, cfg: HankConfig) -> float:
    """Single-channel aggregate relief delivered by a HANK policy.

    Housing-only relief and redistributive relief are summed: HANK does
    not distinguish the two. This is the structural property that produces
    the zero P3 effect in HANK."""
    effective_rent_cap = scenario.rent_cap_intensity * (1.0 - scenario.rent_cap_leakage)
    rent_relief = effective_rent_cap * cfg.dissat_response_to_rent_relief
    transfer_relief = scenario.redistribution_intensity * cfg.dissat_response_to_transfer
    return rent_relief + transfer_relief


def vote_under_scenario(scenario: HankPolicyScenario, cfg: HankConfig) -> float:
    """HANK extreme/anti-system vote share under a policy scenario."""
    relief = aggregate_material_relief(scenario, cfg)
    dissat = max(0.0, cfg.baseline_dissatisfaction - relief)
    logit = cfg.vote_alpha + cfg.vote_beta * dissat
    return float(1.0 / (1.0 + np.exp(-logit)))


# ---------------------------------------------------------------------------
# Structural-zero quantities
# ---------------------------------------------------------------------------


def cross_regional_extreme_share_dispersion(cfg: HankConfig) -> float:
    """Standard deviation of the extreme/anti-system vote across regions.

    HANK has no regional structure. The model produces a single aggregate
    vote share; cross-regional dispersion is mechanically zero."""
    _ = cfg
    return 0.0


def within_region_tenure_vote_gap(cfg: HankConfig) -> float:
    """Renter minus owner extreme/anti-system vote within the same region.

    HANK has aggregate-level voting: vote share is a function of aggregate
    dissatisfaction, not of individual tenure or productivity state. The
    within-region tenure gap is mechanically zero."""
    _ = cfg
    return 0.0


def incomplete_material_repair_effect(cfg: HankConfig, total_relief: float) -> float:
    """Vote-share difference between integrated and housing-only scenarios
    that deliver the SAME total aggregate relief through different mixes.

    HANK reads both scenarios identically because dissatisfaction is
    single-channel. The vote-share difference is mechanically zero. The
    ABM's P3 effect (-0,113 between Scenario E and Scenario C) is not
    representable in HANK by construction."""
    # Two scenarios with the same effective total_relief, different mix.
    housing_only = HankPolicyScenario(
        name="HANK-housing-only-fixed-relief",
        rent_cap_intensity=total_relief / max(cfg.dissat_response_to_rent_relief, 1e-9),
        rent_cap_leakage=0.0,
        redistribution_intensity=0.0,
    )
    half = 0.5 * total_relief
    multichannel = HankPolicyScenario(
        name="HANK-multichannel-fixed-relief",
        rent_cap_intensity=half / max(cfg.dissat_response_to_rent_relief, 1e-9),
        rent_cap_leakage=0.0,
        redistribution_intensity=half / max(cfg.dissat_response_to_transfer, 1e-9),
    )
    v_h = vote_under_scenario(housing_only, cfg)
    v_m = vote_under_scenario(multichannel, cfg)
    return float(v_m - v_h)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(
    cfg: HankConfig,
    target_gini: float = 0.81,
    target_homeownership: float = 0.50,
    target_extreme_share: float = 0.208,
    gini_tol: float = 0.01,
    vote_tol: float = 0.005,
) -> tuple[HankConfig, dict[str, float]]:
    """Sequentially calibrate the three aggregate moments.

    Step 1. Homeownership: scale p_own_high and p_own_low jointly so their
    average matches target_homeownership while preserving the relative
    gap (high types more likely to own than low types).

    Step 2. Wealth Gini: brentq on within_cell_sigma so the sampled Gini
    matches target_gini within gini_tol.

    Step 3. Aggregate extreme-share vote: brentq on baseline_dissatisfaction
    so the sigmoid evaluation matches target_extreme_share within vote_tol.

    Returns the calibrated config and a dict with the realised moments."""

    # Step 1: homeownership.
    base_high, base_low = cfg.p_own_high, cfg.p_own_low
    avg_now = 0.5 * (base_high + base_low)
    if avg_now <= 0:
        scale = 1.0
    else:
        scale = target_homeownership / avg_now
    new_high = float(np.clip(base_high * scale, 0.0, 1.0))
    new_low = float(np.clip(base_low * scale, 0.0, 1.0))
    cfg = replace(cfg, p_own_high=new_high, p_own_low=new_low)
    own_realised = homeownership_rate(cfg)

    # Step 2: Gini via within_cell_sigma.
    def gini_residual(sigma: float) -> float:
        c2 = replace(cfg, within_cell_sigma=float(sigma))
        return aggregate_wealth_gini(c2) - target_gini

    lo, hi = 0.4, 3.5
    g_lo = gini_residual(lo)
    g_hi = gini_residual(hi)
    if g_lo * g_hi > 0:
        # Bracketing failed; fall back to current sigma.
        sigma_star = cfg.within_cell_sigma
    else:
        sigma_star = float(optimize.brentq(gini_residual, lo, hi, xtol=1e-3))
    cfg = replace(cfg, within_cell_sigma=sigma_star)
    gini_realised = aggregate_wealth_gini(cfg)

    # Step 3: extreme-share vote via baseline_dissatisfaction.
    def vote_residual(dissat: float) -> float:
        c2 = replace(cfg, baseline_dissatisfaction=float(dissat))
        return baseline_extreme_share(c2) - target_extreme_share

    lo_d, hi_d = 0.0, 1.0
    if vote_residual(lo_d) * vote_residual(hi_d) > 0:
        dissat_star = cfg.baseline_dissatisfaction
    else:
        dissat_star = float(optimize.brentq(vote_residual, lo_d, hi_d, xtol=1e-4))
    cfg = replace(cfg, baseline_dissatisfaction=dissat_star)
    vote_realised = baseline_extreme_share(cfg)

    realised = {
        "wealth_gini": float(gini_realised),
        "homeownership_rate": float(own_realised),
        "extreme_share_vote": float(vote_realised),
        "within_cell_sigma": float(sigma_star),
        "baseline_dissatisfaction": float(dissat_star),
        "p_own_high": float(new_high),
        "p_own_low": float(new_low),
    }
    return cfg, realised


# ---------------------------------------------------------------------------
# Convenience: paired scenarios analogous to the ABM C vs E comparison
# ---------------------------------------------------------------------------


def make_paired_scenarios(
    cfg: HankConfig,
    rent_cap_intensity: float = 0.6,
    rent_cap_leakage: float = 0.4,
    redistribution_intensity: float = 0.027,
) -> tuple[HankPolicyScenario, HankPolicyScenario]:
    """Two HANK scenarios analogous to ABM Scenario C and Scenario E.

    HANK-housing-only mirrors the ABM Scenario C: rent cap with leakage,
    no redistribution.
    HANK-multichannel mirrors the ABM Scenario E: same rent cap with the
    same leakage, plus a redistribution component.

    Both are HANK scenarios; HANK cannot represent the channel decomposition
    that drives the ABM's wedge between C and E."""
    _ = cfg  # config is read by vote_under_scenario, not here
    housing_only = HankPolicyScenario(
        name="HANK-housing-only",
        rent_cap_intensity=rent_cap_intensity,
        rent_cap_leakage=rent_cap_leakage,
        redistribution_intensity=0.0,
    )
    multichannel = HankPolicyScenario(
        name="HANK-multichannel",
        rent_cap_intensity=rent_cap_intensity,
        rent_cap_leakage=rent_cap_leakage,
        redistribution_intensity=redistribution_intensity,
    )
    return housing_only, multichannel
