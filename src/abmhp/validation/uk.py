"""UK out-of-sample validation: NUTS-1 regional structure for the
bundled-mechanism ABM with German behavioural parameters held fixed.

Behavioural parameters (the eight SMM-identified values from
`outputs/smm_optimum.json`) are LOCKED in this module. Only the
following primitives are recalibrated to UK data:

  - 12 NUTS-1 regions (Greater London, South East, East of England,
    South West, Scotland, North West, West Midlands, East Midlands,
    Yorkshire and Humber, Northern Ireland, North East, Wales).
  - Regional population shares from ONS 2010 mid-year estimates.
  - Regional productivity gradient from ONS regional GVA per capita 2010
    (Greater London is the superstar; North East and Wales are the
    bottom decline).
  - Regional housing supply elasticities from Hilber and Vermeulen (2016).
  - Bequest tax rate at 0,40 (UK Inheritance Tax effective rate, higher
    than German Erbschaftsteuer).
  - Initial house-price gradient calibrated to UK 2005 levels (London
    premium roughly 80 percent above national average).

The Brexit referendum is at simulation period 11 (2005 = t=0, 2016 = t=11,
2019 = t=14). Validation targets are taken at t=11.

Empirical anchors:

  - Aggregate Leave vote share at 2016 referendum: 0,518 (Electoral
    Commission). Acceptance range 0,45 to 0,55.
  - Cross-regional Leave-price-growth correlation (controlling for
    income): approximately -0,20 per Adler and Ansell (2020).
    Acceptance range -0,10 to -0,30.
  - Within-region renter-owner Leave gap: approximately +0,15
    (British Election Study tenure cross-tabulations). Acceptance
    range at least +0,10.

Secondary diagnostic anchors:

  - Greater London Leave share 0,40 (Electoral Commission). Model
    should produce London below national average.
  - East Midlands Leave 0,59, North East 0,58, Wales 0,53. Model
    should produce these above national average.
  - Scotland 0,38 and Northern Ireland 0,44 are Remain-leaning; these
    are political-supply effects the bundled-mechanism model is NOT
    expected to reproduce (the model is silent on partisan direction).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from ..config import (
    BehavioralConfig,
    Config,
    DemographicConfig,
    PolicyConfig,
    RegionalConfig,
    SkillConfig,
    VotingConfig,
)
from ..estimation.smm import load_smm_optimum
from ..simulation import History, simulate


# ---------------------------------------------------------------------------
# Locked behavioural parameters from the German SMM.
#
# Loaded from `outputs/smm_optimum.json` at module import. Previously a
# hardcoded dict of 4-decimal-rounded values; the canonical artefact
# carries full precision, and the UK validation should pin against the
# same numbers the German calibration produced. The audit infrastructure
# in `scripts/audit_parameter_vector.py` will report this entry point as
# AT OPTIMUM once the price_slope override below is in place.
# ---------------------------------------------------------------------------


GERMAN_SMM_PARAMS: dict[str, float] = load_smm_optimum().as_dict()


# ---------------------------------------------------------------------------
# UK regional primitives
# ---------------------------------------------------------------------------


UK_REGION_NAMES: tuple[str, ...] = (
    "Greater London",
    "South East",
    "East of England",
    "South West",
    "Scotland",
    "North West",
    "West Midlands",
    "East Midlands",
    "Yorkshire and Humber",
    "Northern Ireland",
    "North East",
    "Wales",
)

UK_REGION_LABELS: tuple[str, ...] = (
    "LDN", "SE", "EE", "SW", "SCO", "NW", "WM", "EM", "YH", "NI", "NE", "WAL",
)

# Productivity index: ONS regional GVA per capita 2010, normalised to UK = 1,00.
# London is the superstar (~1,65); North East and Wales are the bottom.
UK_PRODUCTIVITY: np.ndarray = np.array(
    [1.65, 1.15, 1.08, 1.00, 1.00, 0.92, 0.88, 0.86, 0.86, 0.85, 0.78, 0.78]
)

# Supply elasticity: Hilber and Vermeulen (2016) regional planning-constraint
# estimates. London is the most inelastic in Europe; Wales and the North are
# the most elastic.
UK_SUPPLY_ELASTICITY: np.ndarray = np.array(
    [0.10, 0.30, 0.40, 0.45, 0.50, 0.55, 0.55, 0.60, 0.60, 0.70, 0.70, 0.80]
)

# ONS 2010 mid-year population estimates, in millions.
UK_POP_SHARE_RAW: np.ndarray = np.array(
    [7.83, 8.52, 5.83, 5.27, 5.22, 6.94, 5.46, 4.53, 5.30, 1.80, 2.60, 3.06]
)

# Classification for region_type reporting: one super (London), eight average,
# three declining (Northern Ireland, North East, Wales).
UK_REGION_TYPE: np.ndarray = np.array(["super"] + ["avg"] * 8 + ["decl"] * 3)


BREXIT_PERIOD: int = 11  # 2005 = t=0, 2016 = t=11.


# Empirical Leave vote share by NUTS-1 region in the 2016 EU referendum.
# Source: Electoral Commission, 23 June 2016.
EMPIRICAL_LEAVE_BY_REGION: dict[str, float] = {
    "Greater London": 0.402,
    "South East": 0.518,
    "East of England": 0.566,
    "South West": 0.527,
    "Scotland": 0.380,
    "North West": 0.535,
    "West Midlands": 0.594,
    "East Midlands": 0.587,
    "Yorkshire and Humber": 0.578,
    "Northern Ireland": 0.440,
    "North East": 0.581,
    "Wales": 0.525,
}

EMPIRICAL_AGGREGATE_LEAVE: float = 0.518


# ---------------------------------------------------------------------------
# UK regional config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UKRegionalConfig(RegionalConfig):
    n_regions: int = 12
    productivity: np.ndarray = field(default_factory=lambda: UK_PRODUCTIVITY.copy())
    supply_elasticity: np.ndarray = field(default_factory=lambda: UK_SUPPLY_ELASTICITY.copy())
    pop_share_raw: np.ndarray = field(default_factory=lambda: UK_POP_SHARE_RAW.copy())
    base_house_price_intercept: float = 180_000.0   # UK 2005 average, GBP-equivalent
    base_house_price_slope: float = 150_000.0       # London premium (productivity range 0,78 to 1,65)
    price_slope: float = 0.0579                     # LOCKED from German SMM
    price_noise: float = 0.008

    @property
    def region_type(self) -> np.ndarray:
        return UK_REGION_TYPE.copy()

    @property
    def region_label(self) -> np.ndarray:
        return np.array(list(UK_REGION_LABELS))


# ---------------------------------------------------------------------------
# UK Config builder with locked German behavioural parameters
# ---------------------------------------------------------------------------


def make_uk_config(seed: int = 73, n_periods: int = 14, n_agents: int = 30_000) -> Config:
    """Build a Config for UK validation.

    The eight SMM-identified behavioural parameters are pinned to their
    German values from `GERMAN_SMM_PARAMS`. UK-specific primitives override:
      - regional config: 12 NUTS-1 regions, UK productivity, elasticity, pop shares
      - bequest_tax_rate = 0,40 (UK Inheritance Tax effective rate)
      - n_periods = 14 (2005 to 2019 simulation horizon)

    Notes:
      - beta_network stays at the model default 0,6 because it is not in
        the SMM free-parameter space.
      - The initial wealth distribution parameters stay at German defaults;
        UK Wealth and Assets Survey shows a similar lognormal-plus-Pareto
        structure with somewhat lower Gini, but the SMM does not free
        init_wealth_sigma so it remains at the German calibration value.
      - Policy block defaults to no activation (UK validation is a
        baseline run; no policy counterfactual is applied).
    """
    voting = VotingConfig(
        rho_aspiration=GERMAN_SMM_PARAMS["rho_aspiration"],
        alpha_local=GERMAN_SMM_PARAMS["alpha_local"],
        beta_0=GERMAN_SMM_PARAMS["beta_0"],
        beta_dissat=GERMAN_SMM_PARAMS["beta_dissat"],
        beta_network=0.6,
        beta_renter=GERMAN_SMM_PARAMS["beta_renter"],
    )

    # Demographic config keeps Gompertz mortality from the German default
    # (UK life tables similar within 1pp) and the assortative exponent and
    # intergenerational skill correlation at their SMM-identified values.
    # The bequest tax rate moves to the UK Inheritance Tax effective rate.
    demographic = DemographicConfig(
        bequest_tax_rate=0.40,
        assortative_exponent=GERMAN_SMM_PARAMS["assortative_exponent"],
        intergenerational_skill_corr=GERMAN_SMM_PARAMS["intergenerational_skill_corr"],
    )

    # price_slope is a SMM-identified parameter, not a UK primitive; the
    # UKRegionalConfig dataclass default carries a truncated copy for
    # construction safety, but the canonical value lives in
    # GERMAN_SMM_PARAMS and must be applied here so the UK Config picks
    # up the full-precision optimum.
    regional = UKRegionalConfig(price_slope=GERMAN_SMM_PARAMS["price_slope"])

    return Config(
        n_agents=n_agents,
        n_periods=n_periods,
        seed=seed,
        regional=regional,
        behavioral=BehavioralConfig(),
        demographic=demographic,
        voting=voting,
        policy=PolicyConfig(),
        skill=SkillConfig(),
    )


# ---------------------------------------------------------------------------
# Sanity check: confirm the SMM behavioural parameters travel correctly
# ---------------------------------------------------------------------------


def assert_smm_parameters_intact(cfg: Config) -> None:
    """Raise if any of the eight SMM-identified parameters differ from
    their German-identified values. Defensive guard against accidental
    refitting during the UK validation."""
    checks = [
        ("beta_dissat", cfg.voting.beta_dissat),
        ("beta_renter", cfg.voting.beta_renter),
        ("rho_aspiration", cfg.voting.rho_aspiration),
        ("alpha_local", cfg.voting.alpha_local),
        ("price_slope", cfg.regional.price_slope),
        ("beta_0", cfg.voting.beta_0),
        ("assortative_exponent", cfg.demographic.assortative_exponent),
        ("intergenerational_skill_corr", cfg.demographic.intergenerational_skill_corr),
    ]
    for name, value in checks:
        target = GERMAN_SMM_PARAMS[name]
        if not np.isclose(value, target, atol=1e-9):
            raise ValueError(
                f"UK validation broken: SMM parameter {name} drifted "
                f"from {target} to {value}. Behavioural-parameter "
                f"invariance violated."
            )


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------


def _aggregate_leave(hist: History, t: int) -> float:
    return float(hist.vote_aggregate[t])


def _cross_regional_leave_price_correlation(hist: History, t: int) -> float:
    """Raw correlation across regions between Leave share and price growth."""
    leave_by_region = hist.vote[t]
    price_growth = hist.price[t] / hist.price[0] - 1.0
    if leave_by_region.std(ddof=1) == 0 or price_growth.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(leave_by_region, price_growth)[0, 1])


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation between x and y controlling for z.
    Apples-to-apples comparator for Adler-Ansell (2020) reported
    income-controlled correlations.

    Returns NaN if any input has zero variance."""
    if x.std(ddof=1) == 0 or y.std(ddof=1) == 0 or z.std(ddof=1) == 0:
        return float("nan")
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt(max((1.0 - r_xz**2) * (1.0 - r_yz**2), 1e-12))
    return float((r_xy - r_xz * r_yz) / denom)


def _cross_regional_leave_price_partial_correlation(hist: History, t: int) -> float:
    """Cross-regional correlation between Leave share and price growth,
    controlling for regional mean income. This is the moment Adler and
    Ansell (2020) report; the raw correlation tends to be larger in
    magnitude because price growth and income covary."""
    leave_by_region = hist.vote[t]
    price_growth = hist.price[t] / hist.price[0] - 1.0
    income_by_region = hist.mean_income[t]
    return _partial_correlation(price_growth, leave_by_region, income_by_region)


def _within_region_renter_owner_gap(hist: History, t: int) -> float:
    rent_v = hist.vote_by_tenure[t, :, 0]
    own_v = hist.vote_by_tenure[t, :, 1]
    return float((rent_v - own_v).mean())


def _aggregate_price_growth(cfg: Config, hist: History, t: int) -> float:
    pop_w = cfg.regional.pop_share
    initial = float((hist.price[0] * pop_w).sum())
    later = float((hist.price[t] * pop_w).sum())
    return later / initial - 1.0


def _regional_dispersion(hist: History, t: int) -> float:
    return float(hist.vote[t].std(ddof=1))


def run_uk_validation(
    seeds: Sequence[int] = (73, 74, 75, 76, 77),
    n_periods: int = 14,
    brexit_t: int = BREXIT_PERIOD,
) -> dict:
    """Simulate UK calibration across seeds and compute validation targets.

    Returns a dict with the three primary targets, the diagnostic
    regional Leave-share comparison, and the German-tension diagnostics
    (aggregate price growth, cross-regional dispersion)."""
    runs: list[tuple[Config, History]] = []
    for s in seeds:
        cfg = make_uk_config(seed=s, n_periods=n_periods)
        assert_smm_parameters_intact(cfg)
        _, hist, _ = simulate(cfg)
        # Re-check after simulation: defensive against any state mutation.
        assert_smm_parameters_intact(cfg)
        runs.append((cfg, hist))

    leave_shares = [_aggregate_leave(h, brexit_t) for _, h in runs]
    correlations = [_cross_regional_leave_price_correlation(h, brexit_t) for _, h in runs]
    partial_correlations = [
        _cross_regional_leave_price_partial_correlation(h, brexit_t) for _, h in runs
    ]
    gaps = [_within_region_renter_owner_gap(h, brexit_t) for _, h in runs]
    agg_price_growth = [_aggregate_price_growth(c, h, brexit_t) for c, h in runs]
    dispersions = [_regional_dispersion(h, brexit_t) for _, h in runs]

    # Regional Leave share averaged across seeds.
    regional_shares = np.stack([h.vote[brexit_t] for _, h in runs]).mean(axis=0)
    regional_dict = {
        name: float(regional_shares[i])
        for i, name in enumerate(UK_REGION_NAMES)
    }

    correlations_finite = [c for c in correlations if np.isfinite(c)]
    partial_correlations_finite = [c for c in partial_correlations if np.isfinite(c)]

    return {
        "aggregate_leave_share": float(np.mean(leave_shares)),
        "aggregate_leave_share_sd": float(np.std(leave_shares, ddof=1)),
        "cross_regional_leave_price_correlation": (
            float(np.mean(correlations_finite)) if correlations_finite else float("nan")
        ),
        "cross_regional_leave_price_partial_correlation": (
            float(np.mean(partial_correlations_finite)) if partial_correlations_finite else float("nan")
        ),
        "within_region_renter_owner_gap": float(np.mean(gaps)),
        "within_region_renter_owner_gap_sd": float(np.std(gaps, ddof=1)),
        "aggregate_price_growth_to_brexit": float(np.mean(agg_price_growth)),
        "cross_regional_dispersion": float(np.mean(dispersions)),
        "regional_leave_model": regional_dict,
        "regional_leave_empirical": dict(EMPIRICAL_LEAVE_BY_REGION),
        "brexit_period": int(brexit_t),
        "n_seeds": len(seeds),
        "n_periods": int(n_periods),
    }


def score_primary_targets(result: dict) -> dict:
    """Apply the three primary acceptance criteria to a validation result.

    The cross-regional correlation uses the income-controlled partial
    correlation as the apples-to-apples comparator with Adler and
    Ansell (2020), who report partial correlations controlling for
    regional income."""
    leave = result["aggregate_leave_share"]
    partial_corr = result["cross_regional_leave_price_partial_correlation"]
    raw_corr = result["cross_regional_leave_price_correlation"]
    gap = result["within_region_renter_owner_gap"]

    return {
        "aggregate_leave_share": {
            "target": EMPIRICAL_AGGREGATE_LEAVE,
            "model": leave,
            "acceptance_range": (0.45, 0.55),
            "passed": 0.45 <= leave <= 0.55,
        },
        "cross_regional_leave_price_correlation": {
            "target": -0.20,
            "model": partial_corr,
            "model_raw_correlation": raw_corr,
            "acceptance_range": (-0.30, -0.10),
            "passed": np.isfinite(partial_corr) and -0.30 <= partial_corr <= -0.10,
        },
        "within_region_renter_owner_gap": {
            "target": 0.15,
            "model": gap,
            "acceptance_range": (0.10, float("inf")),
            "passed": gap >= 0.10,
        },
    }
