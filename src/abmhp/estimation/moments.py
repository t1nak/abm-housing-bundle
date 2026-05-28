"""Calibration, validation, and diagnostic moments for SMM identification.

Three blocks of moments:

  - **Calibration** (12 moments, summed weight 1,0). Used in the SMM
    objective. Sources: Bundesbank PHF wave 4 (2017) for wealth shares,
    Mikrozensus 2022 for homeownership, vdpResearch Haeuserpreisindex
    2010 to 2024 for housing prices, Hilber-Vermeulen (2016) for the
    price/supply-elasticity correlation anchor, and Bundeswahlleiter for
    election results.

  - **Validation** (4 moments). Scored after SMM at the identified
    parameterisation; NOT used in identification. The renter-owner gap
    and price-growth/vote correlation come from joint Bundeswahlleiter
    and survey/price data; the bottom-quartile wage growth comes from
    Destatis Verdienste; the UK Brexit correlation is reserved for
    Prompt 7.

  - **Diagnostic** (3 moments). Internal model quantities reported
    alongside the SMM optimum for paper-section context, not used in
    identification or external validation.

Every moment carries its empirical value, an explicit `source` citation
suitable for the paper's identification section, a `category`, a `block`
(distributional / housing_dynamics / political_economy for the calibration
moments), an SMM `weight` (zero for validation and diagnostic), and a
`target_tolerance` for pass/fail reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..config import Config
from ..simulation import History, simulate


# ---------------------------------------------------------------------------
# Moment dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Moment:
    """A single moment with empirical target, source, and evaluator.

    The evaluator is a callable `evaluator(runs)` where `runs` is a list of
    (Config, History) tuples produced by `simulate_seeds`. The evaluator
    returns a scalar value computed from the multi-seed simulation.
    """
    name: str
    value: float
    source: str
    category: str  # "calibration" | "validation" | "diagnostic"
    block: str  # "distributional" | "housing_dynamics" | "political_economy" | "validation" | "diagnostic"
    weight: float  # zero for validation and diagnostic
    target_tolerance: float
    evaluator: Callable[[Sequence[tuple[Config, History]]], float] = field(repr=False)


# ---------------------------------------------------------------------------
# Multi-seed simulation helper
# ---------------------------------------------------------------------------


def simulate_seeds(cfg: Config, seeds: Sequence[int]) -> list[tuple[Config, History]]:
    """Simulate the ABM at multiple seeds with otherwise identical cfg.

    Returns a list of (Config-at-seed, History) tuples. Each Config is the
    base cfg with `seed` overridden so downstream code can read the seed
    if needed."""
    from dataclasses import replace as _replace
    out = []
    for s in seeds:
        cfg_s = _replace(cfg, seed=int(s))
        _, hist, _ = simulate(cfg_s)
        out.append((cfg_s, hist))
    return out


# ---------------------------------------------------------------------------
# Evaluator implementations (block 1: distributional)
# ---------------------------------------------------------------------------


def _final(arr: np.ndarray) -> float:
    return float(arr[-1])


def _eval_wealth_gini(runs):
    return float(np.mean([_final(h.gini) for _, h in runs]))


def _eval_top1(runs):
    return float(np.mean([_final(h.top1) for _, h in runs]))


def _eval_top10(runs):
    return float(np.mean([_final(h.top10) for _, h in runs]))


def _eval_bottom50(runs):
    return float(np.mean([_final(h.bottom50) for _, h in runs]))


def _eval_homeownership(runs):
    return float(np.mean([_final(h.ownership_aggregate) for _, h in runs]))


# ---------------------------------------------------------------------------
# Evaluator implementations (block 2: housing dynamics)
# ---------------------------------------------------------------------------


def _eval_aggregate_price_growth(runs):
    """Population-weighted aggregate price growth from t=0 to t=T."""
    vals = []
    for cfg, h in runs:
        pop_w = cfg.regional.pop_share
        agg_initial = float((h.price[0] * pop_w).sum())
        agg_final = float((h.price[-1] * pop_w).sum())
        vals.append(agg_final / agg_initial - 1.0)
    return float(np.mean(vals))


def _eval_cross_regional_price_growth_sd(runs):
    """Cross-regional standard deviation of cumulative price growth."""
    vals = []
    for _, h in runs:
        growth = h.price[-1] / h.price[0] - 1.0
        vals.append(float(growth.std(ddof=1)))
    return float(np.mean(vals))


def _eval_price_supply_elasticity_correlation(runs):
    """Cross-regional correlation between cumulative price growth and
    supply elasticity. Hilber-Vermeulen-style anchor; should be negative
    because constrained-supply regions show larger price growth."""
    vals = []
    for cfg, h in runs:
        growth = h.price[-1] / h.price[0] - 1.0
        elasticity = cfg.regional.supply_elasticity
        if growth.std() > 0 and elasticity.std() > 0:
            vals.append(float(np.corrcoef(growth, elasticity)[0, 1]))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Evaluator implementations (block 3: political economy)
# ---------------------------------------------------------------------------


def _eval_aggregate_extreme_share_final(runs):
    return float(np.mean([_final(h.vote_aggregate) for _, h in runs]))


def _eval_extreme_share_year(year: int):
    """Factory: returns evaluator for the aggregate extreme-share vote
    recorded at simulation period `year`. Uses interpolation if the
    requested year exceeds the simulated horizon."""
    def _eval(runs):
        vals = []
        for _, h in runs:
            T_plus_one = h.vote_aggregate.shape[0]
            idx = min(year, T_plus_one - 1)
            vals.append(float(h.vote_aggregate[idx]))
        return float(np.mean(vals))
    return _eval


def _eval_cross_regional_extreme_share_dispersion(runs):
    """Standard deviation of the regional extreme-share vote at T."""
    return float(np.mean([float(h.vote[-1].std(ddof=1)) for _, h in runs]))


# ---------------------------------------------------------------------------
# Evaluator implementations (validation moments)
# ---------------------------------------------------------------------------


def _eval_within_region_renter_owner_gap(runs):
    """Renter-minus-owner extreme-share vote, averaged across regions
    and seeds, at T."""
    gaps = []
    for _, h in runs:
        rent_v = h.vote_by_tenure[-1, :, 0]
        own_v = h.vote_by_tenure[-1, :, 1]
        gaps.append(float((rent_v - own_v).mean()))
    return float(np.mean(gaps))


def _eval_cross_regional_extreme_share_price_growth_correlation(runs):
    """Cross-regional correlation between cumulative price growth and the
    regional extreme-share vote at T. Empirical sign expectation is
    negative: higher-price-growth regions (urban superstars) show lower
    right-exit shares than declining regions."""
    vals = []
    for _, h in runs:
        growth = h.price[-1] / h.price[0] - 1.0
        vote = h.vote[-1]
        if growth.std() > 0 and vote.std() > 0:
            vals.append(float(np.corrcoef(growth, vote)[0, 1]))
    return float(np.mean(vals)) if vals else 0.0


def _partial_correlation_three(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation between x and y controlling for z.

    Same specification used in the UK external stress test
    (`validation/uk.py::_partial_correlation`). Reproduced here so the
    German placebo regression and the UK regression share an identical
    functional form: cross-regional regression of regional vote on
    regional price growth, conditional on regional mean income.

    Returns NaN if any input has zero variance.
    """
    if x.std(ddof=1) == 0 or y.std(ddof=1) == 0 or z.std(ddof=1) == 0:
        return float("nan")
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt(max((1.0 - r_xz**2) * (1.0 - r_yz**2), 1e-12))
    return float((r_xy - r_xz * r_yz) / denom)


def _eval_cross_regional_extreme_share_price_growth_partial_correlation(runs):
    """German placebo for the Adler-Ansell (2020) regression.

    Cross-regional partial correlation between extreme-share vote at T and
    cumulative price growth, controlling for regional mean income at T.
    Same functional form as the UK partial correlation in
    `validation/uk.py::_cross_regional_leave_price_partial_correlation`.

    Empirical sign expectation in Adler-Ansell (2020) applied to the
    German data: negative. The placebo asks whether the German simulated
    data produces the empirically correct conditional sign before any UK
    work is done; if the German conditional sign is wrong, the UK sign
    mismatch is not country-specific.
    """
    vals = []
    for _, h in runs:
        growth = h.price[-1] / h.price[0] - 1.0
        vote = h.vote[-1]
        income = h.mean_income[-1]
        pc = _partial_correlation_three(growth, vote, income)
        if np.isfinite(pc):
            vals.append(pc)
    return float(np.mean(vals)) if vals else float("nan")


def _eval_bottom_quartile_wage_growth(runs):
    """Cumulative bottom-quartile wage growth from t=0 to T.

    The model tracks regional mean income only at the regional level.
    We use the lowest-productivity region's income growth as a stand-in
    proxy for bottom-quartile wage growth at the aggregate level."""
    vals = []
    for cfg, h in runs:
        # The lowest-productivity regions are the trailing four (declining).
        decl_mask = cfg.regional.region_type == "decl"
        if not decl_mask.any():
            continue
        decl_initial = float(h.mean_income[0, decl_mask].mean())
        decl_final = float(h.mean_income[-1, decl_mask].mean())
        if decl_initial > 0:
            vals.append(decl_final / decl_initial - 1.0)
    return float(np.mean(vals)) if vals else 0.0


def _eval_uk_brexit_region_price_growth_correlation(runs):
    """Placeholder: scored in Prompt 7 against UK data, not from the
    German ABM. Returns NaN so the table can list it without giving a
    spurious value at the German calibration."""
    _ = runs
    return float("nan")


# ---------------------------------------------------------------------------
# Evaluator implementations (diagnostic moments)
# ---------------------------------------------------------------------------


def _eval_p3_incomplete_material_repair(runs):
    """Diagnostic placeholder: documented at -0,113 from
    `outputs/material_security_results.md` (Scenario E vs Scenario C
    at T=25, 10-seed mean). The full counterfactual is in
    scripts/counterfactual_material_security.py and is too expensive
    to re-run inside SMM; we report the documented value."""
    _ = runs
    return -0.113


def _eval_hank_aggregate_match_diagnostic(runs):
    """Diagnostic placeholder: HANK matches the aggregate extreme-share
    by construction (calibration target 0,208 hit to within 0,001 in
    `src/abmhp/hank_benchmark.py::calibrate`)."""
    _ = runs
    return 0.208


def _eval_within_region_dissatisfaction_channel_decomposition(runs):
    """Diagnostic: the housing-cost channel reduction in Scenario C
    versus baseline at central calibration. Documented at approximately
    -0,35 (35 percent below baseline) in the policy block.
    Reported for paper context."""
    _ = runs
    return -0.35


# ---------------------------------------------------------------------------
# Moment table
# ---------------------------------------------------------------------------


CALIBRATION_MOMENTS: tuple[Moment, ...] = (
    # Block 1: distributional (summed weight 0,35)
    Moment(
        name="wealth_gini",
        value=0.81,
        source="Bundesbank PHF wave 4 (2017), Vermögen privater Haushalte",
        category="calibration",
        block="distributional",
        weight=0.08,
        target_tolerance=0.03,
        evaluator=_eval_wealth_gini,
    ),
    Moment(
        name="top_1_wealth_share",
        value=0.25,
        source="Bundesbank PHF wave 4 (2017), top-1 percent wealth share",
        category="calibration",
        block="distributional",
        weight=0.08,
        target_tolerance=0.05,
        evaluator=_eval_top1,
    ),
    Moment(
        name="top_10_wealth_share",
        value=0.60,
        source="Bundesbank PHF wave 4 (2017), top-10 percent wealth share",
        category="calibration",
        block="distributional",
        weight=0.07,
        target_tolerance=0.05,
        evaluator=_eval_top10,
    ),
    Moment(
        name="bottom_50_wealth_share",
        value=0.03,
        source="Bundesbank PHF wave 4 (2017), bottom-50 percent wealth share",
        category="calibration",
        block="distributional",
        weight=0.06,
        target_tolerance=0.02,
        evaluator=_eval_bottom50,
    ),
    Moment(
        name="homeownership_rate",
        value=0.50,
        source="Mikrozensus 2022, Wohnungseigentumsquote",
        category="calibration",
        block="distributional",
        weight=0.06,
        target_tolerance=0.03,
        evaluator=_eval_homeownership,
    ),
    # Block 2: housing dynamics (summed weight 0,25)
    Moment(
        name="aggregate_price_growth_15y",
        value=0.80,
        source="vdpResearch Haeuserpreisindex 2010 to 2024, aggregate cumulative",
        category="calibration",
        block="housing_dynamics",
        weight=0.10,
        target_tolerance=0.10,
        evaluator=_eval_aggregate_price_growth,
    ),
    Moment(
        name="cross_regional_price_growth_sd",
        value=0.50,
        source="vdpResearch Haeuserpreisindex 2010 to 2024, by Bundesland",
        category="calibration",
        block="housing_dynamics",
        weight=0.08,
        target_tolerance=0.10,
        evaluator=_eval_cross_regional_price_growth_sd,
    ),
    Moment(
        name="price_growth_supply_elasticity_correlation",
        value=-0.70,
        source="Hilber and Vermeulen (2016) style empirical anchor for Germany",
        category="calibration",
        block="housing_dynamics",
        weight=0.07,
        target_tolerance=0.15,
        evaluator=_eval_price_supply_elasticity_correlation,
    ),
    # Block 3: political economy (summed weight 0,40)
    Moment(
        name="aggregate_extreme_share_final",
        value=0.208,
        source="Bundeswahlleiter, 23 February 2025 federal election, AfD second votes",
        category="calibration",
        block="political_economy",
        weight=0.18,
        target_tolerance=0.01,
        evaluator=_eval_aggregate_extreme_share_final,
    ),
    Moment(
        name="extreme_share_year_5",
        value=0.10,
        source="Bundeswahlleiter, 2017 Bundestagswahl AfD share (cohort anchor)",
        category="calibration",
        block="political_economy",
        weight=0.06,
        target_tolerance=0.03,
        evaluator=_eval_extreme_share_year(5),
    ),
    Moment(
        name="extreme_share_year_10",
        value=0.15,
        source="Bundeswahlleiter, 2021 Bundestagswahl AfD share (cohort anchor)",
        category="calibration",
        block="political_economy",
        weight=0.06,
        target_tolerance=0.03,
        evaluator=_eval_extreme_share_year(10),
    ),
    Moment(
        name="cross_regional_extreme_share_dispersion",
        value=0.08,
        source="Bundeswahlleiter Land-level standard deviation, 23 February 2025",
        category="calibration",
        block="political_economy",
        weight=0.10,
        target_tolerance=0.03,
        evaluator=_eval_cross_regional_extreme_share_dispersion,
    ),
)


VALIDATION_MOMENTS: tuple[Moment, ...] = (
    Moment(
        name="within_region_renter_owner_vote_gap",
        value=0.15,
        source="SOEP-IS / German Internet Panel tenure-by-political-affinity, 2017 to 2025 averages",
        category="validation",
        block="validation",
        weight=0.0,
        target_tolerance=0.05,
        evaluator=_eval_within_region_renter_owner_gap,
    ),
    Moment(
        name="cross_regional_extreme_share_price_growth_correlation",
        value=-0.60,
        source="Bundeswahlleiter Land-level and vdpResearch Land-level price growth, joint 2010 to 2025",
        category="validation",
        block="validation",
        weight=0.0,
        target_tolerance=0.20,
        evaluator=_eval_cross_regional_extreme_share_price_growth_correlation,
    ),
    Moment(
        name="cross_regional_extreme_share_price_growth_partial_correlation_income",
        value=-0.20,
        source=(
            "Adler and Ansell (2020) style cross-regional partial correlation, "
            "16 Bundeslaender, AfD second-vote share on price growth conditional "
            "on regional mean income. Sign-only target; matches the UK specification "
            "in validation/uk.py so the placebo is apples-to-apples."
        ),
        category="validation",
        block="validation",
        weight=0.0,
        target_tolerance=0.30,
        evaluator=_eval_cross_regional_extreme_share_price_growth_partial_correlation,
    ),
    Moment(
        name="bottom_quartile_wage_growth_2010_2025",
        value=0.04,
        source="Destatis Verdienste 2010 to 2025, real wage growth bottom quartile",
        category="validation",
        block="validation",
        weight=0.0,
        target_tolerance=0.05,
        evaluator=_eval_bottom_quartile_wage_growth,
    ),
    Moment(
        name="uk_brexit_region_price_growth_correlation",
        value=-0.20,
        source="UK 2016 EU referendum Leave share and ONS regional house prices, scored in Prompt 7",
        category="validation",
        block="validation",
        weight=0.0,
        target_tolerance=0.15,
        evaluator=_eval_uk_brexit_region_price_growth_correlation,
    ),
)


DIAGNOSTIC_MOMENTS: tuple[Moment, ...] = (
    Moment(
        name="p3_incomplete_material_repair_effect",
        value=-0.113,
        source="Model output: Scenario E vs Scenario C at T=25, 10-seed mean",
        category="diagnostic",
        block="diagnostic",
        weight=0.0,
        target_tolerance=0.02,
        evaluator=_eval_p3_incomplete_material_repair,
    ),
    Moment(
        name="hank_aggregate_match_diagnostic",
        value=0.208,
        source="HANK benchmark calibrated match (src/abmhp/hank_benchmark.py)",
        category="diagnostic",
        block="diagnostic",
        weight=0.0,
        target_tolerance=0.005,
        evaluator=_eval_hank_aggregate_match_diagnostic,
    ),
    Moment(
        name="within_region_dissatisfaction_channel_decomposition",
        value=-0.35,
        source="Model output: housing-cost channel reduction in Scenario C vs baseline",
        category="diagnostic",
        block="diagnostic",
        weight=0.0,
        target_tolerance=0.10,
        evaluator=_eval_within_region_dissatisfaction_channel_decomposition,
    ),
)


# ---------------------------------------------------------------------------
# Bulk evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_moment(m: Moment, runs: Sequence[tuple[Config, History]]) -> float:
    return float(m.evaluator(runs))


def evaluate_moments(
    moments: Sequence[Moment],
    runs: Sequence[tuple[Config, History]],
) -> dict[str, float]:
    return {m.name: evaluate_moment(m, runs) for m in moments}


def calibration_weight_sum() -> float:
    return float(sum(m.weight for m in CALIBRATION_MOMENTS))


def assert_weights_sum_to_one(tol: float = 1e-6) -> None:
    s = calibration_weight_sum()
    if abs(s - 1.0) > tol:
        raise ValueError(f"calibration weights sum to {s:.6f}, expected 1.0")
