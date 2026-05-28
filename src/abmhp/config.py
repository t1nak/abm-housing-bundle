"""Parameter container for the housing and mainstream-exit ABM."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class RegionalConfig:
    n_regions: int = 16
    productivity: np.ndarray = field(
        default_factory=lambda: np.concatenate([
            np.array([1.50, 1.32, 1.22, 1.15]),
            np.ones(8) * 1.00,
            np.array([0.82, 0.76, 0.72, 0.68]),
        ])
    )
    supply_elasticity: np.ndarray = field(
        default_factory=lambda: np.concatenate([
            np.array([0.22, 0.28, 0.34, 0.40]),
            np.ones(8) * 0.85,
            np.array([1.40, 1.70, 2.00, 2.30]),
        ])
    )
    pop_share_raw: np.ndarray = field(
        default_factory=lambda: np.concatenate([
            np.array([0.12, 0.10, 0.08, 0.08]),
            np.ones(8) * 0.045,
            np.array([0.06, 0.055, 0.05, 0.045]),
        ])
    )
    base_house_price_intercept: float = 140_000.0
    base_house_price_slope: float = 80_000.0
    price_slope: float = 0.075
    price_noise: float = 0.008

    # Regional supply-shock AR(1): an exogenous, region-specific component
    # of price growth orthogonal to the productivity gradient. Models supply
    # restrictions, regional credit conditions, and planning shocks that
    # decouple price growth from the productivity-driven wealth-pressure
    # channel. Innovation sd 0,012 is calibrated so that the channel
    # contributes roughly 25 percent of cross-regional price-growth variance
    # in the long run, matching empirical decompositions of regional house-
    # price variation (typically 60 to 70 percent productivity-driven, 25
    # to 35 percent supply/credit/planning shock-driven). Without this
    # channel, regional price growth is collinear with regional income by
    # construction and partialling out income removes all price variation.
    regional_shock_ar: float = 0.85
    regional_shock_innovation_sd: float = 0.012
    regional_shock_persistence: float = 1.0

    @property
    def pop_share(self) -> np.ndarray:
        s = self.pop_share_raw
        return s / s.sum()

    @property
    def region_type(self) -> np.ndarray:
        return np.array(["super"] * 4 + ["avg"] * 8 + ["decl"] * 4)

    @property
    def region_label(self) -> np.ndarray:
        return np.array(
            [f"S{i+1}" for i in range(4)]
            + [f"A{i+1}" for i in range(8)]
            + [f"D{i+1}" for i in range(4)]
        )


@dataclass(frozen=True)
class BehavioralConfig:
    g_wage_high: float = 0.026
    g_wage_low: float = -0.006
    rent_yield: float = 0.040
    rent_burden_share: float = 0.32
    cost_inflation: float = 0.022
    income_shock_sd: float = 0.018
    wealth_shock_prob: float = 0.025
    wealth_shock_size: float = 0.30
    debt_floor: float = -80_000.0

    # Wealth-threshold real returns on positive financial wealth.
    return_schedule: tuple = (
        (50_000.0, -0.015),
        (200_000.0, 0.005),
        (1_000_000.0, 0.040),
        (5_000_000.0, 0.075),
        (np.inf, 0.105),
    )
    debt_interest: float = 0.045

    # Tier-1 diagnostic feature flags. Defaults preserve the current model.
    # realised_gain_share scales the capital-gain term entering own_outcome
    # (1.0 = current; 0.0 = exclude unrealised gains entirely).
    # user_cost_rate multiplies gross levered housing value (wealth times
    # housing_share for owners) to charge a Poterba-style flow cost to
    # owner own_outcome (0.0 = current; 0.04 = standard housing-economics
    # benchmark, interest plus maintenance net of tax deductibility).
    realised_gain_share: float = 1.0
    user_cost_rate: float = 0.0

    # Housing exposure tiers (homeowners only).
    housing_share_tier_low: float = 0.50
    housing_share_tier_mid: float = 1.10
    housing_share_tier_high: float = 2.00
    housing_tier_mid_threshold: float = 500_000.0
    housing_tier_high_threshold: float = 2_000_000.0

    # Savings.
    savings_floor: float = 0.02
    savings_ceiling_extra: float = 0.78
    savings_exponent: float = 2.0
    base_consumption: float = 10_000.0

    # Ownership transitions.
    buy_wealth_to_price: float = 0.30
    buy_income_to_price: float = 0.10
    buy_probability: float = 0.35
    sell_wealth_to_price: float = 0.04

    # Assortative help at first-ownership entry. Operationalises the
    # intergenerational closure channel at the correct life-stage
    # (parental help with first-home purchase).
    #
    # Recipient targeting: near-eligible renters whose wealth lies in
    # [assortative_help_wealth_lower_factor * price, buy_wealth_to_price
    # * price] (both with friction multiplier applied for policy-regime
    # consistency) and whose income passes the buy-income test. These
    # are renters who would not otherwise be able to buy this period
    # but would cross the buy-wealth threshold given a sufficient boost.
    #
    # Each near-eligible renter is offered help with per-period
    # probability p_assortative_help. The transfer amount is
    #     min(buy_threshold - wealth + epsilon, f_assortative_help
    #          * donor.wealth)
    # so the recipient lands just over the buy threshold without
    # exceeding the cap as a fraction of donor wealth.
    #
    # Donor pool: alive agents with age >= 55 and wealth above the
    # national median, sampled with weights wealth ** assortative_exponent.
    # Donors are sampled without replacement within a period: each
    # donor donates at most once per period. If the donor pool is
    # exhausted before all offered events are processed, remaining
    # events are dropped (no double-dipping).
    #
    # The transfer is wealth-conserving: donor.wealth is decremented
    # by the same amount the recipient receives.
    #
    # p and f are placeholder defaults; refine against SOEP
    # intergenerational-transfer summaries in a future revision.
    # TODO: cross-check p_assortative_help and f_assortative_help
    # against SOEP intergenerational-transfer wave (Schupp et al.).
    assortative_help_enabled: bool = False
    p_assortative_help: float = 0.05
    f_assortative_help: float = 0.15
    assortative_help_wealth_lower_factor: float = 0.20
    assortative_help_donor_min_age: float = 55.0

    # Augmented-model SMM flags. When False (default), the SMM free-
    # parameter space is the original 8-parameter set; when True, the
    # corresponding parameter is added to the SMM optimisation. These
    # flags also drive `build_param_space(cfg)` in estimation.smm.
    estimate_beta_n: bool = False
    estimate_gamma_cosmopolitan: bool = False


@dataclass(frozen=True)
class DemographicConfig:
    """Age-cohort and mortality calibration.

    Gompertz A, B fit to destatis 2023 life-table anchors:
        p(70) = 0.018, p(80) = 0.062, p(90) = 0.185.
    Mortality is empirically pinned; do not retune for distribution targets.
    """
    age_min: int = 20
    age_max: int = 85
    gompertz_A: float = 4.5e-5
    gompertz_B: float = 0.085
    mortality_cap: float = 0.40

    # Bequest assortativity and taxation. Calibrated to Bundesbank PHF wave 4
    # (2017) anchors after grid search across the three tuning levers.
    bequest_tax_rate: float = 0.22
    assortative_exponent: float = 2.10
    same_region_prob: float = 0.6

    # Replacement young agent.
    young_age: int = 20
    intergenerational_skill_corr: float = 0.7
    skill_mean: float = 0.6
    skill_innovation_sd: float = 0.25
    new_agent_wealth_mu: float = np.log(5_000.0)
    new_agent_wealth_sigma: float = 0.5

    # Initial wealth: lognormal(log(skill * scale * age), sigma_init).
    init_wealth_age_scale: float = 500.0
    init_wealth_sigma: float = 1.10


@dataclass(frozen=True)
class VotingConfig:
    rho_aspiration: float = 0.92
    alpha_local: float = 0.45
    beta_0: float = -5.2
    beta_dissat: float = 6.5
    beta_network: float = 0.6
    beta_renter: float = 0.5
    # Optional regional intercept shift for the cosmopolitan-proxy channel.
    # When set, the per-agent vote intercept becomes
    #     beta_0 + cosmopolitan_shift_by_region[r]
    # The shift vector is computed as
    #     gamma_cosmopolitan * (grad_share_r - mean_grad_share_country).
    # Two ways to drive it:
    #   1. Pass `cosmopolitan_shift_by_region` directly (the original pilot
    #      path; still supported for backward compatibility).
    #   2. Set `gamma_cosmopolitan != 0.0` together with `grad_share_data_path`;
    #      `simulate()` will materialise the shift vector at simulation init.
    # If both are set, the explicit `cosmopolitan_shift_by_region` wins.
    # When gamma_cosmopolitan == 0.0 and cosmopolitan_shift_by_region is None
    # (the defaults), the model reproduces the scalar-beta_0 specification
    # exactly. Stored as tuple for frozen-dataclass hashability; converted
    # to array inside voting.step_voting.
    cosmopolitan_shift_by_region: tuple[float, ...] | None = None
    gamma_cosmopolitan: float = 0.0
    grad_share_data_path: str | None = None

    # Quantile of regional aspirational-reference income used to anchor the
    # aspiration AR(1) target. Default 0.75 reproduces the existing p75
    # specification; Tier 2C diagnostic switches this to 0.50 (regional
    # median). Applies symmetrically to the national-level quantile so the
    # two share the same definition.
    aspiration_reference_quantile: float = 0.75

    # Reference-rule selector. The default "akerlof_yellen" reproduces the
    # headline normative-anchor specification: aspiration is an AR(1)
    # toward an external p75-of-income target. The alternative
    # "koszegi_rabin" replaces the external normative anchor with a
    # rational-expectations anchor on own outcome: aspiration is an AR(1)
    # toward each agent's own realised material outcome. The smoothing
    # constant rho_aspiration is reused; only the anchor changes.
    # See scripts/run_kr_robustness.py for the Section 9.1 diagnostic.
    reference_rule: str = "akerlof_yellen"


@dataclass(frozen=True)
class PolicyConfig:
    """Housing-only mainstream response block, activated under the
    extreme-share activation regime.

    Each instrument has a nominal intensity and a leakage parameter. The
    effective intensity that reaches the market is
        effective = nominal * (1 - leakage).
    Leakage defaults approximate the Barcelona STR paper estimates.
    """
    incumbency_threshold: float = 0.30
    deactivation_threshold: float = 0.20
    smoothing_window: int = 3

    rent_cap_intensity: float = 0.6
    supply_restriction_intensity: float = 0.5
    transaction_friction: float = 0.3

    # TODO: replace with Barcelona STR paper empirical values, currently approximate placeholders.
    rent_cap_leakage: float = 0.4
    supply_leakage: float = 0.3
    friction_leakage: float = 0.5

    # Test affordance: bypass hysteresis and pin the regime. None = use hysteresis.
    force_regime: str | None = None

    # Integrated material-security intervention: redistributive component.
    # When active in the extreme-share activation regime,
    # regime, a proportional tax tau_K is applied to positive cap_gain and
    # positive fin_yield each period and the revenue is distributed lump-sum
    # to agents in the bottom transfer_recipient_share of the wealth
    # distribution. tau_K is calibrated to deliver an aggregate transfer of
    # approximately 5 percent of aggregate household income (OECD-typical
    # redistributive intensity).
    redistribution_active: bool = False
    capital_tax_rate: float = 0.04
    transfer_recipient_share: float = 0.50


@dataclass(frozen=True)
class SkillConfig:
    skill_sigma: float = 0.55
    skill_min: float = 0.25
    skill_max: float = 5.0
    base_wage: float = 30_000.0


@dataclass(frozen=True)
class Config:
    n_agents: int = 30_000
    n_periods: int = 15
    seed: int = 73
    regional: RegionalConfig = field(default_factory=RegionalConfig)
    behavioral: BehavioralConfig = field(default_factory=BehavioralConfig)
    demographic: DemographicConfig = field(default_factory=DemographicConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
