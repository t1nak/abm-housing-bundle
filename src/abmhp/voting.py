"""Reference-dependent aspiration formation, dissatisfaction, logit extreme-share vote.

The voting block constructs material-security dissatisfaction from three of
the four channels in the bundled-mechanism thesis. own_outcome combines:
    income           : wage component of material security
    rent_paid        : consumption stress (one of the three bundled mechanisms)
    cap_gain         : asset exclusion (gating between owners and renters,
                       via the tenure status that determines whether cap_gain
                       is non-zero)
The third bundled mechanism, intergenerational closure, operates through the
bequest pathway in demography.py / mortality.py: housing wealth transmits
unequally across cohorts, which closes off asset accumulation for households
without family transfers. The bequest mechanism feeds wealth into the voting
block via state.wealth and through cap_gain dynamics on inherited housing.

local_extreme_share is the code-level variable. Its German calibration
interpretation is the right-exit share; the theoretical object is mainstream
exit; the empirical proxy for the calibration is the AfD vote share. See
paper/framing_v2.md for the full hierarchy.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .household import HouseholdState


def step_voting(
    state: HouseholdState,
    cap_gain: np.ndarray,
    rent_paid: np.ndarray,
    transfer_received: np.ndarray,
    housing_value_pre: np.ndarray,
    house_price: np.ndarray,
    regional_price_gain: np.ndarray,
    period: int,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """Updates aspiration, computes dissatisfaction, draws extreme-share votes.
    Returns the regional dissatisfaction mean for recording.

    own_outcome reads income net of current-period housing cost, plus realised
    housing capital gains. This is the channel by which a housing-only
    mainstream response (or its leakage) feeds into political dissatisfaction:
    renters facing a falling rent index keep more of their income inside
    own_outcome and so register less dissatisfaction against their aspiration
    anchor."""
    voting = cfg.voting
    regional = cfg.regional
    behavioral = cfg.behavioral

    # Tier-1 diagnostic feature flags (defaults reproduce the previous model):
    # - realised_gain_share scales the capital-gain term in own_outcome.
    # - user_cost_rate charges a Poterba-style flow cost on gross levered
    #   housing value. The housing_value_pre argument is the pre-transition
    #   housing base (wealth times housing_share, measured before this
    #   period's buy/sell transitions and wealth update), passed in from
    #   step_wealth_and_ownership. Using the pre-transition base keeps the
    #   user-cost charge accounted on the same housing position from which
    #   cap_gain was earned: new buyers do not get charged, new sellers do.
    realised_gain_share = behavioral.realised_gain_share
    user_cost_rate = behavioral.user_cost_rate

    if user_cost_rate > 0.0:
        user_cost = user_cost_rate * housing_value_pre
    else:
        user_cost = np.zeros_like(state.income)

    # Realised outcome: wage, plus redistributive transfer received from the
    # integrated material-security intervention, minus rent (consumption
    # stress channel), plus realised housing capital gains (asset-exclusion
    # channel via tenure), minus owner user cost when active. The
    # aspirational reference is the wage benchmark only; transfers, capital
    # gains, and user cost are policy- or accounting-modulated lines relative
    # to it.
    own_outcome = (
        state.income
        + transfer_received
        - rent_paid
        + realised_gain_share * cap_gain
        - user_cost
    )
    aspirational_reference = state.income

    # Reference-rule selector. Default "akerlof_yellen" is the normative
    # external anchor: AR(1) toward p75 of regional/national income (the
    # existing headline specification, bit-identical to the prior code).
    # "koszegi_rabin" is the rational-expectations anchor: AR(1) toward
    # the agent's own realised outcome.
    if voting.reference_rule == "akerlof_yellen":
        # Aspiration reference quantile; default 0.75 reproduces the previous
        # specification, Tier 2C diagnostic sets 0.50 (regional median).
        q_pct = 100.0 * float(voting.aspiration_reference_quantile)
        regional_p = np.zeros(regional.n_regions)
        for r in range(regional.n_regions):
            mask = state.region == r
            if mask.any():
                regional_p[r] = np.percentile(aspirational_reference[mask], q_pct)
        national_p = float(np.percentile(aspirational_reference, q_pct))

        target = (
            voting.alpha_local * regional_p[state.region]
            + (1.0 - voting.alpha_local) * national_p
        )
        state.aspiration = voting.rho_aspiration * state.aspiration + (1.0 - voting.rho_aspiration) * target
    elif voting.reference_rule == "koszegi_rabin":
        # K-R: per-agent anchor on own realised outcome. No external
        # normative target; no regional/national quantile. Same AR(1)
        # persistence as AY so the two specifications differ only in the
        # anchor variable.
        state.aspiration = voting.rho_aspiration * state.aspiration + (1.0 - voting.rho_aspiration) * own_outcome
    else:
        raise ValueError(
            f"unknown reference_rule {voting.reference_rule!r}; "
            f"expected 'akerlof_yellen' or 'koszegi_rabin'"
        )

    dissat = np.maximum(0.0, state.aspiration - own_outcome) / np.maximum(state.aspiration, 1.0)
    dissat = np.clip(dissat, 0.0, 1.0)

    local_extreme_share = np.zeros(regional.n_regions)
    for r in range(regional.n_regions):
        mask = state.region == r
        if mask.any():
            local_extreme_share[r] = state.extreme_voter[mask].mean()

    renter = (~state.homeowner).astype(float)
    margin_means = None  # (d_rent, d_asset, d_access) population means when decomposed

    # --- Three-margin housing-pressure decomposition (rebuild) ---
    # Each margin is a normalised positive gap in [0, 1]; gamma_* are the
    # per-margin voting responses. The single beta_dissat * dissat term is
    # used only when margin_decomposition is off (legacy specification).
    if voting.margin_decomposition:
        beh = cfg.behavioral
        A = np.maximum(state.aspiration, 1.0)
        # 1. Rent burden / consumption stress: after-rent income shortfall
        #    relative to the regional income aspiration.
        after_rent = state.income - rent_paid
        if voting.rent_margin_spec == "baseline":
            d_rent = np.clip(np.maximum(0.0, A - after_rent) / A, 0.0, 1.0)
        elif voting.rent_margin_spec == "incremental":
            # Construct-validity variant: only the increment in the
            # aspiration shortfall CAUSED by rent (excludes the pre-rent
            # income shortfall a low-income household would register even
            # at zero rent).
            gap_with = np.maximum(0.0, A - after_rent)
            gap_without = np.maximum(0.0, A - state.income)
            d_rent = np.clip((gap_with - gap_without) / A, 0.0, 1.0)
        elif voting.rent_margin_spec == "rent_to_income":
            # Construct-validity variant: conventional rent-burden measure,
            # normalised by the Eurostat 40% housing-cost overburden
            # threshold. No aspiration reference.
            d_rent = np.clip(
                rent_paid / np.maximum(state.income, 1.0) / 0.40, 0.0, 1.0)
        else:
            raise ValueError(
                f"unknown rent_margin_spec {voting.rent_margin_spec!r}")
        if voting.rent_margin_renters_only:
            # Baseline (paper): rent stress is gated on renters, matching
            # the tenure gate carried by the other two margins.
            d_rent = renter * d_rent
        # 2. Asset exclusion: non-owners shut out of sustained regional
        #    house-price appreciation (smoothed multi-period log growth).
        d_asset = renter * np.clip(
            np.maximum(0.0, regional_price_gain[state.region]), 0.0, 1.0)
        # 3. Ownership-access exclusion: non-owners' wealth shortfall below the
        #    regional buy threshold (down payment). Family wealth is the
        #    mechanism that lets some households close this gap.
        buy_threshold = np.maximum(
            beh.buy_wealth_to_price * house_price[state.region], 1.0)
        d_access = renter * np.clip(
            np.maximum(0.0, buy_threshold - state.wealth) / buy_threshold, 0.0, 1.0)
        if voting.access_margin_include_income:
            # Construct-validity variant: the access margin reflects both
            # eligibility constraints of the tenure block (wealth AND
            # income tests), taking the larger of the two shortfalls.
            inc_threshold = np.maximum(
                beh.buy_income_to_price * house_price[state.region], 1.0)
            inc_gap = np.clip(
                np.maximum(0.0, inc_threshold - state.income) / inc_threshold,
                0.0, 1.0)
            d_access = renter * np.maximum(d_access, inc_gap)
        margin_term = (voting.gamma_rent * d_rent
                       + voting.gamma_asset * d_asset
                       + voting.gamma_access * d_access)
        # Record the unweighted mean of the three margins as dissatisfaction.
        dissat = np.clip((d_rent + d_asset + d_access) / 3.0, 0.0, 1.0)
        margin_means = (float(d_rent.mean()), float(d_asset.mean()),
                        float(d_access.mean()))
    else:
        margin_term = voting.beta_dissat * dissat

    # Track-B cosmopolitan-proxy pilot: optional region-specific intercept shift.
    # The shift vector is precomputed externally as
    #     gamma * (grad_share_r - mean_grad_share_country)
    # so a None shift (or gamma = 0) reproduces the scalar beta_0 model exactly.
    if voting.cosmopolitan_shift_by_region is None:
        beta_0_eff = voting.beta_0
    else:
        shift_array = np.asarray(voting.cosmopolitan_shift_by_region, dtype=float)
        if shift_array.shape[0] != regional.n_regions:
            raise ValueError(
                "cosmopolitan_shift_by_region length "
                f"{shift_array.shape[0]} does not match n_regions {regional.n_regions}"
            )
        beta_0_eff = voting.beta_0 + shift_array[state.region]

    # Preset (exogenous) fixed effects: time FE absorb national election-cycle
    # shocks; region FE absorb stable political geography.
    fe_t = 0.0
    if (voting.time_fixed_effects is not None
            and 0 <= period < len(voting.time_fixed_effects)):
        fe_t = float(voting.time_fixed_effects[period])
    if voting.region_fixed_effects is not None:
        fe_r = np.asarray(voting.region_fixed_effects, dtype=float)[state.region]
    else:
        fe_r = 0.0

    logit = (
        beta_0_eff
        + margin_term
        + voting.beta_network * local_extreme_share[state.region]
        + voting.beta_renter * renter
        + fe_t
        + fe_r
    )
    p_ext = 1.0 / (1.0 + np.exp(-logit))
    state.extreme_voter = rng.uniform(0.0, 1.0, size=p_ext.shape) < p_ext

    regional_dissat = np.zeros(regional.n_regions)
    for r in range(regional.n_regions):
        mask = state.region == r
        if mask.any():
            regional_dissat[r] = dissat[mask].mean()
    return regional_dissat, margin_means
