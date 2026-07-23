"""Regional house-price formation, returns, savings, ownership transitions.

Policy-aware: the housing-only mainstream response can dampen rent growth
(`rent_cap`), flatten supply response (`supply_restriction`), and raise the
wealth bar on buying (`transaction_friction`). All three are gated through
EffectiveIntensities so leakage attrition is applied uniformly.
"""
from __future__ import annotations

import numpy as np

from .config import BehavioralConfig, Config
from .household import HouseholdState
from .policy import EffectiveIntensities


def financial_return_rate(wealth: np.ndarray, schedule: tuple) -> np.ndarray:
    out = np.zeros_like(wealth, dtype=float)
    prev_cap = -np.inf
    for cap, ret in schedule:
        mask = (wealth > prev_cap) & (wealth <= cap)
        out[mask] = ret
        prev_cap = cap
    return out


def housing_share_by_wealth(wealth: np.ndarray, homeowner: np.ndarray, beh: BehavioralConfig) -> np.ndarray:
    share = np.zeros_like(wealth, dtype=float)
    tier_low = homeowner & (wealth < beh.housing_tier_mid_threshold)
    tier_mid = homeowner & (wealth >= beh.housing_tier_mid_threshold) & (wealth < beh.housing_tier_high_threshold)
    tier_high = homeowner & (wealth >= beh.housing_tier_high_threshold)
    share[tier_low] = beh.housing_share_tier_low
    share[tier_mid] = beh.housing_share_tier_mid
    share[tier_high] = beh.housing_share_tier_high
    return share


def update_prices(
    state: HouseholdState,
    house_price: np.ndarray,
    regional_shock_state: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    policy: EffectiveIntensities,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-region price update. Under active supply restriction the effective
    elasticity is reduced, steepening the price response to demand pressure.

    A persistent regional supply-shock AR(1) is added to the price delta as
    an exogenous, productivity-orthogonal channel. The shock state is read
    inside the per-region loop; its AR(1) innovations are drawn AFTER the
    loop so the prompt-1 RNG ordering of price-noise draws is preserved.
    The new innovations consume R scalar draws per period (16 for the
    German calibration, 12 for UK validation) and extend the RNG sequence
    rather than reordering it.
    """
    regional = cfg.regional
    R = regional.n_regions
    prev_price = house_price.copy()
    new_price = house_price.copy()
    effective_elasticity = regional.supply_elasticity * (1.0 - policy.supply_restriction)
    effective_elasticity = np.maximum(effective_elasticity, 1e-3)
    for r in range(R):
        mask = state.region == r
        if not mask.any():
            continue
        local_w = state.wealth[mask].mean()
        pressure = local_w / new_price[r]
        delta = regional.price_slope * (pressure - 0.55) / effective_elasticity[r]
        delta = delta + regional.regional_shock_persistence * regional_shock_state[r]
        delta = np.clip(delta, -0.04, 0.10)
        new_price[r] *= 1.0 + delta + rng.normal(0.0, regional.price_noise)
    # Evolve the AR(1) regional-shock state. Placement is immediately after
    # the price-noise loop to preserve the existing RNG ordering for all
    # downstream draws (savings, ownership, voting, next-period mortality).
    innovations = rng.normal(0.0, regional.regional_shock_innovation_sd, size=R)
    new_shock_state = regional.regional_shock_ar * regional_shock_state + innovations
    price_growth = (new_price - prev_price) / prev_price
    return new_price, price_growth, new_shock_state


def update_rent_index(
    rent_index: np.ndarray,
    price_growth: np.ndarray,
    policy: EffectiveIntensities,
) -> np.ndarray:
    """Per-region rent level multiplier. Tracks price growth under MAINSTREAM
    and grows at (1 - rent_cap) * price_growth under the extreme-share
    activation regime. Once a cap has bitten, the lower rent level persists;
    there is no catch-up on deactivation (rent control inertia)."""
    effective_growth = (1.0 - policy.rent_cap) * price_growth
    return rent_index * (1.0 + effective_growth)


def compute_redistribution(
    wealth: np.ndarray,
    cap_gain: np.ndarray,
    fin_yield: np.ndarray,
    cfg: Config,
    policy: EffectiveIntensities,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Capital tax on positive cap_gain plus positive fin_yield, lump-sum
    redistributed to the bottom-recipient_share of the wealth distribution.

    Returns (tax_paid_per_agent, transfer_received_per_agent, diagnostics).
    Both arrays have shape (N,). Transfers go to agents at or below the
    recipient-share quantile of the pre-update wealth distribution; the
    wealth ranking is determined before any redistribution is applied to
    keep the assignment deterministic and observable.
    """
    n = wealth.size
    if policy.redistribution <= 0.0:
        zeros = np.zeros(n)
        return zeros, zeros, {"revenue": 0.0, "n_recipients": 0, "per_recipient": 0.0}

    taxable = np.maximum(cap_gain, 0.0) + np.maximum(fin_yield, 0.0)
    tax_paid = policy.redistribution * taxable
    revenue = float(tax_paid.sum())

    rec_share = cfg.policy.transfer_recipient_share
    threshold = float(np.quantile(wealth, rec_share))
    recipient_mask = wealth <= threshold
    n_recipients = int(recipient_mask.sum())
    transfer = np.zeros(n)
    per_recipient = revenue / n_recipients if n_recipients > 0 else 0.0
    transfer[recipient_mask] = per_recipient

    diagnostics = {
        "revenue": revenue,
        "n_recipients": n_recipients,
        "per_recipient": per_recipient,
    }
    return tax_paid, transfer, diagnostics


def step_wealth_and_ownership(
    state: HouseholdState,
    house_price: np.ndarray,
    price_growth: np.ndarray,
    rent_index: np.ndarray,
    initial_price: np.ndarray,
    period: int,
    cfg: Config,
    rng: np.random.Generator,
    policy: EffectiveIntensities,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update incomes, wealth, ownership for one period.

    Returns (cap_gain, rent_paid, transfer_received, housing_value_pre).
    The first three feed the voting block's own_outcome and aspirational
    reference. housing_value_pre is the gross levered housing position
    measured before tenure transitions and the wealth update, computed
    on the same pre-transition state.wealth and state.homeowner used for
    cap_gain. It is used by the voting block to charge user cost on the
    same housing base from which cap_gain was earned, so the user cost
    and cap_gain are accounted on consistent pre-transition state."""
    beh = cfg.behavioral
    regional = cfg.regional

    # Wage growth.
    skill_rank = np.argsort(np.argsort(state.skill)) / state.skill.size
    g = beh.g_wage_low + (beh.g_wage_high - beh.g_wage_low) * skill_rank
    state.income = state.income * (
        1.0 + g + rng.normal(0.0, beh.income_shock_sd, size=state.income.shape)
    )

    # National value-weighted price index for multi-region top portfolios.
    n = state.region.size
    pop_w = np.bincount(state.region, minlength=regional.n_regions) / n
    prev_price = house_price / (1.0 + price_growth)
    value_w = pop_w * prev_price
    value_w_total = value_w.sum()
    value_w = value_w / value_w_total if value_w_total > 0 else value_w
    national_pg = float((price_growth * value_w).sum())

    housing_share = housing_share_by_wealth(state.wealth, state.homeowner, beh)
    is_multi = state.homeowner & (state.wealth >= beh.housing_tier_mid_threshold)
    effective_pg = np.where(is_multi, national_pg, price_growth[state.region])

    housing_wealth = np.maximum(state.wealth, 0.0) * housing_share
    financial_wealth = state.wealth - housing_wealth
    cap_gain = housing_wealth * effective_pg
    # Pre-transition housing-value base for the voting-block user-cost charge.
    # Computed on the same state used for cap_gain so the two are accounted
    # against the same pre-transition position.
    housing_value_pre = housing_wealth.copy()

    pos_fin = np.maximum(financial_wealth, 0.0)
    neg_fin = np.minimum(financial_wealth, 0.0)
    fin_yield = financial_return_rate(pos_fin, beh.return_schedule) * pos_fin + beh.debt_interest * neg_fin

    # Rent: scale initial-period rent by the regional rent index. Under cap
    # the index grows slower than house_price, so renters keep more cash.
    rent_level = initial_price[state.region] * beh.rent_yield * rent_index[state.region]
    rent_paid = np.where(state.homeowner, 0.0, rent_level * beh.rent_burden_share)

    base_consumption = beh.base_consumption * (1.0 + beh.cost_inflation) ** period
    disposable = state.income - rent_paid - base_consumption

    wealth_rank = np.argsort(np.argsort(state.wealth)) / state.wealth.size
    savings_propensity = beh.savings_floor + beh.savings_ceiling_extra * wealth_rank ** beh.savings_exponent
    savings = disposable * savings_propensity

    tax_paid, transfer_received, _ = compute_redistribution(
        state.wealth, cap_gain, fin_yield, cfg, policy
    )

    new_wealth = (
        financial_wealth + fin_yield + housing_wealth + cap_gain + savings
        - tax_paid + transfer_received
    )

    shock = rng.uniform(0.0, 1.0, size=new_wealth.shape) < beh.wealth_shock_prob
    new_wealth = np.where(
        shock, new_wealth - np.maximum(new_wealth, 0.0) * beh.wealth_shock_size, new_wealth
    )
    new_wealth = np.maximum(new_wealth, beh.debt_floor)
    state.wealth = new_wealth

    # Ownership transitions. Transaction friction raises the wealth bar.
    local_price = house_price[state.region]
    friction_mult = 1.0 + policy.transaction_friction
    buy_threshold = beh.buy_wealth_to_price * local_price * friction_mult
    help_lower = beh.assortative_help_wealth_lower_factor * local_price * friction_mult
    sell_threshold = beh.sell_wealth_to_price * local_price

    # Assortative help is applied BEFORE buy eligibility so a successful
    # transfer can enable an ownership transition that would not otherwise
    # happen this period. The flag gates the entire branch: when off, no
    # RNG draws are consumed inside the help block and the buy_draws
    # below preserve the prior model's RNG ordering exactly.
    if beh.assortative_help_enabled:
        near_eligible = (
            (~state.homeowner)
            & (state.age <= beh.assortative_help_recipient_max_age)
            & (state.wealth >= help_lower)
            & (state.wealth < buy_threshold)
            & (state.income > beh.buy_income_to_price * local_price)
        )
        if near_eligible.any():
            _apply_assortative_help(state, near_eligible, buy_threshold, cfg, rng)

    # Compute final buy eligibility after any help-induced wealth boosts.
    eligible_to_buy = (
        (~state.homeowner)
        & (state.wealth > buy_threshold)
        & (state.income > beh.buy_income_to_price * local_price)
    )

    # Ownership-transition probability, scaled down in regions with deep
    # rental markets (rental_market_depth > 1) and up where owner-occupancy
    # is the institutional norm (depth < 1); depth = 1 recovers the uniform
    # baseline probability.
    p_buy = np.clip(
        beh.buy_probability / cfg.regional.rental_market_depth[state.region],
        0.0, 1.0,
    )
    buy_draws = rng.uniform(0.0, 1.0, size=eligible_to_buy.shape)
    can_buy = eligible_to_buy & (buy_draws < p_buy)
    must_sell = state.homeowner & (state.wealth < sell_threshold)
    state.homeowner = (state.homeowner | can_buy) & (~must_sell)

    return cap_gain, rent_paid, transfer_received, housing_value_pre


def _apply_assortative_help(
    state: HouseholdState,
    near_eligible_mask: np.ndarray,
    buy_threshold: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> None:
    """Wealth-conserving assortative transfer to near-eligible renters.

    Recipients are sampled from `near_eligible_mask` (renters with wealth
    in [assortative_help_wealth_lower_factor * price, buy_wealth_to_price
    * price] and income above the buy-income test) with per-event
    probability p_assortative_help.

    Donor pool: alive agents with age >= assortative_help_donor_min_age
    and wealth above the national median, sampled without replacement
    within a period with weights wealth ** assortative_exponent. Offered
    recipients are excluded from the donor pool so self-donation is
    impossible.

    Each transfer is the smaller of (i) the shortfall to the buy
    threshold plus a 1 EUR epsilon and (ii) f_assortative_help * donor
    wealth. Donor wealth is decremented by the same amount the recipient
    receives (wealth-conserving).

    If a sampled donor's wealth is so low that the transfer would be
    non-positive (edge case under the above filter; shouldn't normally
    fire), the event is skipped without consuming the donor sample. If
    the donor pool is exhausted before all offered events are processed,
    the remaining events are dropped silently.
    """
    beh = cfg.behavioral
    demo = cfg.demographic

    near_eligible_idx = np.where(near_eligible_mask)[0]
    if near_eligible_idx.size == 0:
        return

    receive_draws = rng.uniform(0.0, 1.0, size=near_eligible_idx.size)
    offered = near_eligible_idx[receive_draws < beh.p_assortative_help]
    if offered.size == 0:
        return

    median_wealth = float(np.median(state.wealth))
    donor_mask = (
        (state.age >= beh.assortative_help_donor_min_age)
        & (state.wealth > median_wealth)
    )
    # Exclude offered recipients to prevent self-donation.
    donor_mask[offered] = False
    donor_idx = np.where(donor_mask)[0]
    if donor_idx.size == 0:
        return

    weights = np.maximum(state.wealth[donor_idx], 0.0) ** demo.assortative_exponent
    if weights.sum() <= 0:
        return

    available = np.ones(donor_idx.size, dtype=bool)
    epsilon = 1.0  # 1 EUR, ensures recipient lands strictly over the threshold

    for recipient in offered:
        if not available.any():
            break
        active_w = weights * available
        s = active_w.sum()
        if s <= 0:
            break
        local_donor = int(rng.choice(donor_idx.size, p=active_w / s))
        global_donor = int(donor_idx[local_donor])

        shortfall = float(buy_threshold[recipient] - state.wealth[recipient] + epsilon)
        donor_cap = beh.f_assortative_help * float(state.wealth[global_donor])
        transfer = min(shortfall, donor_cap)

        if transfer <= 0.0:
            # Skip event without consuming the donor sample.
            continue

        state.wealth[recipient] += transfer
        state.wealth[global_donor] -= transfer
        available[local_donor] = False
