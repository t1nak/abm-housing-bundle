"""Main simulation loop: demography, housing, voting per period.

Per-period order (RNG draws are fixed in this order to keep counterfactuals
reproducible at a given seed):
    1. mortality + bequest sampling (mortality.step_mortality_and_bequest)
    2. regime update (deterministic; no rng)
    3. price formation under effective supply restriction
    4. rent-index update (deterministic)
    5. wealth, savings, ownership transitions under effective friction
    6. voting

The policy block introduces no new random draws.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .config import Config
from .cosmopolitan import compute_cosmopolitan_shift
from .household import HouseholdState, initialise, initial_house_price
from .housing_market import step_wealth_and_ownership, update_prices, update_rent_index
from .metrics import gini, top_share, bottom_share, regional_means, regional_shares
from .mortality import step_mortality_and_bequest
from .policy import EffectiveIntensities, PolicyRegime, smoothed_vote, update_regime
from .voting import step_voting


@dataclass
class History:
    price: np.ndarray
    rent_index: np.ndarray
    vote: np.ndarray
    vote_by_tenure: np.ndarray
    ownership: np.ndarray
    dissat: np.ndarray
    mean_wealth: np.ndarray
    mean_income: np.ndarray
    mean_age: np.ndarray
    gini: np.ndarray
    top10: np.ndarray
    top1: np.ndarray
    bottom50: np.ndarray
    vote_aggregate: np.ndarray
    ownership_aggregate: np.ndarray
    deaths: np.ndarray
    regime: np.ndarray
    smoothed_vote: np.ndarray
    effective_rent_cap: np.ndarray
    effective_supply: np.ndarray
    effective_friction: np.ndarray
    transfer_aggregate: np.ndarray
    income_aggregate: np.ndarray
    # Per-period mean of the voting-input outcome o = income - rent + cap_gain
    # + transfer, restricted to agents who were renters / owners at the start
    # of the period. NaN at t=0 (no step has run); filled at t=1..T. Used by
    # the equal-cost counterfactual to integrate renter welfare across the
    # active window. Recording these is observation-only and consumes no RNG,
    # so flags-off bit-for-bit reproducibility is preserved.
    renter_mean_o: np.ndarray
    owner_mean_o: np.ndarray
    n_renters: np.ndarray


def _record(t: int, state: HouseholdState, cfg: Config, hist: History) -> None:
    R = cfg.regional.n_regions
    w = state.wealth
    hist.gini[t] = gini(w)
    hist.top10[t] = top_share(w, 0.10)
    hist.top1[t] = top_share(w, 0.01)
    hist.bottom50[t] = bottom_share(w, 0.50)
    hist.vote_aggregate[t] = state.extreme_voter.mean()
    hist.ownership_aggregate[t] = state.homeowner.mean()
    hist.ownership[t] = regional_shares(state.homeowner, state.region, R)
    hist.mean_wealth[t] = regional_means(state.wealth, state.region, R)
    hist.mean_income[t] = regional_means(state.income, state.region, R)
    hist.mean_age[t] = regional_means(state.age, state.region, R)
    hist.vote[t] = regional_shares(state.extreme_voter, state.region, R)
    for r in range(R):
        for tenure, h in enumerate([False, True]):
            mask = (state.region == r) & (state.homeowner == h)
            if mask.any():
                hist.vote_by_tenure[t, r, tenure] = state.extreme_voter[mask].mean()


def _materialise_cosmopolitan_shift(cfg: Config) -> Config:
    """If gamma_cosmopolitan is set but no explicit shift vector was passed,
    compute the shift from gamma * (grad_share - mean_grad_share) using
    the grad-share data path on VotingConfig. An explicitly-passed
    cosmopolitan_shift_by_region always wins. Defaults (gamma == 0.0,
    shift None) leave the config untouched, so the model reproduces the
    scalar-beta_0 specification bit-for-bit."""
    voting = cfg.voting
    if voting.cosmopolitan_shift_by_region is not None:
        return cfg
    if voting.gamma_cosmopolitan == 0.0 or voting.grad_share_data_path is None:
        return cfg
    shift = compute_cosmopolitan_shift(
        voting.gamma_cosmopolitan,
        voting.grad_share_data_path,
        cfg.regional.n_regions,
    )
    return replace(cfg, voting=replace(voting, cosmopolitan_shift_by_region=shift))


def simulate(cfg: Config) -> tuple[HouseholdState, History, np.ndarray]:
    cfg = _materialise_cosmopolitan_shift(cfg)
    rng = np.random.default_rng(cfg.seed)
    state = initialise(cfg, rng)
    house_price = initial_house_price(cfg)
    initial_price = house_price.copy()
    T = cfg.n_periods
    R = cfg.regional.n_regions

    hist = History(
        price=np.zeros((T + 1, R)),
        rent_index=np.zeros((T + 1, R)),
        vote=np.zeros((T + 1, R)),
        vote_by_tenure=np.zeros((T + 1, R, 2)),
        ownership=np.zeros((T + 1, R)),
        dissat=np.zeros((T + 1, R)),
        mean_wealth=np.zeros((T + 1, R)),
        mean_income=np.zeros((T + 1, R)),
        mean_age=np.zeros((T + 1, R)),
        gini=np.zeros(T + 1),
        top10=np.zeros(T + 1),
        top1=np.zeros(T + 1),
        bottom50=np.zeros(T + 1),
        vote_aggregate=np.zeros(T + 1),
        ownership_aggregate=np.zeros(T + 1),
        deaths=np.zeros(T + 1, dtype=int),
        regime=np.empty(T + 1, dtype=object),
        smoothed_vote=np.zeros(T + 1),
        effective_rent_cap=np.zeros(T + 1),
        effective_supply=np.zeros(T + 1),
        effective_friction=np.zeros(T + 1),
        transfer_aggregate=np.zeros(T + 1),
        income_aggregate=np.zeros(T + 1),
        renter_mean_o=np.full(T + 1, np.nan),
        owner_mean_o=np.full(T + 1, np.nan),
        n_renters=np.zeros(T + 1, dtype=int),
    )
    rent_index = np.ones(R)
    regional_shock_state = np.zeros(R)
    hist.price[0] = house_price.copy()
    hist.rent_index[0] = rent_index.copy()
    hist.regime[0] = PolicyRegime.MAINSTREAM
    regime = PolicyRegime.MAINSTREAM
    _record(0, state, cfg, hist)

    for t in range(T):
        deaths = step_mortality_and_bequest(state, cfg, rng)
        hist.deaths[t + 1] = deaths

        # Regime decision uses the smoothed share of *recorded* votes up to t.
        smoothed = smoothed_vote(hist.vote_aggregate, t, cfg.policy.smoothing_window)
        regime = update_regime(regime, smoothed, cfg.policy)
        intensities = EffectiveIntensities.from_config(cfg.policy, regime)
        hist.smoothed_vote[t + 1] = smoothed
        hist.regime[t + 1] = regime
        hist.effective_rent_cap[t + 1] = intensities.rent_cap
        hist.effective_supply[t + 1] = intensities.supply_restriction
        hist.effective_friction[t + 1] = intensities.transaction_friction

        house_price, price_growth, regional_shock_state = update_prices(
            state, house_price, regional_shock_state, cfg, rng, intensities
        )
        rent_index = update_rent_index(rent_index, price_growth, intensities)
        pre_renter_mask = ~state.homeowner.copy()
        cap_gain, rent_paid, transfer_received, housing_value_pre = step_wealth_and_ownership(
            state, house_price, price_growth, rent_index, initial_price, t, cfg, rng, intensities
        )
        regional_dissat = step_voting(
            state, cap_gain, rent_paid, transfer_received, housing_value_pre, cfg, rng
        )
        hist.transfer_aggregate[t + 1] = float(transfer_received.sum())
        hist.income_aggregate[t + 1] = float(state.income.sum())
        o_per_agent = state.income - rent_paid + cap_gain + transfer_received
        if pre_renter_mask.any():
            hist.renter_mean_o[t + 1] = float(o_per_agent[pre_renter_mask].mean())
        if (~pre_renter_mask).any():
            hist.owner_mean_o[t + 1] = float(o_per_agent[~pre_renter_mask].mean())
        hist.n_renters[t + 1] = int(pre_renter_mask.sum())

        hist.price[t + 1] = house_price
        hist.rent_index[t + 1] = rent_index
        hist.dissat[t + 1] = regional_dissat
        _record(t + 1, state, cfg, hist)

    return state, hist, house_price
