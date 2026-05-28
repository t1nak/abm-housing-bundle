"""Gompertz mortality and assortative-bequest mechanics.

Calibration source: destatis 2023 life tables. Anchor points are
p(70) = 0.018, p(80) = 0.062, p(90) = 0.185. A Gompertz fit
p(a) = A * exp(B * a) through these yields A approx 4.5e-5, B approx 0.085.
The Gompertz parameters are empirically pinned: do not retune to hit
distribution targets. Bequest tax rate, assortative exponent, and
intergenerational skill correlation are the tuning levers.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .household import HouseholdState


def gompertz_mortality(age: np.ndarray, A: float, B: float, cap: float) -> np.ndarray:
    p = A * np.exp(B * age)
    return np.minimum(p, cap)


def step_mortality_and_bequest(
    state: HouseholdState,
    cfg: Config,
    rng: np.random.Generator,
) -> int:
    """Apply one period of mortality, bequest transfer, and young replacement.

    Mutates state in place. Returns the number of deaths this period.
    """
    demo = cfg.demographic
    regional = cfg.regional
    skill_cfg = cfg.skill

    p_death = gompertz_mortality(state.age, demo.gompertz_A, demo.gompertz_B, demo.mortality_cap)
    dies = rng.uniform(0.0, 1.0, size=state.age.shape) < p_death
    dead_idx = np.where(dies)[0]
    n_deaths = int(dead_idx.size)
    if n_deaths == 0:
        state.age += 1.0
        return 0

    alive_mask = ~dies

    # Per-region alive-index caches and sqrt-weight caches for sampling.
    weight_exp = demo.assortative_exponent
    region_indices: dict[int, np.ndarray] = {}
    region_weights: dict[int, np.ndarray] = {}
    for r in range(regional.n_regions):
        idx = np.where(alive_mask & (state.region == r))[0]
        region_indices[r] = idx
        if idx.size:
            w = np.maximum(state.wealth[idx], 0.0) ** weight_exp
            ws = w.sum()
            region_weights[r] = w / ws if ws > 0 else np.full(idx.size, 1.0 / idx.size)
        else:
            region_weights[r] = np.empty(0)

    national_idx = np.where(alive_mask)[0]
    nat_w = np.maximum(state.wealth[national_idx], 0.0) ** weight_exp
    nat_ws = nat_w.sum()
    national_weights = (
        nat_w / nat_ws if nat_ws > 0 else np.full(national_idx.size, 1.0 / national_idx.size)
    )

    # Decide regional vs national pool for each death and sample recipient.
    region_draws = rng.uniform(0.0, 1.0, size=n_deaths)
    bequest_inflow = np.zeros(state.wealth.shape, dtype=float)
    tax_factor = 1.0 - demo.bequest_tax_rate

    for k, i in enumerate(dead_idx):
        r_i = int(state.region[i])
        same_region = region_draws[k] < demo.same_region_prob
        if same_region and region_indices[r_i].size > 0:
            j = int(rng.choice(region_indices[r_i], p=region_weights[r_i]))
        else:
            j = int(rng.choice(national_idx, p=national_weights))
        bequest_inflow[j] += tax_factor * max(state.wealth[i], 0.0)

    # Apply inflows.
    state.wealth = state.wealth + bequest_inflow

    # Replace dead agents with young heirs (same region; AR(1) skill inheritance).
    n_new = n_deaths
    parent_skill = state.skill[dead_idx]
    rho = demo.intergenerational_skill_corr
    innovations = rng.normal(0.0, demo.skill_innovation_sd, size=n_new)
    new_skill = rho * parent_skill + (1.0 - rho) * demo.skill_mean + innovations
    new_skill = np.clip(new_skill, skill_cfg.skill_min, skill_cfg.skill_max)

    new_wealth = rng.lognormal(demo.new_agent_wealth_mu, demo.new_agent_wealth_sigma, size=n_new)

    state.skill[dead_idx] = new_skill
    state.age[dead_idx] = float(demo.young_age)
    state.wealth[dead_idx] = new_wealth
    state.income[dead_idx] = (
        new_skill * regional.productivity[state.region[dead_idx]] * skill_cfg.base_wage
    )
    state.aspiration[dead_idx] = state.income[dead_idx]
    state.extreme_voter[dead_idx] = False
    state.homeowner[dead_idx] = False

    # Survivors age by one year.
    state.age[alive_mask] += 1.0

    return n_deaths
