"""Household state container and initialisation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Config


@dataclass
class HouseholdState:
    region: np.ndarray
    skill: np.ndarray
    age: np.ndarray
    income: np.ndarray
    wealth: np.ndarray
    homeowner: np.ndarray
    aspiration: np.ndarray
    extreme_voter: np.ndarray


def initialise(cfg: Config, rng: np.random.Generator) -> HouseholdState:
    n = cfg.n_agents
    regional = cfg.regional
    skill_cfg = cfg.skill
    demo = cfg.demographic

    region = rng.choice(regional.n_regions, size=n, p=regional.pop_share)
    skill = np.clip(
        rng.lognormal(0.0, skill_cfg.skill_sigma, n),
        skill_cfg.skill_min,
        skill_cfg.skill_max,
    )
    income = skill * regional.productivity[region] * skill_cfg.base_wage

    age = rng.integers(demo.age_min, demo.age_max + 1, size=n).astype(float)

    # Age-correlated lognormal wealth: median grows linearly with age and skill.
    median = skill * demo.init_wealth_age_scale * age
    median = np.maximum(median, 1.0)
    wealth = rng.lognormal(np.log(median), demo.init_wealth_sigma)
    wealth = np.clip(wealth, 200.0, 5e7)

    # Initial tenure: wealth above the national median, scaled by the
    # region's rental-market depth (deep metropolitan rental markets set
    # a higher effective bar for owner-occupancy; depth = 1 recovers the
    # uniform national-median rule).
    ownership_threshold = float(np.percentile(wealth, 50))
    homeowner = wealth > ownership_threshold * regional.rental_market_depth[region]

    aspiration = income.copy()
    extreme_voter = np.zeros(n, dtype=bool)

    return HouseholdState(
        region=region,
        skill=skill,
        age=age,
        income=income,
        wealth=wealth,
        homeowner=homeowner,
        aspiration=aspiration,
        extreme_voter=extreme_voter,
    )


def initial_house_price(cfg: Config) -> np.ndarray:
    reg = cfg.regional
    return reg.base_house_price_intercept + reg.base_house_price_slope * (reg.productivity - 1.0)
