"""Out-of-sample validation moments for the revised manuscript.

Computes model-implied moments that were NOT targeted during calibration:
  V1  wealth-by-age profile (mean net wealth by age band, and each band's
      share of the overall mean) -- compare qualitatively to the hump-shaped
      Bundesbank PHF age profile;
  V2  homeownership rate by region type -- compare to the lower ownership
      rates of expensive metropolitan states vs cheaper peripheral states;
  V3  ownership rate by age band -- rising age gradient (EU-SILC / PHF);
  V4  final-period house-price level dispersion across region types --
      superstar premium over declining regions;
  V5  within-region renter-owner vote gap by region type (already in the
      paper; recomputed here for completeness).

Baseline calibrated configuration, T=15, calibration seeds 73-78. In
addition, the ownership-related moments (V3/V5) are recomputed under two
specification variants that bear on the reported validation failures:
"no_parental_help" / "uniform_depth" (pre-repair tenure block) and
"leverage_decr" (housing exposure decreasing in wealth).
Writes outputs/validation_moments.json.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, simulate
from abmhp.margin_calibration import apply_margin_calibration

SEEDS = range(73, 79)
AGE_BANDS = [(20, 34), (35, 49), (50, 64), (65, 85)]
RTYPE = ["super"] * 4 + ["avg"] * 8 + ["decl"] * 4

VARIANTS = {
    "baseline": {},
    # pre-repair tenure block (round 11): parental help off / uniform depth
    "no_parental_help": dict(assortative_help_enabled=False),
    "uniform_depth": "REGIONAL_UNIFORM_DEPTH",
    "leverage_decr": dict(housing_share_tier_low=2.00,
                          housing_share_tier_mid=1.10,
                          housing_share_tier_high=0.50),
}


def run_variant(mods: dict) -> dict:
    acc: dict = {
        "wealth_by_age": {f"{a}-{b}": [] for a, b in AGE_BANDS},
        "own_by_age": {f"{a}-{b}": [] for a, b in AGE_BANDS},
        "own_by_type": {t: [] for t in ("super", "avg", "decl")},
        "price_by_type": {t: [] for t in ("super", "avg", "decl")},
        "gap_by_type": {t: [] for t in ("super", "avg", "decl")},
        "ownership_rate": [],
    }
    for seed in SEEDS:
        cfg = apply_margin_calibration(Config(seed=seed, n_periods=15))
        if mods == "REGIONAL_UNIFORM_DEPTH":
            cfg = replace(cfg, regional=replace(
                cfg.regional, rental_market_depth=np.ones(16)))
        elif mods:
            cfg = replace(cfg, behavioral=replace(cfg.behavioral, **mods))
        state, hist, prices = simulate(cfg)
        mean_w = state.wealth.mean()
        acc["ownership_rate"].append(float(state.homeowner.mean()))
        for a, b in AGE_BANDS:
            m = (state.age >= a) & (state.age <= b)
            acc["wealth_by_age"][f"{a}-{b}"].append(state.wealth[m].mean() / mean_w)
            acc["own_by_age"][f"{a}-{b}"].append(state.homeowner[m].mean())
        for t in ("super", "avg", "decl"):
            rs = [r for r in range(16) if RTYPE[r] == t]
            m = np.isin(state.region, rs)
            acc["own_by_type"][t].append(state.homeowner[m].mean())
            acc["price_by_type"][t].append(float(np.mean(prices[rs])))
            vt = hist.vote_by_tenure[-1]
            acc["gap_by_type"][t].append(float(np.mean(vt[rs, 0] - vt[rs, 1])))

    out = {k: ({kk: dict(mean=float(np.mean(v)), sd=float(np.std(v)))
                for kk, v in d.items()} if isinstance(d, dict)
               else dict(mean=float(np.mean(d)), sd=float(np.std(d))))
           for k, d in acc.items()}
    decl = out["price_by_type"]["decl"]["mean"]
    out["price_ratio_to_declining"] = {t: out["price_by_type"][t]["mean"] / decl
                                       for t in ("super", "avg", "decl")}
    return out


out = {name: run_variant(mods) for name, mods in VARIANTS.items()}
path = ROOT / "outputs" / "validation_moments.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
