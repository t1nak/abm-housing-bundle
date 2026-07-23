"""Household-level mechanism diagnostics for the revised manuscript.

Reconstructs the three housing-pressure mechanism gaps (d_rent, d_asset,
d_access) household-by-household at the final calibration period, using
exactly the formulas of voting.step_voting, and reports:

  M1  the pairwise correlation matrix of the three gaps (population and
      renters-only) -- evidence that the mechanisms are empirically distinct
      inside the model, i.e. far from collinear;
  M2  mean gap by tenure -- in particular the owner-side component of d_rent
      (owners pay zero rent, so their d_rent is a pure income-aspiration
      shortfall; this quantifies how much of the rent-stress mechanism does
      not run through rent payments);
  M3  mean gaps by region type;
  M4  the share of owners and renters with each gap strictly positive.

Baseline calibrated configuration, T=15, calibration seeds 73-78.
Writes outputs/mechanism_diagnostics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, simulate
from abmhp.margin_calibration import apply_margin_calibration

SEEDS = range(73, 79)
RTYPE = np.array(["super"] * 4 + ["avg"] * 8 + ["decl"] * 4)


def household_margins(cfg: Config):
    """Simulate and reconstruct final-period per-household mechanism gaps."""
    state, hist, house_price = simulate(cfg)
    beh, vot = cfg.behavioral, cfg.voting
    T = cfg.n_periods
    r = state.region

    # rent_paid at the final period (housing_market.step_wealth_and_ownership)
    initial_price = hist.price[0]
    rent_level = initial_price[r] * beh.rent_yield * hist.rent_index[T][r]
    rent_paid = np.where(state.homeowner, 0.0, rent_level * beh.rent_burden_share)

    renter = (~state.homeowner).astype(float)
    A = np.maximum(state.aspiration, 1.0)
    d_rent = np.clip(np.maximum(0.0, A - (state.income - rent_paid)) / A, 0.0, 1.0)
    if vot.rent_margin_renters_only:
        d_rent = renter * d_rent

    w = vot.asset_gain_window
    gain = np.log(hist.price[T] / np.maximum(hist.price[max(T - w, 0)], 1e-9))
    d_asset = renter * np.clip(np.maximum(0.0, gain[r]), 0.0, 1.0)

    thr = np.maximum(beh.buy_wealth_to_price * house_price[r], 1.0)
    d_access = renter * np.clip(np.maximum(0.0, thr - state.wealth) / thr, 0.0, 1.0)
    return state, d_rent, d_asset, d_access


def corrmat(x, y, z, mask=None):
    m = np.ones_like(x, dtype=bool) if mask is None else mask
    M = np.vstack([x[m], y[m], z[m]])
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(M)
    return [[float(C[i, j]) for j in range(3)] for i in range(3)]


acc = {
    "corr_all": [], "corr_renters": [],
    "mean_by_tenure": {"renter": {"d_rent": [], "d_asset": [], "d_access": []},
                       "owner": {"d_rent": [], "d_asset": [], "d_access": []}},
    "share_positive": {"renter_d_rent": [], "owner_d_rent": []},
    "mean_by_type": {t: {"d_rent": [], "d_asset": [], "d_access": []}
                     for t in ("super", "avg", "decl")},
}

for seed in SEEDS:
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=15))
    state, dr, da, dc = household_margins(cfg)
    ren = ~state.homeowner
    acc["corr_all"].append(corrmat(dr, da, dc))
    acc["corr_renters"].append(corrmat(dr, da, dc, ren))
    for name, arr in (("d_rent", dr), ("d_asset", da), ("d_access", dc)):
        acc["mean_by_tenure"]["renter"][name].append(float(arr[ren].mean()))
        acc["mean_by_tenure"]["owner"][name].append(float(arr[~ren].mean()))
    acc["share_positive"]["renter_d_rent"].append(float((dr[ren] > 0).mean()))
    acc["share_positive"]["owner_d_rent"].append(float((dr[~ren] > 0).mean()))
    for t in ("super", "avg", "decl"):
        m = np.isin(state.region, np.where(RTYPE == t)[0])
        for name, arr in (("d_rent", dr), ("d_asset", da), ("d_access", dc)):
            acc["mean_by_type"][t][name].append(float(arr[m].mean()))

out = {
    "corr_all_mean": np.mean(np.array(acc["corr_all"]), axis=0).round(3).tolist(),
    "corr_renters_mean": np.mean(np.array(acc["corr_renters"]), axis=0).round(3).tolist(),
    "mean_by_tenure": {ten: {k: round(float(np.mean(v)), 3) for k, v in d.items()}
                       for ten, d in acc["mean_by_tenure"].items()},
    "share_positive": {k: round(float(np.mean(v)), 3)
                       for k, v in acc["share_positive"].items()},
    "mean_by_type": {t: {k: round(float(np.mean(v)), 3) for k, v in d.items()}
                     for t, d in acc["mean_by_type"].items()},
}
path = ROOT / "outputs" / "mechanism_diagnostics.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
