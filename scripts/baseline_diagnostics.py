"""Consolidated baseline diagnostics for the calibrated (gated) configuration.

Regenerates every headline number the manuscript reports at the calibration
horizon (T=15, seeds 73-78):
  - political anchors: aggregate extreme-share vote; within-region
    renter-owner gap (region-averaged);
  - the three gradient sign checks (cross-regional Pearson correlations,
    evaluators from abmhp.estimation.moments);
  - structural moments: wealth Gini, top-1%, top-10%, bottom-50% shares
    (with seed s.d.), homeownership rate, aggregate 15-year price growth,
    cross-regional price-growth s.d., price-growth/supply-elasticity corr.;
  - the housing-channel ceiling: aggregate vote with gamma = 0
    (margins off), and the implied housing contribution;
  - renter-owner gap by structural region type (mean and within-type s.d.
    over regions x seeds);
  - non-owner wealth relative to the down-payment bar (share below half,
    share within 20%).

Writes outputs/baseline_diagnostics.json.
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

SEEDS = list(range(73, 79))
RTYPE = np.array(["super"] * 4 + ["avg"] * 8 + ["decl"] * 4)


def gini_coef(w):
    w = np.sort(np.maximum(np.asarray(w, dtype=float), 0.0))
    n = w.size
    if w.sum() <= 0:
        return 0.0
    cum = np.cumsum(w)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def top_share(w, q):
    w = np.asarray(w, dtype=float)
    thr = np.percentile(w, 100 - q)
    return float(w[w >= thr].sum() / w.sum())


def bottom_share(w, q):
    w = np.asarray(w, dtype=float)
    thr = np.percentile(w, q)
    return float(np.maximum(w[w <= thr], 0).sum() / np.maximum(w, 0).sum())


def run(cfg):
    return simulate(cfg)


acc = dict(vote=[], gap=[], gini=[], top1=[], top10=[], bot50=[], own=[],
           pgrowth=[], pgrowth_sd=[], elast_corr=[],
           gap_super=[], gap_avg=[], gap_decl=[],
           below_half=[], within20=[], vote_gamma0=[])
grad = dict(pg_renter=[], rent_burden=[], access=[])

for s in SEEDS:
    cfg = apply_margin_calibration(Config(seed=s, n_periods=15))
    state, h, hp = run(cfg)
    acc["vote"].append(float(h.vote_aggregate[-1]))
    vt = h.vote_by_tenure[-1]
    gap_r = vt[:, 0] - vt[:, 1]
    acc["gap"].append(float(np.mean(gap_r)))
    for t, key in (("super", "gap_super"), ("avg", "gap_avg"), ("decl", "gap_decl")):
        acc[key].extend(gap_r[RTYPE == t].tolist())
    w = state.wealth
    acc["gini"].append(gini_coef(w))
    acc["top1"].append(top_share(w, 1))
    acc["top10"].append(top_share(w, 10))
    acc["bot50"].append(bottom_share(w, 50))
    acc["own"].append(float(state.homeowner.mean()))
    growth = h.price[-1] / h.price[0] - 1.0
    # aggregate growth: growth of the population-weighted price index (the
    # definition used by the calibration evaluator in estimation.moments)
    pop_w = cfg.regional.pop_share
    acc["pgrowth"].append(float((h.price[-1] * pop_w).sum() / (h.price[0] * pop_w).sum() - 1.0))
    acc["pgrowth_sd"].append(float(np.std(growth, ddof=1)))
    acc["elast_corr"].append(float(np.corrcoef(growth, cfg.regional.supply_elasticity)[0, 1]))
    # gradients (as in abmhp.estimation.moments)
    grad["pg_renter"].append(float(np.corrcoef(growth, h.vote_by_tenure[-1, :, 0])[0, 1]))
    rent_burden = (h.price[0] * h.rent_index[-1]) / np.maximum(h.mean_income[-1], 1.0)
    grad["rent_burden"].append(float(np.corrcoef(rent_burden, h.vote[-1])[0, 1]))
    access_pressure = h.price[-1] / np.maximum(h.mean_wealth[-1], 1.0)
    grad["access"].append(float(np.corrcoef(access_pressure, h.vote_by_tenure[-1, :, 0])[0, 1]))
    # non-owner wealth vs bar
    thr = cfg.behavioral.buy_wealth_to_price * hp[state.region]
    no = ~state.homeowner
    ratio = state.wealth[no] / np.maximum(thr[no], 1.0)
    acc["below_half"].append(float(np.mean(ratio < 0.5)))
    acc["within20"].append(float(np.mean((ratio >= 0.8) & (ratio < 1.0))))
    # ceiling: margins off
    cfg0 = replace(cfg, voting=replace(cfg.voting, gamma_rent=0.0, gamma_asset=0.0,
                                       gamma_access=0.0))
    _, h0, _ = run(cfg0)
    acc["vote_gamma0"].append(float(h0.vote_aggregate[-1]))

out = {}
for k, v in acc.items():
    out[k] = dict(mean=round(float(np.mean(v)), 4), sd=round(float(np.std(v)), 4))
out["gradients"] = {k: round(float(np.mean(v)), 3) for k, v in grad.items()}
out["ceiling"] = dict(
    baseline_vote=out["vote"]["mean"],
    gamma0_vote=out["vote_gamma0"]["mean"],
    housing_contribution=round(out["vote"]["mean"] - out["vote_gamma0"]["mean"], 4),
    housing_share=round((out["vote"]["mean"] - out["vote_gamma0"]["mean"]) / out["vote"]["mean"], 3),
)

path = ROOT / "outputs" / "baseline_diagnostics.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
