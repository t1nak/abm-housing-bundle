"""Adjusted-anchor band: results over a defensible interval of gap anchors.

The +0.15 renter-owner gap anchor is a raw difference and therefore an upper
bound on the housing-attributable share: composition (age, income, education,
East-West) is embedded in it. The composition-adjusted gap cannot be computed
without SOEP access, so this exercise brackets it instead: the political
block is recalibrated to gap anchors of +0.10 and +0.05 (with the raw +0.15
as reference) by scaling the three mechanism coefficients proportionally
(gamma = s * (0.48, 0.88, 0.80)) and re-solving beta_0 for the level anchor
at each step. For each anchor it reports the solved (s, beta_0), the
achieved anchors, the housing-channel ceiling (gamma = 0 counterfactual),
and the full policy decomposition with the six qualitative checks.

Writes outputs/adjusted_anchor_band.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, PolicyRegime, simulate
from abmhp.margin_calibration import apply_margin_calibration

LEVEL_TARGET = 0.208
LEVEL_TOL = 0.010
GAP_TOL = 0.010
GAMMA_BASE = np.array([0.48, 0.88, 0.80])
GAP_TARGETS = [0.15, 0.10, 0.05]
SEEDS_CAL = list(range(73, 79))
SEEDS_POL = list(range(73, 78))
T_CAL, T_POL = 15, 25

LEAK = dict(rent_cap_leakage=0.4, supply_leakage=0.3, friction_leakage=0.5)
TAU_K = 0.027
SCENARIOS = {
    "none":     dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=0.0),
    "rent":     dict(rent_cap_intensity=0.6, supply_restriction_intensity=0.0, transaction_friction=0.0, **LEAK),
    "supply":   dict(rent_cap_intensity=0.0, supply_restriction_intensity=-0.5, transaction_friction=0.0, **LEAK),
    "access":   dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=-0.3, **LEAK),
    "transfer": dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=0.0,
                     redistribution_active=True, capital_tax_rate=TAU_K),
    "bundle":   dict(rent_cap_intensity=0.6, supply_restriction_intensity=-0.5, transaction_friction=-0.3,
                     redistribution_active=True, capital_tax_rate=TAU_K, **LEAK),
}


def cfg_for(s_scale, b0, seed, n_periods=T_CAL, gamma_zero=False):
    g = np.zeros(3) if gamma_zero else s_scale * GAMMA_BASE
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=n_periods))
    return replace(cfg, voting=replace(
        cfg.voting, beta_0=b0, gamma_rent=float(g[0]), gamma_asset=float(g[1]),
        gamma_access=float(g[2])))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def anchors(s_scale, b0, seeds, gamma_zero=False):
    votes, gaps = [], []
    for s in seeds:
        _, h, _ = simulate(cfg_for(s_scale, b0, s, gamma_zero=gamma_zero))
        votes.append(float(h.vote_aggregate[-1]))
        vt = h.vote_by_tenure[-1]
        gaps.append(float(np.mean(vt[:, 0] - vt[:, 1])))
    return float(np.mean(votes)), float(np.mean(gaps))


def solve(gap_target):
    """Alternate: solve beta_0 for the level, rescale s toward the gap."""
    # Seed the search at the pinned central configuration so the 0.10 row
    # reproduces the headline baseline exactly (margin_calibration.py).
    s_scale, b0 = (0.61, -1.82) if abs(gap_target - 0.10) < 1e-9 else (1.0, -2.17)
    for _ in range(5):
        # inner: beta_0 for level
        for _ in range(3):
            vote, gap = anchors(s_scale, b0, SEEDS_CAL[:3])
            if abs(vote - LEVEL_TARGET) <= LEVEL_TOL:
                break
            b0 = b0 + 0.9 * (logit(LEVEL_TARGET) - logit(vote))
        if abs(gap - gap_target) <= GAP_TOL:
            break
        s_scale = s_scale * (1.0 + 0.9 * (gap_target / max(gap, 1e-6) - 1.0))
    return round(s_scale, 3), round(b0, 3)


def policy_decomposition(s_scale, b0):
    agg = {}
    for sc in SCENARIOS:
        dr, da, dc, vt = [], [], [], []
        for s in SEEDS_POL:
            cfg = cfg_for(s_scale, b0, s, n_periods=T_POL)
            pol = dict(force_regime=PolicyRegime.POPULIST.value)
            pol.update(SCENARIOS[sc])
            cfg = replace(cfg, policy=replace(cfg.policy, **pol))
            _, h, _ = simulate(cfg)
            dr.append(h.d_rent[-1]); da.append(h.d_asset[-1])
            dc.append(h.d_access[-1]); vt.append(h.vote_aggregate[-1])
        agg[sc] = dict(d_rent=float(np.mean(dr)), d_asset=float(np.mean(da)),
                       d_access=float(np.mean(dc)), vote=float(np.mean(vt)))
    base = agg["none"]
    deltas = {sc: {m: round(agg[sc][m] - base[m], 4) for m in base}
              for sc in agg if sc != "none"}
    single = ["rent", "supply", "access", "transfer"]
    dv = {sc: deltas[sc]["vote"] for sc in single}
    margins = ("d_rent", "d_asset", "d_access")
    biggest = {sc: min(margins, key=lambda m: deltas[sc][m]) for sc in single}
    checks = dict(
        q1_supply_all_three=all(deltas["supply"][m] < 0 for m in margins),
        q2_transfer_largest=min(dv, key=dv.get) == "transfer",
        q3_ownership_smallest=max(dv, key=dv.get) == "access",
        q4_rent_hits_rent=biggest["rent"] == "d_rent",
        q5_transfer_hits_access=biggest["transfer"] == "d_access",
        q6_bundle_all_three=all(deltas["bundle"][m] < 0 for m in margins),
    )
    return dict(deltas=deltas, checks=checks, all_pass=all(checks.values()))


def run_target(gap_target):
    s_scale, b0 = solve(gap_target)
    vote, gap = anchors(s_scale, b0, SEEDS_CAL)
    vote0, _ = anchors(s_scale, b0, SEEDS_CAL, gamma_zero=True)
    pol = policy_decomposition(s_scale, b0)
    return gap_target, dict(
        s_scale=s_scale, beta0=b0, vote=round(vote, 4), gap=round(gap, 4),
        gamma=[round(float(x), 3) for x in s_scale * GAMMA_BASE],
        vote_gamma0=round(vote0, 4),
        housing_share=round((vote - vote0) / vote, 3), policy=pol)


def main():
    print(f"Adjusted-anchor band: gap targets {GAP_TARGETS} ...", flush=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        results = {f"{k:.2f}": v for k, v in ex.map(run_target, GAP_TARGETS)}
    (ROOT / "outputs" / "adjusted_anchor_band.json").write_text(
        json.dumps(results, indent=2))
    hdr = (f"{'anchor':>7} {'s':>6} {'beta0':>7} {'vote':>6} {'gap':>7} "
           f"{'share':>6} | {'dV_tra':>8} {'dV_bun':>8} | checks")
    print(hdr); print("-" * len(hdr))
    for k in results:
        r = results[k]; d = r["policy"]["deltas"]
        flags = "".join("Y" if v else "N" for v in r["policy"]["checks"].values())
        print(f"{k:>7} {r['s_scale']:6.2f} {r['beta0']:7.2f} {r['vote']:6.3f} "
              f"{r['gap']:+7.3f} {r['housing_share']:6.1%} | "
              f"{d['transfer']['vote']:+8.4f} {d['bundle']['vote']:+8.4f} | {flags}")


if __name__ == "__main__":
    main()
