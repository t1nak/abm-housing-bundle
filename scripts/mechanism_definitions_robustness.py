"""Construct-validity robustness: alternative mechanism definitions.

Re-runs the calibrated model under alternative operational definitions of
the rent-stress and ownership-access mechanisms, holding all structural
parameters fixed and permitting only the one-parameter beta_0 adjustment
used elsewhere to restore the aggregate-vote anchor:

  baseline         gated after-rent aspiration shortfall (paper Eq. 5)
  rent_incremental only the increment in the aspiration shortfall CAUSED by
                   rent (excludes the pre-rent income shortfall)
  rent_to_income   conventional rent-burden measure, rent/income normalised
                   by the Eurostat 40% overburden threshold (no aspiration
                   reference)
  access_income    access margin includes the income eligibility test
                   (max of wealth and income shortfalls)

For each variant: solved beta_0, anchors (6 seeds, T=15), mean margins by
tenure, full policy decomposition (6 scenarios x 5 seeds, T=25) and the six
qualitative checks. Writes outputs/mechanism_definitions_robustness.json.
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
SEEDS_CAL = list(range(73, 79))
SEEDS_POL = list(range(73, 78))
T_CAL, T_POL = 15, 25

VARIANTS = {
    "baseline": {},
    "rent_incremental": dict(rent_margin_spec="incremental"),
    "rent_to_income": dict(rent_margin_spec="rent_to_income"),
    "access_income": dict(access_margin_include_income=True),
}

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


def cfg_for(mods, b0, seed, n_periods=T_CAL):
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=n_periods))
    return replace(cfg, voting=replace(cfg.voting, beta_0=b0, **mods))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def anchors(mods, b0, seeds):
    votes, gaps = [], []
    for s in seeds:
        _, h, _ = simulate(cfg_for(mods, b0, s))
        votes.append(float(h.vote_aggregate[-1]))
        vt = h.vote_by_tenure[-1]
        gaps.append(float(np.mean(vt[:, 0] - vt[:, 1])))
    return float(np.mean(votes)), float(np.mean(gaps))


def solve_beta0(mods):
    b0 = -1.82
    for _ in range(4):
        vote, _ = anchors(mods, b0, SEEDS_CAL[:3])
        if abs(vote - LEVEL_TARGET) <= LEVEL_TOL:
            break
        b0 = b0 + 0.9 * (logit(LEVEL_TARGET) - logit(vote))
    return round(b0, 3)


def policy_decomposition(mods, b0):
    agg = {}
    for sc in SCENARIOS:
        dr, da, dc, vt = [], [], [], []
        for s in SEEDS_POL:
            cfg = cfg_for(mods, b0, s, n_periods=T_POL)
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


def run_variant(item):
    name, mods = item
    b0 = solve_beta0(mods)
    vote, gap = anchors(mods, b0, SEEDS_CAL)
    pol = policy_decomposition(mods, b0)
    return name, dict(beta0=b0, vote=round(vote, 4), gap=round(gap, 4), policy=pol)


def main():
    print(f"Mechanism-definition robustness: {list(VARIANTS)} ...", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = dict(ex.map(run_variant, VARIANTS.items()))
    (ROOT / "outputs" / "mechanism_definitions_robustness.json").write_text(
        json.dumps(results, indent=2))
    hdr = (f"{'variant':18} {'beta0':>7} {'vote':>6} {'gap':>7} | "
           f"{'dV_rent':>8} {'dV_sup':>8} {'dV_acc':>8} {'dV_tra':>8} {'dV_bun':>8} | checks")
    print(hdr); print("-" * len(hdr))
    for name in VARIANTS:
        r = results[name]; d = r["policy"]["deltas"]
        flags = "".join("Y" if v else "N" for v in r["policy"]["checks"].values())
        print(f"{name:18} {r['beta0']:7.2f} {r['vote']:6.3f} {r['gap']:+7.3f} | "
              f"{d['rent']['vote']:+8.4f} {d['supply']['vote']:+8.4f} "
              f"{d['access']['vote']:+8.4f} {d['transfer']['vote']:+8.4f} "
              f"{d['bundle']['vote']:+8.4f} | {flags}")
    n = sum(results[v]["policy"]["all_pass"] for v in VARIANTS)
    print(f"\nAll six checks pass in {n}/{len(VARIANTS)} definitions.")


if __name__ == "__main__":
    main()
