"""Full three-coefficient calibration manifold for the political block.

Addresses the identification question: which combinations of the three
mechanism coefficients (gamma_rent, gamma_asset, gamma_access) can match the
two political anchors, once the intercept beta_0 is re-solved for each
combination? The exercise makes explicit that gamma_1 is NOT pinned by the
anchors any more than the gamma_2/gamma_3 split is: the anchor-matching set is
a (approximately two-dimensional) manifold in (gamma_1, gamma_2, gamma_3).

Procedure:
  1. Grid over gamma_1 in {0, 0.15, 0.29, 0.44, 0.59},
     gamma_2 in {0.27, 0.54, 0.81}, gamma_3 in {0.24, 0.49, 0.73}
     (the round-3 grid rescaled by 0.61 to bracket the central-scenario
     coefficients).
  2. For each combination, re-solve beta_0 to hit the aggregate level anchor
     (0.208, tolerance 0.010) by damped Newton steps on the mean
     logit, simulating 3 seeds per step (max 3 steps, tolerance 0.008).
  3. Record the resulting renter-owner gap; combinations with
     |gap - 0.100| <= 0.015 form the anchor-matching set.
  4. For the two extreme members of the set in the rent coefficient
     ("rent-light": smallest gamma_1; "rent-heavy": largest gamma_1), run the
     full policy decomposition (6 scenarios x 5 seeds, T=25) and evaluate the
     six qualitative mechanism checks of the robustness sweep.

Writes outputs/gamma_manifold.json.
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

LEVEL_TARGET = 0.208        # empirical level anchor (2025 AfD second-vote share)
GAP_TARGET = 0.100          # CENTRAL gap scenario (literature-informed; not a measured SOEP value)
GAP_TOL = 0.015
LEVEL_TOL = 0.010
SEEDS_CAL = (73, 74, 75)
SEEDS_POL = (73, 74, 75, 76, 77)
T_CAL, T_POL = 15, 25

G1 = (0.0, 0.15, 0.29, 0.44, 0.59)
G2 = (0.27, 0.54, 0.81)
G3 = (0.24, 0.49, 0.73)

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


def cal_cfg(g1, g2, g3, b0, seed, n_periods=T_CAL):
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=n_periods))
    return replace(cfg, voting=replace(cfg.voting, gamma_rent=g1, gamma_asset=g2,
                                       gamma_access=g3, beta_0=b0))


def eval_point(g1, g2, g3, b0):
    votes, gaps = [], []
    for s in SEEDS_CAL:
        _, h, _ = simulate(cal_cfg(g1, g2, g3, b0, s))
        votes.append(h.vote_aggregate[-1])
        vt = h.vote_by_tenure[-1]
        gaps.append(float(np.mean(vt[:, 0] - vt[:, 1])))
    return float(np.mean(votes)), float(np.mean(gaps))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def solve_combo(args):
    g1, g2, g3 = args
    b0 = -1.82
    vote = gap = None
    for _ in range(3):
        vote, gap = eval_point(g1, g2, g3, b0)
        if abs(vote - LEVEL_TARGET) <= LEVEL_TOL:
            break
        b0 = b0 + 0.9 * (logit(LEVEL_TARGET) - logit(vote))
    return dict(g1=g1, g2=g2, g3=g3, beta0=round(b0, 3),
                vote=round(vote, 4), gap=round(gap, 4),
                level_ok=abs(vote - LEVEL_TARGET) <= LEVEL_TOL,
                gap_ok=abs(gap - GAP_TARGET) <= GAP_TOL)


def policy_cfg(g1, g2, g3, b0, scenario, seed):
    cfg = cal_cfg(g1, g2, g3, b0, seed, n_periods=T_POL)
    pol = dict(force_regime=PolicyRegime.POPULIST.value)
    pol.update(SCENARIOS[scenario])
    return replace(cfg, policy=replace(cfg.policy, **pol))


def policy_decomposition(g1, g2, g3, b0):
    agg = {}
    for sc in SCENARIOS:
        dr, da, dc, vt = [], [], [], []
        for s in SEEDS_POL:
            _, h, _ = simulate(policy_cfg(g1, g2, g3, b0, sc, s))
            dr.append(h.d_rent[-1]); da.append(h.d_asset[-1])
            dc.append(h.d_access[-1]); vt.append(h.vote_aggregate[-1])
        agg[sc] = dict(d_rent=float(np.mean(dr)), d_asset=float(np.mean(da)),
                       d_access=float(np.mean(dc)), vote=float(np.mean(vt)))
    base = agg["none"]
    deltas = {sc: {m: agg[sc][m] - base[m] for m in base} for sc in agg if sc != "none"}
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


def main():
    combos = [(a, b, c) for a in G1 for b in G2 for c in G3]
    print(f"Solving beta_0 for {len(combos)} gamma combinations "
          f"({len(SEEDS_CAL)} seeds per evaluation, T={T_CAL}) ...", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(solve_combo, combos, chunksize=2))

    matching = [r for r in rows if r["level_ok"] and r["gap_ok"]]
    print(f"\nAnchor-matching set: {len(matching)} of {len(rows)} combinations "
          f"(|vote-{LEVEL_TARGET}|<={LEVEL_TOL}, |gap-{GAP_TARGET}|<={GAP_TOL})")
    hdr = f"{'g1':>5} {'g2':>5} {'g3':>5} {'beta0':>7} {'vote':>7} {'gap':>7}  match"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["g1"], r["g2"], r["g3"])):
        print(f"{r['g1']:5.2f} {r['g2']:5.2f} {r['g3']:5.2f} {r['beta0']:7.2f} "
              f"{r['vote']:7.3f} {r['gap']:7.3f}  "
              f"{'YES' if (r['level_ok'] and r['gap_ok']) else ''}")

    result = dict(grid=rows, matching=[dict(r) for r in matching])

    if matching:
        lo = min(matching, key=lambda r: r["g1"])
        hi = max(matching, key=lambda r: r["g1"])
        for name, r in (("rent_light", lo), ("rent_heavy", hi)):
            print(f"\nPolicy decomposition at {name}: gamma=({r['g1']}, {r['g2']}, "
                  f"{r['g3']}), beta0={r['beta0']}", flush=True)
            dec = policy_decomposition(r["g1"], r["g2"], r["g3"], r["beta0"])
            result[name] = dict(point=r, **dec)
            flags = "".join("Y" if v else "N" for v in dec["checks"].values())
            dv = {sc: round(dec["deltas"][sc]["vote"], 4) for sc in dec["deltas"]}
            print(f"  checks {flags}  dVote {dv}")

    out = ROOT / "outputs" / "gamma_manifold.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
