"""Robustness of the calibrated (gated) model to the aspiration-locality
parameter alpha_local.

alpha_local weights the regional vs. national reference income in the
aspiration anchor (Eq. for y*): alpha_local = 1 purely regional, 0 purely
national. The baseline is a judgemental design choice, alpha_local = 0.45.

This exercise varies ONLY alpha_local across {0.00, 0.25, 0.45, 0.65, 0.85,
1.00}, holding every other parameter at its calibrated value. For each value
the aggregate-vote anchor is restored by re-solving beta_0 alone (damped
Newton on the mean logit, exactly the one-parameter adjustment used for the
rent-gating revision); no other parameter is recalibrated.

For each alpha_local it reports:
  - calibration: solved beta_0, aggregate vote, renter-owner gap;
  - mechanisms: mean d_rent / d_asset / d_access by tenure and region type;
  - policy: final-period Delta(vote) for rent / supply / access / transfer /
    bundle, the mechanism-target of rent and transfer, and the six
    qualitative checks (Q1-Q6);
  - validation: regional renter-owner gap gradient, ownership by region type,
    aggregate ownership, wealth-by-age profile.

Calibration diagnostics at T=15 (6 seeds); policy decomposition at T=25
(20 seeds). Writes outputs/alpha_loc_robustness.json.
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

ALPHAS = [0.00, 0.25, 0.45, 0.65, 0.85, 1.00]
BASELINE_ALPHA = 0.45
LEVEL_TARGET = 0.208
LEVEL_TOL = 0.010
SEEDS_CAL = list(range(73, 79))     # 6 seeds, T=15
SEEDS_POL = list(range(73, 93))     # 20 seeds, T=25
T_CAL, T_POL = 15, 25
RTYPE = np.array(["super"] * 4 + ["avg"] * 8 + ["decl"] * 4)
AGE_BANDS = [(20, 34), (35, 49), (50, 64), (65, 85)]

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


def cal_cfg(alpha, b0, seed, n_periods=T_CAL):
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=n_periods))
    return replace(cfg, voting=replace(cfg.voting, alpha_local=alpha, beta_0=b0))


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def eval_anchors(alpha, b0, seeds=SEEDS_CAL):
    votes, gaps = [], []
    for s in seeds:
        _, h, _ = simulate(cal_cfg(alpha, b0, s))
        votes.append(float(h.vote_aggregate[-1]))
        vt = h.vote_by_tenure[-1]
        gaps.append(float(np.mean(vt[:, 0] - vt[:, 1])))
    return float(np.mean(votes)), float(np.mean(gaps))


def solve_beta0(alpha):
    """Re-solve beta_0 (only) to restore the aggregate-vote anchor."""
    b0 = -1.82
    for _ in range(4):
        vote, gap = eval_anchors(alpha, b0, seeds=SEEDS_CAL[:3])
        if abs(vote - LEVEL_TARGET) <= LEVEL_TOL:
            break
        b0 = b0 + 0.9 * (logit(LEVEL_TARGET) - logit(vote))
    return round(b0, 3)


def household_margins(cfg):
    """Reconstruct per-household gated mechanism gaps at the final period."""
    state, hist, house_price = simulate(cfg)
    beh, vot = cfg.behavioral, cfg.voting
    T = cfg.n_periods
    r = state.region
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
    return state, hist, house_price, d_rent, d_asset, d_access


def calibration_and_validation(alpha, b0):
    acc = dict(vote=[], gap=[], own=[],
               gap_super=[], gap_avg=[], gap_decl=[],
               own_super=[], own_avg=[], own_decl=[])
    tenure = {t: {m: [] for m in ("d_rent", "d_asset", "d_access")}
              for t in ("renter", "owner")}
    reg = {t: {m: [] for m in ("d_rent", "d_asset", "d_access")}
           for t in ("super", "avg", "decl")}
    wealth_age = {f"{a}-{b}": [] for a, b in AGE_BANDS}
    for s in SEEDS_CAL:
        cfg = cal_cfg(alpha, b0, s)
        state, hist, hp, dr, da, dc = household_margins(cfg)
        acc["vote"].append(float(hist.vote_aggregate[-1]))
        vt = hist.vote_by_tenure[-1]
        gap_r = vt[:, 0] - vt[:, 1]
        acc["gap"].append(float(np.mean(gap_r)))
        acc["own"].append(float(state.homeowner.mean()))
        ren = ~state.homeowner
        for m, arr in (("d_rent", dr), ("d_asset", da), ("d_access", dc)):
            tenure["renter"][m].append(float(arr[ren].mean()))
            tenure["owner"][m].append(float(arr[~ren].mean()))
        for t in ("super", "avg", "decl"):
            rs = np.where(RTYPE == t)[0]
            acc[f"gap_{t}"].append(float(np.mean(gap_r[rs])))
            mask = np.isin(state.region, rs)
            acc[f"own_{t}"].append(float(state.homeowner[mask].mean()))
            for m, arr in (("d_rent", dr), ("d_asset", da), ("d_access", dc)):
                reg[t][m].append(float(arr[mask].mean()))
        mean_w = state.wealth.mean()
        for a, b in AGE_BANDS:
            mm = (state.age >= a) & (state.age <= b)
            wealth_age[f"{a}-{b}"].append(float(state.wealth[mm].mean() / mean_w))

    def ms(v):
        return dict(mean=round(float(np.mean(v)), 4), sd=round(float(np.std(v)), 4))
    return dict(
        beta0=b0,
        vote=ms(acc["vote"]), gap=ms(acc["gap"]), own=ms(acc["own"]),
        gap_by_type={t: ms(acc[f"gap_{t}"]) for t in ("super", "avg", "decl")},
        own_by_type={t: ms(acc[f"own_{t}"]) for t in ("super", "avg", "decl")},
        margins_by_tenure={t: {m: round(float(np.mean(tenure[t][m])), 3)
                               for m in tenure[t]} for t in tenure},
        margins_by_region={t: {m: round(float(np.mean(reg[t][m])), 3)
                               for m in reg[t]} for t in reg},
        wealth_by_age={k: round(float(np.mean(v)), 3) for k, v in wealth_age.items()},
    )


def policy_cfg(alpha, b0, scenario, seed):
    cfg = cal_cfg(alpha, b0, seed, n_periods=T_POL)
    pol = dict(force_regime=PolicyRegime.POPULIST.value)
    pol.update(SCENARIOS[scenario])
    return replace(cfg, policy=replace(cfg.policy, **pol))


def policy_decomposition(alpha, b0):
    agg = {}
    for sc in SCENARIOS:
        dr, da, dc, vt = [], [], [], []
        for s in SEEDS_POL:
            _, h, _ = simulate(policy_cfg(alpha, b0, sc, s))
            dr.append(h.d_rent[-1]); da.append(h.d_asset[-1])
            dc.append(h.d_access[-1]); vt.append(h.vote_aggregate[-1])
        agg[sc] = dict(d_rent=float(np.mean(dr)), d_asset=float(np.mean(da)),
                       d_access=float(np.mean(dc)), vote=float(np.mean(vt)))
    base = agg["none"]
    deltas = {sc: {m: agg[sc][m] - base[m] for m in base}
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
    return dict(
        deltas={sc: {m: round(deltas[sc][m], 4) for m in deltas[sc]} for sc in deltas},
        biggest=biggest, checks=checks, all_pass=all(checks.values()),
        base_vote=round(base["vote"], 4))


def run_alpha(alpha):
    b0 = solve_beta0(alpha)
    cal = calibration_and_validation(alpha, b0)
    pol = policy_decomposition(alpha, b0)
    return alpha, dict(calibration=cal, policy=pol)


def main():
    print(f"alpha_loc robustness: {ALPHAS}, re-solving beta_0 each; "
          f"cal T={T_CAL}/{len(SEEDS_CAL)} seeds, policy T={T_POL}/"
          f"{len(SEEDS_POL)} seeds ...", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = dict(ex.map(run_alpha, ALPHAS))

    out = {f"{a:.2f}": results[a] for a in ALPHAS}
    (ROOT / "outputs" / "alpha_loc_robustness.json").write_text(json.dumps(out, indent=2))

    hdr = (f"{'alpha':>6} {'beta0':>7} {'vote':>6} {'gap':>7} | "
           f"{'dV_rent':>8} {'dV_sup':>8} {'dV_acc':>8} {'dV_tra':>8} {'dV_bun':>8} | "
           f"{'gapS/A/D':>16} | checks")
    print(hdr); print("-" * len(hdr))
    for a in ALPHAS:
        c = results[a]["calibration"]; p = results[a]["policy"]
        d = p["deltas"]; g = c["gap_by_type"]
        flags = "".join("Y" if v else "N" for v in p["checks"].values())
        star = " (baseline)" if a == BASELINE_ALPHA else ""
        print(f"{a:6.2f} {c['beta0']:7.2f} {c['vote']['mean']:6.3f} "
              f"{c['gap']['mean']:+7.3f} | "
              f"{d['rent']['vote']:+8.4f} {d['supply']['vote']:+8.4f} "
              f"{d['access']['vote']:+8.4f} {d['transfer']['vote']:+8.4f} "
              f"{d['bundle']['vote']:+8.4f} | "
              f"{g['super']['mean']:+.2f}/{g['avg']['mean']:+.2f}/{g['decl']['mean']:+.2f} | "
              f"{flags}{star}")
    n_pass = sum(results[a]['policy']['all_pass'] for a in ALPHAS)
    print(f"\nAll six qualitative checks pass in {n_pass}/{len(ALPHAS)} alpha values.")


if __name__ == "__main__":
    main()
