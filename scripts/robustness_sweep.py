"""Systematic one-at-a-time robustness sweep for the policy decomposition.

For each perturbed configuration, re-runs the full exogenous policy
decomposition (none / rent / supply / access / transfer / bundle) and records
(i) the final-period change in each housing-pressure margin and in the
extreme-share vote relative to that variant's own no-policy baseline, and
(ii) the variant's baseline anchors (aggregate vote, renter-owner gap).

The object of interest is the stability of the QUALITATIVE mechanism results:
  Q1  supply expansion relieves all three margins (broadest coverage);
  Q2  the transfer is the largest single mover of the vote;
  Q3  ownership support is the smallest single mover of the vote;
  Q4  rent relief's largest relieved margin is the rent margin;
  Q5  the transfer's largest relieved margin is the access margin;
  Q6  the bundle relieves all three margins.

Perturbations cover: down-payment requirement, ownership transition
probability, aspiration persistence, wealth-return schedule, housing exposure,
inheritance (assortativity, bequest tax), policy leakage, and the political
coefficients (proportional scale and the asset/access split).

Writes outputs/robustness_sweep.json and prints a summary table.
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

T = 25
SEEDS = list(range(73, 78))          # 5 seeds per scenario per variant
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


def scaled_returns(factor: float) -> tuple:
    base = ((50_000.0, -0.015), (200_000.0, 0.005), (1_000_000.0, 0.040),
            (5_000_000.0, 0.075), (float(np.inf), 0.105))
    return tuple((thr, r * factor) for thr, r in base)


# variant -> dict of block -> {param: value}
VARIANTS: dict[str, dict] = {
    "baseline": {},
    # down-payment requirement (buy wealth-to-price threshold 0.30)
    "downpay_0.25": {"behavioral": dict(buy_wealth_to_price=0.25)},
    "downpay_0.35": {"behavioral": dict(buy_wealth_to_price=0.35)},
    # ownership transition probability (0.35)
    "buyprob_0.25": {"behavioral": dict(buy_probability=0.25)},
    "buyprob_0.45": {"behavioral": dict(buy_probability=0.45)},
    # aspiration persistence (0.92)
    "rho_asp_0.85": {"voting": dict(rho_aspiration=0.85)},
    "rho_asp_0.96": {"voting": dict(rho_aspiration=0.96)},
    # wealth-return schedule (+/- 25% on all tier rates)
    "returns_x0.75": {"behavioral": dict(return_schedule=scaled_returns(0.75))},
    "returns_x1.25": {"behavioral": dict(return_schedule=scaled_returns(1.25))},
    # housing exposure tiers (0.50 / 1.10 / 2.00, +/- 20%)
    "housing_exp_x0.8": {"behavioral": dict(housing_share_tier_low=0.40,
                                            housing_share_tier_mid=0.88,
                                            housing_share_tier_high=1.60)},
    "housing_exp_x1.2": {"behavioral": dict(housing_share_tier_low=0.60,
                                            housing_share_tier_mid=1.32,
                                            housing_share_tier_high=2.40)},
    # inheritance: bequest assortativity (2.10) and bequest tax (0.22)
    "assort_1.6": {"demographic": dict(assortative_exponent=1.6)},
    "assort_2.6": {"demographic": dict(assortative_exponent=2.6)},
    "beqtax_0.15": {"demographic": dict(bequest_tax_rate=0.15)},
    "beqtax_0.30": {"demographic": dict(bequest_tax_rate=0.30)},
    # policy leakage (baseline 0.4 / 0.3 / 0.5, scaled x0.5 and x1.5)
    "leak_x0.5": {"leak": dict(rent_cap_leakage=0.2, supply_leakage=0.15, friction_leakage=0.25)},
    "leak_x1.5": {"leak": dict(rent_cap_leakage=0.6, supply_leakage=0.45, friction_leakage=0.75)},
    # political coefficients: proportional scale of (gamma1, gamma2, gamma3)
    "gamma_x0.75": {"voting": dict(gamma_rent=0.2196, gamma_asset=0.4026, gamma_access=0.3660)},
    "gamma_x1.25": {"voting": dict(gamma_rent=0.3660, gamma_asset=0.6710, gamma_access=0.6100)},
    # political coefficients: asset/access split along the identified manifold
    "split_asset_heavy": {"voting": dict(gamma_asset=0.7198, gamma_access=0.3050)},
    "split_access_heavy": {"voting": dict(gamma_asset=0.3050, gamma_access=0.6710)},
    # specification variants (reviewer round 2)
    # UNGATED rent margin (baseline is gated on renters; this variant
    # restores the pre-round-3 specification in which owners' d_rent is a
    # pure income-aspiration shortfall)
    "rent_ungated": {"voting": dict(rent_margin_renters_only=False)},
    # Poterba-style owner flow cost on gross levered housing value
    "user_cost_0.04": {"behavioral": dict(user_cost_rate=0.04)},
    # leverage decreasing in wealth (reversed exposure tiers)
    "leverage_decr": {"behavioral": dict(housing_share_tier_low=2.00,
                                         housing_share_tier_mid=1.10,
                                         housing_share_tier_high=0.50)},
    # pre-repair tenure block (round 11): parental transfers OFF
    "no_parental_help": {"behavioral": dict(assortative_help_enabled=False)},
    # pre-repair tenure block (round 11): uniform rental-market depth
    "uniform_depth": {"regional": dict(uniform_depth=True)},
    # network reinforcement coefficient (fixed design value 0.60)
    "beta_n_0.3": {"voting": dict(beta_network=0.3)},
    "beta_n_0.9": {"voting": dict(beta_network=0.9)},
}


def build_cfg(variant: str, scenario: str, seed: int) -> Config:
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=T))
    mods = VARIANTS[variant]
    if "behavioral" in mods:
        cfg = replace(cfg, behavioral=replace(cfg.behavioral, **mods["behavioral"]))
    if "regional" in mods:
        cfg = replace(cfg, regional=replace(
            cfg.regional, rental_market_depth=np.ones(16)))
    if "demographic" in mods:
        cfg = replace(cfg, demographic=replace(cfg.demographic, **mods["demographic"]))
    if "voting" in mods:
        cfg = replace(cfg, voting=replace(cfg.voting, **mods["voting"]))
    pol = dict(force_regime=PolicyRegime.POPULIST.value)
    scen = dict(SCENARIOS[scenario])
    if "leak" in mods:
        # leakage variant replaces the LEAK entries wherever the scenario has them
        for k, v in mods["leak"].items():
            if k in scen:
                scen[k] = v
    pol.update(scen)
    return replace(cfg, policy=replace(cfg.policy, **pol))


def run_cell(args):
    variant, scenario, seed = args
    state, h, _ = simulate(build_cfg(variant, scenario, seed))
    out = dict(variant=variant, scenario=scenario, seed=seed,
               d_rent=float(h.d_rent[-1]), d_asset=float(h.d_asset[-1]),
               d_access=float(h.d_access[-1]), vote=float(h.vote_aggregate[-1]))
    if scenario == "none":
        vt = h.vote_by_tenure[-1]                     # (R, 2): renter, owner
        out["gap"] = float(np.mean(vt[:, 0] - vt[:, 1]))
    return out


def main():
    cells = [(v, s, sd) for v in VARIANTS for s in SCENARIOS for sd in SEEDS]
    print(f"Running {len(cells)} simulations ({len(VARIANTS)} variants x "
          f"{len(SCENARIOS)} scenarios x {len(SEEDS)} seeds), T={T} ...", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(run_cell, cells, chunksize=4))

    # aggregate: mean over seeds per (variant, scenario)
    agg: dict = {}
    for v in VARIANTS:
        agg[v] = {}
        for s in SCENARIOS:
            sub = [r for r in rows if r["variant"] == v and r["scenario"] == s]
            agg[v][s] = {k: float(np.mean([r[k] for r in sub]))
                         for k in ("d_rent", "d_asset", "d_access", "vote")}
            agg[v][s]["sd_vote"] = float(np.std([r["vote"] for r in sub]))
            if s == "none":
                agg[v][s]["gap"] = float(np.mean([r["gap"] for r in sub]))

    # deltas vs each variant's own baseline + qualitative checks
    results = {}
    single = ["rent", "supply", "access", "transfer"]
    for v in VARIANTS:
        base = agg[v]["none"]
        deltas = {}
        for s in SCENARIOS:
            if s == "none":
                continue
            deltas[s] = {m: agg[v][s][m] - base[m]
                         for m in ("d_rent", "d_asset", "d_access", "vote")}
        dv = {s: deltas[s]["vote"] for s in single}
        margin_names = ("d_rent", "d_asset", "d_access")
        biggest = {s: min(margin_names, key=lambda m: deltas[s][m]) for s in single}
        checks = dict(
            q1_supply_all_three=all(deltas["supply"][m] < 0 for m in margin_names),
            q2_transfer_largest=min(dv, key=dv.get) == "transfer",
            q3_ownership_smallest=max(dv, key=dv.get) == "access",
            q4_rent_hits_rent=biggest["rent"] == "d_rent",
            q5_transfer_hits_access=biggest["transfer"] == "d_access",
            q6_bundle_all_three=all(deltas["bundle"][m] < 0 for m in margin_names),
        )
        results[v] = dict(baseline_vote=base["vote"], baseline_gap=base["gap"],
                          deltas=deltas, checks=checks,
                          all_pass=all(checks.values()))

    out_path = ROOT / "outputs" / "robustness_sweep.json"
    out_path.write_text(json.dumps(dict(T=T, seeds=SEEDS, variants=results), indent=2))
    print(f"\nWrote {out_path}\n")

    hdr = (f"{'variant':20} {'vote0':>6} {'gap0':>6} | "
           f"{'dV_rent':>8} {'dV_sup':>8} {'dV_acc':>8} {'dV_tra':>8} {'dV_bun':>8} | checks")
    print(hdr); print("-" * len(hdr))
    for v, r in results.items():
        d = r["deltas"]
        flags = "".join("Y" if x else "N" for x in r["checks"].values())
        print(f"{v:20} {r['baseline_vote']:6.3f} {r['baseline_gap']:+6.3f} | "
              f"{d['rent']['vote']:+8.4f} {d['supply']['vote']:+8.4f} "
              f"{d['access']['vote']:+8.4f} {d['transfer']['vote']:+8.4f} "
              f"{d['bundle']['vote']:+8.4f} | {flags}"
              + ("" if r["all_pass"] else "   <-- check"))


if __name__ == "__main__":
    main()
