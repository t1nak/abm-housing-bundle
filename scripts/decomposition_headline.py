"""Headline policy decomposition and ownership-support dose scan.

Regenerates the manuscript's decomposition table (final-period change in each
mechanism gap and in the aggregate extreme-share vote under each exogenously
imposed instrument, relative to the no-policy baseline) and the dose table,
at the calibrated (gated) configuration: T=25, 20 seeds (73-92), seed
standard deviations reported.

Writes outputs/decomposition_headline.json.
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
SEEDS = list(range(73, 93))
LEAK = dict(rent_cap_leakage=0.4, supply_leakage=0.3, friction_leakage=0.5)
TAU_K = 0.027

SCENARIOS = {
    "none":     dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=0.0),
    "rent":     dict(rent_cap_intensity=0.6, supply_restriction_intensity=0.0, transaction_friction=0.0, **LEAK),
    "supply":   dict(rent_cap_intensity=0.0, supply_restriction_intensity=-0.5, transaction_friction=0.0, **LEAK),
    "access":   dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=-0.3, **LEAK),
    "access_x2": dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=-0.6, **LEAK),
    "access_x4": dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=-1.2, **LEAK),
    "transfer": dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=0.0,
                     redistribution_active=True, capital_tax_rate=TAU_K),
    "bundle":   dict(rent_cap_intensity=0.6, supply_restriction_intensity=-0.5, transaction_friction=-0.3,
                     redistribution_active=True, capital_tax_rate=TAU_K, **LEAK),
}


def run_cell(args):
    scen, seed = args
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=T))
    pol = dict(force_regime=PolicyRegime.POPULIST.value)
    pol.update(SCENARIOS[scen])
    cfg = replace(cfg, policy=replace(cfg.policy, **pol))
    _, h, _ = simulate(cfg)
    return dict(scenario=scen, seed=seed,
                d_rent=float(h.d_rent[-1]), d_asset=float(h.d_asset[-1]),
                d_access=float(h.d_access[-1]), vote=float(h.vote_aggregate[-1]))


def main():
    cells = [(sc, sd) for sc in SCENARIOS for sd in SEEDS]
    print(f"Running {len(cells)} simulations ...", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(run_cell, cells, chunksize=4))

    per = {}
    for sc in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == sc]
        per[sc] = {k: dict(mean=float(np.mean([r[k] for r in sub])),
                           sd=float(np.std([r[k] for r in sub])))
                   for k in ("d_rent", "d_asset", "d_access", "vote")}

    base = per["none"]
    out = dict(T=T, n_seeds=len(SEEDS), levels=per, deltas={})
    for sc in SCENARIOS:
        if sc == "none":
            continue
        out["deltas"][sc] = {
            k: dict(mean=round(per[sc][k]["mean"] - base[k]["mean"], 4),
                    # sd of the difference across seeds (paired by seed)
                    sd=round(float(np.std(
                        [a[k] - b[k] for a, b in zip(
                            [r for r in rows if r["scenario"] == sc],
                            [r for r in rows if r["scenario"] == "none"])])), 4))
            for k in ("d_rent", "d_asset", "d_access", "vote")}

    path = ROOT / "outputs" / "decomposition_headline.json"
    path.write_text(json.dumps(out, indent=2))
    hdr = f"{'scenario':10} {'dRent':>9} {'dAsset':>9} {'dAccess':>9} {'dVote':>9}"
    print(hdr); print("-" * len(hdr))
    for sc, d in out["deltas"].items():
        print(f"{sc:10} {d['d_rent']['mean']:+9.4f} {d['d_asset']['mean']:+9.4f} "
              f"{d['d_access']['mean']:+9.4f} {d['vote']['mean']:+9.4f}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
