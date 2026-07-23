"""Path A diagnostic decomposition: which instrument reaches which margin.

Exogenous policies (regime forced active from period 1, NO activation
threshold) on the CALIBRATED three-margin model. This is a comparative
within-model DIAGNOSTIC of mechanism coverage, not a policy evaluation and not
a quantitative prediction of AfD voting. It reports, for each instrument, the
final-period population mean of each housing-pressure margin (d_rent, d_asset,
d_access) and the extreme-share vote, against the no-policy baseline.

Substantive point: rent-side relief reaches the consumption-stress margin only;
wealth/ownership-access instruments are needed to reach the asset-exclusion and
ownership-access margins.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, PolicyRegime, simulate
from abmhp.margin_calibration import apply_margin_calibration

T = 25
SEEDS = range(73, 81)
LEAK = dict(rent_cap_leakage=0.4, supply_leakage=0.3, friction_leakage=0.5)
TAU_K = 0.027

# Instrument settings per scenario. force_regime activates the housing-policy
# regime from period 1 (exogenous; no vote-threshold gating).
# Instruments are signed in the RELIEVING direction: rent cap lowers rents;
# supply EXPANSION (negative supply_restriction) raises supply elasticity and
# lowers price growth; ownership support (negative transaction_friction) lowers
# the down-payment/buy threshold; the transfer raises bottom-50% wealth.
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


def cfg_for(name: str, seed: int) -> Config:
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=T))
    pol = dict(force_regime=PolicyRegime.POPULIST.value)
    pol.update(SCENARIOS[name])
    return replace(cfg, policy=replace(cfg.policy, **pol))


def summarise(name: str):
    dr, da, dc, vt = [], [], [], []
    for s in SEEDS:
        _, h, _ = simulate(cfg_for(name, s))
        dr.append(h.d_rent[-1]); da.append(h.d_asset[-1])
        dc.append(h.d_access[-1]); vt.append(h.vote_aggregate[-1])
    return np.mean(dr), np.mean(da), np.mean(dc), np.mean(vt)


base = summarise("none")
print("Final-period margins and extreme-share vote (mean over 8 seeds), and")
print("the change vs the no-policy baseline. Diagnostic decomposition only.\n")
print(f"{'scenario':10} {'d_rent':>8} {'d_asset':>8} {'d_access':>8} {'vote':>8}   "
      f"{'Δrent':>7} {'Δasset':>7} {'Δaccess':>7} {'Δvote':>7}")
for nm in SCENARIOS:
    dr, da, dc, v = summarise(nm)
    print(f"{nm:10} {dr:8.3f} {da:8.3f} {dc:8.3f} {v:8.3f}   "
          f"{dr-base[0]:+7.3f} {da-base[1]:+7.3f} {dc-base[2]:+7.3f} {v-base[3]:+7.3f}")
print("\nReads as: a negative Δ means the instrument relieves that margin.")
