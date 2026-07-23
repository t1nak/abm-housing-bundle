"""Threshold-sensitivity counterfactual at the corrected SMM optimum.

Rebuild design (no hidden intercept shift):
  * beta_0 stays at the SMM-estimated value for ALL scenarios (apply_smm_optimum).
  * the activation threshold theta+ is a FIXED, non-estimated design parameter,
    varied as a transparent sensitivity dimension (0.30 original / 0.25 main /
    0.20 low), with hysteresis theta- moved proportionally.
  * an "always-on from t=1" diagnostic isolates instrument effects from the
    activation rule.
  * the no-policy baseline (A) shares the estimated beta_0 and the same theta+,
    so scenario differences are a clean policy effect.

Within-model comparative diagnostics only; not forecasts.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, PolicyRegime, simulate
from abmhp.estimation.smm import apply_smm_optimum

T = 25
SEEDS = list(range(73, 83))  # 10 seeds
TAU_K = 0.027
MED = dict(rent_cap_leakage=0.40, supply_leakage=0.30, friction_leakage=0.50)


def scen_cfg(scenario: str, seed: int, theta_plus: float, theta_minus: float,
             force_on: bool) -> Config:
    """All scenarios start from the estimated optimum (beta_0 unchanged)."""
    cfg = apply_smm_optimum(Config(seed=seed, n_periods=T))
    pol = dict(incumbency_threshold=theta_plus, deactivation_threshold=theta_minus)
    if force_on:
        pol["force_regime"] = PolicyRegime.POPULIST.value
    if scenario == "A":            # no-policy baseline: instruments off
        pol.update(rent_cap_intensity=0.0, supply_restriction_intensity=0.0,
                   transaction_friction=0.0)
    elif scenario == "B":          # housing-only, leak-free
        pol.update(rent_cap_leakage=0.0, supply_leakage=0.0, friction_leakage=0.0)
    elif scenario == "C":          # housing-only, central leakage
        pol.update(**MED)
    elif scenario == "E":          # bundle: housing (leaky) + capital-tax transfer
        pol.update(**MED, redistribution_active=True, capital_tax_rate=TAU_K)
    return replace(cfg, policy=replace(cfg.policy, **pol))


def final_vote(scenario, theta_plus, theta_minus, force_on=False):
    vals, activ = [], []
    for s in SEEDS:
        cfg = scen_cfg(scenario, s, theta_plus, theta_minus, force_on)
        _, h, _ = simulate(cfg)
        vals.append(float(h.vote_aggregate[-1]))
        activ.append(any(r is PolicyRegime.POPULIST for r in h.regime[1:]))
    return float(np.mean(vals)), sum(activ) / len(activ)


CASES = [
    ("theta+=0.30 (original)", 0.30, 0.20, False),
    ("theta+=0.25 (main)",     0.25, 0.17, False),
    ("theta+=0.20 (low)",      0.20, 0.15, False),
    ("always-on (t=1)",        0.30, 0.20, True),
]

print(f"estimated beta_0 = {apply_smm_optimum(Config(seed=73)).voting.beta_0:.4f}\n")
print(f"{'case':24s} {'A':>7} {'B':>7} {'C':>7} {'E':>7} | {'B-A':>7} {'C-A':>7} {'E-A':>7} | actE")
print("-" * 92)
for label, tp, tm, fon in CASES:
    res = {sc: final_vote(sc, tp, tm, fon) for sc in ["A", "B", "C", "E"]}
    A, B, C, E = (res[sc][0] for sc in ["A", "B", "C", "E"])
    print(f"{label:24s} {A:7.3f} {B:7.3f} {C:7.3f} {E:7.3f} | "
          f"{B-A:+7.3f} {C-A:+7.3f} {E-A:+7.3f} | {res['E'][1]*100:3.0f}%")
print("\nNote: negative B-A/C-A/E-A = policy reduces the model-implied extreme vote.")
