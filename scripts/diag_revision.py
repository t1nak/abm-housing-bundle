"""Revision diagnostics for the JEBO calibrated-decomposition paper.

All runs are on the EXISTING calibrated model (apply_margin_calibration); no
re-estimation. Computes, with clear TASK labels:
  3a  housing-channel ceiling (gamma=0 margins-off vote vs calibrated baseline)
  3b  within-region renter-owner extreme-share gap by region type
  2a  non-owner wealth distribution relative to the down-payment bar
  2b  ownership-support dose robustness (1x / 2x / 4x calibrated friction)
  4d  seed dispersion on the Table 4 policy decomposition
  4b  gamma-split sensitivity band (asset/access reweighting along the manifold)
"""
from __future__ import annotations
import sys
from dataclasses import replace
sys.path.insert(0, "src")
import numpy as np
from abmhp import Config, simulate, PolicyRegime
from abmhp.margin_calibration import apply_margin_calibration

T_CAL = 15          # calibration horizon (2010-2025)
T_DEC = 25          # decomposition horizon (matches counterfactual_decomposition.py)
S_CAL = range(73, 79)        # 6 seeds for distributional diagnostics
S_DEC = range(73, 93)        # 20 seeds for the decomposition (Table 4 robustness)
LEAK = dict(rent_cap_leakage=0.4, supply_leakage=0.3, friction_leakage=0.5)
TAU_K = 0.027
RTYPE = Config().regional.region_type  # 0-3 super, 4-11 avg, 12-15 decl


def cal_cfg(seed, T=T_CAL, **vot):
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=T))
    if vot:
        cfg = replace(cfg, voting=replace(cfg.voting, **vot))
    return cfg


def mean_final_vote(cfg_fn, seeds=S_CAL):
    v = [simulate(cfg_fn(s))[1].vote_aggregate[-1] for s in seeds]
    return float(np.mean(v)), float(np.std(v))


# ===== Task 3a: housing-channel ceiling =====
print("="*70)
base = mean_final_vote(lambda s: cal_cfg(s))
g0 = mean_final_vote(lambda s: cal_cfg(s, gamma_rent=0.0, gamma_asset=0.0, gamma_access=0.0))
contrib = base[0] - g0[0]
print(f"3a CEILING: calibrated baseline vote = {base[0]:.3f} (sd {base[1]:.3f})")
print(f"           margins-off (gamma=0)    = {g0[0]:.3f} (sd {g0[1]:.3f})  [beta0-only floor]")
print(f"           housing-margin contribution = {contrib:.3f} "
      f"= {100*contrib/base[0]:.0f}% of {base[0]:.3f}; beta0 floor = {100*g0[0]/base[0]:.0f}%")

# ===== Task 3b: renter-owner gap by region type =====
print("="*70)
gaps = {"super": [], "avg": [], "decl": []}
for s in S_CAL:
    h = simulate(cal_cfg(s))[1]
    vt = h.vote_by_tenure[-1]          # (R,2): [:,0]=renter, [:,1]=owner
    gap_r = vt[:, 0] - vt[:, 1]
    for r in range(16):
        gaps[RTYPE[r]].append(gap_r[r])
print("3b RENTER-OWNER GAP by region type (final period, mean over regions x seeds):")
for k in ("super", "avg", "decl"):
    a = np.array(gaps[k]); print(f"     {k:6s} gap = {a.mean():+.3f} (sd {a.std():.3f}, n={a.size})")
allg = np.array(gaps['super']+gaps['avg']+gaps['decl'])
print(f"     ALL    gap = {allg.mean():+.3f}  (target anchor +0.148)")

# ===== Task 2a: non-owner wealth vs down-payment bar =====
print("="*70)
bins = {"ge_bar": [], "w80_100": [], "w50_80": [], "below50": []}
for s in S_CAL:
    state, h, hp = simulate(cal_cfg(s))
    beh = cal_cfg(s).behavioral
    thr = beh.buy_wealth_to_price * hp[state.region]
    no = ~state.homeowner
    ratio = state.wealth[no] / np.maximum(thr[no], 1.0)
    n = ratio.size
    bins["ge_bar"].append(np.mean(ratio >= 1.0))
    bins["w80_100"].append(np.mean((ratio >= 0.8) & (ratio < 1.0)))
    bins["w50_80"].append(np.mean((ratio >= 0.5) & (ratio < 0.8)))
    bins["below50"].append(np.mean(ratio < 0.5))
print("2a NON-OWNER WEALTH vs down-payment bar (share of non-owners, mean over seeds):")
print(f"     at/above bar (>=1.0)      : {np.mean(bins['ge_bar']):.3f}")
print(f"     within 20% below (0.8-1.0): {np.mean(bins['w80_100']):.3f}")
print(f"     0.5-0.8 of bar            : {np.mean(bins['w50_80']):.3f}")
print(f"     far below (<0.5 of bar)   : {np.mean(bins['below50']):.3f}")

# ===== decomposition machinery =====
SCEN = {
    "none":     dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=0.0),
    "rent":     dict(rent_cap_intensity=0.6, supply_restriction_intensity=0.0, transaction_friction=0.0, **LEAK),
    "supply":   dict(rent_cap_intensity=0.0, supply_restriction_intensity=-0.5, transaction_friction=0.0, **LEAK),
    "access":   dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=-0.3, **LEAK),
    "transfer": dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=0.0,
                     redistribution_active=True, capital_tax_rate=TAU_K),
    "bundle":   dict(rent_cap_intensity=0.6, supply_restriction_intensity=-0.5, transaction_friction=-0.3,
                     redistribution_active=True, capital_tax_rate=TAU_K, **LEAK),
}


def dec_cfg(seed, pol, gam=None):
    cfg = apply_margin_calibration(Config(seed=seed, n_periods=T_DEC))
    if gam:
        cfg = replace(cfg, voting=replace(cfg.voting, **gam))
    p = dict(force_regime=PolicyRegime.POPULIST.value); p.update(pol)
    return replace(cfg, policy=replace(cfg.policy, **p))


def run_scen(pol, seeds, gam=None):
    dr = da = dc = vt = None
    R = []
    for s in seeds:
        _, h, _ = simulate(dec_cfg(s, pol, gam))
        R.append((h.d_rent[-1], h.d_asset[-1], h.d_access[-1], h.vote_aggregate[-1]))
    A = np.array(R)
    return A.mean(0), A.std(0)


# ===== Task 4d: seed dispersion on Table 4 (24 seeds) =====
print("="*70)
print(f"4d SEED DISPERSION on Table 4 ({len(list(S_DEC))} seeds): mean Delta [sd] vs no-policy baseline")
base_m, _ = run_scen(SCEN["none"], S_DEC)
print(f"     {'instrument':10} {'Dd_rent':>16} {'Dd_asset':>16} {'Dd_access':>16} {'Dvote':>16}")
table4 = {}
for nm in ("rent", "supply", "transfer", "access", "bundle"):
    m, sd = run_scen(SCEN[nm], S_DEC)
    d = m - base_m
    table4[nm] = (d, sd)
    print(f"     {nm:10} "
          + " ".join(f"{d[i]:+.3f}[{sd[i]:.3f}]".rjust(16) for i in range(4)))

# ===== Task 2b: ownership-support dose robustness =====
print("="*70)
print(f"2b OWNERSHIP-SUPPORT DOSE ROBUSTNESS ({len(list(S_DEC))} seeds): Delta vs no-policy baseline")
print(f"     friction_mult is (1+effective_friction); leakage 0.5 halves the nominal dose")
for mult, fr in [("1x", -0.3), ("2x", -0.6), ("4x", -1.2)]:
    pol = dict(rent_cap_intensity=0.0, supply_restriction_intensity=0.0, transaction_friction=fr, **LEAK)
    m, sd = run_scen(pol, S_DEC)
    d = m - base_m
    print(f"     {mult:3s} (friction={fr:+.1f}, eff={fr*0.5:+.2f}, bar={1+fr*0.5:.2f}x): "
          f"Dd_access={d[2]:+.3f}[{sd[2]:.3f}]  Dvote={d[3]:+.3f}[{sd[3]:.3f}]")

# ===== Task 4b: gamma-split sensitivity band =====
print("="*70)
print("4b GAMMA-SPLIT SENSITIVITY BAND (asset/access reweighting, gamma1=0.48 fixed)")
# Splits chosen to roughly hold gamma2*E[d_asset]+gamma3*E[d_access] (renter contribution
# -> gap) constant; baseline E[d_asset]~0.254, E[d_access]~0.522 from the none scenario.
SPLITS = {"baseline (0.88,0.80)": (0.88, 0.80),
          "asset-heavy (1.30,0.60)": (1.30, 0.60),
          "access-heavy (0.37,1.05)": (0.37, 1.05)}


def tune_beta0(g2, g3, target=0.200):
    lo, hi = -3.2, -1.4
    for _ in range(5):
        mid = 0.5*(lo+hi)
        v, _ = mean_final_vote(lambda s: cal_cfg(s, gamma_asset=g2, gamma_access=g3, beta_0=mid),
                               seeds=range(73, 76))
        if v > target: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)


for name, (g2, g3) in SPLITS.items():
    b0 = tune_beta0(g2, g3)
    gam = dict(gamma_asset=g2, gamma_access=g3, beta_0=b0)
    # check level + gap at the tuned beta0
    lvl, _ = mean_final_vote(lambda s: cal_cfg(s, **gam))
    gg = []
    for s in S_CAL:
        vt = simulate(cal_cfg(s, **gam))[1].vote_by_tenure[-1]
        gg.extend((vt[:, 0]-vt[:, 1]).tolist())
    gap = float(np.mean(gg))
    # re-run the decomposition vote column for this split
    base_g, _ = run_scen(SCEN["none"], range(73, 79), gam)
    row = {}
    for nm in ("rent", "supply", "transfer", "access", "bundle"):
        m, _ = run_scen(SCEN[nm], range(73, 79), gam)
        row[nm] = (m - base_g)
    print(f"  {name}: beta0={b0:.2f}  level={lvl:.3f}  gap={gap:+.3f}")
    print("     Dvote: " + "  ".join(f"{nm}={row[nm][3]:+.3f}" for nm in ("rent","supply","transfer","access","bundle")))
    print("     Dd_access (should be ~invariant): " + "  ".join(f"{nm}={row[nm][2]:+.3f}" for nm in ("supply","transfer","bundle")))
print("="*70)
print("DONE")
