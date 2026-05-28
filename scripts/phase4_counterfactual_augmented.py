"""Phase 4 counterfactuals at the p75-augmented theta_hat.

Re-runs scenarios A-E from scripts/counterfactual_material_security.py with
the augmented 10-parameter calibration applied on top of the existing
scenario-specific activation overrides. Question: does Scenario E
peak-and-decay survive, and are housing-only scenarios still censored?

Configuration (same for every scenario):
    Apply outputs/smm_augmented_reduced.json theta_hat (10 params)
    assortative_help_enabled = True
    grad_share_data_path = data/cosmopolitan_grad_share_de.json
    aspiration_reference_quantile = 0.75 (default; explicit)
    estimate_beta_n / estimate_gamma_cosmopolitan flags on (SMM-only;
        do not affect simulation behaviour but kept for config provenance)

Scenario activation overrides (preserved from the original protocol):
    A: incumbency_threshold = 1.0 (prevents activation)
    B-E: beta_0 = -3.5 (forces activation)
    C, E: leakage profile

10 seeds, 25-period horizon. Central leakage profile (0.40 / 0.30 / 0.50)
for scenarios C and E. Plus Scenario E at low (0.20/0.15/0.25) and high
(0.60/0.50/0.70) leakage as the robustness sweep.

Output: outputs/augmented_phase4_counterfactual.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, PolicyRegime, simulate
from abmhp.estimation.smm import apply_params, build_param_space

# Reuse helpers from the original counterfactual script.
from counterfactual_material_security import (
    LOW, MEDIUM, HIGH, LeakageProfile,
    SCENARIO_ORDER, T_HORIZON, SEEDS,
    aggregate_rent_burden, aggregate_dissat, bottom50_resource_trajectory,
    years_extreme_share, tenure_dissat_gap,
    compute_half_life, half_life_summary,
)


OUTPUTS = ROOT / "outputs"
DE_GRAD_PATH = "data/cosmopolitan_grad_share_de.json"
TAU_K = 0.027  # matches original script


def _load_theta() -> tuple[np.ndarray, tuple]:
    payload = json.loads((OUTPUTS / "smm_augmented_reduced.json").read_text())
    theta = np.array(payload["theta_hat"], dtype=float)
    names = list(payload["param_names"])

    # Build the corresponding param_space (assumes augmented flags = True).
    sentinel = Config()
    sentinel = replace(sentinel, behavioral=replace(
        sentinel.behavioral,
        estimate_beta_n=True,
        estimate_gamma_cosmopolitan=True,
    ))
    ps = build_param_space(sentinel)
    assert [p.name for p in ps] == names, (
        f"param order mismatch: {[p.name for p in ps]} vs {names}"
    )
    return theta, ps


def make_augmented_config(
    scenario: str,
    seed: int,
    theta: np.ndarray,
    param_space: tuple,
    leakage: LeakageProfile = MEDIUM,
    n_periods: int = T_HORIZON,
) -> Config:
    """Build the augmented version of one scenario.

    1. Start from the augmented base flags (help on, grad-share path set).
    2. Apply the 10-parameter augmented theta_hat (sets beta_0 to ~-6.05).
    3. Apply the scenario-specific override on top, exactly as the original
       make_config does. For scenarios B-E this re-sets beta_0 to -3.5
       (forcing activation as in the original protocol); for scenario A it
       sets incumbency_threshold = 1.0 (preventing activation).
    """
    cfg = Config(seed=seed, n_periods=n_periods)
    cfg = replace(
        cfg,
        behavioral=replace(
            cfg.behavioral,
            assortative_help_enabled=True,
            estimate_beta_n=True,
            estimate_gamma_cosmopolitan=True,
        ),
        voting=replace(cfg.voting, grad_share_data_path=DE_GRAD_PATH),
    )
    cfg = apply_params(cfg, theta, param_space=param_space)

    if scenario == "A":
        cfg = replace(cfg, policy=replace(cfg.policy, incumbency_threshold=1.0))
        return cfg

    cfg = replace(cfg, voting=replace(cfg.voting, beta_0=-3.5))
    if scenario == "B":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_leakage=0.0, supply_leakage=0.0, friction_leakage=0.0,
        ))
    elif scenario == "C":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_leakage=leakage.rent_cap,
            supply_leakage=leakage.supply,
            friction_leakage=leakage.friction,
        ))
    elif scenario == "D":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_intensity=0.0, supply_restriction_intensity=0.0,
            transaction_friction=0.0,
        ))
    elif scenario == "E":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_leakage=leakage.rent_cap,
            supply_leakage=leakage.supply,
            friction_leakage=leakage.friction,
            redistribution_active=True,
            capital_tax_rate=TAU_K,
        ))
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return cfg


def run_scenario(scenario, theta, param_space, leakage=MEDIUM):
    runs = []
    for s in SEEDS:
        cfg = make_augmented_config(scenario, s, theta, param_space, leakage)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def summarise(runs, scenario_name):
    votes = np.stack([h.vote_aggregate for _, h in runs])
    rent = np.stack([aggregate_rent_burden(h, c) for c, h in runs])
    diss = np.stack([aggregate_dissat(h, c) for c, h in runs])
    years = np.array([years_extreme_share(h) for _, h in runs])
    gap = np.array([tenure_dissat_gap(h, c) for c, h in runs])

    pol_share = []
    for c, h in runs:
        active = np.array([r is PolicyRegime.POPULIST for r in h.regime])
        inc = h.income_aggregate[active]
        tr = h.transfer_aggregate[active]
        pol_share.append(float(tr.sum() / inc.sum()) if inc.sum() > 0 else 0.0)
    transfer_to_income = float(np.mean(pol_share)) if pol_share else 0.0

    return {
        "scenario": scenario_name,
        "vote_mean": votes.mean(axis=0).tolist(),
        "rent_mean": rent.mean(axis=0).tolist(),
        "vote_peak_period_mean": float(np.mean([int(np.argmax(h.vote_aggregate)) for _, h in runs])),
        "vote_peak_mean": float(np.mean([h.vote_aggregate.max() for _, h in runs])),
        "vote_final_mean": float(votes[:, -1].mean()),
        "rent_burden_final_mean": float(rent[:, -1].mean()),
        "dissat_final_mean": float(diss[:, -1].mean()),
        "years_extreme_share_mean": float(years.mean()),
        "tenure_gap_final_mean": float(gap.mean()),
        "transfer_to_income_share": transfer_to_income,
    }


def main() -> None:
    t0 = time.time()
    theta, param_space = _load_theta()
    names = [p.name for p in param_space]
    print(f"theta_hat (augmented): {dict(zip(names, [round(v, 4) for v in theta]))}")
    print(f"Running {len(SCENARIO_ORDER)} scenarios x {len(SEEDS)} seeds x "
          f"{T_HORIZON} periods at MEDIUM leakage\n")

    scenario_runs = {}
    summaries = {}
    for s in SCENARIO_ORDER:
        ts = time.time()
        scenario_runs[s] = run_scenario(s, theta, param_space, MEDIUM)
        summaries[s] = summarise(scenario_runs[s], s)
        print(f"  scenario {s} done ({time.time()-ts:.1f}s)  "
              f"vote_peak={summaries[s]['vote_peak_mean']:.3f}  "
              f"vote_final={summaries[s]['vote_final_mean']:.3f}  "
              f"years_ext={summaries[s]['years_extreme_share_mean']:.1f}")

    # Half-lives.
    half_lives = {"A": {"median": 0.0, "share_censored": 0.0, "n_censored": 0,
                          "n_total": len(SEEDS), "mean_uncensored": 0.0, "sd_uncensored": 0.0,
                          "raw": [0.0]*len(SEEDS)}}
    for s in ["B", "C", "D", "E"]:
        half_lives[s] = half_life_summary(scenario_runs[s], scenario_runs["A"])

    # Leakage robustness for Scenario E (and C for context).
    print("\nLeakage robustness: Scenario C, E at LOW and HIGH")
    rob = []
    for profile in (LOW, HIGH):
        for s in ("C", "E"):
            ts = time.time()
            runs = run_scenario(s, theta, param_space, profile)
            hl = half_life_summary(runs, scenario_runs["A"])
            rob.append({
                "scenario": s, "leakage": profile.label,
                "rent_leakage": profile.rent_cap,
                "supply_leakage": profile.supply,
                "friction_leakage": profile.friction,
                "median": hl["median"],
                "mean_uncensored": hl["mean_uncensored"],
                "share_censored": hl["share_censored"],
                "n_censored": hl["n_censored"],
                "n_total": hl["n_total"],
                "vote_peak_mean": float(np.mean([h.vote_aggregate.max() for _, h in runs])),
                "vote_final_mean": float(np.mean([h.vote_aggregate[-1] for _, h in runs])),
            })
            print(f"  {s} @ {profile.label}: median_half_life={hl['median']:.1f}  "
                  f"censored={hl['n_censored']}/{hl['n_total']}  ({time.time()-ts:.1f}s)")
    # Add medium row for completeness.
    for s in ("C", "E"):
        hl = half_lives[s]
        rob.append({
            "scenario": s, "leakage": "medium",
            "rent_leakage": MEDIUM.rent_cap,
            "supply_leakage": MEDIUM.supply,
            "friction_leakage": MEDIUM.friction,
            "median": hl["median"],
            "mean_uncensored": hl["mean_uncensored"],
            "share_censored": hl["share_censored"],
            "n_censored": hl["n_censored"],
            "n_total": hl["n_total"],
            "vote_peak_mean": summaries[s]["vote_peak_mean"],
            "vote_final_mean": summaries[s]["vote_final_mean"],
        })

    # Load original Scenario E for comparison.
    orig_npz = OUTPUTS / "counterfactual_data.npz"
    orig_e_comparison = None
    if orig_npz.exists():
        with np.load(orig_npz, allow_pickle=True) as data:
            orig_e_vote = data["E_vote_mean"]
            orig_e_peak_t = int(np.argmax(orig_e_vote))
            orig_e_comparison = {
                "vote_peak_mean": float(orig_e_vote.max()),
                "vote_peak_period": orig_e_peak_t,
                "vote_final_mean": float(orig_e_vote[-1]),
                "source": str(orig_npz.name),
            }

    payload = {
        "phase": "4_counterfactual_augmented",
        "theta_hat": theta.tolist(),
        "param_names": names,
        "seeds": SEEDS,
        "t_horizon": T_HORIZON,
        "scenarios": summaries,
        "half_lives": {k: {kk: vv for kk, vv in v.items() if kk != "raw"}
                         for k, v in half_lives.items()},
        "half_life_raw": {k: v.get("raw", []) for k, v in half_lives.items()},
        "leakage_robustness": rob,
        "original_e_comparison": orig_e_comparison,
        "config_flags": {
            "assortative_help_enabled": True,
            "grad_share_data_path": DE_GRAD_PATH,
            "aspiration_reference_quantile": 0.75,
        },
        "elapsed_seconds": time.time() - t0,
    }
    out = OUTPUTS / "augmented_phase4_counterfactual.json"
    out.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------- surface
    print()
    print("=" * 100)
    print("PHASE 4 COUNTERFACTUAL: AUGMENTED THETA_HAT (medium leakage)")
    print("=" * 100)
    print(f"{'scenario':>4s}  {'years_ext':>10s}  {'vote_peak':>10s}  "
          f"{'peak_t':>7s}  {'vote_final':>10s}  {'rent_T':>7s}  "
          f"{'transfer/Y':>11s}  {'half_life':>10s}  {'censored':>9s}")
    for s in SCENARIO_ORDER:
        sm = summaries[s]
        hl = half_lives[s]
        hl_str = (f"{hl['median']:.1f}" if hl['share_censored'] < 0.5
                  else f">={T_HORIZON}")
        print(
            f"{s:>4s}  {sm['years_extreme_share_mean']:>10.1f}  "
            f"{sm['vote_peak_mean']:>10.3f}  "
            f"{sm['vote_peak_period_mean']:>7.1f}  "
            f"{sm['vote_final_mean']:>10.3f}  "
            f"{sm['rent_burden_final_mean']:>7.3f}  "
            f"{sm['transfer_to_income_share']:>11.4f}  "
            f"{hl_str:>10s}  "
            f"{hl['n_censored']:>3d}/{hl['n_total']:>3d}"
        )

    if orig_e_comparison is not None:
        print()
        print("=" * 100)
        print("SCENARIO E: original vs augmented")
        print("=" * 100)
        eo = orig_e_comparison
        ea = summaries["E"]
        print(f"  {'metric':22s}  {'original':>10s}  {'augmented':>10s}  {'delta':>10s}")
        print(f"  {'vote peak (mean)':22s}  {eo['vote_peak_mean']:>10.3f}  "
              f"{ea['vote_peak_mean']:>10.3f}  {ea['vote_peak_mean']-eo['vote_peak_mean']:+10.3f}")
        print(f"  {'peak period':22s}  {eo['vote_peak_period']:>10d}  "
              f"{ea['vote_peak_period_mean']:>10.1f}  "
              f"{ea['vote_peak_period_mean']-eo['vote_peak_period']:+10.1f}")
        print(f"  {'vote final':22s}  {eo['vote_final_mean']:>10.3f}  "
              f"{ea['vote_final_mean']:>10.3f}  "
              f"{ea['vote_final_mean']-eo['vote_final_mean']:+10.3f}")

    print()
    print("=" * 100)
    print("LEAKAGE ROBUSTNESS (Table 7 equivalent)")
    print("=" * 100)
    print(f"{'scen':>4s}  {'leak':>7s}  {'rent/sup/fric':>14s}  {'half_life':>10s}  "
          f"{'censored':>9s}  {'peak':>7s}  {'final':>7s}")
    rob_order = sorted(rob, key=lambda r: (r["scenario"],
                                              {"low":0,"medium":1,"high":2}[r["leakage"]]))
    for r in rob_order:
        hl_str = (f"{r['median']:.1f}" if r['share_censored'] < 0.5
                  else f">={T_HORIZON}")
        leak = f"{r['rent_leakage']:.2f}/{r['supply_leakage']:.2f}/{r['friction_leakage']:.2f}"
        print(f"{r['scenario']:>4s}  {r['leakage']:>7s}  {leak:>14s}  {hl_str:>10s}  "
              f"{r['n_censored']:>3d}/{r['n_total']:>3d}  "
              f"{r['vote_peak_mean']:>7.3f}  {r['vote_final_mean']:>7.3f}")

    print(f"\nSaved {out.relative_to(ROOT)}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
