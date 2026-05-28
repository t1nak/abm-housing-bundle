"""Equal-cost counterfactual: isolate channel composition from fiscal intensity.

Scenario E (the bundled intervention) commits a transfer worth approximately
5 percent of aggregate household income. Scenarios B, C, D commit no public
budget. A referee can attribute E's peak-and-decay to the fiscal-intensity
gap rather than to channel composition.

Two new scenarios disambiguate:

  E-light: same instruments as E, but capital-tax rate scaled down so the
           realised transfer / aggregate-income share is approximately 1.5
           percent (a modest housing-allowance expansion).

  C-plus:  housing-only at central leakage (rent_cap_leakage=0.40,
           supply_leakage=0.30, friction_leakage=0.50), but rent_cap_intensity
           and supply_restriction_intensity dialled up until the integrated
           renter mean delta-o over the active period matches Scenario E.

Both scenarios are run at 10 seeds, T=25, central leakage, with the headline
8-parameter calibration (augmented-model flags off).

Output:
  outputs/equal_cost_counterfactual.json (full per-seed payload + calibration log)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace, dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, PolicyRegime, simulate
from abmhp.config import PolicyConfig, VotingConfig
from abmhp.estimation.smm import apply_smm_optimum

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

T_HORIZON = 25
SEEDS = list(range(73, 83))
ACTIVE_WINDOW = (11, 25)  # inclusive on both ends

TAU_K_HEADLINE = 0.027
TAU_K_LIGHT_START = 0.0081
CENTRAL_LEAKAGE = {"rent_cap_leakage": 0.40, "supply_leakage": 0.30, "friction_leakage": 0.50}


def make_baseline_A(seed: int) -> Config:
    cfg = apply_smm_optimum(Config(seed=seed, n_periods=T_HORIZON))
    return replace(cfg, policy=replace(cfg.policy, incumbency_threshold=1.0))


def make_E(seed: int, tau_k: float = TAU_K_HEADLINE) -> Config:
    cfg = apply_smm_optimum(Config(seed=seed, n_periods=T_HORIZON))
    cfg = replace(cfg, voting=replace(cfg.voting, beta_0=-3.5))
    return replace(cfg, policy=replace(
        cfg.policy,
        rent_cap_leakage=CENTRAL_LEAKAGE["rent_cap_leakage"],
        supply_leakage=CENTRAL_LEAKAGE["supply_leakage"],
        friction_leakage=CENTRAL_LEAKAGE["friction_leakage"],
        redistribution_active=True,
        capital_tax_rate=tau_k,
    ))


def make_C_plus(seed: int, rent_mult: float, supply_mult: float) -> Config:
    """Housing-only at central leakage with rent/supply intensities dialled up."""
    cfg = apply_smm_optimum(Config(seed=seed, n_periods=T_HORIZON))
    cfg = replace(cfg, voting=replace(cfg.voting, beta_0=-3.5))
    base_rent = cfg.policy.rent_cap_intensity
    base_supply = cfg.policy.supply_restriction_intensity
    return replace(cfg, policy=replace(
        cfg.policy,
        rent_cap_intensity=base_rent * rent_mult,
        supply_restriction_intensity=base_supply * supply_mult,
        rent_cap_leakage=CENTRAL_LEAKAGE["rent_cap_leakage"],
        supply_leakage=CENTRAL_LEAKAGE["supply_leakage"],
        friction_leakage=CENTRAL_LEAKAGE["friction_leakage"],
    ))


def make_C_central(seed: int) -> Config:
    """Reference Scenario C at central leakage, headline rent/supply intensities."""
    return make_C_plus(seed, rent_mult=1.0, supply_mult=1.0)


def run_scenario(make_cfg, seeds: Iterable[int] = SEEDS, **kwargs):
    runs = []
    for s in seeds:
        cfg = make_cfg(s, **kwargs) if kwargs else make_cfg(s)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def integrated_renter_o(hist) -> float:
    """Renter-period-weighted mean of renter_mean_o over the active window.

    integral = sum_t n_renters[t] * renter_mean_o[t] / sum_t n_renters[t],
    summed over t in [ACTIVE_WINDOW[0], ACTIVE_WINDOW[1]] inclusive.

    Zero-renter periods are excluded: simulate() leaves renter_mean_o[t] as
    NaN when n_renters[t] == 0, and 0 * NaN is NaN in NumPy, so a single
    zero-renter period would poison the sum even when other periods carry
    valid renter data. Masking to n > 0 is the defensive guard.
    """
    t0, t1 = ACTIVE_WINDOW
    n = hist.n_renters[t0 : t1 + 1].astype(float)
    o = hist.renter_mean_o[t0 : t1 + 1]
    mask = n > 0
    if not mask.any():
        return float("nan")
    n_m = n[mask]
    o_m = o[mask]
    denom = n_m.sum()
    if denom <= 0:
        return float("nan")
    return float((n_m * o_m).sum() / denom)


def transfer_share_active(hist) -> float:
    """Cumulative transfer / cumulative income across periods where the
    activation regime is in force."""
    active = np.array([r is PolicyRegime.POPULIST for r in hist.regime])
    if not active.any():
        return 0.0
    inc = hist.income_aggregate[active]
    tr = hist.transfer_aggregate[active]
    return float(tr.sum() / inc.sum()) if inc.sum() > 0 else 0.0


def rent_index_attenuation(hist_scenario, hist_baseline) -> float:
    """Mean rent_index ratio over active window: scenario / baseline.

    A value < 0.10 indicates the rent cap has attenuated rent growth by more
    than 90 percent relative to no-policy baseline (Scenario A). Population-
    weighted across regions is left implicit; rent_index already starts at
    1.0 per region so the comparison is per-region symmetric. We average
    the per-region ratio over time and regions.
    """
    t0, t1 = ACTIVE_WINDOW
    r_scen = hist_scenario.rent_index[t0 : t1 + 1]
    r_base = hist_baseline.rent_index[t0 : t1 + 1]
    ratio = r_scen / np.where(r_base > 0, r_base, 1.0)
    return float(ratio.mean())


def years_extreme_share(hist) -> int:
    return int(sum(1 for r in hist.regime[1:] if r is PolicyRegime.POPULIST))


def summarise(runs, baseline_runs) -> dict:
    """Per-scenario summary metrics. baseline_runs is Scenario A at the
    same seeds (paired)."""
    votes = np.stack([h.vote_aggregate for _, h in runs])
    seed_integrals = np.array([integrated_renter_o(h) for _, h in runs])
    seed_baseline = np.array([integrated_renter_o(h) for _, h in baseline_runs])
    seed_delta_o = seed_integrals - seed_baseline
    transfer_shares = np.array([transfer_share_active(h) for _, h in runs])
    years = np.array([years_extreme_share(h) for _, h in runs])

    # Half-life from peak vs scenario A.
    half_lives = []
    censored = 0
    for (_, h), (_, hb) in zip(runs, baseline_runs):
        scen_v = h.vote_aggregate
        base_v = hb.vote_aggregate
        peak_t = int(np.argmax(scen_v))
        target = base_v * 1.25
        hl = None
        for tt in range(peak_t, len(scen_v)):
            if scen_v[tt] <= target[tt]:
                hl = float(tt - peak_t)
                break
        if hl is None:
            censored += 1
            half_lives.append(np.nan)
        else:
            half_lives.append(hl)
    hl_arr = np.array(half_lives, dtype=float)
    finite = hl_arr[np.isfinite(hl_arr)]
    median_hl = float(np.nanmedian(hl_arr)) if finite.size else float("nan")

    # Peak / final vote
    peak_t_per_seed = np.array([int(np.argmax(v)) for v in votes])
    peak_v_per_seed = votes.max(axis=1)
    vote_T = votes[:, -1]

    # Rent burden at T (cribbed from counterfactual_material_security).
    rent_burden = []
    for c, h in runs:
        rent_yield = c.behavioral.rent_yield
        burden_share = c.behavioral.rent_burden_share
        initial_price = h.price[0]
        renter_share = 1.0 - h.ownership
        pop_share = c.regional.pop_share
        rent_level = initial_price * rent_yield * h.rent_index[-1] * burden_share
        income_mean = np.where(h.mean_income[-1] > 0, h.mean_income[-1], 1.0)
        per_region_burden = rent_level / income_mean
        renter_count_share = pop_share * renter_share[-1]
        denom = renter_count_share.sum()
        rent_burden.append(float((per_region_burden * renter_count_share).sum() / max(denom, 1e-12)))
    rent_burden = np.array(rent_burden)

    return {
        "n_seeds": len(runs),
        "years_extreme_share_mean": float(years.mean()),
        "years_extreme_share_min": int(years.min()),
        "years_extreme_share_max": int(years.max()),
        "vote_peak_mean": float(peak_v_per_seed.mean()),
        "vote_peak_period_mean": float(peak_t_per_seed.mean()),
        "vote_T_mean": float(vote_T.mean()),
        "rent_burden_T_mean": float(rent_burden.mean()),
        "transfer_to_income_share_mean": float(transfer_shares.mean()),
        "transfer_to_income_share_sd": float(transfer_shares.std(ddof=1)) if len(transfer_shares) > 1 else 0.0,
        "median_half_life": median_hl,
        "n_censored": int(censored),
        "renter_o_integral_mean": float(seed_integrals.mean()),
        "renter_o_integral_baseline_mean": float(seed_baseline.mean()),
        "renter_delta_o_paired_mean": float(seed_delta_o.mean()),
        "renter_delta_o_paired_sd": float(seed_delta_o.std(ddof=1)) if len(seed_delta_o) > 1 else 0.0,
        "per_seed_vote_T": vote_T.tolist(),
        "per_seed_renter_delta_o": seed_delta_o.tolist(),
        "per_seed_transfer_share": transfer_shares.tolist(),
        "per_seed_half_life": half_lives,
    }


def calibrate_tau_k_light(runs_A, target_share: float = 0.015, tol: float = 0.001,
                          max_iter: int = 6) -> tuple[float, list[dict]]:
    """Iteratively tune TAU_K_light so the realised transfer/income share
    sits in [target - tol, target + tol]. Linear-search using last two points."""
    log = []
    tau = TAU_K_LIGHT_START
    history = []  # (tau, share)
    for it in range(max_iter):
        runs = run_scenario(make_E, tau_k=tau)
        shares = np.array([transfer_share_active(h) for _, h in runs])
        share_mean = float(shares.mean())
        log.append({"iter": it, "tau_k": tau, "transfer_share": share_mean})
        print(f"  iter {it}: tau_k={tau:.6f}, realised transfer/income={share_mean:.4f}")
        history.append((tau, share_mean))
        if abs(share_mean - target_share) <= tol:
            return tau, log
        # Linear rescale: tau_new = tau * (target / current)
        if share_mean > 0:
            tau = tau * (target_share / share_mean)
        else:
            tau = tau * 2.0
    return tau, log


def calibrate_C_plus(runs_A, target_delta_o: float, rel_tol: float = 0.05,
                     max_iter: int = 15, mult_ceiling: float = 12.0) -> tuple[float, float, list[dict], dict]:
    """Tune rent_mult and supply_mult proportionally so integrated renter
    delta-o for C-plus matches target_delta_o within rel_tol. Starts at
    (rent_mult=2.0, supply_mult=1.5) and rescales both by a common factor.
    Returns (rent_mult, supply_mult, log, final_runs_summary).

    Feasibility: if mean rent_index ratio (C-plus / A) over active window
    drops below 0.10, the rent cap is at >90% attenuation; we stop tuning
    and flag the binding constraint.
    """
    log = []
    rent_mult = 2.0
    supply_mult = 1.5
    last_attenuation = None
    last_delta_o = None
    last_runs = None
    for it in range(max_iter):
        runs = run_scenario(lambda s: make_C_plus(s, rent_mult, supply_mult))
        seed_integrals = np.array([integrated_renter_o(h) for _, h in runs])
        seed_baseline = np.array([integrated_renter_o(h) for _, h in runs_A])
        seed_delta_o = seed_integrals - seed_baseline
        delta_o_mean = float(seed_delta_o.mean())
        # Attenuation: average rent_index ratio C-plus / A across active window
        attens = []
        for (_, h_s), (_, h_a) in zip(runs, runs_A):
            attens.append(rent_index_attenuation(h_s, h_a))
        atten_mean = float(np.mean(attens))
        last_attenuation = atten_mean
        last_delta_o = delta_o_mean
        last_runs = runs
        rel_gap = (delta_o_mean - target_delta_o) / target_delta_o if target_delta_o != 0 else 0.0
        log.append({
            "iter": it, "rent_mult": rent_mult, "supply_mult": supply_mult,
            "delta_o": delta_o_mean, "target": target_delta_o,
            "rel_gap": rel_gap, "rent_atten_mean": atten_mean,
        })
        print(f"  iter {it}: rent_mult={rent_mult:.3f}, supply_mult={supply_mult:.3f}, "
              f"delta_o={delta_o_mean:.1f}, target={target_delta_o:.1f}, "
              f"rel_gap={rel_gap:+.3f}, rent_atten={atten_mean:.3f}")
        if atten_mean < 0.10:
            print(f"  ** Binding constraint: rent attenuation {atten_mean:.3f} < 0.10, halting")
            break
        if abs(rel_gap) <= rel_tol:
            return rent_mult, supply_mult, log, {
                "renter_delta_o": delta_o_mean,
                "rent_atten_mean": atten_mean,
                "converged": True,
            }
        # Rescale both multipliers proportionally to close gap.
        # Heuristic: increase intensity ⇒ increase delta-o (lower rents help renters).
        # If delta_o < target, intensities need to go up. Use sqrt for damping.
        if delta_o_mean < target_delta_o and delta_o_mean > 0:
            factor = float(np.sqrt(target_delta_o / delta_o_mean))
        elif delta_o_mean > target_delta_o and target_delta_o > 0:
            factor = float(np.sqrt(target_delta_o / delta_o_mean))
        else:
            factor = 1.2
        rent_mult = float(np.clip(rent_mult * factor, 1.0, mult_ceiling))
        supply_mult = float(np.clip(supply_mult * factor, 1.0, mult_ceiling))
        # If both already at ceiling and we haven't converged, halt.
        if rent_mult >= mult_ceiling and supply_mult >= mult_ceiling and abs(rel_gap) > rel_tol:
            # Run once more with ceiling to confirm last-point delta-o, then exit.
            runs = run_scenario(lambda s: make_C_plus(s, rent_mult, supply_mult))
            seed_integrals = np.array([integrated_renter_o(h) for _, h in runs])
            seed_baseline = np.array([integrated_renter_o(h) for _, h in runs_A])
            seed_delta_o = seed_integrals - seed_baseline
            delta_o_mean = float(seed_delta_o.mean())
            attens = [rent_index_attenuation(h_s, h_a)
                      for (_, h_s), (_, h_a) in zip(runs, runs_A)]
            atten_mean = float(np.mean(attens))
            log.append({
                "iter": it + 1, "rent_mult": rent_mult, "supply_mult": supply_mult,
                "delta_o": delta_o_mean, "target": target_delta_o,
                "rel_gap": (delta_o_mean - target_delta_o) / target_delta_o,
                "rent_atten_mean": atten_mean,
                "note": "ceiling-pinned highest-feasible",
            })
            print(f"  ** Both multipliers at ceiling ({mult_ceiling}), "
                  f"halting at highest-feasible: delta_o={delta_o_mean:.1f}, "
                  f"rent_atten={atten_mean:.3f}")
            return rent_mult, supply_mult, log, {
                "renter_delta_o": delta_o_mean,
                "rent_atten_mean": atten_mean,
                "converged": False,
                "ceiling_pinned": True,
            }
    return rent_mult, supply_mult, log, {
        "renter_delta_o": last_delta_o,
        "rent_atten_mean": last_attenuation,
        "converged": False,
        "final_runs": last_runs,
    }


def main():
    t_start = time.time()
    payload: dict = {
        "protocol": {
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "horizon": T_HORIZON,
            "active_window": list(ACTIVE_WINDOW),
            "leakage_profile": "central",
            "central_leakage": CENTRAL_LEAKAGE,
            "tau_k_headline": TAU_K_HEADLINE,
            "tau_k_light_start": TAU_K_LIGHT_START,
            "headline_calibration": True,
            "augmented_model_flags": {"assortative_help_enabled": False,
                                       "estimate_beta_n": False,
                                       "estimate_gamma_cosmopolitan": False},
        },
    }

    print("=" * 78)
    print("STEP 1: Scenario A (baseline) and reference Scenario C (central leakage)")
    print("=" * 78)
    runs_A = run_scenario(make_baseline_A)
    runs_C = run_scenario(make_C_central)
    print(f"  A: {len(runs_A)} seeds done, t={time.time()-t_start:.1f}s")

    print()
    print("=" * 78)
    print("STEP 2: Headline Scenario E (TAU_K = 0.027)")
    print("=" * 78)
    runs_E = run_scenario(make_E, tau_k=TAU_K_HEADLINE)
    summary_E = summarise(runs_E, runs_A)
    target_delta_o = summary_E["renter_delta_o_paired_mean"]
    print(f"  E transfer/income: {summary_E['transfer_to_income_share_mean']:.4f}")
    print(f"  E renter delta-o integral (paired vs A): {target_delta_o:.1f}")

    print()
    print("=" * 78)
    print("STEP 3: Calibrate TAU_K_light to hit 1.5% transfer/income")
    print("=" * 78)
    tau_k_light, light_log = calibrate_tau_k_light(runs_A, target_share=0.015, tol=0.001)
    runs_E_light = run_scenario(make_E, tau_k=tau_k_light)
    summary_E_light = summarise(runs_E_light, runs_A)
    print(f"  Final TAU_K_light = {tau_k_light:.6f}")
    print(f"  Realised transfer/income: {summary_E_light['transfer_to_income_share_mean']:.4f}")
    print(f"  E-light vote_T: {summary_E_light['vote_T_mean']:.3f}, "
          f"peak: {summary_E_light['vote_peak_mean']:.3f} at "
          f"t={summary_E_light['vote_peak_period_mean']:.1f}")

    print()
    print("=" * 78)
    print("STEP 4: Calibrate C-plus rent/supply multipliers to match E's delta-o")
    print("=" * 78)
    rent_mult, supply_mult, cplus_log, cplus_meta = calibrate_C_plus(
        runs_A, target_delta_o, rel_tol=0.05
    )
    # If calibration didn't converge but didn't bind, do a final clean run
    # to populate metrics. If it bound, also run the highest-feasible spec.
    runs_C_plus = run_scenario(lambda s: make_C_plus(s, rent_mult, supply_mult))
    summary_C_plus = summarise(runs_C_plus, runs_A)
    final_delta_o = summary_C_plus["renter_delta_o_paired_mean"]
    final_atten = float(np.mean([
        rent_index_attenuation(h, ha)
        for (_, h), (_, ha) in zip(runs_C_plus, runs_A)
    ]))
    converged = abs((final_delta_o - target_delta_o) / target_delta_o) <= 0.05
    binding = final_atten < 0.10
    print(f"  Final rent_mult={rent_mult:.3f}, supply_mult={supply_mult:.3f}")
    print(f"  C-plus delta-o: {final_delta_o:.1f}, target {target_delta_o:.1f}, "
          f"residual gap {(final_delta_o - target_delta_o):+.1f} "
          f"({((final_delta_o - target_delta_o) / target_delta_o):+.1%})")
    print(f"  Rent attenuation (mean ratio C-plus/A): {final_atten:.3f} "
          f"{'(BINDING)' if binding else ''}")

    # Summarise headline Scenario C for table parity
    summary_C = summarise(runs_C, runs_A)

    # Also summarise A vs itself (so per-seed baselines are present)
    summary_A = summarise(runs_A, runs_A)

    payload["scenarios"] = {
        "A": summary_A,
        "C_central": summary_C,
        "E": summary_E,
        "E_light": summary_E_light,
        "C_plus": summary_C_plus,
    }
    payload["calibration"] = {
        "tau_k_light_final": tau_k_light,
        "tau_k_light_realised_share": summary_E_light["transfer_to_income_share_mean"],
        "tau_k_light_target": 0.015,
        "tau_k_light_log": light_log,
        "C_plus_rent_mult": rent_mult,
        "C_plus_supply_mult": supply_mult,
        "C_plus_delta_o_target": target_delta_o,
        "C_plus_delta_o_realised": final_delta_o,
        "C_plus_residual_gap_relative": (final_delta_o - target_delta_o) / target_delta_o,
        "C_plus_rent_atten_mean": final_atten,
        "C_plus_binding_constraint": binding,
        "C_plus_converged_within_5pct": converged,
        "C_plus_log": cplus_log,
    }
    payload["wallclock_seconds"] = time.time() - t_start

    out_path = OUTPUTS / "equal_cost_counterfactual.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print()
    print(f"Wrote {out_path}")
    print(f"Total wallclock: {payload['wallclock_seconds']:.1f}s")

    # Brief Table 6 preview
    print()
    print("=" * 78)
    print("TABLE 6 PREVIEW (E-light, C-plus columns)")
    print("=" * 78)
    cols = ["A", "C_central", "E", "E_light", "C_plus"]
    rows = [
        ("years_extreme_share_mean", "Years activated"),
        ("vote_peak_mean", "Peak vote"),
        ("vote_peak_period_mean", "Period of peak"),
        ("vote_T_mean", "Vote at T=25"),
        ("rent_burden_T_mean", "Rent burden T=25"),
        ("transfer_to_income_share_mean", "Transfer/income"),
        ("median_half_life", "Median half-life"),
        ("n_censored", "Censored seeds"),
        ("renter_delta_o_paired_mean", "Renter integral o vs A"),
    ]
    header = f"{'metric':<28s}" + "".join(f"{c:>14s}" for c in cols)
    print(header)
    for key, label in rows:
        line = f"{label:<28s}"
        for c in cols:
            v = payload["scenarios"][c][key]
            if isinstance(v, float):
                if "share" in key or "rent_burden" in key:
                    line += f"{v:>14.4f}"
                elif "vote" in key or "half_life" in key:
                    line += f"{v:>14.3f}"
                else:
                    line += f"{v:>14.2f}"
            else:
                line += f"{v:>14}"
        print(line)


if __name__ == "__main__":
    main()
