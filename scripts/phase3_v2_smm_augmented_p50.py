"""Phase 3 v2 augmented SMM (reduced budget, p50 reference, cold-start).

Configuration:
    aspiration_reference_quantile = 0.50 (switch from p75 to p50; the Tier
        2C diagnostic showed p50 alone moves DE cleavage from 0.278 to
        0.166 at fixed theta_hat; this run gives the SMM both the
        augmented free-parameter space AND the reference-point change)
    assortative_help_enabled = True (Phase 1A revised mechanism)
    p_assortative_help = 0.05, f_assortative_help = 0.15
    estimate_beta_n = True
    estimate_gamma_cosmopolitan = True
    grad_share_data_path = data/cosmopolitan_grad_share_de.json

SMM parameter space (10 parameters, 8 original + beta_n + gamma):
    beta_dissat upper bound widened 9.0 -> 12.0 (was pinned at the bound
    in Phase 3 reduced).

Initialization: cold-start (no x0_stage1). Warm-starting from the original
theta_hat (estimated under p75) would put the BO in the wrong basin under
p50 dynamics. 24 Sobol initial points (up from 12 in Phase 3 reduced).

Budget (same as Phase 3 reduced):
    Stage 1: 60 calls
    Stage 2: 30 calls
    Variance replications: 10
    Sobol initial: 24

Output: outputs/smm_augmented_p50_reduced.json
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

from abmhp import Config
from abmhp.estimation.moments import (
    CALIBRATION_MOMENTS,
    VALIDATION_MOMENTS,
    evaluate_moments,
    simulate_seeds,
)
from abmhp.estimation.smm import (
    apply_params,
    build_param_space,
    run_smm,
)


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DE_GRAD_PATH = "data/cosmopolitan_grad_share_de.json"

N_FIRST_STAGE = 60
N_SECOND_STAGE = 30
N_VARIANCE_REPLICATIONS = 10
N_INITIAL_POINTS_STAGE1 = 24
N_INITIAL_POINTS_STAGE2 = 10
N_SEEDS_PER_EVAL = 5
SEEDS_OFFSET = 73
RANDOM_STATE = 42


def _build_base_cfg() -> Config:
    """Augmented p50 base config."""
    cfg = Config()
    behavioral = replace(
        cfg.behavioral,
        assortative_help_enabled=True,
        estimate_beta_n=True,
        estimate_gamma_cosmopolitan=True,
    )
    voting = replace(
        cfg.voting,
        aspiration_reference_quantile=0.50,  # p50 reference, Tier 2C lever
        grad_share_data_path=DE_GRAD_PATH,
    )
    return replace(cfg, behavioral=behavioral, voting=voting)


def _uk_cleavage_at_theta(theta_hat: np.ndarray, param_space, base_cfg: Config,
                           seeds: list[int]) -> dict:
    """Score UK cleavage under the augmented + p50 specification at theta_hat."""
    from abmhp.validation.uk import BREXIT_PERIOD, make_uk_config
    from abmhp import simulate

    UK_GRAD_PATH = "data/cosmopolitan_grad_share_uk.json"
    runs = []
    for s in seeds:
        cfg = make_uk_config(seed=s, n_periods=14)
        cfg = apply_params(cfg, theta_hat, param_space=param_space)
        # Apply the same configuration toggles the DE base_cfg has.
        cfg = replace(
            cfg,
            behavioral=replace(
                cfg.behavioral,
                assortative_help_enabled=True,
            ),
            voting=replace(
                cfg.voting,
                aspiration_reference_quantile=0.50,
                grad_share_data_path=UK_GRAD_PATH,
            ),
        )
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))

    t = BREXIT_PERIOD
    cleavage_vals = []
    dispersion_vals = []
    agg_vals = []
    for _, h in runs:
        rent_v = h.vote_by_tenure[t, :, 0]
        own_v = h.vote_by_tenure[t, :, 1]
        cleavage_vals.append(float((rent_v - own_v).mean()))
        dispersion_vals.append(float(h.vote[t].std(ddof=1)))
        agg_vals.append(float(h.vote_aggregate[t]))
    return {
        "uk_aggregate_leave": float(np.mean(agg_vals)),
        "uk_cleavage": float(np.mean(cleavage_vals)),
        "uk_dispersion": float(np.mean(dispersion_vals)),
    }


def main() -> None:
    t0 = time.time()
    base_cfg = _build_base_cfg()
    param_space = build_param_space(base_cfg)
    param_names = [ps.name for ps in param_space]
    assert len(param_space) == 10, f"expected 10 params, got {len(param_space)}"

    print("=" * 78)
    print("PHASE 3 v2 augmented SMM (p50, reduced budget, cold-start)")
    print("=" * 78)
    print(f"Param space ({len(param_space)} parameters):")
    for ps in param_space:
        print(f"  {ps.name:32s}  bounds = [{ps.low:+.3f}, {ps.high:+.3f}]")
    print(f"Budget: Stage1={N_FIRST_STAGE} (Sobol={N_INITIAL_POINTS_STAGE1}, "
          f"COLD-START), Stage2={N_SECOND_STAGE} (Sobol={N_INITIAL_POINTS_STAGE2}), "
          f"VarReps={N_VARIANCE_REPLICATIONS}")
    print(f"base_cfg flags: aspiration_reference_quantile="
          f"{base_cfg.voting.aspiration_reference_quantile}, "
          f"assortative_help_enabled={base_cfg.behavioral.assortative_help_enabled}, "
          f"grad_share_data_path={base_cfg.voting.grad_share_data_path}")
    print()

    result = run_smm(
        base_cfg=base_cfg,
        n_seeds_per_eval=N_SEEDS_PER_EVAL,
        seeds_offset=SEEDS_OFFSET,
        n_first_stage=N_FIRST_STAGE,
        n_second_stage=N_SECOND_STAGE,
        n_variance_replications=N_VARIANCE_REPLICATIONS,
        random_state=RANDOM_STATE,
        verbose=True,
        param_space=param_space,
        n_initial_points_stage1=N_INITIAL_POINTS_STAGE1,
        n_initial_points_stage2=N_INITIAL_POINTS_STAGE2,
        x0_stage1=None,  # cold-start
    )

    # Validation moments at the augmented optimum (DE).
    cfg_at_opt = apply_params(base_cfg, result.theta_hat, param_space=param_space)
    val_runs = simulate_seeds(cfg_at_opt, list(range(73, 78)))
    val_sim = evaluate_moments(VALIDATION_MOMENTS, val_runs)
    validation = {}
    for m in VALIDATION_MOMENTS:
        model = float(val_sim[m.name])
        err = float("nan") if not np.isfinite(model) else float(model - m.value)
        passed = (
            bool(np.isfinite(err) and abs(err) <= m.target_tolerance)
            if np.isfinite(err) else False
        )
        validation[m.name] = {
            "target": float(m.value),
            "model": model,
            "error": err,
            "tolerance": float(m.target_tolerance),
            "passed": passed,
        }

    # UK cleavage / dispersion / aggregate at the same theta_hat.
    uk_metrics = _uk_cleavage_at_theta(
        result.theta_hat, param_space, base_cfg, list(range(73, 83))
    )

    # Calibration fit summary.
    fit_rows = []
    n_pass = 0
    for m, tgt, mod in zip(CALIBRATION_MOMENTS, result.moment_targets,
                             result.moment_at_optimum):
        err = float(mod - tgt)
        passed = bool(abs(err) <= m.target_tolerance)
        if passed:
            n_pass += 1
        fit_rows.append({
            "name": m.name,
            "target": float(tgt),
            "model": float(mod),
            "error": err,
            "tolerance": float(m.target_tolerance),
            "passed": passed,
        })

    payload = {
        "phase": "3_v2_p50_reduced_budget",
        "budget": {
            "n_first_stage": N_FIRST_STAGE,
            "n_second_stage": N_SECOND_STAGE,
            "n_variance_replications": N_VARIANCE_REPLICATIONS,
            "n_initial_points_stage1": N_INITIAL_POINTS_STAGE1,
            "n_initial_points_stage2": N_INITIAL_POINTS_STAGE2,
            "n_seeds_per_eval": N_SEEDS_PER_EVAL,
            "cold_start": True,
        },
        "base_cfg_flags": {
            "aspiration_reference_quantile":
                base_cfg.voting.aspiration_reference_quantile,
            "assortative_help_enabled": base_cfg.behavioral.assortative_help_enabled,
            "p_assortative_help": base_cfg.behavioral.p_assortative_help,
            "f_assortative_help": base_cfg.behavioral.f_assortative_help,
            "assortative_help_wealth_lower_factor":
                base_cfg.behavioral.assortative_help_wealth_lower_factor,
            "assortative_help_donor_min_age":
                base_cfg.behavioral.assortative_help_donor_min_age,
            "grad_share_data_path": base_cfg.voting.grad_share_data_path,
        },
        "param_space_bounds": [{"name": ps.name, "low": ps.low, "high": ps.high}
                                 for ps in param_space],
        "theta_hat": [float(v) for v in result.theta_hat],
        "se": [float(v) for v in result.se],
        "param_names": list(param_names),
        "j_statistic": float(result.j_statistic),
        "dof": int(result.dof),
        "p_value": float(result.p_value),
        "moment_names": [m.name for m in CALIBRATION_MOMENTS],
        "moment_targets": [float(v) for v in result.moment_targets],
        "moment_at_optimum": [float(v) for v in result.moment_at_optimum],
        "moment_residuals": [float(v) for v in result.moment_residuals_at_optimum],
        "moment_weights": [float(m.weight) for m in CALIBRATION_MOMENTS],
        "moment_tolerances": [float(m.target_tolerance) for m in CALIBRATION_MOMENTS],
        "fit_rows": fit_rows,
        "fit_n": n_pass,
        "fit_total": len(CALIBRATION_MOMENTS),
        "validation_de": validation,
        "uk_metrics": uk_metrics,
        "stage1_path_y": [float(v) for v in result.stage1_path_y],
        "stage2_path_y": [float(v) for v in result.stage2_path_y],
        "elapsed_seconds": time.time() - t0,
    }

    out = OUTPUTS / "smm_augmented_p50_reduced.json"
    out.write_text(json.dumps(payload, indent=2))

    # ----------------------------------------------------------------- surface
    print()
    print("=" * 92)
    print("PHASE 3 v2 AUGMENTED SMM (p50, REDUCED BUDGET, COLD-START) RESULTS")
    print("=" * 92)
    print(f"J-statistic = {result.j_statistic:.4f}   dof = {result.dof}   "
          f"p-value = {result.p_value:.4f}")
    print(f"Table 4 fit: {n_pass}/{len(CALIBRATION_MOMENTS)}")
    print(f"Elapsed: {payload['elapsed_seconds']:.1f}s "
          f"({payload['elapsed_seconds']/60:.1f} min)")
    print()
    print(f"{'parameter':32s}  {'theta_hat':>10s}  {'se':>10s}  "
          f"{'|se/theta|':>10s}  {'at bound?':>10s}")
    for ps, theta, se in zip(param_space, result.theta_hat, result.se):
        denom = max(abs(theta), 1e-9)
        ratio = se / denom
        at_lo = abs(theta - ps.low) < 1e-3 * (ps.high - ps.low)
        at_hi = abs(theta - ps.high) < 1e-3 * (ps.high - ps.low)
        at = "lo" if at_lo else "hi" if at_hi else "no"
        print(f"  {ps.name:30s}  {theta:+10.4f}  {se:>10.4f}  "
              f"{ratio:>10.3f}  {at:>10s}")
    print()
    print("Calibration moments at optimum:")
    print(f"  {'moment':50s}  {'target':>8s}  {'model':>8s}  {'err':>8s}  "
          f"{'tol':>6s}  pass")
    for r in fit_rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['name']:50s}  {r['target']:+8.4f}  {r['model']:+8.4f}  "
              f"{r['error']:+8.4f}  {r['tolerance']:>6.3f}  {mark}")
    print()
    print("DE validation (held-out) moments:")
    for name, v in validation.items():
        if not np.isfinite(v["model"]):
            print(f"  {name:55s}  target={v['target']:+.4f}  model=NaN")
            continue
        mark = "PASS" if v["passed"] else "FAIL"
        print(f"  {name:55s}  target={v['target']:+.4f}  model={v['model']:+.4f}  "
              f"err={v['error']:+.4f}  tol={v['tolerance']:.3f}  {mark}")
    print()
    print("UK metrics at theta_hat (Brexit period, 10 seeds, augmented + p50):")
    for k, v in uk_metrics.items():
        print(f"  {k:30s}  {v:+.4f}")
    print()
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
