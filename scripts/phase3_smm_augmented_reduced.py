"""Phase 3 augmented SMM (reduced budget).

Joint estimation of 10 parameters:
    8 original (beta_dissat, beta_renter, rho_aspiration, alpha_local,
                price_slope, beta_0, assortative_exponent,
                intergenerational_skill_corr)
    + beta_network (bounds [0.3, 1.5], cold-start at 0.6)
    + gamma_cosmopolitan (bounds [-15.0, 5.0], cold-start at -5.156)

Configuration held fixed throughout:
    assortative_help_enabled = True
    grad_share_data_path = data/cosmopolitan_grad_share_de.json
    p_assortative_help = 0.05, f_assortative_help = 0.15
    assortative_help_wealth_lower_factor = 0.20
    assortative_help_donor_min_age = 55.0

Reduced budget (viability check; full budget run is a separate file if this
result motivates committing more wall time):
    Stage 1:  60 gp_minimize calls (12 Sobol initial + 1 warm-start point)
    Stage 2:  30 gp_minimize calls (10 Sobol initial, warm-started at theta1)
    Variance: 10 replications at theta0 and at theta1
    Warm-start theta_hat for the 8 original parameters (from
        outputs/smm_optimum.json); cold-start beta_n and gamma_cosmopolitan.

Output: outputs/smm_augmented_reduced.json
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
    score_validation_moments,
)


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DE_GRAD_PATH = "data/cosmopolitan_grad_share_de.json"

# Reduced budget per Phase 3 spec.
N_FIRST_STAGE = 60
N_SECOND_STAGE = 30
N_VARIANCE_REPLICATIONS = 10
N_INITIAL_POINTS_STAGE1 = 12
N_INITIAL_POINTS_STAGE2 = 10
N_SEEDS_PER_EVAL = 5
SEEDS_OFFSET = 73
RANDOM_STATE = 42

# Cold-start values for the two new parameters.
COLD_BETA_N = 0.6
COLD_GAMMA = -5.156


def _build_base_cfg() -> Config:
    """Augmented base config: all Phase 1 flags on, grad-share data path set
    so simulate() materialises the cosmopolitan shift from gamma at each
    evaluation."""
    cfg = Config()
    behavioral = replace(
        cfg.behavioral,
        assortative_help_enabled=True,
        estimate_beta_n=True,
        estimate_gamma_cosmopolitan=True,
    )
    voting = replace(
        cfg.voting,
        grad_share_data_path=DE_GRAD_PATH,
        # cosmopolitan_shift_by_region stays None so simulate() recomputes
        # the shift from gamma each evaluation (gamma is being optimised).
    )
    return replace(cfg, behavioral=behavioral, voting=voting)


def main() -> None:
    t0 = time.time()
    base_cfg = _build_base_cfg()
    param_space = build_param_space(base_cfg)
    param_names = [ps.name for ps in param_space]
    assert len(param_space) == 10, f"expected 10 params, got {len(param_space)}"

    # Warm-start vector.
    smm = json.loads((OUTPUTS / "smm_optimum.json").read_text())
    theta_orig = np.array(smm["theta_hat"], dtype=float)
    orig_names = list(smm["param_names"])
    assert orig_names == param_names[:8], (
        f"param_name ordering drift: orig={orig_names}, new[:8]={param_names[:8]}"
    )
    x0 = list(theta_orig) + [COLD_BETA_N, COLD_GAMMA]

    print("=" * 78)
    print("PHASE 3 augmented SMM (reduced budget)")
    print("=" * 78)
    print(f"Param space ({len(param_space)} parameters):")
    for ps in param_space:
        print(f"  {ps.name:32s}  bounds = [{ps.low:+.3f}, {ps.high:+.3f}]")
    print(f"Warm-start x0: {dict(zip(param_names, [round(v, 4) for v in x0]))}")
    print(f"Budget: Stage1={N_FIRST_STAGE} (Sobol={N_INITIAL_POINTS_STAGE1}), "
          f"Stage2={N_SECOND_STAGE} (Sobol={N_INITIAL_POINTS_STAGE2}), "
          f"VarReps={N_VARIANCE_REPLICATIONS}")
    print(f"base_cfg flags: assortative_help_enabled="
          f"{base_cfg.behavioral.assortative_help_enabled}, "
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
        x0_stage1=x0,
    )

    # Validation moments at the augmented optimum.
    validation = score_validation_moments(
        result.theta_hat,
        base_cfg,
        seeds=list(range(73, 78)),
    )
    # NOTE: score_validation_moments uses the module-level PARAM_SPACE inside
    # its apply_params call; that's wrong here because theta_hat has 10
    # entries. Recompute manually.
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

    # Per-moment fit summary at the augmented optimum.
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
        "phase": "3_reduced_budget",
        "budget": {
            "n_first_stage": N_FIRST_STAGE,
            "n_second_stage": N_SECOND_STAGE,
            "n_variance_replications": N_VARIANCE_REPLICATIONS,
            "n_initial_points_stage1": N_INITIAL_POINTS_STAGE1,
            "n_initial_points_stage2": N_INITIAL_POINTS_STAGE2,
            "n_seeds_per_eval": N_SEEDS_PER_EVAL,
        },
        "base_cfg_flags": {
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
        "x0_warm_start": x0,
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
        "validation": validation,
        "stage1_path_y": [float(v) for v in result.stage1_path_y],
        "stage2_path_y": [float(v) for v in result.stage2_path_y],
        "elapsed_seconds": time.time() - t0,
    }

    out = OUTPUTS / "smm_augmented_reduced.json"
    out.write_text(json.dumps(payload, indent=2))

    # ---------------------------------------------------------------- surface
    print()
    print("=" * 90)
    print("PHASE 3 AUGMENTED SMM (REDUCED BUDGET) RESULTS")
    print("=" * 90)
    print(f"J-statistic = {result.j_statistic:.4f}   dof = {result.dof}   "
          f"p-value = {result.p_value:.4f}")
    print(f"Table 4 fit: {n_pass}/{len(CALIBRATION_MOMENTS)}")
    print(f"Elapsed: {payload['elapsed_seconds']:.1f}s")
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
    print("Validation (held-out) moments:")
    for name, v in validation.items():
        if not np.isfinite(v["model"]):
            print(f"  {name:55s}  target={v['target']:+.4f}  model=NaN")
            continue
        mark = "PASS" if v["passed"] else "FAIL"
        print(f"  {name:55s}  target={v['target']:+.4f}  model={v['model']:+.4f}  "
              f"err={v['error']:+.4f}  tol={v['tolerance']:.3f}  {mark}")
    print()
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
