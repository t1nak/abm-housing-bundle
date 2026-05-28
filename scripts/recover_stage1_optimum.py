"""Recover stage-1 optimum and use it as the SMM final estimate.

The full SMM run produced a stage-2 optimum where aggregate-extreme-share
collapsed to 0,004 (vs target 0,208). This is a known pathology when the
optimal weighting matrix W2 = Sigma^{-1} is poorly conditioned at the
stage-1 covariance estimate; gp_minimize then converges to a point that
fits the W2-weighted objective but has bad moment matching.

Standard SMM practice when stage-2 diverges: fall back to stage-1.
Stage 1 uses the diagonal variance-normalised weighting matrix and is
asymptotically less efficient but finite-sample more robust.

This script:
  1. Loads outputs/smm_state.pkl
  2. Extracts the stage-1 best point
  3. Re-evaluates the Jacobian, SE, J-stat AT stage-1 theta with the
     stage-2 W2 (for comparability)
  4. Re-runs validation at stage-1
  5. Overwrites outputs/smm_optimum.json with the stage-1 result
  6. Documents stage-2 as a robustness check
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import stats

from abmhp import Config
from abmhp.estimation.smm import (
    CALIBRATION_MOMENTS,
    PARAM_NAMES,
    PARAM_SPACE,
    SMMResult,
    apply_params,
    estimate_moment_covariance,
    moment_jacobian,
    moment_residuals,
    moment_targets,
    optimal_weighting_matrix,
    plot_sensitivity_jacobian,
    score_validation_moments,
    simulated_moments,
)


OUTPUTS = ROOT / "outputs"


def main() -> None:
    print("Loading stage-2 result")
    with (OUTPUTS / "smm_state.pkl").open("rb") as f:
        prev: SMMResult = pickle.load(f)

    stage1_x = prev.stage1_path_x
    stage1_y = prev.stage1_path_y
    best_idx = int(np.argmin(stage1_y))
    theta1 = np.array(stage1_x[best_idx])
    stage1_obj = float(stage1_y[best_idx])
    print(f"Stage-1 best at iteration {best_idx + 1}, obj = {stage1_obj:.4e}")
    for ps, v in zip(PARAM_SPACE, theta1):
        print(f"  {ps.name:30s} = {v:+.4f}")

    print("\nRe-estimating moment covariance at stage-1 optimum (for SE)")
    cfg1 = apply_params(Config(), theta1)
    _, cov_at_theta1 = estimate_moment_covariance(
        cfg1, n_replications=20, seeds_per_replication=5, seed_offset=4000,
    )
    W2 = optimal_weighting_matrix(cov_at_theta1)

    seeds = [73, 74, 75, 76, 77]
    print("\nComputing moment vector and Jacobian at stage-1 optimum")
    m_at_opt = simulated_moments(theta1, Config(), seeds)
    g_hat = m_at_opt - moment_targets()
    G = moment_jacobian(theta1, Config(), seeds)

    # J-statistic under W2 evaluated at stage-1 theta.
    j_stat = float(g_hat @ W2 @ g_hat)
    j_stat_stage2 = float(prev.j_statistic)
    print(f"J-statistic (stage-1 theta under W2): {j_stat:.4f}")
    print(f"J-statistic (stage-2 theta under W2): {j_stat_stage2:.4f}")
    if j_stat < j_stat_stage2:
        print(f"  -> Stage-1 is the better SMM optimum.")
    else:
        print(f"  -> Stage-2 is the better SMM optimum despite degenerate moments.")

    dof = len(CALIBRATION_MOMENTS) - len(PARAM_SPACE)
    p_value = float(1.0 - stats.chi2.cdf(j_stat, dof))

    # Sandwich SE.
    bread = np.linalg.pinv(G.T @ W2 @ G)
    meat = G.T @ W2 @ cov_at_theta1 @ W2 @ G
    cov_theta = bread @ meat @ bread / 20  # n_variance_replications
    se = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))

    print(f"\nStage-1 parameter estimates and SE (sandwich formula):")
    print(f"{'parameter':<32s} {'estimate':>10s} {'std error':>10s} {'SE/|theta|':>10s} {'identified':>10s}")
    for name, t, s in zip(PARAM_NAMES, theta1, se):
        ratio = s / max(abs(t), 1e-9)
        identified = "yes" if ratio <= 0.25 else "no"
        print(f"  {name:<30s} {t:+10.4f} {s:>10.4f} {ratio:>10.3f} {identified:>10s}")

    print("\nCalibration moments at stage-1 optimum:")
    for name, tgt, mod in zip(
        [m.name for m in CALIBRATION_MOMENTS],
        moment_targets(),
        m_at_opt,
    ):
        err = mod - tgt
        print(f"  {name:<55s} target={tgt:+.4f}  model={mod:+.4f}  err={err:+.4f}")

    print("\nScoring validation moments at stage-1 optimum")
    validation = score_validation_moments(theta1, Config())
    for name, info in validation.items():
        if np.isfinite(info["model"]):
            status = "PASS" if info["passed"] else "FAIL"
            print(f"  {name:<55s} target={info['target']:+.4f}  "
                  f"model={info['model']:+.4f}  err={info['error']:+.4f}  {status}")
        else:
            print(f"  {name:<55s} target={info['target']:+.4f}  model=NaN  (deferred)")

    # Update result struct with stage-1 as the official optimum,
    # preserving stage-2 for robustness reporting.
    updated = SMMResult(
        theta_hat=theta1,
        se=se,
        j_statistic=j_stat,
        dof=dof,
        p_value=p_value,
        moment_targets=moment_targets(),
        moment_at_optimum=m_at_opt,
        moment_residuals_at_optimum=g_hat,
        weighting_matrix_stage1=prev.weighting_matrix_stage1,
        weighting_matrix_stage2=W2,
        covariance_at_optimum=cov_at_theta1,
        jacobian=G,
        parameter_covariance=cov_theta,
        stage1_path_x=prev.stage1_path_x,
        stage1_path_y=prev.stage1_path_y,
        stage2_path_x=prev.stage2_path_x,
        stage2_path_y=prev.stage2_path_y,
        n_seeds_per_eval=prev.n_seeds_per_eval,
        n_first_stage=prev.n_first_stage,
        n_second_stage=prev.n_second_stage,
    )

    payload = {
        "theta_hat": [float(v) for v in updated.theta_hat],
        "se": [float(v) for v in updated.se],
        "param_names": list(PARAM_NAMES),
        "j_statistic": float(updated.j_statistic),
        "dof": int(updated.dof),
        "p_value": float(updated.p_value),
        "moment_names": [m.name for m in CALIBRATION_MOMENTS],
        "moment_targets": [float(v) for v in updated.moment_targets],
        "moment_at_optimum": [float(v) for v in updated.moment_at_optimum],
        "moment_residuals": [float(v) for v in updated.moment_residuals_at_optimum],
        "moment_weights": [float(m.weight) for m in CALIBRATION_MOMENTS],
        "validation": validation,
        "n_seeds_per_eval": int(updated.n_seeds_per_eval),
        "n_first_stage": int(updated.n_first_stage),
        "n_second_stage": int(updated.n_second_stage),
        "stage2_theta": [float(v) for v in prev.theta_hat],
        "stage2_j_statistic": float(prev.j_statistic),
        "stage_used": "stage_1",
        "fallback_reason": (
            "Stage-2 gp_minimize converged to a degenerate optimum where "
            "aggregate_extreme_share_final fell to 0,004 versus target 0,208. "
            "This is a known SMM pathology when the optimal weighting matrix "
            "W2 = Sigma^{-1} is poorly conditioned at the stage-1 covariance "
            "estimate. Standard practice: fall back to stage-1, which is "
            "asymptotically less efficient but finite-sample more robust."
        ),
    }
    json_path = OUTPUTS / "smm_optimum.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {json_path.relative_to(ROOT)}")

    pkl_path = OUTPUTS / "smm_state.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(updated, f)
    print(f"Saved {pkl_path.relative_to(ROOT)}")

    jac_path = OUTPUTS / "sensitivity_jacobian.png"
    plot_sensitivity_jacobian(updated, jac_path)
    print(f"Saved {jac_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
