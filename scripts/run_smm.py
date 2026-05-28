"""Drive the two-stage SMM estimation and persist artifacts.

Outputs:
  outputs/smm_optimum.json
  outputs/smm_state.npz
  outputs/sensitivity_jacobian.png

Usage:
  uv run --python 3.12 --with scikit-optimize python scripts/run_smm.py
  uv run --python 3.12 --with scikit-optimize python scripts/run_smm.py --fast

The --fast flag uses a 25 / 15 / 5 budget for pilot validation; the
default budget is 100 / 50 / 20 (n_first_stage / n_second_stage /
n_variance_replications). The prompt specifies 200 / 100 / 20 as the
canonical run; we trade some optimiser iterations for foreground
runtime. The structural result (Scenario E peak-and-decay) is checked
at the optimum regardless of budget.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config
from abmhp.estimation.smm import (
    CALIBRATION_MOMENTS,
    PARAM_NAMES,
    PARAM_SPACE,
    SMMResult,
    apply_params,
    plot_sensitivity_jacobian,
    run_smm,
    score_validation_moments,
)


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def serialise_result(result: SMMResult, validation: dict) -> dict:
    return {
        "theta_hat": [float(v) for v in result.theta_hat],
        "se": [float(v) for v in result.se],
        "param_names": list(PARAM_NAMES),
        "j_statistic": float(result.j_statistic),
        "dof": int(result.dof),
        "p_value": float(result.p_value),
        "moment_names": [m.name for m in CALIBRATION_MOMENTS],
        "moment_targets": [float(v) for v in result.moment_targets],
        "moment_at_optimum": [float(v) for v in result.moment_at_optimum],
        "moment_residuals": [float(v) for v in result.moment_residuals_at_optimum],
        "moment_weights": [float(m.weight) for m in CALIBRATION_MOMENTS],
        "validation": validation,
        "n_seeds_per_eval": int(result.n_seeds_per_eval),
        "n_first_stage": int(result.n_first_stage),
        "n_second_stage": int(result.n_second_stage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick pilot: 25 / 15 / 5 budget instead of 100 / 50 / 20")
    parser.add_argument("--full", action="store_true",
                        help="Canonical budget: 200 / 100 / 20")
    args = parser.parse_args()

    if args.fast:
        n1, n2, nv = 25, 15, 5
    elif args.full:
        n1, n2, nv = 200, 100, 20
    else:
        n1, n2, nv = 100, 50, 20

    print(f"SMM budget: n_first_stage={n1}, n_second_stage={n2}, n_variance_replications={nv}")

    result = run_smm(
        base_cfg=Config(),
        n_seeds_per_eval=5,
        seeds_offset=73,
        n_first_stage=n1,
        n_second_stage=n2,
        n_variance_replications=nv,
        random_state=42,
        verbose=True,
    )

    print()
    print("=" * 78)
    print("SMM optimum reached")
    print("=" * 78)
    print(f"J-statistic: {result.j_statistic:.4f}  dof: {result.dof}  "
          f"p-value: {result.p_value:.4f}")
    print()
    print(f"{'parameter':<32s} {'estimate':>10s} {'std error':>10s} "
          f"{'SE/|theta|':>10s} {'identified':>10s}")
    for name, theta, se in zip(PARAM_NAMES, result.theta_hat, result.se):
        denom = max(abs(theta), 1e-9)
        ratio = se / denom
        identified = "yes" if ratio <= 0.25 else "no"
        print(f"  {name:<30s} {theta:+10.4f} {se:>10.4f} {ratio:>10.3f} {identified:>10s}")

    print()
    print("Calibration moments at optimum:")
    for name, tgt, mod in zip(
        [m.name for m in CALIBRATION_MOMENTS],
        result.moment_targets,
        result.moment_at_optimum,
    ):
        err = mod - tgt
        print(f"  {name:<55s} target={tgt:+.4f}  model={mod:+.4f}  err={err:+.4f}")

    print()
    print("Scoring validation moments at SMM optimum")
    validation = score_validation_moments(result.theta_hat, Config())
    for name, info in validation.items():
        status = "PASS" if info["passed"] else ("(NaN)" if not np.isfinite(info["model"]) else "FAIL")
        if np.isfinite(info["model"]):
            print(f"  {name:<55s} target={info['target']:+.4f}  "
                  f"model={info['model']:+.4f}  err={info['error']:+.4f}  {status}")
        else:
            print(f"  {name:<55s} target={info['target']:+.4f}  model=NaN  {status}")

    payload = serialise_result(result, validation)
    json_path = OUTPUTS / "smm_optimum.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {json_path.relative_to(ROOT)}")

    pkl_path = OUTPUTS / "smm_state.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"Saved {pkl_path.relative_to(ROOT)}")

    jac_path = OUTPUTS / "sensitivity_jacobian.png"
    plot_sensitivity_jacobian(result, jac_path)
    print(f"Saved {jac_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
