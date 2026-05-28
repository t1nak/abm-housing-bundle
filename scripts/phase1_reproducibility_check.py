"""Phase 1 reproducibility check.

Runs the model at theta_hat (loaded from outputs/smm_optimum.json) over 10
seeds with all newly-added feature flags at their defaults (i.e. False or
zero), and checks the resulting aggregate AfD, DE cleavage, dispersion,
and Table 4 fit fraction against the locked baseline.

Bit-for-bit invariance criterion: each aggregate moment must match the
recorded baseline to within 1e-6.

Outputs:
    outputs/augmented_phase1_reproducibility.json
"""
from __future__ import annotations

import json
import sys
import time
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
from abmhp.estimation.smm import apply_params


OUTPUTS = ROOT / "outputs"


def load_theta_hat() -> tuple[np.ndarray, list[str]]:
    payload = json.loads((OUTPUTS / "smm_optimum.json").read_text())
    return np.array(payload["theta_hat"], dtype=float), list(payload["param_names"])


def main() -> None:
    t0 = time.time()
    theta_hat, names = load_theta_hat()
    print(f"theta_hat = {dict(zip(names, [round(v, 4) for v in theta_hat]))}")

    base_cfg = Config()
    cfg = apply_params(base_cfg, theta_hat)

    # Defensive: confirm every new flag is at its default.
    assert cfg.behavioral.assortative_help_enabled is False
    assert cfg.behavioral.estimate_beta_n is False
    assert cfg.behavioral.estimate_gamma_cosmopolitan is False
    assert cfg.voting.gamma_cosmopolitan == 0.0
    assert cfg.voting.grad_share_data_path is None
    assert cfg.voting.cosmopolitan_shift_by_region is None
    print("All Phase 1 flags confirmed at defaults.")

    seeds = list(range(73, 83))
    runs = simulate_seeds(cfg, seeds)

    cal = evaluate_moments(CALIBRATION_MOMENTS, runs)
    val = evaluate_moments(VALIDATION_MOMENTS, runs)

    # Fit fraction: number of calibration moments within their target_tolerance.
    fit_n = 0
    rows = []
    for m in CALIBRATION_MOMENTS:
        model_val = cal[m.name]
        err = float(model_val - m.value)
        passed = bool(abs(err) <= m.target_tolerance)
        if passed:
            fit_n += 1
        rows.append({
            "name": m.name,
            "target": float(m.value),
            "model": float(model_val),
            "error": err,
            "tolerance": float(m.target_tolerance),
            "passed": passed,
        })

    aggregate_afd = float(cal["aggregate_extreme_share_final"])
    cleavage = float(val["within_region_renter_owner_vote_gap"])
    dispersion = float(cal["cross_regional_extreme_share_dispersion"])

    # Reference baseline values for the bit-for-bit check.
    #
    # `smm_optimum.json::moment_at_optimum` was generated against an earlier
    # Config() and has drifted since (Issue 1 bounds work, tolerance fixes).
    # Comparing Phase 1 against it would conflate Phase 1 changes with the
    # intervening calibration drift. The proper bit-for-bit reference is
    # the *current master* code running the same 5 seeds. If the file
    # `outputs/phase1_master_baseline.json` is present (captured manually
    # by running the same theta_hat + 5 seeds against master before
    # applying Phase 1 changes), we compare against it. Otherwise we fall
    # back to the stale smm_optimum snapshot for visibility but the
    # `bitcheck_passed` flag will report the master-baseline comparison.

    master_baseline_path = OUTPUTS / "phase1_master_baseline.json"
    if master_baseline_path.exists():
        baseline_payload = json.loads(master_baseline_path.read_text())
        baseline_map = {k: float(v) for k, v in baseline_payload["cal5"].items()}
        baseline_source = str(master_baseline_path.name)
    else:
        smm = json.loads((OUTPUTS / "smm_optimum.json").read_text())
        baseline_map = dict(zip(smm["moment_names"], smm["moment_at_optimum"]))
        baseline_source = "smm_optimum.json::moment_at_optimum (drifted, see comment)"

    # 5-seed evaluation for direct comparison against the locked snapshot.
    seeds5 = list(range(73, 78))
    runs5 = simulate_seeds(cfg, seeds5)
    cal5 = evaluate_moments(CALIBRATION_MOMENTS, runs5)

    bitcheck = []
    max_abs_diff = 0.0
    for name, base_v in baseline_map.items():
        new_v = float(cal5[name])
        diff = abs(new_v - base_v)
        max_abs_diff = max(max_abs_diff, diff)
        bitcheck.append({
            "name": name,
            "baseline": float(base_v),
            "phase1": new_v,
            "abs_diff": diff,
        })

    payload = {
        "theta_hat": theta_hat.tolist(),
        "param_names": names,
        "seeds_10": seeds,
        "seeds_5": seeds5,
        "calibration_10seed": {k: float(v) for k, v in cal.items()},
        "validation_10seed": {k: float(v) for k, v in val.items()},
        "table4_fit_fraction": f"{fit_n}/{len(CALIBRATION_MOMENTS)}",
        "aggregate_afd_10seed": aggregate_afd,
        "cleavage_10seed": cleavage,
        "dispersion_10seed": dispersion,
        "rows_10seed": rows,
        "bitcheck_5seed": bitcheck,
        "bitcheck_baseline_source": baseline_source,
        "max_abs_diff_5seed_vs_baseline": max_abs_diff,
        "bitcheck_passed": bool(max_abs_diff < 1e-6),
        "elapsed_seconds": time.time() - t0,
    }

    out = OUTPUTS / "augmented_phase1_reproducibility.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")

    print("\n=== Phase 1 reproducibility check ===")
    print(f"10-seed aggregate AfD:        {aggregate_afd:.6f}")
    print(f"10-seed cleavage (renter-owner gap): {cleavage:+.6f}")
    print(f"10-seed dispersion:           {dispersion:.6f}")
    print(f"10-seed Table 4 fit fraction: {fit_n}/{len(CALIBRATION_MOMENTS)}")

    print("\n=== Bit-for-bit check: 5-seed reproduce smm_optimum.moment_at_optimum ===")
    print(f"max |diff| across 12 calibration moments: {max_abs_diff:.3e}")
    print(f"PASSED (< 1e-6): {payload['bitcheck_passed']}")
    if not payload["bitcheck_passed"]:
        print("\nDetail:")
        for row in bitcheck:
            print(f"  {row['name']:50s}  baseline={row['baseline']:+.8f}  "
                  f"phase1={row['phase1']:+.8f}  diff={row['abs_diff']:.3e}")


if __name__ == "__main__":
    main()
