"""German placebo for the Adler-Ansell (2020) cross-regional regression.

Run the income-controlled partial correlation between the extreme-share
vote and cumulative price growth across the 16 Laender, using the same
specification as the UK external stress test in `validation/uk.py`.

Empirical sign expectation: negative (Adler-Ansell 2020 housing-channel
direction). The placebo's role is to confirm the housing-channel model
produces the empirically correct conditional sign in Germany before any
work is done on the UK sign mismatch. If the German conditional sign is
wrong as well, the UK sign mismatch is not country-specific and Issue 3
becomes substantively harder.

Inputs:
  outputs/smm_optimum.json   the headline theta_hat
  --theta-from outputs/widen_bounds_<variant>.json (optional)
                              evaluate the placebo at a widened-bound
                              variant's optimum, for cross-variant
                              comparison.

Outputs:
  outputs/german_partial_correlation_placebo.json

Usage:
  uv run python scripts/german_partial_correlation_placebo.py
  uv run python scripts/german_partial_correlation_placebo.py \
      --theta-from outputs/widen_bounds_widen_ceiling_joint.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config
from abmhp.estimation.moments import (
    VALIDATION_MOMENTS,
    evaluate_moments,
    simulate_seeds,
)
from abmhp.estimation.smm import apply_params, PARAM_NAMES


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


SEEDS = list(range(73, 83))  # 10 seeds for placebo stability


def load_theta(path: Path) -> tuple[np.ndarray, str]:
    data = json.loads(path.read_text())
    if "theta_hat" in data:
        return np.array(data["theta_hat"], dtype=float), str(path)
    raise KeyError(f"{path} has no theta_hat field")


def evaluate_placebo(theta: np.ndarray, seeds: list[int] = SEEDS) -> dict:
    base_cfg = Config()
    cfg = apply_params(base_cfg, theta)
    runs = simulate_seeds(cfg, seeds)
    vals = evaluate_moments(VALIDATION_MOMENTS, runs)

    raw_corr = float(vals["cross_regional_extreme_share_price_growth_correlation"])
    partial_corr = float(vals["cross_regional_extreme_share_price_growth_partial_correlation_income"])
    gap = float(vals["within_region_renter_owner_vote_gap"])

    # Sign-only judgment vs. Adler-Ansell direction (negative).
    raw_sign_ok = raw_corr < 0
    partial_sign_ok = np.isfinite(partial_corr) and partial_corr < 0

    return {
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "theta": theta.tolist(),
        "param_names": list(PARAM_NAMES),
        "raw_correlation": raw_corr,
        "raw_correlation_sign_negative": bool(raw_sign_ok),
        "partial_correlation_income_controlled": partial_corr,
        "partial_correlation_sign_negative": bool(partial_sign_ok),
        "within_region_renter_owner_gap": gap,
        "spec": (
            "Cross-regional regression of regional extreme-share vote (16 Laender) "
            "on regional cumulative price growth, controlling for regional mean income. "
            "Same functional form as validation/uk.py::_cross_regional_leave_price_partial_correlation."
        ),
        "interpretation": (
            "Raw correlation matches the German held-out moment in moments.py. "
            "Partial correlation is the placebo for Adler-Ansell (2020). "
            "If the partial correlation is negative, the housing-channel mechanism "
            "produces the empirically correct conditional sign in Germany, "
            "isolating the UK sign mismatch as country-specific."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--theta-from", type=Path, default=OUTPUTS / "smm_optimum.json",
                   help="JSON file containing a 'theta_hat' field; defaults to the headline optimum.")
    p.add_argument("--out", type=Path, default=OUTPUTS / "german_partial_correlation_placebo.json")
    args = p.parse_args()

    theta, src = load_theta(args.theta_from)
    print(f"Loading theta_hat from {src}")
    for name, v in zip(PARAM_NAMES, theta):
        print(f"  {name:32s} = {v:+.4f}")
    print(f"Evaluating placebo with {len(SEEDS)} seeds")

    result = evaluate_placebo(theta)
    result["theta_source"] = src
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Saved {args.out.relative_to(ROOT)}")

    print()
    print("=" * 72)
    print("GERMAN PLACEBO RESULT")
    print("=" * 72)
    print(f"  raw correlation                              = {result['raw_correlation']:+.4f}  "
          f"(sign expected: negative; got "
          f"{'negative' if result['raw_correlation_sign_negative'] else 'positive'})")
    pc = result["partial_correlation_income_controlled"]
    if np.isfinite(pc):
        print(f"  partial correlation | income (placebo)       = {pc:+.4f}  "
              f"(sign expected: negative; got "
              f"{'negative' if result['partial_correlation_sign_negative'] else 'positive'})")
    else:
        print("  partial correlation | income (placebo)       = NaN (degenerate variance)")
    print(f"  within-region renter-owner cleavage           = {result['within_region_renter_owner_gap']:+.4f}")


if __name__ == "__main__":
    main()
