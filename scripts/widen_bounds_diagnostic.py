"""Bound-widening SMM diagnostic for Issue 1.

For each variant, override one or more bounds on the binding side and
re-run a reduced-budget two-stage SMM, warm-started at the existing
Stage 1 optimum theta_hat. Variants:

  control                  : original bounds, reduced budget, warm start.
                             Sanity check for budget vs. full-budget.
  widen_beta_renter        : low 0.20 -> 0.00.
  widen_rho_aspiration     : high 0.90 -> 0.99.
  widen_alpha_local        : low 0.20 -> 0.00.
  widen_assortative_exponent : high 3.00 -> 4.50.
  widen_intergenerational_skill_corr : high 0.85 -> 0.99.
  widen_ceiling_joint      : rho 0.90->0.99, alpha_assort 3.0->4.5,
                             intergen 0.85->0.99 simultaneously.

Reduced budget: 40 Stage 1 calls, 20 Stage 2 calls, 10 variance reps.
The warm-start point counts as one Stage 1 evaluation; gp_minimize
then runs 10 Sobol initial points plus 29 BO calls. The user's
asymmetric-verification protocol: any variant whose new optimum
moves more than 5 percent of the bound range into the new interior
should be re-run at full budget (100 / 50 / 20).

Per variant the script writes:
  outputs/widen_bounds_<variant>.json

Aggregated summary at the end:
  outputs/widen_bounds_summary.json

The diagnostic reports, for each variant:
  - theta at the new optimum;
  - whether each widened bound is still binding (within 1 percent of
    the new bound on the binding side);
  - aggregate AfD share at the new optimum;
  - within-region renter-owner cleavage at the new optimum;
  - full Table 4 fit (12 SMM moments).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
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
    PARAM_NAMES,
    PARAM_SPACE,
    apply_params,
    diagonal_weighting_matrix,
    estimate_moment_covariance,
    moment_residuals,
    moment_targets,
    objective,
    optimal_weighting_matrix,
    simulated_moments,
)


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


SEEDS = list(range(73, 78))
N_FIRST_STAGE = 40
N_SECOND_STAGE = 20
N_VARIANCE_REPLICATIONS = 10


@dataclass(frozen=True)
class BoundOverride:
    """Override one parameter's low and/or high bound."""
    name: str
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Variant:
    label: str
    overrides: tuple[BoundOverride, ...]
    description: str


# Bound widening: 50 percent of the original bound range, extended on the
# binding side; clipped at semantic limits (rho and intergen_corr at 0.99
# below 1.0; alpha_local and beta_renter at 0.0).
VARIANTS: tuple[Variant, ...] = (
    Variant(
        label="control",
        overrides=(),
        description="Original bounds at reduced budget; calibration anchor for the diagnostic.",
    ),
    Variant(
        label="widen_beta_renter",
        overrides=(BoundOverride("beta_renter", low=0.0),),
        description="Renter premium low bound widened: 0.20 -> 0.00.",
    ),
    Variant(
        label="widen_rho_aspiration",
        overrides=(BoundOverride("rho_aspiration", high=0.99),),
        description="Aspiration persistence high bound widened: 0.90 -> 0.99.",
    ),
    Variant(
        label="widen_alpha_local",
        overrides=(BoundOverride("alpha_local", low=0.0),),
        description="Local-reference share low bound widened: 0.20 -> 0.00.",
    ),
    Variant(
        label="widen_assortative_exponent",
        overrides=(BoundOverride("assortative_exponent", high=4.5),),
        description="Assortative-bequest exponent high bound widened: 3.0 -> 4.5.",
    ),
    Variant(
        label="widen_intergenerational_skill_corr",
        overrides=(BoundOverride("intergenerational_skill_corr", high=0.99),),
        description="Intergenerational skill correlation high bound widened: 0.85 -> 0.99.",
    ),
    Variant(
        label="widen_ceiling_joint",
        overrides=(
            BoundOverride("rho_aspiration", high=0.99),
            BoundOverride("assortative_exponent", high=4.5),
            BoundOverride("intergenerational_skill_corr", high=0.99),
        ),
        description="Three ceiling-bound parameters widened simultaneously.",
    ),
)


def resolve_bounds(variant: Variant) -> dict[str, tuple[float, float]]:
    """Return effective (low, high) for each parameter given the variant."""
    bounds: dict[str, tuple[float, float]] = {p.name: (p.low, p.high) for p in PARAM_SPACE}
    for ov in variant.overrides:
        lo, hi = bounds[ov.name]
        if ov.low is not None:
            lo = ov.low
        if ov.high is not None:
            hi = ov.high
        bounds[ov.name] = (lo, hi)
    return bounds


def is_still_binding(name: str, value: float, bounds: dict[str, tuple[float, float]],
                     overrides: tuple[BoundOverride, ...]) -> str:
    """Return 'low', 'high', or 'interior' describing where value sits."""
    lo, hi = bounds[name]
    rng = max(hi - lo, 1e-12)
    if (value - lo) / rng < 0.01:
        return "low"
    if (hi - value) / rng < 0.01:
        return "high"
    return "interior"


def run_warm_started_smm(
    variant: Variant,
    theta_warm: np.ndarray,
    seeds: list[int] = SEEDS,
    n_first_stage: int = N_FIRST_STAGE,
    n_second_stage: int = N_SECOND_STAGE,
    n_variance_replications: int = N_VARIANCE_REPLICATIONS,
    seeds_offset: int = 73,
    random_state: int = 42,
) -> dict:
    """Run a two-stage SMM under variant-specific bounds, warm-started at
    theta_warm. Returns a dict with theta_hat, moment fit, and validation
    moments."""
    from skopt import gp_minimize
    from skopt.space import Real

    bounds = resolve_bounds(variant)

    base_cfg = Config()

    # If the warm point is outside the (possibly shrunken) widened space,
    # clip it. In our variants we only ever WIDEN, so this is a no-op.
    theta_warm_in_space = theta_warm.copy()
    for j, ps in enumerate(PARAM_SPACE):
        lo, hi = bounds[ps.name]
        theta_warm_in_space[j] = float(np.clip(theta_warm[j], lo, hi))

    # Stage 0: variance estimation at the warm point.
    print(f"[{variant.label}] Stage 0: variance reps = {n_variance_replications}, "
          f"5 seeds each, at warm point")
    sys.stdout.flush()
    cfg_warm = apply_params(base_cfg, theta_warm_in_space)
    variances, _ = estimate_moment_covariance(
        cfg_warm, n_replications=n_variance_replications,
        seeds_per_replication=5, seed_offset=2000,
    )
    W1 = diagonal_weighting_matrix(variances)

    # Stage 1: BO with warm start.
    space = [Real(bounds[ps.name][0], bounds[ps.name][1], name=ps.name) for ps in PARAM_SPACE]
    stage1_x: list = []
    stage1_y: list = []

    def stage1_obj(theta_list):
        theta = np.array(theta_list)
        val = objective(theta, base_cfg, seeds, W1)
        stage1_x.append(list(theta_list))
        stage1_y.append(val)
        if len(stage1_y) % 5 == 0:
            print(f"[{variant.label}]   stage1 iter {len(stage1_y):>3d}/{n_first_stage}: "
                  f"obj = {val:.4e}, best = {min(stage1_y):.4e}")
            sys.stdout.flush()
        return val

    print(f"[{variant.label}] Stage 1: warm-start at theta_hat, n_calls = {n_first_stage}, "
          f"n_initial_points = 10")
    sys.stdout.flush()
    res1 = gp_minimize(
        stage1_obj, space,
        n_calls=n_first_stage,
        n_initial_points=10,
        initial_point_generator="sobol",
        random_state=random_state,
        x0=[list(theta_warm_in_space)],
        verbose=False,
    )
    theta1 = np.array(res1.x)
    obj1 = float(res1.fun)

    # Stage 2: covariance at theta1, optimal weighting, warm-started at theta1.
    print(f"[{variant.label}] Stage 2: variance reps = {n_variance_replications} at theta1")
    sys.stdout.flush()
    cfg1 = apply_params(base_cfg, theta1)
    _, cov_at_theta1 = estimate_moment_covariance(
        cfg1, n_replications=n_variance_replications,
        seeds_per_replication=5, seed_offset=3000,
    )
    W2 = optimal_weighting_matrix(cov_at_theta1)

    stage2_x: list = []
    stage2_y: list = []

    def stage2_obj(theta_list):
        theta = np.array(theta_list)
        val = objective(theta, base_cfg, seeds, W2)
        stage2_x.append(list(theta_list))
        stage2_y.append(val)
        if len(stage2_y) % 5 == 0:
            print(f"[{variant.label}]   stage2 iter {len(stage2_y):>3d}/{n_second_stage}: "
                  f"obj = {val:.4e}, best = {min(stage2_y):.4e}")
            sys.stdout.flush()
        return val

    print(f"[{variant.label}] Stage 2: gp_minimize, n_calls = {n_second_stage}, "
          f"warm-started at stage 1 optimum")
    sys.stdout.flush()
    res2 = gp_minimize(
        stage2_obj, space,
        n_calls=n_second_stage,
        n_initial_points=5,
        initial_point_generator="sobol",
        random_state=random_state + 1,
        x0=[list(theta1)],
        verbose=False,
    )
    theta_hat = np.array(res2.x)
    obj2 = float(res2.fun)

    # Compute moment fit at theta_hat using the same 5-seed pool used in BO.
    m_at_opt = simulated_moments(theta_hat, base_cfg, seeds)
    targets = moment_targets()
    errors = m_at_opt - targets
    cal_within_tol = [
        bool(abs(errors[i]) <= CALIBRATION_MOMENTS[i].target_tolerance)
        for i in range(len(CALIBRATION_MOMENTS))
    ]

    # Validation moments (renter-owner gap is the headline one).
    cfg_at_opt = apply_params(base_cfg, theta_hat)
    runs = simulate_seeds(cfg_at_opt, seeds)
    val = evaluate_moments(VALIDATION_MOMENTS, runs)

    # Tag each parameter's binding state at the new optimum.
    binding_state = {
        ps.name: is_still_binding(ps.name, float(theta_hat[j]), bounds, variant.overrides)
        for j, ps in enumerate(PARAM_SPACE)
    }

    return {
        "variant": variant.label,
        "description": variant.description,
        "bounds": {k: list(v) for k, v in bounds.items()},
        "theta_warm": theta_warm_in_space.tolist(),
        "theta1": theta1.tolist(),
        "theta_hat": theta_hat.tolist(),
        "obj_stage1": obj1,
        "obj_stage2": obj2,
        "param_names": list(PARAM_NAMES),
        "binding_state": binding_state,
        "calibration_moment_names": [m.name for m in CALIBRATION_MOMENTS],
        "calibration_targets": targets.tolist(),
        "calibration_values": m_at_opt.tolist(),
        "calibration_errors": errors.tolist(),
        "calibration_within_tol": cal_within_tol,
        "calibration_tolerances": [m.target_tolerance for m in CALIBRATION_MOMENTS],
        "validation_renter_owner_gap": float(val["within_region_renter_owner_vote_gap"]),
        "validation_price_growth_correlation": float(
            val.get("cross_regional_extreme_share_price_growth_correlation", float("nan"))
        ),
        "validation_bottom_quartile_wage_growth": float(
            val.get("bottom_quartile_wage_growth_2010_2025", float("nan"))
        ),
        "budget": {
            "n_first_stage": n_first_stage,
            "n_second_stage": n_second_stage,
            "n_variance_replications": n_variance_replications,
            "seeds": seeds,
        },
    }


def load_warm_theta() -> np.ndarray:
    path = OUTPUTS / "smm_optimum.json"
    data = json.loads(path.read_text())
    return np.array(data["theta_hat"], dtype=float)


def main() -> None:
    theta_warm = load_warm_theta()
    print(f"Warm-start theta_hat = {theta_warm.tolist()}")
    print(f"Variants: {[v.label for v in VARIANTS]}")
    print(f"Budget per variant: {N_FIRST_STAGE} Stage 1, {N_SECOND_STAGE} Stage 2, "
          f"{N_VARIANCE_REPLICATIONS} variance reps")
    print()

    results: list[dict] = []
    for variant in VARIANTS:
        print("=" * 78)
        print(f"VARIANT: {variant.label}")
        print(f"  {variant.description}")
        for ov in variant.overrides:
            print(f"  override: {ov.name} low={ov.low} high={ov.high}")
        print("=" * 78)
        sys.stdout.flush()

        t_start = time.time()
        try:
            result = run_warm_started_smm(variant, theta_warm)
            result["elapsed_seconds"] = time.time() - t_start
            results.append(result)
            out_path = OUTPUTS / f"widen_bounds_{variant.label}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"[{variant.label}] saved {out_path.relative_to(ROOT)} "
                  f"(elapsed {result['elapsed_seconds']:.0f} s)")

            # Compact per-variant summary.
            print(f"[{variant.label}] theta_hat:")
            for name, v in zip(PARAM_NAMES, result["theta_hat"]):
                state = result["binding_state"][name]
                print(f"    {name:32s} = {v:+.4f}  ({state})")
            print(f"[{variant.label}] aggregate AfD share = "
                  f"{result['calibration_values'][8]:+.4f}  "
                  f"(target {result['calibration_targets'][8]:+.4f})")
            print(f"[{variant.label}] renter-owner cleavage = "
                  f"{result['validation_renter_owner_gap']:+.4f}  "
                  f"(target +0.150)")
            sys.stdout.flush()

        except Exception as e:
            print(f"[{variant.label}] FAILED: {e}")
            results.append({"variant": variant.label, "error": str(e)})

        print()

    summary_path = OUTPUTS / "widen_bounds_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {summary_path.relative_to(ROOT)}")

    # Print the headline comparison table.
    print()
    print("=" * 100)
    print("HEADLINE COMPARISON (across variants)")
    print("=" * 100)
    print(f"{'variant':<40s}  {'agg AfD':>10s}  {'cleavage':>10s}  "
          f"{'binding state of widened param(s)':>40s}")
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<40s}  FAILED")
            continue
        widened_names = [n for n in r["binding_state"]
                         if any(ov.name == n for v in VARIANTS if v.label == r["variant"]
                                for ov in v.overrides)]
        widened_state = ", ".join(
            f"{n}={r['binding_state'][n]}" for n in widened_names
        ) if widened_names else "(none, control)"
        print(
            f"{r['variant']:<40s}  "
            f"{r['calibration_values'][8]:+10.4f}  "
            f"{r['validation_renter_owner_gap']:+10.4f}  "
            f"{widened_state}"
        )


if __name__ == "__main__":
    main()
