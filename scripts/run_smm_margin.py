"""SMM for the three-margin (Version A) model.

Estimates the small political set {gamma_rent, gamma_asset, gamma_access,
rho_aspiration, beta_0} with margin_decomposition on and beta_renter = 0 (the
margins must explain the tenure cleavage, not a renter dummy). Demographic,
wealth-return, bequest and housing-price parameters are held fixed at defaults
(not estimated) to avoid the boundary-estimate pathology.

The renter-owner tenure gap and the price-growth/vote correlation are reported
as HELD-OUT diagnostics (not in the objective) -- the make-or-break test of
whether the architecture produces the right cross-sectional gradients.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, simulate
from abmhp.estimation.smm import (
    CALIBRATION_MOMENTS, apply_params, build_param_space, run_smm,
)

OUTPUTS = ROOT / "outputs"


def margin_base_cfg(n_periods: int = 15) -> Config:
    # The base config doubles as the point at which the SMM estimates moment
    # variances (Stage 0). It MUST sit at a realistic operating point (margins
    # active, vote ~0.19) -- not the degenerate gamma=0 / vote~0 corner, which
    # gives the political moments microscopic variance, astronomical effective
    # weights, and a spurious near-zero-vote local optimum. The SMM still
    # estimates gamma/rho/beta_0; these are only the variance-base + warm start.
    c = Config(n_periods=n_periods)
    return replace(c, voting=replace(
        c.voting, margin_decomposition=True, beta_renter=0.0, asset_gain_window=5,
        gamma_rent=0.6, gamma_asset=1.1, gamma_access=1.0,
        rho_aspiration=0.85, beta_0=-2.6))


def heldout(theta, base_cfg, pspace, seeds=(73, 74, 75)):
    from abmhp.estimation.moments import (
        _eval_within_region_renter_owner_gap as gap_e,
        _eval_aggregate_extreme_share_final as agg_e,
        _eval_cross_regional_extreme_share_dispersion as disp_e,
        _eval_price_growth_renter_vote_correlation as pg_e,
        _eval_rent_burden_vote_correlation as rb_e,
        _eval_access_gap_vote_correlation as ac_e)
    runs = []
    for s in seeds:
        cfg = replace(apply_params(base_cfg, theta, pspace), seed=s)
        _, h, _ = simulate(cfg)
        runs.append((cfg, h))
    return dict(agg=agg_e(runs), gap=gap_e(runs), disp=disp_e(runs),
                pg=pg_e(runs), rent=rb_e(runs), access=ac_e(runs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    n1, n2, nv = (25, 15, 5) if a.fast else (100, 50, 20)

    base = margin_base_cfg()
    pspace = build_param_space(base)
    names = [p.name for p in pspace]
    print(f"Three-margin SMM (Version A). estimated params: {names}")
    print(f"budget {n1}/{n2}/{nv}")

    # Warm-start near the reachable region (mapped manually): moderate margins,
    # beta_0 ~ -2.6 gives aggregate ~0.19 and tenure gap ~0.16. Without this the
    # small-budget global search falls into a degenerate near-zero-vote corner.
    # Order matches _MARGIN_PARAM_SPACE: [g_rent, g_asset, g_access, rho, beta_0].
    x0 = [0.6, 1.1, 1.0, 0.85, -2.6]
    res = run_smm(base_cfg=base, n_seeds_per_eval=5, seeds_offset=73,
                  n_first_stage=n1, n_second_stage=n2, n_variance_replications=nv,
                  random_state=42, param_space=pspace, x0_stage1=x0, verbose=True)

    print("\n=== optimum ===")
    for n, t, se in zip(names, res.theta_hat, res.se):
        ratio = se / max(abs(t), 1e-9)
        print(f"  {n:16s} = {t:+.4f}  (se {se:.4f}, se/|θ| {ratio:.2f}, "
              f"{'on bound' if ratio < 1e-3 else 'identified' if ratio <= 0.25 else 'weak'})")
    print(f"J={res.j_statistic:.2f} dof={res.dof} p={res.p_value:.4f}")

    print("\nCalibration moments at optimum:")
    for n, tgt, mo in zip([m.name for m in CALIBRATION_MOMENTS],
                          res.moment_targets, res.moment_at_optimum):
        print(f"  {n:52s} target={tgt:+.4f} model={mo:+.4f} err={mo-tgt:+.4f}")

    ho = heldout(res.theta_hat, base, pspace)
    print(f"\n*** HELD-OUT / raw gradients (uncapped) ***")
    print(f"  aggregate 2025            = {ho['agg']:+.4f}   (approx target 0.208)")
    print(f"  regional dispersion       = {ho['disp']:+.4f}   (target 0.08)")
    print(f"  tenure gap (renter-owner) = {ho['gap']:+.4f}   (target ~ +0.15)")
    print(f"  price-growth/renter-vote  = {ho['pg']:+.4f}   (sign restriction > 0.15)")
    print(f"  rent-burden/vote          = {ho['rent']:+.4f}   (sign restriction > 0.15)")
    print(f"  access-pressure/vote      = {ho['access']:+.4f}   (sign restriction > 0.15)")

    payload = dict(
        param_names=names, theta_hat=[float(v) for v in res.theta_hat],
        se=[float(v) for v in res.se], j_statistic=float(res.j_statistic),
        p_value=float(res.p_value),
        moment_names=[m.name for m in CALIBRATION_MOMENTS],
        moment_targets=[float(v) for v in res.moment_targets],
        moment_at_optimum=[float(v) for v in res.moment_at_optimum],
        heldout=ho)
    (OUTPUTS / "smm_margin_optimum.json").write_text(json.dumps(payload, indent=2))
    print("\nSaved outputs/smm_margin_optimum.json")


if __name__ == "__main__":
    main()
