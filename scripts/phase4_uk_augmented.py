"""Phase 4 UK external stress test at the p75-augmented theta_hat.

Holds the 10 augmented behavioural parameters at their German calibration
values; recalibrates only the UK regional and demographic primitives via
make_uk_config(). Adds the augmented configuration on top:
    assortative_help_enabled = True
    grad_share_data_path = data/cosmopolitan_grad_share_uk.json
    beta_network from theta_hat (0.775)
    gamma_cosmopolitan from theta_hat (-3.965)

Surfaces:
    aggregate Leave share at Brexit (t = BREXIT_PERIOD = 11)
    within-region renter-owner cleavage at Brexit
    cross-regional dispersion at Brexit
    regional ordering correlation with empirical Leave (Pearson and Spearman)
    side-by-side comparison vs original values (0.176 / +0.230 / 0.041)

Output: outputs/augmented_phase4_uk.json
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

from abmhp import simulate
from abmhp.estimation.smm import apply_params, build_param_space
from abmhp.validation.uk import (
    BREXIT_PERIOD,
    EMPIRICAL_AGGREGATE_LEAVE,
    EMPIRICAL_LEAVE_BY_REGION,
    UK_REGION_NAMES,
    make_uk_config,
)
from abmhp import Config

OUTPUTS = ROOT / "outputs"
UK_GRAD_PATH = "data/cosmopolitan_grad_share_uk.json"
UK_SEEDS = list(range(73, 83))


def _load_theta() -> tuple[np.ndarray, tuple]:
    payload = json.loads((OUTPUTS / "smm_augmented_reduced.json").read_text())
    theta = np.array(payload["theta_hat"], dtype=float)
    names = list(payload["param_names"])
    sentinel = Config()
    sentinel = replace(sentinel, behavioral=replace(
        sentinel.behavioral,
        estimate_beta_n=True,
        estimate_gamma_cosmopolitan=True,
    ))
    ps = build_param_space(sentinel)
    assert [p.name for p in ps] == names
    return theta, ps


def main() -> None:
    t0 = time.time()
    theta, param_space = _load_theta()
    names = [p.name for p in param_space]
    print(f"theta_hat (augmented): {dict(zip(names, [round(v, 4) for v in theta]))}")
    print(f"UK seeds: {UK_SEEDS}")
    print(f"Running UK stress test at augmented theta_hat\n")

    runs = []
    for s in UK_SEEDS:
        cfg = make_uk_config(seed=s, n_periods=14)
        cfg = apply_params(cfg, theta, param_space=param_space)
        cfg = replace(
            cfg,
            behavioral=replace(
                cfg.behavioral,
                assortative_help_enabled=True,
            ),
            voting=replace(
                cfg.voting,
                grad_share_data_path=UK_GRAD_PATH,
            ),
        )
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))

    t = BREXIT_PERIOD
    leave_by_seed = np.array([float(h.vote_aggregate[t]) for _, h in runs])
    cleavage_by_seed = []
    dispersion_by_seed = []
    regional_by_seed = []
    for _, h in runs:
        rent_v = h.vote_by_tenure[t, :, 0]
        own_v = h.vote_by_tenure[t, :, 1]
        cleavage_by_seed.append(float((rent_v - own_v).mean()))
        dispersion_by_seed.append(float(h.vote[t].std(ddof=1)))
        regional_by_seed.append(h.vote[t].copy())
    regional_mean = np.stack(regional_by_seed).mean(axis=0)

    # Regional ordering correlation with empirical.
    empirical = np.array([EMPIRICAL_LEAVE_BY_REGION[n] for n in UK_REGION_NAMES])
    pearson = float(np.corrcoef(regional_mean, empirical)[0, 1])
    # Spearman = corr of ranks.
    rank_m = np.argsort(np.argsort(regional_mean))
    rank_e = np.argsort(np.argsort(empirical))
    spearman = float(np.corrcoef(rank_m, rank_e)[0, 1])

    aggregate_leave = float(leave_by_seed.mean())
    cleavage_mean = float(np.mean(cleavage_by_seed))
    dispersion_mean = float(np.mean(dispersion_by_seed))
    empirical_dispersion = float(np.std(list(EMPIRICAL_LEAVE_BY_REGION.values()), ddof=1))

    # Load original UK validation for comparison.
    orig_path = OUTPUTS / "uk_validation_payload.json"
    orig = None
    if orig_path.exists():
        d = json.loads(orig_path.read_text())
        r = d.get("result", d)  # original payload nests under "result"
        orig = {
            "aggregate_leave_share": r.get("aggregate_leave_share"),
            "within_region_renter_owner_gap": r.get("within_region_renter_owner_gap"),
            "cross_regional_dispersion": r.get("cross_regional_dispersion"),
            "cross_regional_leave_price_partial_correlation":
                r.get("cross_regional_leave_price_partial_correlation"),
        }

    payload = {
        "phase": "4_uk_augmented",
        "theta_hat": theta.tolist(),
        "param_names": names,
        "uk_seeds": UK_SEEDS,
        "brexit_period": t,
        "aggregate_leave_share": aggregate_leave,
        "aggregate_leave_share_sd": float(leave_by_seed.std(ddof=1)),
        "within_region_renter_owner_gap": cleavage_mean,
        "within_region_renter_owner_gap_sd": float(np.std(cleavage_by_seed, ddof=1)),
        "cross_regional_dispersion": dispersion_mean,
        "cross_regional_dispersion_sd": float(np.std(dispersion_by_seed, ddof=1)),
        "regional_ordering_pearson_with_empirical": pearson,
        "regional_ordering_spearman_with_empirical": spearman,
        "regional_leave_model": {n: float(v) for n, v in zip(UK_REGION_NAMES, regional_mean)},
        "regional_leave_empirical": dict(EMPIRICAL_LEAVE_BY_REGION),
        "empirical_aggregate_leave": EMPIRICAL_AGGREGATE_LEAVE,
        "empirical_dispersion": empirical_dispersion,
        "original_comparison": orig,
        "config_flags": {
            "assortative_help_enabled": True,
            "grad_share_data_path": UK_GRAD_PATH,
        },
        "elapsed_seconds": time.time() - t0,
    }
    out = OUTPUTS / "augmented_phase4_uk.json"
    out.write_text(json.dumps(payload, indent=2))

    print("=" * 96)
    print("PHASE 4 UK STRESS TEST (augmented theta_hat)")
    print("=" * 96)
    print(f"  {'metric':40s}  {'original':>10s}  {'augmented':>10s}  {'empirical':>10s}")
    if orig:
        print(f"  {'aggregate Leave share':40s}  "
              f"{orig['aggregate_leave_share']:>10.4f}  {aggregate_leave:>10.4f}  "
              f"{EMPIRICAL_AGGREGATE_LEAVE:>10.4f}")
        print(f"  {'within-region renter-owner gap':40s}  "
              f"{orig['within_region_renter_owner_gap']:>10.4f}  {cleavage_mean:>10.4f}  "
              f"{'+0.15':>10s}")
        print(f"  {'cross-regional dispersion':40s}  "
              f"{orig['cross_regional_dispersion']:>10.4f}  {dispersion_mean:>10.4f}  "
              f"{empirical_dispersion:>10.4f}")
        print(f"  {'partial corr (income-ctrl)':40s}  "
              f"{orig['cross_regional_leave_price_partial_correlation']:>10.4f}  "
              f"{'see val':>10s}  "
              f"{-0.20:>10.4f}")
    print()
    print(f"  {'regional Pearson with empirical':40s}  "
          f"{'--':>10s}  {pearson:>10.4f}")
    print(f"  {'regional Spearman with empirical':40s}  "
          f"{'--':>10s}  {spearman:>10.4f}")
    print()
    print("Regional Leave (model vs empirical):")
    for n in UK_REGION_NAMES:
        m = float(dict(zip(UK_REGION_NAMES, regional_mean))[n])
        e = float(EMPIRICAL_LEAVE_BY_REGION[n])
        print(f"  {n:25s}  model={m:+.4f}  empirical={e:+.4f}  delta={m-e:+.4f}")

    print(f"\nSaved {out.relative_to(ROOT)}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
