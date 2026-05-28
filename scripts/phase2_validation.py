"""Phase 2 validation at fixed theta_hat across feature-flag cells.

Runs each cell at the headline SMM theta_hat with 10 seeds DE + 10 seeds UK
and reports:
    - aggregate AfD (DE, final period)
    - DE cleavage (within-region renter-owner vote gap, final period)
    - UK cleavage (within-region renter-owner Leave gap, t = BREXIT_PERIOD)
    - DE cross-regional dispersion (sd of regional vote share, final)
    - UK cross-regional dispersion (sd of regional Leave share at Brexit)
    - Table 4 fit fraction (# of the 12 SMM calibration moments within
      target_tolerance, evaluated on the DE runs)

Cells (10 sub-cells; the original spec lists 7 cells, where cell 7 has 4
beta_n sub-cells):
    1. baseline                         (all flags False)
    2. assortative_help only
    3. gamma_cosmopolitan = -5.156      (shared pilot)
    4. estimate_beta_n=True, beta_n=0.6 (plumbing sanity; should be == cell 1)
    5. assortative_help + gamma=-5.156
    6. assortative_help + gamma=-7.5    (DE-specific pilot)
    7a-7d. all-three with beta_n in {0.4, 0.6, 0.8, 1.0}, gamma=-5.156

Output: outputs/augmented_phase2_results.json
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

from abmhp import Config, simulate
from abmhp.estimation.moments import (
    CALIBRATION_MOMENTS,
    VALIDATION_MOMENTS,
    evaluate_moments,
    simulate_seeds,
)
from abmhp.estimation.smm import apply_params
from abmhp.validation.uk import BREXIT_PERIOD, make_uk_config


OUTPUTS = ROOT / "outputs"
DE_GRAD_PATH = "data/cosmopolitan_grad_share_de.json"
UK_GRAD_PATH = "data/cosmopolitan_grad_share_uk.json"

DE_SEEDS = list(range(73, 83))
UK_SEEDS = list(range(73, 83))


# ---------------------------------------------------------------------------
# Cell specification
# ---------------------------------------------------------------------------


def _cell(
    name: str,
    *,
    assortative_help: bool = False,
    gamma: float = 0.0,
    estimate_beta_n_flag: bool = False,
    beta_n: float | None = None,
) -> dict:
    return {
        "name": name,
        "assortative_help": assortative_help,
        "gamma": gamma,
        "estimate_beta_n_flag": estimate_beta_n_flag,
        "beta_n": beta_n,
    }


CELLS = [
    _cell("1_baseline"),
    _cell("2_assortative_help", assortative_help=True),
    _cell("3_gamma_shared", gamma=-5.156),
    _cell("4_beta_n_flag_plumbing", estimate_beta_n_flag=True, beta_n=0.6),
    _cell("5_help_plus_gamma_shared", assortative_help=True, gamma=-5.156),
    _cell("6_help_plus_gamma_DE_specific", assortative_help=True, gamma=-7.5),
    _cell("7a_all_three_beta_n_0.4", assortative_help=True, gamma=-5.156, beta_n=0.4),
    _cell("7b_all_three_beta_n_0.6", assortative_help=True, gamma=-5.156, beta_n=0.6),
    _cell("7c_all_three_beta_n_0.8", assortative_help=True, gamma=-5.156, beta_n=0.8),
    _cell("7d_all_three_beta_n_1.0", assortative_help=True, gamma=-5.156, beta_n=1.0),
]


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def _apply_cell_de(theta_hat: np.ndarray, cell: dict) -> Config:
    cfg = apply_params(Config(), theta_hat)
    voting = cfg.voting
    behavioral = cfg.behavioral

    voting_kwargs: dict = {}
    if cell["gamma"] != 0.0:
        voting_kwargs["gamma_cosmopolitan"] = cell["gamma"]
        voting_kwargs["grad_share_data_path"] = DE_GRAD_PATH
    if cell["beta_n"] is not None:
        voting_kwargs["beta_network"] = cell["beta_n"]
    if voting_kwargs:
        voting = replace(voting, **voting_kwargs)

    beh_kwargs: dict = {}
    if cell["assortative_help"]:
        beh_kwargs["assortative_help_enabled"] = True
    if cell["estimate_beta_n_flag"]:
        beh_kwargs["estimate_beta_n"] = True
    if beh_kwargs:
        behavioral = replace(behavioral, **beh_kwargs)

    return replace(cfg, voting=voting, behavioral=behavioral)


def _apply_cell_uk(theta_hat: np.ndarray, cell: dict, seed: int) -> Config:
    cfg = make_uk_config(seed=seed, n_periods=14)
    cfg = apply_params(cfg, theta_hat)
    voting = cfg.voting
    behavioral = cfg.behavioral

    voting_kwargs: dict = {}
    if cell["gamma"] != 0.0:
        voting_kwargs["gamma_cosmopolitan"] = cell["gamma"]
        voting_kwargs["grad_share_data_path"] = UK_GRAD_PATH
    if cell["beta_n"] is not None:
        voting_kwargs["beta_network"] = cell["beta_n"]
    if voting_kwargs:
        voting = replace(voting, **voting_kwargs)

    beh_kwargs: dict = {}
    if cell["assortative_help"]:
        beh_kwargs["assortative_help_enabled"] = True
    if cell["estimate_beta_n_flag"]:
        beh_kwargs["estimate_beta_n"] = True
    if beh_kwargs:
        behavioral = replace(behavioral, **beh_kwargs)

    return replace(cfg, voting=voting, behavioral=behavioral)


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------


def run_de(theta_hat: np.ndarray, cell: dict) -> tuple[list, dict[str, float]]:
    cfg = _apply_cell_de(theta_hat, cell)
    runs = simulate_seeds(cfg, DE_SEEDS)
    cal = evaluate_moments(CALIBRATION_MOMENTS, runs)
    val = evaluate_moments(VALIDATION_MOMENTS, runs)
    return runs, {**cal, **val}


def run_uk(theta_hat: np.ndarray, cell: dict) -> tuple[list, dict[str, float]]:
    runs = []
    for s in UK_SEEDS:
        cfg = _apply_cell_uk(theta_hat, cell, s)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))

    # UK metrics at BREXIT_PERIOD (t=11).
    t = BREXIT_PERIOD
    agg = float(np.mean([float(h.vote_aggregate[t]) for _, h in runs]))
    cleavage_vals = []
    dispersion_vals = []
    for _, h in runs:
        rent_v = h.vote_by_tenure[t, :, 0]
        own_v = h.vote_by_tenure[t, :, 1]
        cleavage_vals.append(float((rent_v - own_v).mean()))
        dispersion_vals.append(float(h.vote[t].std(ddof=1)))
    return runs, {
        "uk_aggregate_leave": agg,
        "uk_cleavage": float(np.mean(cleavage_vals)),
        "uk_dispersion": float(np.mean(dispersion_vals)),
    }


def table4_fit_fraction(de_moments: dict[str, float]) -> tuple[int, int, list[dict]]:
    rows = []
    n_pass = 0
    for m in CALIBRATION_MOMENTS:
        model_val = float(de_moments[m.name])
        err = float(model_val - m.value)
        passed = bool(abs(err) <= m.target_tolerance)
        if passed:
            n_pass += 1
        rows.append({
            "name": m.name,
            "target": float(m.value),
            "model": model_val,
            "error": err,
            "tolerance": float(m.target_tolerance),
            "passed": passed,
        })
    return n_pass, len(CALIBRATION_MOMENTS), rows


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    smm = json.loads((OUTPUTS / "smm_optimum.json").read_text())
    theta_hat = np.array(smm["theta_hat"], dtype=float)
    param_names = list(smm["param_names"])
    print(f"theta_hat = {dict(zip(param_names, theta_hat.tolist()))}")
    print(f"DE seeds: {DE_SEEDS}")
    print(f"UK seeds: {UK_SEEDS}")
    print(f"Running {len(CELLS)} cells (DE 10 seeds + UK 10 seeds each)\n")

    results: list[dict] = []
    haywire_flags: list[str] = []

    for i, cell in enumerate(CELLS, 1):
        cell_t0 = time.time()
        print(f"  [{i}/{len(CELLS)}] {cell['name']}... ", end="", flush=True)

        _, de_moments = run_de(theta_hat, cell)
        _, uk_metrics = run_uk(theta_hat, cell)
        n_pass, n_total, fit_rows = table4_fit_fraction(de_moments)

        agg_afd_de = float(de_moments["aggregate_extreme_share_final"])
        de_cleavage = float(de_moments["within_region_renter_owner_vote_gap"])
        de_dispersion = float(de_moments["cross_regional_extreme_share_dispersion"])

        # Haywire heuristics.
        notes = []
        if agg_afd_de < 0.05 or agg_afd_de > 0.95:
            notes.append(f"aggregate AfD outside [0.05, 0.95]: {agg_afd_de:.4f}")
            haywire_flags.append(f"{cell['name']}: aggregate AfD = {agg_afd_de:.4f}")
        if n_pass < 5:
            notes.append(f"fit fraction below 5/12: {n_pass}/{n_total}")
            haywire_flags.append(f"{cell['name']}: fit = {n_pass}/{n_total}")
        for v in (agg_afd_de, de_cleavage, de_dispersion,
                   uk_metrics["uk_aggregate_leave"], uk_metrics["uk_cleavage"],
                   uk_metrics["uk_dispersion"]):
            if not np.isfinite(v):
                notes.append("non-finite value detected")
                haywire_flags.append(f"{cell['name']}: non-finite output")
                break

        elapsed = time.time() - cell_t0
        results.append({
            "cell": cell["name"],
            "spec": {
                "assortative_help": cell["assortative_help"],
                "gamma_cosmopolitan": cell["gamma"],
                "estimate_beta_n_flag": cell["estimate_beta_n_flag"],
                "beta_network": cell["beta_n"] if cell["beta_n"] is not None else 0.6,
            },
            "de_calibration_moments": {k: float(v) for k, v in de_moments.items()
                                         if any(k == m.name for m in CALIBRATION_MOMENTS)},
            "de_validation_moments": {k: float(v) for k, v in de_moments.items()
                                        if any(k == m.name for m in VALIDATION_MOMENTS)},
            "uk_metrics": uk_metrics,
            "fit_n": n_pass,
            "fit_total": n_total,
            "fit_rows": fit_rows,
            "summary": {
                "aggregate_afd_de": agg_afd_de,
                "de_cleavage": de_cleavage,
                "de_dispersion": de_dispersion,
                "uk_cleavage": uk_metrics["uk_cleavage"],
                "uk_dispersion": uk_metrics["uk_dispersion"],
                "fit_fraction": f"{n_pass}/{n_total}",
            },
            "notes": notes,
            "elapsed_seconds": elapsed,
        })
        print(f"done ({elapsed:.1f}s) fit={n_pass}/{n_total}"
              + (f"  NOTES: {notes}" if notes else ""))

    payload = {
        "theta_hat": theta_hat.tolist(),
        "param_names": param_names,
        "de_seeds": DE_SEEDS,
        "uk_seeds": UK_SEEDS,
        "cells": results,
        "haywire_flags": haywire_flags,
        "elapsed_seconds": time.time() - t0,
    }
    out = OUTPUTS / "augmented_phase2_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)} ({time.time() - t0:.1f}s total)")

    # Consolidated table.
    print("\n" + "=" * 124)
    print("PHASE 2 CONSOLIDATED TABLE (at theta_hat, 10 seeds DE + 10 seeds UK)")
    print("=" * 124)
    hdr = (f"{'cell':36s}  {'agg AfD':>9s}  {'DE cleav':>9s}  {'UK cleav':>9s}  "
           f"{'DE disp':>9s}  {'UK disp':>9s}  {'Table 4 fit':>12s}")
    print(hdr)
    print("-" * 124)
    for r in results:
        s = r["summary"]
        print(f"{r['cell']:36s}  {s['aggregate_afd_de']:+9.4f}  "
              f"{s['de_cleavage']:+9.4f}  {s['uk_cleavage']:+9.4f}  "
              f"{s['de_dispersion']:>9.4f}  {s['uk_dispersion']:>9.4f}  "
              f"{s['fit_fraction']:>12s}")

    if haywire_flags:
        print("\n*** HAYWIRE FLAGS RAISED ***")
        for flag in haywire_flags:
            print(f"  - {flag}")
    else:
        print("\nNo haywire flags raised.")


if __name__ == "__main__":
    main()
