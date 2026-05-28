"""Tier 1 + Tier 2C diagnostic: o-construction sensitivity.

Tier 1A: realised_gain_share in {1.0, 0.5, 0.25, 0.0} with user_cost_rate = 0
Tier 1B: user_cost_rate = 0.04 with realised_gain_share = 1.0
Tier 1C: joint cell, realised_gain_share = 0.25 AND user_cost_rate = 0.04
Tier 2C: aspiration reference quantile switched from p75 (default) to p50

All cells at headline theta_hat. No SMM re-estimation. Defaults preserve
the current model so the gain=1.0, cost=0.0, p75 cell reproduces the
baseline.

Per cell, the diagnostic reports:
  - aggregate AfD share at T (DE; target 0.208)
  - within-region renter-owner cleavage (DE; target 0.150)
  - within-region renter-owner cleavage (UK at Brexit period; target 0.150)
  - cross-regional vote dispersion (DE at T; target 0.080)
  - Table 4 fit fraction (DE; 12 SMM moments within scoring tolerance)

10 seeds per country per cell.

Outputs:
  outputs/o_construction_diagnostic.json
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
    evaluate_moments,
    simulate_seeds,
)
from abmhp.estimation.smm import apply_params, moment_targets
from abmhp.validation.uk import BREXIT_PERIOD, make_uk_config


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(73, 83))  # 10 seeds


CELLS: list[dict] = [
    {"label": "baseline (gain=1.0, cost=0.0, p75)",
     "gain": 1.0, "cost": 0.0, "quantile": 0.75},
    {"label": "1A: gain=0.5, cost=0.0, p75",
     "gain": 0.5, "cost": 0.0, "quantile": 0.75},
    {"label": "1A: gain=0.25, cost=0.0, p75",
     "gain": 0.25, "cost": 0.0, "quantile": 0.75},
    {"label": "1A: gain=0.0, cost=0.0, p75",
     "gain": 0.0, "cost": 0.0, "quantile": 0.75},
    {"label": "1B: gain=1.0, cost=0.04, p75",
     "gain": 1.0, "cost": 0.04, "quantile": 0.75},
    {"label": "1C joint: gain=0.25, cost=0.04, p75",
     "gain": 0.25, "cost": 0.04, "quantile": 0.75},
    {"label": "2C: gain=1.0, cost=0.0, p50",
     "gain": 1.0, "cost": 0.0, "quantile": 0.50},
]


def load_theta_hat() -> np.ndarray:
    return np.array(json.loads((OUTPUTS / "smm_optimum.json").read_text())["theta_hat"], dtype=float)


def apply_cell(cfg: Config, gain: float, cost: float, quantile: float) -> Config:
    """Apply Tier-1/Tier-2C feature flags to a config."""
    cfg = replace(cfg, behavioral=replace(
        cfg.behavioral,
        realised_gain_share=float(gain),
        user_cost_rate=float(cost),
    ))
    cfg = replace(cfg, voting=replace(
        cfg.voting,
        aspiration_reference_quantile=float(quantile),
    ))
    return cfg


def simulate_de(theta_hat: np.ndarray, gain: float, cost: float, quantile: float) -> list:
    base = Config()
    cfg_theta = apply_params(base, theta_hat)
    runs = []
    for s in SEEDS:
        cfg = apply_cell(replace(cfg_theta, seed=int(s)), gain, cost, quantile)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def simulate_uk(theta_hat: np.ndarray, gain: float, cost: float, quantile: float) -> list:
    """UK validation. SMM-identified parameters already locked inside
    make_uk_config; the cell overrides are applied on top via replace."""
    runs = []
    for s in SEEDS:
        cfg = make_uk_config(seed=s, n_periods=14)
        cfg = apply_cell(cfg, gain, cost, quantile)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def aggregate_afd(runs) -> float:
    return float(np.mean([float(h.vote_aggregate[-1]) for _, h in runs]))


def cleavage(runs, t_idx: int = -1) -> float:
    gaps = []
    for _, h in runs:
        rent_v = h.vote_by_tenure[t_idx, :, 0]
        own_v = h.vote_by_tenure[t_idx, :, 1]
        gaps.append(float((rent_v - own_v).mean()))
    return float(np.mean(gaps))


def dispersion(runs, t_idx: int = -1) -> float:
    return float(np.mean([float(h.vote[t_idx].std(ddof=1)) for _, h in runs]))


def calibration_fit_count(runs) -> tuple[int, list]:
    """Return (number-within-tolerance, list-of-(name, target, model, tol, ok))."""
    sim = evaluate_moments(CALIBRATION_MOMENTS, runs)
    targets = moment_targets()
    details = []
    n_ok = 0
    for i, m in enumerate(CALIBRATION_MOMENTS):
        model_v = float(sim[m.name])
        target_v = float(targets[i])
        err = model_v - target_v
        ok = bool(abs(err) <= m.target_tolerance)
        n_ok += int(ok)
        details.append({
            "name": m.name,
            "target": target_v,
            "model": model_v,
            "error": err,
            "tolerance": float(m.target_tolerance),
            "within_tol": ok,
        })
    return n_ok, details


def run_cell(cell: dict, theta_hat: np.ndarray) -> dict:
    t0 = time.time()
    label = cell["label"]
    gain = cell["gain"]
    cost = cell["cost"]
    quantile = cell["quantile"]

    print(f"\n=== {label} ===")
    print(f"  Simulating DE (10 seeds)...")
    sys.stdout.flush()
    de_runs = simulate_de(theta_hat, gain, cost, quantile)

    print(f"  Simulating UK (10 seeds)...")
    sys.stdout.flush()
    uk_runs = simulate_uk(theta_hat, gain, cost, quantile)

    afd = aggregate_afd(de_runs)
    cleav_de = cleavage(de_runs, t_idx=-1)
    cleav_uk = cleavage(uk_runs, t_idx=BREXIT_PERIOD)
    disp_de = dispersion(de_runs, t_idx=-1)
    n_within, fit_details = calibration_fit_count(de_runs)

    print(f"  aggregate AfD = {afd:+.4f}  (target +0.2080, error {afd-0.2080:+.4f})")
    print(f"  DE cleavage   = {cleav_de:+.4f}  (target +0.1500, error {cleav_de-0.150:+.4f})")
    print(f"  UK cleavage   = {cleav_uk:+.4f}  (target +0.1500, error {cleav_uk-0.150:+.4f})")
    print(f"  DE dispersion = {disp_de:.4f}   (target 0.0800)")
    print(f"  Moment fit    = {n_within}/12")
    sys.stdout.flush()

    return {
        "label": label,
        "realised_gain_share": gain,
        "user_cost_rate": cost,
        "aspiration_reference_quantile": quantile,
        "aggregate_afd": afd,
        "aggregate_afd_error": afd - 0.208,
        "cleavage_de": cleav_de,
        "cleavage_de_error": cleav_de - 0.150,
        "cleavage_uk": cleav_uk,
        "cleavage_uk_error": cleav_uk - 0.150,
        "dispersion_de": disp_de,
        "moment_fit_count": n_within,
        "moment_fit_details": fit_details,
        "elapsed_seconds": time.time() - t0,
    }


def main() -> None:
    theta_hat = load_theta_hat()
    print(f"theta_hat = {theta_hat.tolist()}")
    print(f"Seeds = {SEEDS}")
    print(f"Cells = {[c['label'] for c in CELLS]}")

    rows: list[dict] = []
    for cell in CELLS:
        rows.append(run_cell(cell, theta_hat))

    payload = {
        "theta_hat": theta_hat.tolist(),
        "n_seeds": len(SEEDS),
        "cells": rows,
        "calibration_moment_names": [m.name for m in CALIBRATION_MOMENTS],
    }
    out_path = OUTPUTS / "o_construction_diagnostic.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path.relative_to(ROOT)}")

    # Surface a single consolidated table.
    print()
    print("=" * 120)
    print("CONSOLIDATED TABLE")
    print("=" * 120)
    hdr = (f"{'cell':45s}  {'agg AfD':>9s}  {'AfD err':>9s}  "
           f"{'DE cleav':>10s}  {'UK cleav':>10s}  {'DE disp':>9s}  {'fit':>5s}")
    print(hdr)
    print("-" * 120)
    for r in rows:
        print(
            f"{r['label']:45s}  "
            f"{r['aggregate_afd']:+9.4f}  "
            f"{r['aggregate_afd_error']:+9.4f}  "
            f"{r['cleavage_de']:+10.4f}  "
            f"{r['cleavage_uk']:+10.4f}  "
            f"{r['dispersion_de']:9.4f}  "
            f"{r['moment_fit_count']:>2d}/12"
        )


if __name__ == "__main__":
    main()
