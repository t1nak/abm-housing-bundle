"""Addendum to the Tier 1 + 2C o-construction diagnostic.

Two cells that combine the Tier 2C p50 reference with Tier 1A/1C
o-construction changes:

  Cell 4A: gain=0.25 + cost=0.04 + p50  (joint o-construction + median ref)
  Cell 4B: gain=0.5  + cost=0.00 + p50  (mild gain discount + median ref)

Same harness, same metrics, same 10-seed protocol as the parent diagnostic.
Headline theta_hat fixed; no SMM re-estimation in this step.

Outputs:
  outputs/o_construction_addendum.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import json
import time

import numpy as np

# Reuse the parent diagnostic's helpers so the metric definitions are
# identical between the two surfaced tables.
from o_construction_diagnostic import (
    SEEDS,
    aggregate_afd,
    apply_cell,
    calibration_fit_count,
    cleavage,
    dispersion,
    load_theta_hat,
    run_cell,
)


OUTPUTS = ROOT / "outputs"


ADDENDUM_CELLS: list[dict] = [
    {"label": "4A: gain=0.25, cost=0.04, p50",
     "gain": 0.25, "cost": 0.04, "quantile": 0.50},
    {"label": "4B: gain=0.5, cost=0.00, p50",
     "gain": 0.5, "cost": 0.00, "quantile": 0.50},
]


def main() -> None:
    t0 = time.time()
    theta_hat = load_theta_hat()
    print(f"theta_hat = {theta_hat.tolist()}")
    print(f"Seeds = {SEEDS}")
    print(f"Cells = {[c['label'] for c in ADDENDUM_CELLS]}")

    rows = [run_cell(c, theta_hat) for c in ADDENDUM_CELLS]
    payload = {
        "theta_hat": theta_hat.tolist(),
        "n_seeds": len(SEEDS),
        "cells": rows,
        "elapsed_seconds": time.time() - t0,
    }
    (OUTPUTS / "o_construction_addendum.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved outputs/o_construction_addendum.json")

    print()
    print("=" * 120)
    print("ADDENDUM TABLE")
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
