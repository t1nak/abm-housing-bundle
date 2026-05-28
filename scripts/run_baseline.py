"""Run the baseline (bequest-cohort) simulation and print headline diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from abmhp import Config, simulate
from abmhp.estimation.smm import apply_smm_optimum


def main(seed: int = 73) -> None:
    cfg = apply_smm_optimum(Config(seed=seed))
    state, hist, _ = simulate(cfg)
    T = cfg.n_periods

    print("=" * 78)
    print(f"BASELINE  seed={seed}")
    print("=" * 78)
    print(f"  Gini             : {hist.gini[0]:.3f} -> {hist.gini[T]:.3f}")
    print(f"  Top-1% share     : {hist.top1[0]:.3f} -> {hist.top1[T]:.3f}")
    print(f"  Top-10% share    : {hist.top10[0]:.3f} -> {hist.top10[T]:.3f}")
    print(f"  Bottom-50% share : {hist.bottom50[0]:.3f} -> {hist.bottom50[T]:.3f}")
    print(f"  Extreme-share vote: {hist.vote_aggregate[0]:.3f} -> {hist.vote_aggregate[T]:.3f}")
    print(f"  Ownership rate   : {hist.ownership_aggregate[0]:.3f} -> {hist.ownership_aggregate[T]:.3f}")
    print(f"  Deaths/period    : mean {hist.deaths[1:].mean():.1f}, total {hist.deaths.sum()}")

    region_type = cfg.regional.region_type
    region_label = cfg.regional.region_label
    total_price_growth = (hist.price[T] / hist.price[0]) - 1.0
    df = pd.DataFrame({
        "label": region_label,
        "type": region_type,
        "price_growth": total_price_growth,
        "renter_share": 1.0 - hist.ownership[T],
        "dissat": hist.dissat[T],
        "vote": hist.vote[T],
    })
    print()
    print(df.to_string(index=False, float_format="%.3f"))

    print()
    print("VOTE SHARE BY TENURE (final period)")
    print("                renters     owners    gap")
    for rt in ["super", "avg", "decl"]:
        idx = np.where(region_type == rt)[0]
        rent_v = hist.vote_by_tenure[T, idx, 0].mean()
        own_v = hist.vote_by_tenure[T, idx, 1].mean()
        print(f"  {rt:>6s}        {rent_v:.3f}      {own_v:.3f}    {rent_v - own_v:+.3f}")


if __name__ == "__main__":
    main()
