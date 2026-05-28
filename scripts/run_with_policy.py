"""Run the baseline simulation with the housing-only mainstream response active.

At the prompt 2 calibration the aggregate extreme-share vote saturates around
0.22, which is below the activation threshold of 0.30. The expected behaviour
for this baseline is therefore: regime never activates, no policy ever engages,
all effective intensities remain at zero. The script prints this explicitly.

Use --force-extreme-share or --beta0 to exercise the policy block in
non-baseline scenarios; that path is also useful as a sanity check before
the counterfactual in prompt 4.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from abmhp import Config, PolicyConfig, PolicyRegime, simulate
from abmhp.config import VotingConfig
from abmhp.estimation.smm import apply_smm_optimum


def first_switch(regime_history: np.ndarray) -> int | None:
    for t, r in enumerate(regime_history):
        if r is PolicyRegime.POPULIST:
            return t
    return None


def extreme_share_years(regime_history: np.ndarray) -> int:
    """Count of periods governed by the extreme-share activation regime."""
    return int(sum(1 for r in regime_history[1:] if r is PolicyRegime.POPULIST))


def aggregate_rent_growth(rent_index: np.ndarray) -> float:
    """Population-weighted (uniform across regions here) rent growth from
    period 0 to T."""
    final_mean = float(rent_index[-1].mean())
    return final_mean - 1.0


def regime_display(regime: PolicyRegime | None) -> str:
    """Human-readable regime label for terminal output."""
    if regime is None:
        return "-"
    if regime is PolicyRegime.MAINSTREAM:
        return "mainstream"
    return "extreme-share"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--force-extreme-share", action="store_true",
                        help="Pin regime to the extreme-share activation regime for the full simulation")
    parser.add_argument("--beta0", type=float, default=None,
                        help="Override voting beta_0 (default -5.2). Use -3.5 to cross the threshold.")
    args = parser.parse_args()

    cfg = apply_smm_optimum(Config(seed=args.seed))
    if args.force_extreme_share:
        cfg = replace(cfg, policy=replace(cfg.policy, force_regime=PolicyRegime.POPULIST.value))
    if args.beta0 is not None:
        cfg = replace(cfg, voting=replace(cfg.voting, beta_0=args.beta0))

    state, hist, _ = simulate(cfg)
    T = cfg.n_periods

    print("=" * 78)
    print(f"RUN WITH POLICY  seed={args.seed}  beta_0={cfg.voting.beta_0:.2f}  "
          f"force_regime={cfg.policy.force_regime}")
    print("=" * 78)
    print()
    print("regime trajectory (year : regime  smoothed-vote  vote-aggregate  "
          "rent-cap  supply-restr  friction)")
    for t in range(T + 1):
        regime_label = regime_display(hist.regime[t])
        print(
            f"  t={t:2d}  {regime_label:14s}  "
            f"sm={hist.smoothed_vote[t]:.3f}  v={hist.vote_aggregate[t]:.3f}  "
            f"rc={hist.effective_rent_cap[t]:.2f}  sr={hist.effective_supply[t]:.2f}  "
            f"fr={hist.effective_friction[t]:.2f}"
        )

    switch = first_switch(hist.regime)
    years_active = extreme_share_years(hist.regime)
    rent_growth = aggregate_rent_growth(hist.rent_index)
    price_growth = float((hist.price[T] / hist.price[0] - 1.0).mean())
    rent_to_price_ratio = rent_growth / max(price_growth, 1e-9)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Years until first regime switch                  : {switch if switch is not None else 'never'}")
    print(f"  Years of extreme-share governance                : {years_active} / {T}")
    print(f"  Aggregate extreme-share vote (final)             : {hist.vote_aggregate[T]:.3f}")
    print(f"  Aggregate dissatisfaction (final)                : {hist.dissat[T].mean():.3f}")
    print(f"  Mean regional rent growth (cumulative)           : {rent_growth:+.2%}")
    print(f"  Mean regional price growth (cumulative)          : {price_growth:+.2%}")
    print(f"  Rent-to-price growth ratio                       : {rent_to_price_ratio:.3f}")
    print(f"  Final aggregate homeownership rate               : {hist.ownership_aggregate[T]:.3f}")

    region_type = cfg.regional.region_type
    region_label = cfg.regional.region_label
    rows = []
    for r in range(cfg.regional.n_regions):
        rows.append({
            "label": region_label[r],
            "type": region_type[r],
            "price_growth": hist.price[T, r] / hist.price[0, r] - 1.0,
            "rent_growth": hist.rent_index[T, r] - 1.0,
            "renter_share": 1.0 - hist.ownership[T, r],
            "vote": hist.vote[T, r],
        })
    print()
    print("REGIONAL FINALS")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

    if switch is None:
        print()
        print("=" * 78)
        print("BASELINE BEHAVIOUR: regime never activates.")
        print("=" * 78)
        print("At the prompt 2 calibration the aggregate extreme-share vote stabilises")
        print("near 0.22, below the activation threshold of 0.30. No policy instruments")
        print("engage, all effective intensities stay at zero, and rent tracks price")
        print("one-for-one. This is the documented baseline; the policy block is")
        print("exercised by tests in tests/test_policy.py and by the counterfactual")
        print("in prompt 4.")


if __name__ == "__main__":
    main()
