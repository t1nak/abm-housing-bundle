"""Validate the bequest-cohort calibration against Bundesbank PHF wave 4 (2017).

Source note: Bundesbank PHF (2017) is the canonical anchor for German wealth
distribution. HFCS and DIW estimates of the top-1% share differ by 5 to 10
percentage points because of business-equity treatment; Bundesbank is used here.

Targets:
    Gini      0.78 to 0.83
    Top-1%    0.22 to 0.30
    Top-10%   0.55 to 0.65
    Bottom-50 0.02 to 0.05
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from abmhp import Config, DemographicConfig, simulate
from abmhp.estimation.smm import apply_smm_optimum

TARGETS = {
    "gini":      (0.78, 0.83),
    "top1":      (0.22, 0.30),
    "top10":     (0.55, 0.65),
    "bottom50":  (0.02, 0.05),
}


def run_one(seed: int, demo_overrides: dict | None = None) -> dict:
    base_demo = DemographicConfig()
    if demo_overrides:
        base_demo = replace(base_demo, **demo_overrides)
    cfg = apply_smm_optimum(Config(seed=seed, demographic=base_demo))
    state, hist, _ = simulate(cfg)
    T = cfg.n_periods
    g_path = hist.gini
    monotonic_rise = bool(np.all(np.diff(g_path) >= -0.005))
    in_band_path = float(((g_path[1:] >= TARGETS["gini"][0]) & (g_path[1:] <= TARGETS["gini"][1])).mean())

    region_type = cfg.regional.region_type
    rent_v = hist.vote_by_tenure[T, :, 0]
    own_v = hist.vote_by_tenure[T, :, 1]
    tenure_gap = float(rent_v.mean() - own_v.mean())

    super_mask = region_type == "super"
    decl_mask = region_type == "decl"
    super_vote = float(hist.vote[T, super_mask].mean())
    decl_vote = float(hist.vote[T, decl_mask].mean())

    return {
        "gini": float(hist.gini[T]),
        "top1": float(hist.top1[T]),
        "top10": float(hist.top10[T]),
        "bottom50": float(hist.bottom50[T]),
        "gini_min": float(g_path[1:].min()),
        "gini_max": float(g_path[1:].max()),
        "gini_in_band_share": in_band_path,
        "gini_monotonic": monotonic_rise,
        "vote_aggregate": float(hist.vote_aggregate[T]),
        "ownership": float(hist.ownership_aggregate[T]),
        "tenure_gap": tenure_gap,
        "decl_vote": decl_vote,
        "super_vote": super_vote,
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    keys = rows[0].keys()
    for k in keys:
        if isinstance(rows[0][k], bool):
            out[k] = sum(int(r[k]) for r in rows) / len(rows)
            continue
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "lo": float(np.percentile(vals, 5)) if len(vals) > 1 else float(vals[0]),
            "hi": float(np.percentile(vals, 95)) if len(vals) > 1 else float(vals[0]),
        }
    return out


def check(summary: dict) -> tuple[bool, list[str]]:
    issues = []
    for k, (lo, hi) in TARGETS.items():
        m = summary[k]["mean"]
        if not (lo <= m <= hi):
            issues.append(f"{k} mean {m:.3f} outside [{lo:.2f}, {hi:.2f}]")
    # Trajectory: either monotonic rise into the band by year T (preferred per
    # task spec: "monotonically rises through the target band ... better
    # mechanics") or sustained presence in the band across the run.
    mono = summary["gini_monotonic"]
    if isinstance(mono, dict):
        mono = mono["mean"]
    gini_band = summary["gini_in_band_share"]
    if isinstance(gini_band, dict):
        gini_band = gini_band["mean"]
    trajectory_ok = mono >= 0.9 or gini_band >= 0.6
    if not trajectory_ok:
        issues.append(
            f"Trajectory unsatisfactory: monotonic-rise {mono:.0%}, in-band {gini_band:.0%}"
        )
    if summary["tenure_gap"]["mean"] < 0.25:
        issues.append(f"Renter-owner gap {summary['tenure_gap']['mean']:+.3f} below 0.25 threshold")
    return (len(issues) == 0, issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bequest-tax", type=float, default=None)
    parser.add_argument("--assortative", type=float, default=None)
    parser.add_argument("--skill-corr", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.bequest_tax is not None:
        overrides["bequest_tax_rate"] = args.bequest_tax
    if args.assortative is not None:
        overrides["assortative_exponent"] = args.assortative
    if args.skill_corr is not None:
        overrides["intergenerational_skill_corr"] = args.skill_corr

    rows = []
    for k in range(args.seeds):
        seed = 73 + k
        r = run_one(seed, overrides if overrides else None)
        rows.append(r)
        print(
            f"  seed {seed}: gini {r['gini']:.3f}  top1 {r['top1']:.3f}  "
            f"top10 {r['top10']:.3f}  bot50 {r['bottom50']:.3f}  "
            f"vote {r['vote_aggregate']:.3f}  tenure_gap {r['tenure_gap']:+.3f}"
        )

    summary = summarize(rows)
    print()
    print("=" * 78)
    print(f"SUMMARY  n={len(rows)} seeds")
    if overrides:
        print(f"  overrides: {overrides}")
    print("=" * 78)
    for k in ("gini", "top1", "top10", "bottom50"):
        m = summary[k]
        lo, hi = TARGETS[k]
        flag = "OK " if lo <= m["mean"] <= hi else "OFF"
        print(
            f"  {k:9s} mean {m['mean']:.3f}  sd {m['std']:.3f}  "
            f"[{m['lo']:.3f}, {m['hi']:.3f}]  target [{lo:.2f}, {hi:.2f}]  {flag}"
        )

    gini_min = summary["gini_min"]
    gini_max = summary["gini_max"]
    in_band = summary["gini_in_band_share"]
    if isinstance(in_band, dict):
        in_band_mean = in_band["mean"]
    else:
        in_band_mean = in_band
    mono = summary["gini_monotonic"]
    print(
        f"\n  Gini trajectory: min {gini_min['mean']:.3f}  max {gini_max['mean']:.3f}  "
        f"in-band share {in_band_mean:.0%}  monotonic-rise {mono:.0%}"
    )
    print(
        f"  Politics: vote_aggregate {summary['vote_aggregate']['mean']:.3f}, "
        f"decl {summary['decl_vote']['mean']:.3f} vs super {summary['super_vote']['mean']:.3f}, "
        f"tenure gap {summary['tenure_gap']['mean']:+.3f}"
    )

    ok, issues = check(summary)
    print()
    if ok:
        print("PASS: all distribution and political-economy targets met")
    else:
        print("FAIL:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)


if __name__ == "__main__":
    main()
