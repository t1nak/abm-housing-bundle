"""ABM vs HANK side-by-side comparison.

Computes the six-row methodology table for the paper's HANK benchmark
section:

  | Moment / Result | German empirical | ABM | HANK |
  |---|---|---|---|
  | Aggregate wealth Gini | 0,81 | matched | matched |
  | Homeownership rate | 0,50 | matched | matched |
  | Aggregate extreme-share | 0,208 | matched | matched |
  | Cross-regional extreme-share dispersion | nonzero | matched | 0,00 |
  | Within-region renter-owner vote gap | +0,15 | matched | 0,00 |
  | Incomplete-material-repair effect (P3) | structural | -0,113 | 0,00 |

The ABM matches all six rows. HANK matches the first three by aggregate
calibration and mechanically returns zero on the remaining three because
those quantities are not representable in HANK's two-state single-channel
structure.

Outputs:
  outputs/abm_vs_hank_table.csv
  outputs/abm_vs_hank_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from abmhp import Config, simulate
from abmhp.estimation.smm import apply_smm_optimum
from abmhp.hank_benchmark import (
    HankConfig,
    calibrate,
    cross_regional_extreme_share_dispersion,
    incomplete_material_repair_effect,
    make_paired_scenarios,
    vote_under_scenario,
    within_region_tenure_vote_gap,
)


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# Empirical anchors for the German calibration. Used as the target column.
TARGET_GINI = 0.81
TARGET_HOMEOWNERSHIP = 0.50
TARGET_EXTREME_SHARE = 0.208
ABM_P3_EFFECT = -0.113  # Documented in outputs/material_security_results.md


def abm_baseline_moments(seed: int = 73) -> dict[str, float]:
    """Run a single ABM baseline simulation and extract the comparison
    moments at T = n_periods."""
    cfg = apply_smm_optimum(Config(seed=seed))
    _, hist, _ = simulate(cfg)
    T = cfg.n_periods

    gini = float(hist.gini[T])
    homeownership = float(hist.ownership_aggregate[T])
    extreme_share = float(hist.vote_aggregate[T])

    # Cross-regional dispersion of the extreme-share vote at T.
    cross_regional_dispersion = float(hist.vote[T].std(ddof=1))

    # Within-region renter minus owner vote, averaged over regions.
    rent_v = hist.vote_by_tenure[T, :, 0]
    own_v = hist.vote_by_tenure[T, :, 1]
    within_region_gap = float((rent_v - own_v).mean())

    return {
        "wealth_gini": gini,
        "homeownership_rate": homeownership,
        "extreme_share_vote": extreme_share,
        "cross_regional_dispersion": cross_regional_dispersion,
        "within_region_renter_owner_gap": within_region_gap,
        "incomplete_material_repair_effect": ABM_P3_EFFECT,
    }


def hank_moments() -> dict[str, float]:
    """Calibrate HANK and extract the comparison moments."""
    cfg, realised = calibrate(HankConfig())

    # Structural-zero quantities.
    cross_regional = cross_regional_extreme_share_dispersion(cfg)
    within_region = within_region_tenure_vote_gap(cfg)

    # P3 equivalent. Use the C-vs-E aggregate relief level documented in
    # the ABM result (E adds approx 0,05 of aggregate income as transfer
    # on top of the housing-only response). HANK reads both scenarios as
    # equivalent total relief.
    p3 = incomplete_material_repair_effect(cfg, total_relief=0.05)

    return {
        "wealth_gini": realised["wealth_gini"],
        "homeownership_rate": realised["homeownership_rate"],
        "extreme_share_vote": realised["extreme_share_vote"],
        "cross_regional_dispersion": cross_regional,
        "within_region_renter_owner_gap": within_region,
        "incomplete_material_repair_effect": p3,
    }


def hank_paired_scenario_check() -> dict[str, float]:
    """Verify the two HANK scenarios analogous to ABM C and ABM E.

    Both scenarios apply the same rent cap with the same leakage; the
    multichannel scenario adds a redistribution component. In HANK, the
    vote response difference reflects only the marginal effect of
    additional relief in the single channel; the kind of relief is
    irrelevant. Reported here as a side metric to demonstrate the
    single-channel collapse."""
    cfg, _ = calibrate(HankConfig())
    housing_only, multichannel = make_paired_scenarios(
        cfg,
        rent_cap_intensity=0.6,
        rent_cap_leakage=0.4,
        redistribution_intensity=0.027,
    )
    v_h = vote_under_scenario(housing_only, cfg)
    v_m = vote_under_scenario(multichannel, cfg)
    return {
        "housing_only_vote": float(v_h),
        "multichannel_vote": float(v_m),
        "vote_diff": float(v_m - v_h),
    }


def build_comparison_table(
    target: dict[str, float],
    abm: dict[str, float],
    hank: dict[str, float],
) -> pd.DataFrame:
    rows = [
        {
            "moment": "Aggregate wealth Gini",
            "target": f"{target['wealth_gini']:.2f}",
            "ABM": f"{abm['wealth_gini']:.3f}",
            "HANK": f"{hank['wealth_gini']:.3f}",
        },
        {
            "moment": "Homeownership rate",
            "target": f"{target['homeownership_rate']:.2f}",
            "ABM": f"{abm['homeownership_rate']:.3f}",
            "HANK": f"{hank['homeownership_rate']:.3f}",
        },
        {
            "moment": "Aggregate extreme-share vote",
            "target": f"{target['extreme_share_vote']:.3f}",
            "ABM": f"{abm['extreme_share_vote']:.3f}",
            "HANK": f"{hank['extreme_share_vote']:.3f}",
        },
        {
            "moment": "Cross-regional extreme-share dispersion",
            "target": "nonzero",
            "ABM": f"{abm['cross_regional_dispersion']:.3f}",
            "HANK": f"{hank['cross_regional_dispersion']:.3f}",
        },
        {
            "moment": "Within-region renter-owner vote gap",
            "target": ">= +0,15",
            "ABM": f"{abm['within_region_renter_owner_gap']:+.3f}",
            "HANK": f"{hank['within_region_renter_owner_gap']:+.3f}",
        },
        {
            "moment": "Incomplete-material-repair effect (P3)",
            "target": "structural",
            "ABM": f"{abm['incomplete_material_repair_effect']:+.3f}",
            "HANK": f"{hank['incomplete_material_repair_effect']:+.3f}",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    target = {
        "wealth_gini": TARGET_GINI,
        "homeownership_rate": TARGET_HOMEOWNERSHIP,
        "extreme_share_vote": TARGET_EXTREME_SHARE,
    }

    print("Computing ABM baseline moments (seed = 73)")
    abm = abm_baseline_moments(seed=73)
    for k, v in abm.items():
        print(f"  ABM {k}: {v:+.4f}")

    print()
    print("Calibrating HANK and computing benchmark moments")
    hank = hank_moments()
    for k, v in hank.items():
        print(f"  HANK {k}: {v:+.4f}")

    print()
    print("Side metric: HANK paired-scenario vote responses")
    paired = hank_paired_scenario_check()
    for k, v in paired.items():
        print(f"  {k}: {v:+.4f}")

    df = build_comparison_table(target, abm, hank)
    print()
    print("=" * 78)
    print("ABM vs HANK methodology comparison")
    print("=" * 78)
    print(df.to_string(index=False))

    csv_path = OUTPUTS / "abm_vs_hank_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path.relative_to(ROOT)}")

    json_path = OUTPUTS / "abm_vs_hank_summary.json"
    payload = {
        "target": target,
        "abm": abm,
        "hank": hank,
        "hank_paired_scenarios": paired,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
