"""Grid sweep over the three calibration levers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itertools
import numpy as np

from validate_distribution import run_one, TARGETS  # type: ignore


def score(row: dict) -> float:
    """Squared-deviation from band midpoint, normalised by band width.
    Penalises being out-of-band on any of four metrics."""
    keys = ["gini", "top1", "top10", "bottom50"]
    s = 0.0
    for k in keys:
        lo, hi = TARGETS[k]
        mid = 0.5 * (lo + hi)
        width = hi - lo
        s += ((row[k] - mid) / width) ** 2
    return s


def main() -> None:
    grid = {
        "bequest_tax_rate":            [0.10, 0.15, 0.20],
        "assortative_exponent":        [1.5, 2.0, 2.5, 3.0],
        "intergenerational_skill_corr":[0.5, 0.6, 0.7],
    }
    seeds = [73, 74, 75]
    results = []
    for combo in itertools.product(*grid.values()):
        ov = dict(zip(grid.keys(), combo))
        rows = [run_one(s, ov) for s in seeds]
        avg = {k: np.mean([r[k] for r in rows]) for k in rows[0] if isinstance(rows[0][k], (int, float))}
        avg["score"] = score(avg)
        avg.update(ov)
        results.append(avg)
        in_band = (
            (TARGETS["gini"][0]      <= avg["gini"]     <= TARGETS["gini"][1])
            and (TARGETS["top1"][0]     <= avg["top1"]     <= TARGETS["top1"][1])
            and (TARGETS["top10"][0]    <= avg["top10"]    <= TARGETS["top10"][1])
            and (TARGETS["bottom50"][0] <= avg["bottom50"] <= TARGETS["bottom50"][1])
        )
        flag = "OK " if in_band else "   "
        print(
            f"  tax {ov['bequest_tax_rate']:.2f}  ass {ov['assortative_exponent']:.1f}  "
            f"sk {ov['intergenerational_skill_corr']:.1f}  | "
            f"gini {avg['gini']:.3f}  top1 {avg['top1']:.3f}  "
            f"top10 {avg['top10']:.3f}  bot50 {avg['bottom50']:.3f}  "
            f"gap {avg['tenure_gap']:+.3f}  score {avg['score']:.2f}  {flag}"
        )

    print()
    print("BEST BY SCORE")
    results.sort(key=lambda r: r["score"])
    for r in results[:5]:
        print(
            f"  tax {r['bequest_tax_rate']:.2f}  ass {r['assortative_exponent']:.1f}  "
            f"sk {r['intergenerational_skill_corr']:.1f}  | "
            f"gini {r['gini']:.3f}  top1 {r['top1']:.3f}  "
            f"top10 {r['top10']:.3f}  bot50 {r['bottom50']:.3f}  score {r['score']:.2f}"
        )


if __name__ == "__main__":
    main()
