"""SOEP renter-owner anchor pipeline (plug-and-play once data access exists).

Computes, from a SOEP-Core person-year extract (2021-2024 waves):

  A1  the RAW within-state renter-owner gap in the share leaning AfD
      (weighted, within federal state, averaged across states) -- the
      calibration anchor of manuscript Section 4.4;
  A2  the COMPOSITION-ADJUSTED gap: weighted linear-probability regression
      of leans-AfD on the renter dummy with age, log income, education,
      East dummy (or state FE), and wave FE; household-clustered SEs;
  A3  household-block BOOTSTRAP confidence intervals for A1, A2 and the
      regional gradient spread (superstar-minus-declining terciles);
  A4  the regional price-growth-tercile gradient (the pre-specified
      empirical test of manuscript Section 5.2);
  A5  per-state renter/owner sample sizes and a reporting block (edition,
      DOI, extraction date) for manuscript Appendix A;
  A6  LaTeX table rows for direct inclusion.

INPUT: a CSV at --extract with one row per person-year and columns
  pid, hid, syear, bula (1-16), leans_afd (0/1), renter (0/1), weight,
  age, hh_income, edu_years, east (0/1)
built from SOEP-Core v41 files pl (plh0011/plh0012), hgen (hgowner,
hghinc), ppfad (bula, gebjahr), phrf -- see scripts/
regional_gap_validation.py for variable-level guidance and paneldata.org
for the v41 codes. Record edition and DOI via --edition/--doi.

REGIONAL GROUPS: price-growth terciles of the 16 states, supplied as
--terciles CSV (bula, tercile in {0,1,2}; 2 = superstar). Keep this file
model-free (observed growth only).

TEST MODE: --selftest generates a synthetic extract with a known gap and
verifies the pipeline end-to-end (no SOEP data involved; synthetic output
is clearly labelled and never written to outputs/).

Usage:
  uv run scripts/soep_anchor_pipeline.py --extract soep_extract.csv \
      --terciles state_terciles.csv --edition "SOEP-Core v41" \
      --doi 10.5684/soep.core.v41eu --n-boot 1000
  uv run scripts/soep_anchor_pipeline.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["pid", "hid", "syear", "bula", "leans_afd", "renter", "weight",
            "age", "hh_income", "edu_years", "east"]


def raw_gap(df: pd.DataFrame) -> float:
    """Weighted renter-minus-owner AfD-leaning share, within state, averaged
    across states with equal state weights (manuscript definition)."""
    gaps = []
    for _, g in df.groupby("bula"):
        r = g[g.renter == 1]
        o = g[g.renter == 0]
        if len(r) == 0 or len(o) == 0:
            continue
        pr = np.average(r.leans_afd, weights=r.weight)
        po = np.average(o.leans_afd, weights=o.weight)
        gaps.append(pr - po)
    return float(np.mean(gaps))


def adjusted_gap(df: pd.DataFrame):
    """Weighted LPM of leans_afd on renter + controls + state FE + wave FE.
    Returns (coefficient on renter, household-clustered SE)."""
    import statsmodels.formula.api as smf
    d = df.copy()
    d["log_inc"] = np.log(np.maximum(d.hh_income, 1.0))
    model = smf.wls(
        "leans_afd ~ renter + age + log_inc + edu_years + C(bula) + C(syear)",
        data=d, weights=d.weight)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d.hid})
    return float(res.params["renter"]), float(res.bse["renter"])


def tercile_gradient(df: pd.DataFrame, terciles: pd.Series):
    """Raw gap by price-growth tercile; returns dict and the 2-minus-0 spread."""
    d = df.copy()
    d["tercile"] = d.bula.map(terciles)
    out = {}
    for t, g in d.groupby("tercile"):
        out[int(t)] = raw_gap(g)
    spread = out.get(2, np.nan) - out.get(0, np.nan)
    return out, float(spread)


def household_bootstrap(df, terciles, n_boot, seed=73):
    """Resample households (all their person-years) with replacement."""
    rng = np.random.default_rng(seed)
    hids = df.hid.unique()
    stats = {"raw": [], "adj": [], "spread": []}
    groups = dict(tuple(df.groupby("hid")))
    for b in range(n_boot):
        draw = rng.choice(hids, size=len(hids), replace=True)
        sample = pd.concat([groups[h] for h in draw], ignore_index=True)
        stats["raw"].append(raw_gap(sample))
        try:
            stats["adj"].append(adjusted_gap(sample)[0])
        except Exception:
            stats["adj"].append(np.nan)
        _, sp = tercile_gradient(sample, terciles)
        stats["spread"].append(sp)
    def ci(v):
        v = np.asarray([x for x in v if np.isfinite(x)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: ci(v) for k, v in stats.items()}


def per_state_ns(df: pd.DataFrame) -> dict:
    tab = df.groupby(["bula", "renter"]).size().unstack(fill_value=0)
    return {int(b): {"renters": int(row.get(1, 0)), "owners": int(row.get(0, 0))}
            for b, row in tab.iterrows()}


def latex_rows(res: dict) -> str:
    r = res["raw_gap"]; a = res["adjusted_gap"]; s = res["spread"]
    return "\n".join([
        f"Raw renter--owner gap & ${r['estimate']:+.3f}$ & "
        f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] \\\\",
        f"Composition-adjusted gap & ${a['estimate']:+.3f}$ & "
        f"[{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] \\\\",
        f"Superstar$-$declining spread & ${s['estimate']:+.3f}$ & "
        f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] \\\\",
    ])


def run(extract, terciles_csv, edition, doi, n_boot, out_path):
    df = pd.read_csv(extract)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"extract is missing columns: {missing}")
    df = df.dropna(subset=REQUIRED)
    terciles = pd.read_csv(terciles_csv).set_index("bula")["tercile"]

    raw = raw_gap(df)
    adj, adj_se = adjusted_gap(df)
    by_terc, spread = tercile_gradient(df, terciles)
    cis = household_bootstrap(df, terciles, n_boot)

    res = dict(
        reporting=dict(edition=edition, doi=doi,
                       extraction_date=str(date.today()),
                       waves=sorted(int(x) for x in df.syear.unique()),
                       n_person_years=int(len(df)),
                       n_households=int(df.hid.nunique()),
                       per_state_sample_sizes=per_state_ns(df)),
        raw_gap=dict(estimate=round(raw, 4), ci=cis["raw"]),
        adjusted_gap=dict(estimate=round(adj, 4), se_clustered=round(adj_se, 4),
                          ci=cis["adj"]),
        gap_by_tercile={str(k): round(v, 4) for k, v in by_terc.items()},
        spread=dict(estimate=round(spread, 4), ci=cis["spread"]),
        n_boot=n_boot,
    )
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\n--- LaTeX rows (Appendix A) ---")
    print(latex_rows(res))
    return res


def selftest():
    """Synthetic end-to-end check with known structure (NOT SOEP data)."""
    rng = np.random.default_rng(7)
    rows = []
    hid = 0
    for bula in range(1, 17):
        terc = 0 if bula <= 5 else (1 if bula <= 11 else 2)
        for _ in range(220):
            hid += 1
            renter = rng.random() < 0.5
            east = 1 if bula >= 12 else 0
            age = rng.integers(20, 80)
            inc = float(np.exp(rng.normal(10.2, 0.5)))
            edu = float(rng.normal(12, 2))
            # true structure: baseline .12, renter effect .06 + .02*tercile,
            # east effect .08 (composition!), income gradient
            p = (0.12 + (0.06 + 0.02 * terc) * renter + 0.08 * east
                 - 0.02 * (np.log(inc) - 10.2))
            for wave in (2021, 2022):
                rows.append(dict(pid=hid * 10 + wave % 10, hid=hid, syear=wave,
                                 bula=bula, leans_afd=int(rng.random() < p),
                                 renter=int(renter), weight=1.0, age=age,
                                 hh_income=inc, edu_years=edu, east=east))
    df = pd.DataFrame(rows)
    tmp = Path("/tmp/soep_selftest")
    tmp.mkdir(exist_ok=True)
    df.to_csv(tmp / "extract.csv", index=False)
    pd.DataFrame({"bula": range(1, 17),
                  "tercile": [0]*5 + [1]*6 + [2]*5}).to_csv(
        tmp / "terciles.csv", index=False)
    res = run(tmp / "extract.csv", tmp / "terciles.csv",
              "SYNTHETIC SELFTEST", "none", n_boot=60,
              out_path=tmp / "result.json")
    raw = res["raw_gap"]["estimate"]
    spread = res["spread"]["estimate"]
    assert 0.03 < raw < 0.12, f"raw gap {raw} outside plausible synthetic range"
    assert spread > 0, f"spread {spread} should be positive by construction"
    print("\nSELFTEST PASSED (synthetic data; results not written to outputs/)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract")
    ap.add_argument("--terciles")
    ap.add_argument("--edition", default="SOEP-Core v41")
    ap.add_argument("--doi", default="")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="outputs/soep_anchor.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not a.extract or not a.terciles:
            sys.exit("provide --extract and --terciles, or run --selftest")
        run(a.extract, a.terciles, a.edition, a.doi, a.n_boot, a.out)
