"""Anchor + regional validation of the renter--owner gap, from ONE SOEP-Core extract.

Recomputes both load-bearing empirical quantities in the same data universe (SOEP):
  (1) the aggregate renter--owner gap anchor   (paper Section 4.3 / Appendix A; ~+0.15)
  (2) the regional gradient                    (paper Section 5.4; model +0.187/+0.148/+0.108,
                                                superstar-minus-declining spread ~0.08)

Outcome is SOEP PARTY LEANING ("leans AfD", Parteineigung), not a Sunday-question
vote intention -- the prose in Section 4.3 / Appendix A matches this.

FOUR EXECUTION CAVEATS (baked in):
  - SOEP-Core, not SOEP-IS (IS is too thin once cut to renter x AfD x price-tercile).
  - Pool the 2021-2024 waves with wave fixed effects (C(syear)) -- AfD leaners are a minority.
    (SOEP-Core v41, data 1984-2024, is the current release; there is no 2025 wave.)
  - Outcome = party leaning (leans AfD), weighted.
  - Use the SOEP cross-sectional person weight.

Region groups are price-growth TERCILES of the sixteen federal states (bula), the
empirical analogue of the model's sixteen regions -- no Kreis geography needed.

EXTRACT (SOEPlong; one row per person-year): pull from pl, hgen, pgen, ppfad + phrf.
Confirm exact codes on paneldata.org for your SOEP version before relying on them.
  pl    : plh0011 (leans a party: 1=yes), plh0012 (which party -> AfD code), syear, pid
  hgen  : hgowner (owner vs tenant), hghinc (hh net income)      [link by hid+syear]
  pgen  : pgisced11 or pgbilzeit (education), pgpartz/age inputs  [link by pid+syear]
  ppfad : gebjahr (birth year -> age), bula (federal state), sex
  phrf  : the cross-sectional person weight for the pooled waves
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

# ============================ CONFIG -- FILL THESE ============================
EXTRACT_CSV = "soep_extract.csv"     # merged person-year extract
COL = dict(
    bula="bula",            # federal state (1..16)
    syear="syear",          # survey year -> wave fixed effects
    leans="plh0011",        # leans a party at all (1=yes)
    party="plh0012",        # which party (AfD code below)
    tenure="hgowner",       # SOEP generated tenure
    weight="phrf",          # SOEP cross-sectional person weight
    controls=["age", "hghinc", "edu_years", "east"],   # for the adjusted gradient
)
AFD_CODE = 27               # plh0012 value for AfD -- VERIFY per SOEP version
OWNER_CODES = (1,)          # hgowner values meaning owner-occupier -- VERIFY

def is_renter(v):  return v not in OWNER_CODES and v > 0     # positive, non-owner
def leans_afd(row):                                          # 1 if leans AfD, else 0
    if row[COL["leans"]] != 1:        # does not lean any party -> not AfD (kept in base)
        return 0
    return int(row[COL["party"]] == AFD_CODE)
def valid(row):                                              # drop item non-response
    return row[COL["leans"]] in (1, 2) and (row[COL["leans"]] != 1 or row[COL["party"]] > 0)

# Cumulative regional house-price growth by bula (vdpResearch/Destatis -- the model series).
REGION_PRICE_GROWTH = {
    # 1:"Schleswig-Holstein", 2:"Hamburg", ... 16:"Thueringen"  -> fill cumulative growth
}
N_BUCKETS = 3            # 0=declining, 1=average, 2=superstar (terciles of bula by price growth)
MODEL_SPREAD = 0.08
# ============================================================================


def wmean(x, w):
    x = np.asarray(x, float); w = np.ones_like(x) if w is None else np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    return float(np.sum(x[m]*w[m]) / np.sum(w[m])) if m.any() else np.nan


def load():
    df = pd.read_csv(EXTRACT_CSV)
    df = df[df.apply(valid, axis=1)].copy()
    df["afd"] = df.apply(leans_afd, axis=1).astype(float)
    df["renter"] = df[COL["tenure"]].map(is_renter)
    df = df.dropna(subset=[COL["bula"], "renter"])
    if not REGION_PRICE_GROWTH:
        sys.exit("Fill REGION_PRICE_GROWTH (cumulative price growth per bula 1..16).")
    df["pg"] = df[COL["bula"]].map(REGION_PRICE_GROWTH)
    df = df.dropna(subset=["pg"])
    # tercile of states by price growth, weighting states equally (not respondents)
    state_pg = df.groupby(COL["bula"])["pg"].first()
    cut = pd.qcut(state_pg.rank(method="first"), N_BUCKETS, labels=False)
    df["bucket"] = df[COL["bula"]].map(cut)
    return df


def raw_gap_by_bucket(df):
    w = COL["weight"]; rows = []
    for b in sorted(df["bucket"].dropna().unique()):
        sub = df[df["bucket"] == b]; wt = sub[w] if w in sub else None
        g_r = wmean(sub.loc[sub.renter, "afd"], None if wt is None else wt[sub.renter])
        g_o = wmean(sub.loc[~sub.renter, "afd"], None if wt is None else wt[~sub.renter])
        rows.append((int(b), g_r-g_o, g_r, g_o, len(sub)))
    return pd.DataFrame(rows, columns=["bucket", "gap", "renter_afd", "owner_afd", "n"])


def anchor(df):
    """Aggregate renter-owner gap = paper Section 4.3 anchor (~+0.15)."""
    w = COL["weight"]; wt = df[w] if w in df else None
    g_r = wmean(df.loc[df.renter, "afd"], None if wt is None else wt[df.renter])
    g_o = wmean(df.loc[~df.renter, "afd"], None if wt is None else wt[~df.renter])
    return g_r - g_o


def adjusted_gradient(df):
    """renter x price-growth interaction, controls + wave fixed effects.
    Positive => the renter premium rises with appreciation, as the model predicts;
    this is the composition-adjusted gradient that backs the 47%-as-upper-bound point."""
    import statsmodels.formula.api as smf
    d = df.copy(); d["pg_z"] = (d["pg"]-d["pg"].mean())/d["pg"].std()
    ctrl = " + ".join(c for c in COL["controls"] if c in d.columns)
    f = "afd ~ renter * pg_z + C({})".format(COL["syear"]) + (f" + {ctrl}" if ctrl else "")
    w = COL["weight"]; fw = d[w] if w in d else None
    return smf.logit(f, d, freq_weights=fw).fit(disp=0) if fw is not None else smf.logit(f, d).fit(disp=0)


def main():
    df = load()
    print(f"pooled SOEP-Core person-years: {len(df)}  waves: {sorted(df[COL['syear']].unique())}")
    print(f"\n(1) ANCHOR  renter-owner gap = {anchor(df):+.3f}   (paper target ~ +0.15)")
    raw = raw_gap_by_bucket(df)
    print("\n(2) REGIONAL  renter-owner gap by price-growth tercile:")
    print(raw.to_string(index=False))
    spread = raw.loc[raw.bucket==raw.bucket.max(),"gap"].iat[0] - raw.loc[raw.bucket==raw.bucket.min(),"gap"].iat[0]
    print(f"\nEMPIRICAL superstar-minus-declining spread = {spread:+.3f}")
    print(f"MODEL predicted spread                      = {MODEL_SPREAD:+.3f}")
    try:
        m = adjusted_gradient(df)
        b = m.params.get("renter:pg_z", float("nan")); p = m.pvalues.get("renter:pg_z", float("nan"))
        print(f"\nADJUSTED renter x price-growth (controls + wave FE) = {b:+.3f} (p={p:.3f})")
    except Exception as e:
        print(f"\n[adjusted gradient skipped: {e}]")


if __name__ == "__main__":
    main()
