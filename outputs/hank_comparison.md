# HANK methodological benchmark vs the ABM

This note contains the methodology-section table and accompanying prose
for the HANK comparison. The ABM and a stripped two-state HANK are
compared on six moments: three aggregate moments that both models match
by calibration, and three structural moments that the ABM produces and
HANK mechanically cannot.

## Comparison table

| Moment / Result | German empirical | ABM | HANK |
|---|---|---|---|
| Aggregate wealth Gini | 0,81 | 0,791 | 0,810 |
| Homeownership rate | 0,50 | 0,470 | 0,500 |
| Aggregate extreme-share vote | 0,208 | 0,145 | 0,208 |
| Cross-regional extreme-share dispersion | nonzero | 0,022 | 0,000 |
| Within-region renter-owner vote gap | >= +0,15 | +0,213 | +0,000 |
| Incomplete-material-repair effect (P3) | structural | -0,113 | +0,000 |

The ABM numbers are taken from the seed-73 baseline simulation
(`scripts/compare_abm_hank.py`); the seed-73 baseline sits comfortably
inside the five-seed bands documented in `tests/test_baseline.py`
(Gini in [0,78, 0,83]; aggregate extreme-share in [0,10, 0,35];
renter-owner gap above +0,20). The HANK numbers come from the
closed-form steady-state calibration in `src/abmhp/hank_benchmark.py`.
The ABM P3 effect of -0,113 is the Scenario E versus Scenario C
vote-share difference at T = 25, documented in
`outputs/material_security_results.md`.

## Scope of the methodological argument

The structural-zero results below describe HANK's capacity to represent
the bundled-mechanism question, not HANK's broader usefulness. HANK is
an appropriate tool for many macro questions where the relevant
heterogeneity is at the aggregate or income-bin level and the policy
question is about a single aggregate-relief channel; the existing
literature uses it productively for monetary transmission, fiscal
multipliers, MPC heterogeneity, and similar questions. The argument
here is narrow: HANK is structurally inappropriate for the bundled-
mechanism question because it averages individual-level heterogeneity
into aggregate response and has single-channel dissatisfaction by
construction. Both properties are reasonable design choices in HANK's
home applications; they are disqualifying for the question this paper
asks.

## Methodology-section paragraph (publication prose)

We benchmark the ABM against a stripped two-state HANK-style model with
four cells (high productivity owner, high productivity renter, low
productivity owner, low productivity renter), lognormal within-cell
wealth dispersion, and a sigmoid vote function of a single aggregate
dissatisfaction scalar. The benchmark is calibrated to match three
German aggregate moments: wealth Gini at 0,81, homeownership rate at
0,50, and the aggregate right-exit vote share at 0,208 (AfD second
votes in the 23 February 2025 federal election). HANK matches these
aggregate moments by construction. It mechanically returns zero on the
three structural moments the paper relies on. Cross-regional dispersion
in the extreme-share vote is zero because HANK has no regional
structure. The within-region renter-owner vote gap is zero because HANK
has aggregate-level voting; vote share is a function of aggregate
dissatisfaction, not of individual tenure or productivity state. The
incomplete-material-repair effect (P3) is zero because HANK reads
housing-only relief and redistribution as the same aggregate-relief
channel; any two policy scenarios delivering the same total relief
through different mixes produce identical vote responses. The ABM is the
appropriate method when within-region tenure status, regional housing-
market structure, and the bundled-mechanism dissatisfaction
decomposition are quantities of interest. All three are required by the
empirical German right-exit geography, the renter-owner vote gap, and
the P2 / P3 counterfactual results.

## Side metric: the marginal effect of redistribution in HANK

The HANK paired-scenarios analogous to ABM Scenario C and ABM Scenario E
do differ in HANK by a small margin:

  - HANK-housing-only (rent cap 0,6 with leakage 0,4, no redistribution):
    vote share 0,170.
  - HANK-multichannel (same rent cap, plus redistribution intensity
    0,027): vote share 0,154.
  - Vote difference: -0,016.

This is the marginal effect of adding redistribution to the same single
channel; it is NOT the channel-decomposition effect (P3) that the ABM
identifies. The methodologically important test, reported in the
comparison table, is whether HANK can produce a P3-style effect when two
scenarios deliver the same total aggregate relief through different
mixes. That test gives zero in HANK by construction (verified in
`tests/test_hank_benchmark.py::test_incomplete_material_repair_effect_is_structurally_zero`
and the direct equality test
`test_paired_scenarios_with_equal_total_relief_give_equal_votes`).

## What HANK matches by construction

The three aggregate calibrations are independent levers:

  - The within-cell wealth dispersion parameter `within_cell_sigma`
    controls the aggregate wealth Gini. Calibrated value: 1,346.
  - The cell ownership probabilities `p_own_high`, `p_own_low` control
    the homeownership rate. Calibrated values: 0,780 and 0,220.
  - The aggregate baseline dissatisfaction controls the extreme-share
    vote via the sigmoid. Calibrated value: 0,197.

The calibration routine (`abmhp.hank_benchmark.calibrate`) hits all three
targets to within 1e-3 in a single sequential pass.

## What HANK cannot produce: the three structural-zero results

  1. Cross-regional dispersion. HANK has no regional state variable. The
     extreme-share vote is a function of aggregate dissatisfaction only.
     Cross-regional dispersion is mechanically zero. Any reduction of the
     ABM's cross-regional structure to HANK necessarily collapses this
     moment to a constant.

  2. Within-region renter-owner vote gap. HANK has aggregate-level
     voting. Two agents in the same cell at the same wealth percentile
     differ only in productivity state and tenure; their vote share is
     identical because vote share is computed at the aggregate. The
     within-region gap is mechanically zero. The ABM produces a +0,213
     gap because voting is logit-individual with a tenure-conditioned
     coefficient (`beta_renter`) and a network coefficient
     (`beta_network`) on the local extreme-share rate.

  3. Incomplete-material-repair effect (P3). HANK reads housing-only
     relief and redistribution as the same aggregate-relief channel. The
     vote function is single-channel; the policy lever decomposition is
     structurally invisible. The ABM produces a -0,113 difference between
     Scenarios E and C because the bundled-mechanism dissatisfaction
     reads rent (consumption stress), capital gain access (asset
     exclusion), and disposable resources (the material-security
     channel) as distinct contributors to individual dissatisfaction.
     Removing this individual-level dual-channel structure removes the
     P3 effect by construction.

## Conclusion of the methodology section

HANK is appropriate when the moments of interest are aggregate and the
policy question maps to a single aggregate-relief channel. It is not
appropriate for the paper's three structural findings: the German
right-exit geography (Bundeswahlleiter Land-level evidence), the
within-region renter-owner vote gap (Mikrozensus / survey evidence), and
the P3 incomplete-material-repair result (the Scenario C versus
Scenario E comparison in the present model). The ABM is the right tool
because the relevant heterogeneity (regional structure, individual
tenure status, dual-channel dissatisfaction) is in the model by
construction. The argument is narrow and tool-specific: a different
question, even one closely related (aggregate fiscal multipliers, MPC
heterogeneity by income, the monetary transmission channel), would
make HANK the appropriate choice.

## Source files

  - `src/abmhp/hank_benchmark.py`
  - `scripts/compare_abm_hank.py`
  - `tests/test_hank_benchmark.py`
  - `outputs/abm_vs_hank_table.csv`
  - `outputs/abm_vs_hank_summary.json`
