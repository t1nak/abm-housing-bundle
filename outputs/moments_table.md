# Identification: calibration, validation, and diagnostic moments

This table is the load-bearing artifact for the paper's identification
section. It separates moments used in SMM identification (calibration)
from moments scored after estimation as external tests (validation), and
reports a small set of diagnostic quantities used for paper-section
context but not for identification.

The separation is required for a structural-empirical claim: SMM hits
the calibration moments by construction; the validation moments are
external tests of the identified model. Honesty about validation
misses is what makes the contribution structural-empirical rather than
calibrated-story.

## Calibration moments (12; SMM objective; summed weight 1,0)

| # | Moment | Target | Weight | Block | Source |
|---|---|---|---|---|---|
| 1 | wealth_gini | 0,81 | 0,08 | distributional | Bundesbank PHF wave 4 (2017), Vermoegen privater Haushalte |
| 2 | top_1_wealth_share | 0,25 | 0,08 | distributional | Bundesbank PHF wave 4 (2017), top-1 percent wealth share |
| 3 | top_10_wealth_share | 0,60 | 0,07 | distributional | Bundesbank PHF wave 4 (2017), top-10 percent wealth share |
| 4 | bottom_50_wealth_share | 0,03 | 0,06 | distributional | Bundesbank PHF wave 4 (2017), bottom-50 percent wealth share |
| 5 | homeownership_rate | 0,50 | 0,06 | distributional | Mikrozensus 2022, Wohnungseigentumsquote |
| 6 | aggregate_price_growth_15y | 0,80 | 0,10 | housing_dynamics | vdpResearch Haeuserpreisindex 2010 to 2024, aggregate cumulative |
| 7 | cross_regional_price_growth_sd | 0,50 | 0,08 | housing_dynamics | vdpResearch Haeuserpreisindex 2010 to 2024, by Bundesland |
| 8 | price_growth_supply_elasticity_correlation | -0,70 | 0,07 | housing_dynamics | Hilber and Vermeulen (2016) style empirical anchor for Germany |
| 9 | aggregate_extreme_share_final | 0,208 | 0,18 | political_economy | Bundeswahlleiter, 23 February 2025 federal election, AfD second votes |
| 10 | extreme_share_year_5 | 0,10 | 0,06 | political_economy | Bundeswahlleiter, 2017 Bundestagswahl AfD share |
| 11 | extreme_share_year_10 | 0,15 | 0,06 | political_economy | Bundeswahlleiter, 2021 Bundestagswahl AfD share |
| 12 | cross_regional_extreme_share_dispersion | 0,08 | 0,10 | political_economy | Bundeswahlleiter Land-level standard deviation, 23 February 2025 |

### Block weight rationale

- **Distributional block, summed weight 0,35.** Five Bundesbank PHF
  moments from a single internally consistent wave. These pin the wealth
  Lorenz curve and ownership margin; without them the model has no
  anchor for the asset-exclusion channel of the bundled mechanism.
- **Housing dynamics block, summed weight 0,25.** Three moments. The
  aggregate price growth pins the level of housing pressure over the
  calibration horizon. The cross-regional standard deviation pins the
  dispersion that drives the right-exit geography. The Hilber-Vermeulen-
  style correlation pins the supply-elasticity channel without which
  supply restriction has no identified effect.
- **Political economy block, summed weight 0,40.** Four moments. The
  aggregate right-exit share (0,18, the heaviest single moment) is the
  binding identification target: it sets the level of the model's
  central political quantity at the empirical AfD 2025 vote share. Two
  cohort anchors (2017 and 2021 Bundestagswahl shares) pin the
  trajectory's shape. The cross-regional dispersion (0,10) prevents
  the model from matching the aggregate by uniform inflation rather
  than by recovering the geographic structure of mainstream exit.

The 0,18 weight on `aggregate_extreme_share_final` reflects that this
is the most empirically anchored moment in the table and the moment
the paper most directly claims to explain. If the SMM optimum produces
an aggregate extreme-share materially below 0,208 (more than 1
percentage point), the calibration has failed even if the J-statistic
does not reject. This is enforced as an acceptance criterion for SMM.

**Footnote on the 0,208 target.** The 0,208 number is the empirical
AfD second-vote total from the 23 February 2025 federal election, which
is a multi-driver outcome: housing-mediated dissatisfaction is one
input, alongside cultural backlash, declining institutional trust,
trade exposure, immigration salience, post-crisis economic insecurity,
social-media ecosystem effects, and East-West historical persistence
(see `paper/scope_and_limitations.md` for citations). The model is
calibrated to this total under the convention that the housing channel
is the operative mechanism in the model. The resulting parameter
estimates (in particular BETA_0) capture a country-specific residual
component of the total that is not separately identified from the
housing channel within the single-channel model. This is a known
limitation of single-channel structural identification against multi-
channel empirical totals and is documented in `outputs/smm_results.md`
under "Scope of identification". The UK validation makes this absorption
visible: when BETA_0 is held at its German-identified value, the model
produces 17,6 percent UK aggregate Leave (upper end of the Adler-Ansell
(2020) 5 to 10 percentage point housing-channel decomposition), not
the 51,8 percent empirical total.

## Validation moments (4; scored post-estimation; not in SMM objective)

| # | Moment | Target | Tolerance | Source |
|---|---|---|---|---|
| 1 | within_region_renter_owner_vote_gap | +0,15 | 0,05 | SOEP-IS / German Internet Panel tenure-by-political-affinity, 2017 to 2025 averages |
| 2 | cross_regional_extreme_share_price_growth_correlation | -0,60 | 0,20 | Bundeswahlleiter Land-level and vdpResearch Land-level price growth, joint 2010 to 2025 |
| 3 | bottom_quartile_wage_growth_2010_2025 | +0,04 | 0,05 | Destatis Verdienste 2010 to 2025, real wage growth bottom quartile |
| 4 | uk_brexit_region_price_growth_correlation | -0,20 | 0,15 | UK 2016 EU referendum Leave share and ONS regional house prices, scored in Prompt 7 |

### Why these moments belong in validation, not calibration

  - **Within-region renter-owner gap** is the model's signature
    distributional moment. The renter coefficient (BETA_R) is in the
    SMM free-parameter space. Including this moment in calibration would
    let SMM tune BETA_R to match the gap by construction; that is
    circular. Holding it out tests whether the rest of the calibration
    is consistent with the observed tenure cleavage.
  - **Cross-regional extreme-share / price-growth correlation** uses the
    SAME regional price growth that calibration moment 7
    (`cross_regional_price_growth_sd`) calibrates against, but the
    correlation with the extreme-share is an emergent property of the
    model's regional dissatisfaction structure. Including it in
    calibration would conflate level with co-movement.
  - **Bottom-quartile wage growth** uses external Destatis data not
    used in calibration. It tests whether the model's wage-growth
    structure (skill rank linearisation of `g_wage_low` to `g_wage_high`)
    produces a defensible bottom-quartile trajectory. The model's
    regional-mean proxy (lowest-productivity quartile of regions) is
    an approximation; the validation tolerance is set wide accordingly.
  - **UK Brexit correlation** is scored in Prompt 7 against UK-specific
    data; the value enters this table for completeness only.

## Diagnostic moments (3; reported in paper context; not used in identification)

| # | Moment | Value | Source |
|---|---|---|---|
| 1 | p3_incomplete_material_repair_effect | -0,113 | Model output: Scenario E versus Scenario C at T=25, 10-seed mean (`outputs/material_security_results.md`) |
| 2 | hank_aggregate_match_diagnostic | 0,208 | HANK benchmark calibrated match (`src/abmhp/hank_benchmark.py`) |
| 3 | within_region_dissatisfaction_channel_decomposition | -0,35 | Model output: housing-cost channel reduction in Scenario C versus baseline |

Diagnostic moments are model outputs and methodology cross-checks. They
appear in the paper's results section but are not part of the
identification or external validation. The P3 quantity is the headline
counterfactual finding; the HANK match is the methodology benchmark; the
channel decomposition is the bundled-mechanism evidence inside the model.

## Identification budget

| Quantity | Value |
|---|---|
| Calibration moments | 12 |
| Free SMM parameters | 8 |
| Degrees of freedom for overidentification | 4 |
| Acceptance threshold for J-statistic | p > 0,05 (chi-squared, 4 d.f.) |

The eight SMM free parameters are: BETA_D, BETA_R, RHO_ASP, ALPHA_LOCAL,
PRICE_SLOPE, BETA_0, assortative_exponent, skill_intergenerational_correlation.
Bounds are specified in `src/abmhp/estimation/smm.py`. Mortality parameters
(Gompertz A, B) and the bequest tax rate are fixed from prior calibration
(Destatis 2023 life-table) and not in the SMM free-parameter space.

## Current model fit at default configuration (5-seed mean, no SMM)

This is the pre-SMM baseline; the SMM stage moves these toward the targets.

| Moment | Target | Model | Error |
|---|---|---|---|
| wealth_gini | 0,810 | 0,788 | -0,022 |
| top_1_wealth_share | 0,250 | 0,230 | -0,020 |
| top_10_wealth_share | 0,600 | 0,648 | +0,048 |
| bottom_50_wealth_share | 0,030 | 0,022 | -0,008 |
| homeownership_rate | 0,500 | 0,471 | -0,029 |
| aggregate_price_growth_15y | 0,800 | 0,789 | -0,011 |
| cross_regional_price_growth_sd | 0,500 | 0,437 | -0,063 |
| price_growth_supply_elasticity_correlation | -0,700 | -0,827 | -0,127 |
| aggregate_extreme_share_final | 0,208 | 0,143 | -0,065 |
| extreme_share_year_5 | 0,100 | 0,059 | -0,041 |
| extreme_share_year_10 | 0,150 | 0,104 | -0,046 |
| cross_regional_extreme_share_dispersion | 0,080 | 0,020 | -0,060 |

The model is too gentle on the political-economy block at default
parameterisation. The aggregate extreme-share is 6,5 percentage points
below target, and the cross-regional dispersion is materially
compressed. SMM should pull BETA_0 up (closer to -3,0) and likely
ALPHA_LOCAL up to recover regional variance.

## Source files

  - `src/abmhp/estimation/moments.py`
  - `src/abmhp/estimation/__init__.py`
