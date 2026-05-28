"""Out-of-sample validation of the German-calibrated bundled-mechanism ABM.

The German calibration is in `src/abmhp/estimation/`. This package
contains country-specific structural primitives (regional shares,
productivity gradients, supply elasticities, country-specific bequest
tax rates) that swap the regional and demographic blocks of the model
WITHOUT touching the eight SMM-identified behavioural parameters
(BETA_D, BETA_R, RHO_ASP, ALPHA_LOCAL, PRICE_SLOPE, BETA_0,
assortative_exponent, intergenerational_skill_corr).

The behavioural-parameter invariance is enforced by tests in
`tests/test_uk_validation.py`.
"""
from .uk import (
    GERMAN_SMM_PARAMS,
    UK_REGION_NAMES,
    UKRegionalConfig,
    make_uk_config,
    run_uk_validation,
    BREXIT_PERIOD,
)

__all__ = [
    "GERMAN_SMM_PARAMS",
    "UK_REGION_NAMES",
    "UKRegionalConfig",
    "make_uk_config",
    "run_uk_validation",
    "BREXIT_PERIOD",
]
