"""SMM identification and moment evaluation for the bundled-mechanism ABM.

Two layers:
  - `moments`: the calibration / validation / diagnostic moment table,
    each moment carrying its empirical target, source citation, and a
    callable that evaluates the moment on a multi-seed simulation result.
  - `smm`: the two-stage SMM estimator targeting the calibration moments,
    with overidentification testing, asymptotic standard errors, and
    sensitivity diagnostics.

See `outputs/moments_table.md` for the moment definitions and
`outputs/smm_results.md` for the identification report.
"""
from .moments import (
    CALIBRATION_MOMENTS,
    DIAGNOSTIC_MOMENTS,
    Moment,
    VALIDATION_MOMENTS,
    evaluate_moment,
    evaluate_moments,
    simulate_seeds,
)

__all__ = [
    "Moment",
    "CALIBRATION_MOMENTS",
    "VALIDATION_MOMENTS",
    "DIAGNOSTIC_MOMENTS",
    "evaluate_moment",
    "evaluate_moments",
    "simulate_seeds",
]
