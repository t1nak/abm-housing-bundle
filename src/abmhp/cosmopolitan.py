"""Cosmopolitan-proxy regional intercept shift, promoted to first-class.

Computes the per-region beta_0 shift used by the voting block:

    shift_r = gamma_cosmopolitan * (grad_share_r - mean_grad_share)

The shift vector is materialised once at simulation initialisation
(see `simulation.simulate`) and stored on the VotingConfig as
`cosmopolitan_shift_by_region`. When gamma_cosmopolitan = 0.0 (the
default) or grad_share_data_path is None, no shift is applied and the
model reproduces the scalar-beta_0 specification exactly.

Country routing is by `n_regions`:
    16 -> Germany (DE_LAND_BY_MODEL_REGION ordering)
    12 -> United Kingdom (UK_REGION_NAMES ordering, from validation.uk)
Other values raise a ValueError so unintended applications fail loud.

The Germany Land-to-model-region ordering was first hand-constructed in
scripts/cosmopolitan_pilot.py for the cosmopolitan pilot; it is reproduced
here so the mechanism is a first-class part of the package rather than a
script-side hack. The ordering and rationale are documented inline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Germany: hand-constructed Land-to-model-region ordering
# ---------------------------------------------------------------------------
#
# The 16-region model has 4 super + 8 avg + 4 decl by productivity. Two
# principles guided the assignment:
#   (1) Berlin (highest graduate share) sits in "avg", not at the top of
#       "super", because Berlin's productivity is closer to the national
#       mean than to Hamburg or Bavaria. This breaks the productivity-
#       graduate-share collinearity that would otherwise make the income-
#       controlled partial correlation mechanically zero.
#   (2) Within each productivity class, Laender are roughly ordered by
#       graduate share so cross-region grad-share variance is maximised
#       given the productivity constraints.
DE_LAND_BY_MODEL_REGION: tuple[str, ...] = (
    "Hamburg",
    "Hessen",
    "Bayern",
    "Baden-Wuerttemberg",
    "Berlin",
    "Bremen",
    "Nordrhein-Westfalen",
    "Niedersachsen",
    "Schleswig-Holstein",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Brandenburg",
    "Thueringen",
    "Sachsen-Anhalt",
    "Mecklenburg-Vorpommern",
)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    # Resolve relative paths against the repository root: src/abmhp/cosmopolitan.py -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


def load_grad_share_vector(grad_share_data_path: str, n_regions: int) -> np.ndarray:
    """Return the per-model-region graduate-share vector for the country
    implied by `n_regions` (16 = DE, 12 = UK).

    Reads the JSON at `grad_share_data_path` (relative to repo root, or
    absolute) and maps its `values` dict onto the model-region ordering."""
    data = json.loads(_resolve_path(grad_share_data_path).read_text())
    values_by_name: dict[str, float] = data["values"]

    if n_regions == 16:
        names = DE_LAND_BY_MODEL_REGION
    elif n_regions == 12:
        # Lazy import to avoid a circular import via validation.uk -> config.
        from .validation.uk import UK_REGION_NAMES
        names = UK_REGION_NAMES
    else:
        raise ValueError(
            f"load_grad_share_vector: n_regions={n_regions} not supported "
            "(expected 16 for DE or 12 for UK)"
        )

    missing = [n for n in names if n not in values_by_name]
    if missing:
        raise KeyError(
            f"grad-share data at {grad_share_data_path} is missing entries "
            f"for {missing}"
        )

    return np.array([values_by_name[n] for n in names], dtype=float)


def compute_cosmopolitan_shift(
    gamma: float,
    grad_share_data_path: str,
    n_regions: int,
) -> tuple[float, ...]:
    """Compute the per-region beta_0 shift vector. Returned as a tuple so
    it can live inside a frozen-dataclass VotingConfig."""
    grad = load_grad_share_vector(grad_share_data_path, n_regions)
    mean_grad = float(grad.mean())
    shift = gamma * (grad - mean_grad)
    return tuple(float(s) for s in shift)
