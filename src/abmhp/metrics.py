"""Distributional and regional aggregation metrics."""
from __future__ import annotations

import numpy as np


def gini(x: np.ndarray) -> float:
    """Gini coefficient. Negative values are clipped to zero (debt is treated
    as zero net wealth for the inequality measure; the negative tail is
    reported separately as debt share)."""
    x = np.clip(x, 0.0, None)
    s = np.sort(x)
    n = len(s)
    total = s.sum()
    if total == 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * s)) / (n * total) - (n + 1) / n)


def top_share(wealth: np.ndarray, fraction: float) -> float:
    """Share of total positive wealth held by the top fraction."""
    w = np.clip(wealth, 0.0, None)
    s = np.sort(w)
    n = len(s)
    k = max(1, int(round(n * fraction)))
    total = s.sum()
    if total == 0:
        return 0.0
    return float(s[-k:].sum() / total)


def bottom_share(wealth: np.ndarray, fraction: float) -> float:
    w = np.clip(wealth, 0.0, None)
    s = np.sort(w)
    n = len(s)
    k = max(1, int(round(n * fraction)))
    total = s.sum()
    if total == 0:
        return 0.0
    return float(s[:k].sum() / total)


def regional_means(values: np.ndarray, region: np.ndarray, n_regions: int) -> np.ndarray:
    out = np.zeros(n_regions)
    for r in range(n_regions):
        mask = region == r
        if mask.any():
            out[r] = values[mask].mean()
    return out


def regional_shares(boolean: np.ndarray, region: np.ndarray, n_regions: int) -> np.ndarray:
    return regional_means(boolean.astype(float), region, n_regions)
