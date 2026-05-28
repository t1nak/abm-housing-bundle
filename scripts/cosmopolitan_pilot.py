"""Cosmopolitan-proxy pilot for Issue 3 (Track B).

Tests whether a single scalar gamma on a region-array beta_0 shifter keyed
on graduate share is enough to flip the partial-correlation sign in BOTH
Germany and the UK without breaking the within-region renter-owner cleavage.

Spec:
    beta_0_r = beta_0 + gamma * (grad_share_r - mean_grad_share_country)

Per-country gamma is calibrated to one moment each:
    Germany: empirical differential in AfD vote share between the lowest-
        graduate-share Land (Mecklenburg-Vorpommern) and the highest-
        graduate-share Land (Berlin), at the 2025 federal election.
    UK: empirical differential in Leave vote share between the lowest-
        graduate-share NUTS-1 region (North East) and the highest-graduate-
        share region (Greater London), at the 2016 referendum.

Then a shared gamma (average of the two country-specific values) is tested.

The eight SMM-identified behavioural parameters are held at their Stage 1
values throughout. This is a one-scalar extension at fixed theta_hat, no
SMM re-estimation.

Outputs:
    outputs/cosmopolitan_pilot.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config, simulate
from abmhp.estimation.smm import apply_params
from abmhp.validation.uk import (
    BREXIT_PERIOD,
    UK_REGION_NAMES,
    make_uk_config,
)


OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Land-to-model-region mapping for Germany (16 stylised productivity classes)
# ---------------------------------------------------------------------------
#
# The model has 4 super + 8 avg + 4 decl regions ordered by productivity.
# The named-Land mapping below is the pilot's hand-construction; documented
# explicitly so the dependency of the partial-correlation result on this
# mapping is auditable. Two principles:
#   (1) The Bundesland with highest graduate share (Berlin, 0.37) is placed
#       in the "avg" group, not at the top of "super", because Berlin's
#       economic productivity is closer to the national mean than to
#       Hamburg or Bavaria. This breaks the productivity-graduate-share
#       collinearity that would otherwise make the partial correlation
#       conditional on income mechanically zero.
#   (2) Within each productivity class, Laender are roughly ordered by
#       graduate share so that the cross-region grad-share variance is
#       maximised given the productivity constraints.
#
# The mapping is debatable; a referee can argue for alternatives. The pilot's
# job is to test whether ANY reasonable mapping with a single gamma can
# rescue the partial-correlation sign. If yes, we revisit the mapping for
# robustness. If no, the cosmopolitan addition is rejected.

DE_LAND_BY_MODEL_REGION: tuple[str, ...] = (
    # super (productivity 1.50, 1.32, 1.22, 1.15)
    "Hamburg",            # S1: productivity 1.40, grad 0.34
    "Hessen",             # S2: productivity 1.30, grad 0.30
    "Bayern",             # S3: productivity 1.25, grad 0.30
    "Baden-Wuerttemberg", # S4: productivity 1.20, grad 0.29
    # avg (productivity 1.00 x 8)
    "Berlin",             # A1: HIGH-GRAD outlier; productivity 1.10, grad 0.37
    "Bremen",             # A2: productivity 1.05, grad 0.29
    "Nordrhein-Westfalen",# A3: productivity 1.02, grad 0.26
    "Niedersachsen",      # A4: productivity 0.98, grad 0.26
    "Schleswig-Holstein", # A5: productivity 0.95, grad 0.26
    "Rheinland-Pfalz",    # A6: productivity 0.95, grad 0.25
    "Saarland",           # A7: productivity 0.92, grad 0.25
    "Sachsen",            # A8: productivity 0.92, grad 0.28
    # decl (productivity 0.82, 0.76, 0.72, 0.68)
    "Brandenburg",        # D1: productivity 0.85, grad 0.28
    "Thueringen",         # D2: productivity 0.82, grad 0.27
    "Sachsen-Anhalt",     # D3: productivity 0.78, grad 0.24
    "Mecklenburg-Vorpommern",  # D4: productivity 0.75, grad 0.24
)


def load_de_data() -> dict:
    return json.loads((DATA / "cosmopolitan_grad_share_de.json").read_text())


def load_uk_data() -> dict:
    return json.loads((DATA / "cosmopolitan_grad_share_uk.json").read_text())


def de_grad_vector_by_model_region() -> np.ndarray:
    de = load_de_data()
    grad_by_land = de["values"]
    return np.array([grad_by_land[land] for land in DE_LAND_BY_MODEL_REGION])


def uk_grad_vector_by_model_region() -> np.ndarray:
    uk = load_uk_data()
    grad_by_region = uk["values"]
    return np.array([grad_by_region[name] for name in UK_REGION_NAMES])


def de_afd_vector_by_model_region() -> np.ndarray:
    de = load_de_data()
    afd_by_land = de["afd_2025_share_by_land"]
    return np.array([afd_by_land[land] for land in DE_LAND_BY_MODEL_REGION])


def uk_leave_vector_by_model_region() -> np.ndarray:
    uk = load_uk_data()
    leave_by_region = uk["leave_2016_share_by_region"]
    return np.array([leave_by_region[name] for name in UK_REGION_NAMES])


# ---------------------------------------------------------------------------
# Theta_hat loader
# ---------------------------------------------------------------------------


def load_theta_hat() -> np.ndarray:
    path = OUTPUTS / "smm_optimum.json"
    return np.array(json.loads(path.read_text())["theta_hat"], dtype=float)


# ---------------------------------------------------------------------------
# Per-country simulation runner
# ---------------------------------------------------------------------------


def build_shift(gamma: float, grad_vector: np.ndarray) -> tuple[float, ...]:
    mean_grad = float(grad_vector.mean())
    shift = gamma * (grad_vector - mean_grad)
    return tuple(float(s) for s in shift)


def simulate_de_at_gamma(theta_hat: np.ndarray, gamma: float, seeds: list[int]) -> list:
    """Simulate the German calibration at theta_hat with cosmopolitan shift."""
    grad_vector = de_grad_vector_by_model_region()
    shift = build_shift(gamma, grad_vector)
    base_cfg = Config()
    cfg_theta = apply_params(base_cfg, theta_hat)
    runs = []
    for s in seeds:
        cfg = replace(
            cfg_theta,
            seed=int(s),
            voting=replace(cfg_theta.voting, cosmopolitan_shift_by_region=shift),
        )
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def simulate_uk_at_gamma(theta_hat: np.ndarray, gamma: float, seeds: list[int]) -> list:
    """Simulate the UK external stress test at the passed theta_hat with shift.

    make_uk_config builds the UK config with the eight SMM-identified
    behavioural parameters initialised from the locked GERMAN_SMM_PARAMS
    constants in validation/uk.py. We then call apply_params with the
    theta_hat argument so the UK runs use the same parameter vector the
    DE runs are using. If theta_hat matches GERMAN_SMM_PARAMS bit-for-bit
    (the current pilot case), the override is a no-op; if a future caller
    passes a re-estimated theta_hat or runs the pilot at a different
    optimum, DE and UK runs stay in sync rather than silently diverging.
    """
    grad_vector = uk_grad_vector_by_model_region()
    shift = build_shift(gamma, grad_vector)
    runs = []
    for s in seeds:
        cfg = make_uk_config(seed=s, n_periods=14)
        # Apply theta_hat to the UK config so DE and UK runs share the
        # same behavioural parameter vector regardless of whether the
        # passed theta_hat differs from the locked GERMAN_SMM_PARAMS.
        cfg = apply_params(cfg, theta_hat)
        cfg = replace(cfg, voting=replace(cfg.voting, cosmopolitan_shift_by_region=shift))
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


# ---------------------------------------------------------------------------
# Moments computed from a list of (cfg, hist) tuples
# ---------------------------------------------------------------------------


def regional_vote_at_t(runs, t: int) -> np.ndarray:
    """Cross-seed mean of the per-region vote share at time t."""
    return np.stack([h.vote[t] for _, h in runs]).mean(axis=0)


def partial_correlation_at_t(runs, t: int) -> float:
    """Income-controlled cross-regional partial correlation between vote
    share and cumulative price growth across regions, averaged over seeds."""
    vals = []
    for _, h in runs:
        vote = h.vote[t]
        growth = h.price[t] / h.price[0] - 1.0
        income = h.mean_income[t]
        if vote.std(ddof=1) == 0 or growth.std(ddof=1) == 0 or income.std(ddof=1) == 0:
            continue
        r_xy = np.corrcoef(growth, vote)[0, 1]
        r_xz = np.corrcoef(growth, income)[0, 1]
        r_yz = np.corrcoef(vote, income)[0, 1]
        denom = np.sqrt(max((1.0 - r_xz**2) * (1.0 - r_yz**2), 1e-12))
        vals.append((r_xy - r_xz * r_yz) / denom)
    return float(np.mean(vals)) if vals else float("nan")


def renter_owner_gap_at_t(runs, t: int) -> float:
    gaps = []
    for _, h in runs:
        rent_v = h.vote_by_tenure[t, :, 0]
        own_v = h.vote_by_tenure[t, :, 1]
        gaps.append(float((rent_v - own_v).mean()))
    return float(np.mean(gaps))


def regional_dispersion_at_t(runs, t: int) -> float:
    return float(np.mean([float(h.vote[t].std(ddof=1)) for _, h in runs]))


def aggregate_vote_at_t(runs, t: int) -> float:
    return float(np.mean([float(h.vote_aggregate[t]) for _, h in runs]))


# ---------------------------------------------------------------------------
# Gamma calibration via 1-D bisection
# ---------------------------------------------------------------------------


def differential_at_gamma_de(theta_hat: np.ndarray, gamma: float,
                              high_grad_idx: int, low_grad_idx: int,
                              seeds: list[int], t: int = 14) -> float:
    """Model differential = vote(low-grad-region) - vote(high-grad-region) at t."""
    runs = simulate_de_at_gamma(theta_hat, gamma, seeds)
    vote = regional_vote_at_t(runs, t)
    return float(vote[low_grad_idx] - vote[high_grad_idx])


def differential_at_gamma_uk(theta_hat: np.ndarray, gamma: float,
                              high_grad_idx: int, low_grad_idx: int,
                              seeds: list[int], t: int = BREXIT_PERIOD) -> float:
    runs = simulate_uk_at_gamma(theta_hat, gamma, seeds)
    vote = regional_vote_at_t(runs, t)
    return float(vote[low_grad_idx] - vote[high_grad_idx])


def bisect_gamma(diff_func, target: float, seeds: list[int],
                 lo: float = -30.0, hi: float = 0.0,
                 tol: float = 0.005, max_iter: int = 12) -> tuple[float, list]:
    """Bisect gamma in [lo, hi] until model differential is within tol of target.

    diff_func(gamma, seeds) returns the model differential at gamma.
    Returns (gamma_hat, history) where history is the list of (gamma, diff)
    pairs evaluated.
    """
    history: list[dict] = []
    d_lo = diff_func(lo, seeds)
    d_hi = diff_func(hi, seeds)
    history.append({"gamma": lo, "diff": d_lo})
    history.append({"gamma": hi, "diff": d_hi})

    # We expect: as gamma -> negative infinity, high-grad regions get
    # suppressed beta_0 shift, vote drops there, differential (low - high)
    # grows. So diff is monotone increasing as gamma -> -infinity.
    # If the target is between d_hi and d_lo, bisect.

    if target < min(d_lo, d_hi) or target > max(d_lo, d_hi):
        # Target outside the bracket; return the boundary that minimises error
        if abs(target - d_lo) < abs(target - d_hi):
            return lo, history
        return hi, history

    a, b = lo, hi
    f_a, f_b = d_lo, d_hi
    # Standardise so f(a) <= target <= f(b)
    if f_a > f_b:
        a, b = b, a
        f_a, f_b = f_b, f_a

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = diff_func(mid, seeds)
        history.append({"gamma": mid, "diff": f_mid})
        if abs(f_mid - target) < tol:
            return mid, history
        if f_mid < target:
            a = mid
        else:
            b = mid

    return 0.5 * (a + b), history


# ---------------------------------------------------------------------------
# Top-level pilot
# ---------------------------------------------------------------------------


CAL_SEEDS = [73, 74, 75, 76, 77]
FINAL_SEEDS = list(range(73, 83))


def main() -> None:
    t0 = time.time()
    theta_hat = load_theta_hat()
    print(f"theta_hat = {theta_hat.tolist()}")

    grad_de = de_grad_vector_by_model_region()
    grad_uk = uk_grad_vector_by_model_region()
    afd_emp = de_afd_vector_by_model_region()
    leave_emp = uk_leave_vector_by_model_region()

    # Identify high and low graduate-share regions (by model-region index).
    de_high = int(np.argmax(grad_de))
    de_low = int(np.argmin(grad_de))
    uk_high = int(np.argmax(grad_uk))
    uk_low = int(np.argmin(grad_uk))

    target_de = float(afd_emp[de_low] - afd_emp[de_high])
    target_uk = float(leave_emp[uk_low] - leave_emp[uk_high])

    print(f"\nDE: high-grad region idx {de_high} (grad {grad_de[de_high]:.2f}, "
          f"Land = {DE_LAND_BY_MODEL_REGION[de_high]}, AfD {afd_emp[de_high]:.2f})")
    print(f"DE: low-grad region idx {de_low} (grad {grad_de[de_low]:.2f}, "
          f"Land = {DE_LAND_BY_MODEL_REGION[de_low]}, AfD {afd_emp[de_low]:.2f})")
    print(f"DE: target differential AfD(low-grad) - AfD(high-grad) = {target_de:+.3f}")

    print(f"\nUK: high-grad region idx {uk_high} (grad {grad_uk[uk_high]:.2f}, "
          f"region = {UK_REGION_NAMES[uk_high]}, Leave {leave_emp[uk_high]:.3f})")
    print(f"UK: low-grad region idx {uk_low} (grad {grad_uk[uk_low]:.2f}, "
          f"region = {UK_REGION_NAMES[uk_low]}, Leave {leave_emp[uk_low]:.3f})")
    print(f"UK: target differential Leave(low-grad) - Leave(high-grad) = {target_uk:+.3f}")

    # Control: gamma = 0 in both countries (reproduces existing pipeline exactly).
    print("\n=== Control (gamma = 0) ===")
    runs_de_ctrl = simulate_de_at_gamma(theta_hat, 0.0, FINAL_SEEDS)
    runs_uk_ctrl = simulate_uk_at_gamma(theta_hat, 0.0, FINAL_SEEDS)
    ctrl_de = {
        "partial_corr": partial_correlation_at_t(runs_de_ctrl, 14),
        "cleavage": renter_owner_gap_at_t(runs_de_ctrl, 14),
        "dispersion": regional_dispersion_at_t(runs_de_ctrl, 14),
        "aggregate_vote": aggregate_vote_at_t(runs_de_ctrl, 14),
        "differential_low_minus_high_grad": float(
            regional_vote_at_t(runs_de_ctrl, 14)[de_low]
            - regional_vote_at_t(runs_de_ctrl, 14)[de_high]
        ),
    }
    ctrl_uk = {
        "partial_corr": partial_correlation_at_t(runs_uk_ctrl, BREXIT_PERIOD),
        "cleavage": renter_owner_gap_at_t(runs_uk_ctrl, BREXIT_PERIOD),
        "dispersion": regional_dispersion_at_t(runs_uk_ctrl, BREXIT_PERIOD),
        "aggregate_vote": aggregate_vote_at_t(runs_uk_ctrl, BREXIT_PERIOD),
        "differential_low_minus_high_grad": float(
            regional_vote_at_t(runs_uk_ctrl, BREXIT_PERIOD)[uk_low]
            - regional_vote_at_t(runs_uk_ctrl, BREXIT_PERIOD)[uk_high]
        ),
    }
    print(f"  DE control: partial_corr={ctrl_de['partial_corr']:+.4f}, "
          f"cleavage={ctrl_de['cleavage']:+.4f}, "
          f"dispersion={ctrl_de['dispersion']:.4f}, "
          f"aggregate={ctrl_de['aggregate_vote']:.4f}, "
          f"differential={ctrl_de['differential_low_minus_high_grad']:+.4f}")
    print(f"  UK control: partial_corr={ctrl_uk['partial_corr']:+.4f}, "
          f"cleavage={ctrl_uk['cleavage']:+.4f}, "
          f"dispersion={ctrl_uk['dispersion']:.4f}, "
          f"aggregate={ctrl_uk['aggregate_vote']:.4f}, "
          f"differential={ctrl_uk['differential_low_minus_high_grad']:+.4f}")

    # Calibrate gamma per country.
    print(f"\n=== Calibrating gamma (DE) to target differential {target_de:+.3f} ===")
    def de_diff(gamma, seeds):
        d = differential_at_gamma_de(theta_hat, gamma, de_high, de_low, seeds, 14)
        print(f"    DE gamma={gamma:+.3f}: differential={d:+.4f}")
        sys.stdout.flush()
        return d
    gamma_de, hist_de = bisect_gamma(de_diff, target_de, CAL_SEEDS,
                                       lo=-30.0, hi=0.0, tol=0.01, max_iter=10)
    print(f"  DE gamma_hat = {gamma_de:+.4f}")

    print(f"\n=== Calibrating gamma (UK) to target differential {target_uk:+.3f} ===")
    def uk_diff(gamma, seeds):
        d = differential_at_gamma_uk(theta_hat, gamma, uk_high, uk_low, seeds, BREXIT_PERIOD)
        print(f"    UK gamma={gamma:+.3f}: differential={d:+.4f}")
        sys.stdout.flush()
        return d
    gamma_uk, hist_uk = bisect_gamma(uk_diff, target_uk, CAL_SEEDS,
                                       lo=-30.0, hi=0.0, tol=0.01, max_iter=10)
    print(f"  UK gamma_hat = {gamma_uk:+.4f}")

    # Final 10-seed recomputation at country-specific gamma.
    print("\n=== Recomputing moments at country-specific gamma, 10 seeds ===")
    runs_de_pilot = simulate_de_at_gamma(theta_hat, gamma_de, FINAL_SEEDS)
    runs_uk_pilot = simulate_uk_at_gamma(theta_hat, gamma_uk, FINAL_SEEDS)
    pilot_de = {
        "gamma": gamma_de,
        "partial_corr": partial_correlation_at_t(runs_de_pilot, 14),
        "cleavage": renter_owner_gap_at_t(runs_de_pilot, 14),
        "dispersion": regional_dispersion_at_t(runs_de_pilot, 14),
        "aggregate_vote": aggregate_vote_at_t(runs_de_pilot, 14),
        "differential_low_minus_high_grad": float(
            regional_vote_at_t(runs_de_pilot, 14)[de_low]
            - regional_vote_at_t(runs_de_pilot, 14)[de_high]
        ),
    }
    pilot_uk = {
        "gamma": gamma_uk,
        "partial_corr": partial_correlation_at_t(runs_uk_pilot, BREXIT_PERIOD),
        "cleavage": renter_owner_gap_at_t(runs_uk_pilot, BREXIT_PERIOD),
        "dispersion": regional_dispersion_at_t(runs_uk_pilot, BREXIT_PERIOD),
        "aggregate_vote": aggregate_vote_at_t(runs_uk_pilot, BREXIT_PERIOD),
        "differential_low_minus_high_grad": float(
            regional_vote_at_t(runs_uk_pilot, BREXIT_PERIOD)[uk_low]
            - regional_vote_at_t(runs_uk_pilot, BREXIT_PERIOD)[uk_high]
        ),
    }

    # Shared gamma test.
    gamma_shared = 0.5 * (gamma_de + gamma_uk)
    print(f"\n=== Shared gamma = mean(DE, UK) = {gamma_shared:+.4f}, 10 seeds ===")
    runs_de_shared = simulate_de_at_gamma(theta_hat, gamma_shared, FINAL_SEEDS)
    runs_uk_shared = simulate_uk_at_gamma(theta_hat, gamma_shared, FINAL_SEEDS)
    shared_de = {
        "gamma": gamma_shared,
        "partial_corr": partial_correlation_at_t(runs_de_shared, 14),
        "cleavage": renter_owner_gap_at_t(runs_de_shared, 14),
        "dispersion": regional_dispersion_at_t(runs_de_shared, 14),
        "aggregate_vote": aggregate_vote_at_t(runs_de_shared, 14),
        "differential_low_minus_high_grad": float(
            regional_vote_at_t(runs_de_shared, 14)[de_low]
            - regional_vote_at_t(runs_de_shared, 14)[de_high]
        ),
    }
    shared_uk = {
        "gamma": gamma_shared,
        "partial_corr": partial_correlation_at_t(runs_uk_shared, BREXIT_PERIOD),
        "cleavage": renter_owner_gap_at_t(runs_uk_shared, BREXIT_PERIOD),
        "dispersion": regional_dispersion_at_t(runs_uk_shared, BREXIT_PERIOD),
        "aggregate_vote": aggregate_vote_at_t(runs_uk_shared, BREXIT_PERIOD),
        "differential_low_minus_high_grad": float(
            regional_vote_at_t(runs_uk_shared, BREXIT_PERIOD)[uk_low]
            - regional_vote_at_t(runs_uk_shared, BREXIT_PERIOD)[uk_high]
        ),
    }

    payload = {
        "theta_hat": theta_hat.tolist(),
        "de_land_by_model_region": list(DE_LAND_BY_MODEL_REGION),
        "uk_region_by_model_region": list(UK_REGION_NAMES),
        "grad_share_de": grad_de.tolist(),
        "grad_share_uk": grad_uk.tolist(),
        "afd_2025_by_model_region": afd_emp.tolist(),
        "leave_2016_by_model_region": leave_emp.tolist(),
        "target_differential_de": target_de,
        "target_differential_uk": target_uk,
        "control_de": ctrl_de,
        "control_uk": ctrl_uk,
        "pilot_de_country_specific": pilot_de,
        "pilot_uk_country_specific": pilot_uk,
        "pilot_de_shared_gamma": shared_de,
        "pilot_uk_shared_gamma": shared_uk,
        "gamma_de": gamma_de,
        "gamma_uk": gamma_uk,
        "gamma_shared": gamma_shared,
        "bisection_history_de": hist_de,
        "bisection_history_uk": hist_uk,
        "n_calibration_seeds": len(CAL_SEEDS),
        "n_final_seeds": len(FINAL_SEEDS),
        "elapsed_seconds": time.time() - t0,
    }

    out_path = OUTPUTS / "cosmopolitan_pilot.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path.relative_to(ROOT)}")

    # Surface a single table.
    print("\n" + "=" * 110)
    print("PILOT RESULTS TABLE")
    print("=" * 110)
    hdr = f"{'country':10s}  {'spec':18s}  {'gamma':>9s}  {'partial':>9s}  {'cleavage':>9s}  {'dispersion':>10s}  {'aggregate':>9s}"
    print(hdr)
    print("-" * 110)
    for ctry, label, data in [
        ("Germany", "control (gamma=0)", ctrl_de),
        ("Germany", "country-specific", pilot_de),
        ("Germany", f"shared gamma     ", shared_de),
        ("UK",      "control (gamma=0)", ctrl_uk),
        ("UK",      "country-specific", pilot_uk),
        ("UK",      f"shared gamma     ", shared_uk),
    ]:
        gam = data.get("gamma", 0.0)
        print(f"{ctry:10s}  {label:18s}  {gam:+9.3f}  {data['partial_corr']:+9.4f}  "
              f"{data['cleavage']:+9.4f}  {data['dispersion']:>10.4f}  {data['aggregate_vote']:.4f}")


if __name__ == "__main__":
    main()
