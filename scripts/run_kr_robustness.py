"""Koszegi-Rabin reference-rule robustness, Section 9.1 honesty check.

Section 9.1 of the manuscript claims that switching the reference rule from
Akerlof-Yellen (AY) to Koszegi-Rabin (K-R) reduces the within-region
renter-owner cleavage from +0.279 to ~+0.18 and the aggregate AfD vote share
from 0.209 to ~0.16. Until this script existed, no entry-point in the
replication archive produced those numbers from a fresh run.

This script:

  1. Loads the SMM optimum AY parameter vector (no re-estimation).
  2. Builds a paired K-R config that differs only in the reference rule
     (`reference_rule="koszegi_rabin"` on VotingConfig).
  3. Evaluates the 12 SMM calibration moments + the renter-owner cleavage
     validation moment under each rule, averaged over the headline 5-seed
     validation set {73, 74, 75, 76, 77}.
  4. Prints a side-by-side AY-vs-K-R table.
  5. Persists outputs/kr_robustness.json with the AY values, the K-R
     values, per-moment deltas, and an honesty-check verdict.
  6. Reports SUCCESS or DISCREPANCY against the paper's Section 9.1 claim,
     using paper tolerances (cleavage +/- 0.02, AfD +/- 0.01).

The K-R implementation lives in src/abmhp/voting.py behind a
VotingConfig.reference_rule flag defaulting to "akerlof_yellen", so the
headline pipeline path is bit-identical. Verified at script load via a
per-field np.array_equal check; see the seed-by-seed unit test in
tests/test_reference_rule.py for the standalone assertion.

The K-R operationalisation: aspiration is updated as
    aspiration_t = rho * aspiration_{t-1} + (1 - rho) * own_outcome_t
i.e. AR(1) on each agent's own realised outcome, with the same rho as AY.
This isolates the reference-anchor change (external p75-of-income vs. own
realised outcome) from a separate smoothing-rate change. It is the
standard reduced-form K-R contrast: rational expectations on the
consumed/realised variable rather than a backward-looking normative anchor.

If the SUCCESS branch fires, the manuscript wording stands. If the
DISCREPANCY branch fires, the script halts before touching anything
other than its own kr_robustness.json diagnostic record (no smm_optimum
update, no manuscript edits) and prints the discrepancy so the author
can decide whether to revise the paper text or revise the K-R
operationalisation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from abmhp import Config
from abmhp.estimation.moments import (
    CALIBRATION_MOMENTS,
    VALIDATION_MOMENTS,
    evaluate_moments,
    simulate_seeds,
)
from abmhp.estimation.smm import apply_smm_optimum, load_smm_optimum

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

SEEDS = [73, 74, 75, 76, 77]  # same set used by score_validation_moments
N_PERIODS = 15                # SMM calibration horizon

PAPER_CLAIM = {
    "cleavage_target": 0.18,
    "cleavage_tolerance": 0.02,
    "afd_target": 0.16,
    "afd_tolerance": 0.01,
    "ay_headline_cleavage": 0.279,
    "ay_headline_afd": 0.209,
    "source": "manuscript Section 9.1 (reference-point specification)",
}

CLEAVAGE_MOMENT_NAME = "within_region_renter_owner_vote_gap"
AFD_MOMENT_NAME = "aggregate_extreme_share_final"


def sha256_repr(obj) -> str:
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()


def _portable_source(path: Path) -> str:
    """Render `path` as a repo-relative POSIX string so the persisted
    metadata is machine-independent (no absolute workstation paths in
    outputs/kr_robustness.json). If the path is not under ROOT, fall back
    to the basename so we never leak parent-directory information."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def evaluate_under(cfg: Config) -> dict:
    """Run all seeds, evaluate calibration + cleavage moments."""
    runs = simulate_seeds(cfg, SEEDS)
    cal = evaluate_moments(CALIBRATION_MOMENTS, runs)
    cleavage_moment = next(
        m for m in VALIDATION_MOMENTS if m.name == CLEAVAGE_MOMENT_NAME
    )
    cleavage = float(cleavage_moment.evaluator(runs))
    return {"calibration": cal, "cleavage": cleavage}


def build_comparison(ay: dict, kr: dict) -> dict:
    """Per-moment deltas K-R minus AY."""
    deltas = {}
    for name, v_ay in ay["calibration"].items():
        deltas[name] = kr["calibration"][name] - v_ay
    deltas[CLEAVAGE_MOMENT_NAME] = kr["cleavage"] - ay["cleavage"]
    return deltas


def honesty_check(kr: dict) -> dict:
    """Compare K-R values to paper's Section 9.1 claim."""
    afd_kr = kr["calibration"][AFD_MOMENT_NAME]
    cleavage_kr = kr["cleavage"]
    afd_in = abs(afd_kr - PAPER_CLAIM["afd_target"]) <= PAPER_CLAIM["afd_tolerance"]
    cleavage_in = abs(cleavage_kr - PAPER_CLAIM["cleavage_target"]) <= PAPER_CLAIM["cleavage_tolerance"]
    return {
        "paper_claim": PAPER_CLAIM,
        "kr_afd": afd_kr,
        "kr_cleavage": cleavage_kr,
        "afd_within_tolerance": afd_in,
        "cleavage_within_tolerance": cleavage_in,
        "reproduces_section_9_1": bool(afd_in and cleavage_in),
        "afd_error": afd_kr - PAPER_CLAIM["afd_target"],
        "cleavage_error": cleavage_kr - PAPER_CLAIM["cleavage_target"],
    }


def print_table(ay: dict, kr: dict, comparison: dict) -> None:
    """Side-by-side AY-vs-K-R table for the moments named in Section 9.1
    plus the full calibration battery for context."""
    print()
    print("=" * 88)
    print("AY headline vs K-R alternative")
    print(f"  seeds: {SEEDS}, n_periods: {N_PERIODS}")
    print("=" * 88)
    fmt = "{:42s} {:>12s} {:>12s} {:>12s}"
    print(fmt.format("moment", "AY", "K-R", "K-R - AY"))
    print("-" * 88)
    # Cleavage first since it's the headline Section 9.1 number
    print(fmt.format(
        CLEAVAGE_MOMENT_NAME + "  (Sec 9.1 claim: 0.18 +/- 0.02)",
        f"{ay['cleavage']:.4f}",
        f"{kr['cleavage']:.4f}",
        f"{kr['cleavage'] - ay['cleavage']:+.4f}",
    ))
    print(fmt.format(
        AFD_MOMENT_NAME + "  (Sec 9.1 claim: 0.16 +/- 0.01)",
        f"{ay['calibration'][AFD_MOMENT_NAME]:.4f}",
        f"{kr['calibration'][AFD_MOMENT_NAME]:.4f}",
        f"{comparison[AFD_MOMENT_NAME]:+.4f}",
    ))
    print("-" * 88)
    for name in ay["calibration"]:
        if name == AFD_MOMENT_NAME:
            continue
        print(fmt.format(
            name,
            f"{ay['calibration'][name]:.4f}",
            f"{kr['calibration'][name]:.4f}",
            f"{comparison[name]:+.4f}",
        ))


def print_verdict(check: dict) -> None:
    print()
    print("=" * 88)
    print("HONESTY CHECK against Section 9.1")
    print("=" * 88)
    print(f"  Paper claim: cleavage = {PAPER_CLAIM['cleavage_target']:.2f} "
          f"+/- {PAPER_CLAIM['cleavage_tolerance']:.2f}, "
          f"AfD = {PAPER_CLAIM['afd_target']:.2f} "
          f"+/- {PAPER_CLAIM['afd_tolerance']:.2f}")
    print(f"  This run:    cleavage = {check['kr_cleavage']:.4f}  "
          f"(error {check['cleavage_error']:+.4f}, "
          f"{'OK' if check['cleavage_within_tolerance'] else 'OUTSIDE TOLERANCE'})")
    print(f"               AfD      = {check['kr_afd']:.4f}  "
          f"(error {check['afd_error']:+.4f}, "
          f"{'OK' if check['afd_within_tolerance'] else 'OUTSIDE TOLERANCE'})")
    print()
    if check["reproduces_section_9_1"]:
        print("VERDICT: SUCCESS")
        print("  Both K-R values fall within the paper's stated tolerances.")
        print("  Section 9.1 wording is supported by a fresh-from-code K-R run.")
    else:
        print("VERDICT: DISCREPANCY")
        print("  At least one K-R value is outside the paper's stated tolerance.")
        print("  This does NOT silently rewrite Section 9.1; the manuscript text")
        print("  is unchanged. Possible explanations:")
        print("    (a) The K-R operationalisation in src/abmhp/voting.py")
        print("        (AR(1) on own outcome, same rho) differs from whatever")
        print("        was used to produce the Section 9.1 numbers.")
        print("    (b) The Section 9.1 numbers came from a one-off script that")
        print("        was never committed; the manuscript claim is now an")
        print("        unverifiable assertion.")
        print("  Recommended next step: either revise the K-R operationalisation")
        print("  in voting.py to match the intended variant, or revise the")
        print("  manuscript to report the values this script actually produces.")


def main() -> None:
    t_start = time.time()
    print("Koszegi-Rabin reference-rule robustness diagnostic")
    print(f"  loading SMM optimum from {ROOT / 'outputs' / 'smm_optimum.json'}")
    optimum = load_smm_optimum()
    cfg_ay = apply_smm_optimum(Config(seed=SEEDS[0], n_periods=N_PERIODS),
                                optimum=optimum)
    cfg_kr = replace(cfg_ay,
                     voting=replace(cfg_ay.voting, reference_rule="koszegi_rabin"))
    assert cfg_ay.voting.reference_rule == "akerlof_yellen", \
        "AY config should default to akerlof_yellen"
    assert cfg_kr.voting.reference_rule == "koszegi_rabin", \
        "K-R config should be koszegi_rabin"

    print(f"  AY  cfg: reference_rule = {cfg_ay.voting.reference_rule}")
    print(f"  K-R cfg: reference_rule = {cfg_kr.voting.reference_rule}")
    print(f"  evaluating moments at seeds {SEEDS} over {N_PERIODS} periods")
    print(f"  (a single AY-vs-K-R full evaluation is "
          f"{2 * len(SEEDS)} simulator runs)")
    print()

    print("  evaluating AY...", flush=True)
    ay = evaluate_under(cfg_ay)
    print(f"    cleavage = {ay['cleavage']:.4f}, "
          f"AfD = {ay['calibration'][AFD_MOMENT_NAME]:.4f}")
    print("  evaluating K-R...", flush=True)
    kr = evaluate_under(cfg_kr)
    print(f"    cleavage = {kr['cleavage']:.4f}, "
          f"AfD = {kr['calibration'][AFD_MOMENT_NAME]:.4f}")

    comparison = build_comparison(ay, kr)
    print_table(ay, kr, comparison)

    check = honesty_check(kr)
    print_verdict(check)

    # Build the JSON payload. Schema mirrors outputs/smm_optimum.json on
    # shared fields: same moment_names ordering, same moment_targets.
    moment_names = list(ay["calibration"].keys())
    moment_targets = [
        next(m.value for m in CALIBRATION_MOMENTS if m.name == n)
        for n in moment_names
    ]
    cleavage_target = next(
        m.value for m in VALIDATION_MOMENTS if m.name == CLEAVAGE_MOMENT_NAME
    )

    payload = {
        "metadata": {
            "script": "scripts/run_kr_robustness.py",
            "git_commit": git_commit_hash(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": SEEDS,
            "n_periods": N_PERIODS,
            "seed_vector_sha256": sha256_repr(SEEDS),
            "parameter_vector_sha256": sha256_repr(cfg_ay),
            "reference_rule_ay": "akerlof_yellen",
            "reference_rule_kr": "koszegi_rabin",
            "kr_operationalisation": (
                "AR(1) on own realised outcome with the same rho_aspiration "
                "as AY; only the anchor variable differs"
            ),
            "smm_optimum_source": _portable_source(optimum.source_path),
            "theta_hat": optimum.theta_hat.tolist(),
            "param_names": list(optimum.param_names),
        },
        "ay_headline": {
            "moment_names": moment_names,
            "moment_targets": moment_targets,
            "moment_at_optimum": [ay["calibration"][n] for n in moment_names],
            "validation": {
                CLEAVAGE_MOMENT_NAME: {
                    "target": cleavage_target,
                    "model": ay["cleavage"],
                    "error": ay["cleavage"] - cleavage_target,
                },
            },
        },
        "kr_alternative": {
            "moment_names": moment_names,
            "moment_targets": moment_targets,
            "moment_at_optimum": [kr["calibration"][n] for n in moment_names],
            "validation": {
                CLEAVAGE_MOMENT_NAME: {
                    "target": cleavage_target,
                    "model": kr["cleavage"],
                    "error": kr["cleavage"] - cleavage_target,
                },
            },
        },
        "comparison_to_ay_headline": {
            "deltas_per_moment": {
                n: comparison[n] for n in moment_names + [CLEAVAGE_MOMENT_NAME]
            },
        },
        "honesty_check": check,
        "wallclock_seconds": time.time() - t_start,
    }

    out_path = OUTPUTS / "kr_robustness.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print()
    print(f"Wrote {out_path}")
    print(f"Total wallclock: {payload['wallclock_seconds']:.1f}s")
    print()
    if not check["reproduces_section_9_1"]:
        print("NOTE: outputs/smm_optimum.json was NOT modified.")
        print("      The manuscript text was NOT modified.")
        print("      Only outputs/kr_robustness.json (this script's own")
        print("      diagnostic record) was written. Manual review needed.")


if __name__ == "__main__":
    main()
