"""Audit the runtime parameter vector of each manuscript-producing entry
point against the SMM optimum in ``outputs/smm_optimum.json``.

Usage:
    python scripts/audit_parameter_vector.py                 # audit all
    python scripts/audit_parameter_vector.py --list          # show entries
    python scripts/audit_parameter_vector.py run_baseline    # one entry
    python scripts/audit_parameter_vector.py --json out.json # machine-readable

For each registered entry point, the script builds the Config that
entry point would use (without running the simulation) and runs
``audit_config_against_optimum``. It prints a per-parameter table and an
overall status line of ``AT OPTIMUM``, ``BOUND-ONLY DETACHED``, or
``FULLY DETACHED`` for each entry point, then a summary table.

The Config-building functions in ENTRY_POINTS mirror the inline Config
construction done by each entry-point script as of this audit. They are
intentionally explicit rather than imported, both because most scripts
inline their Config in ``main()`` and because the audit needs to read
the *runtime* parameter vector that the entry point would use, not a
parameter vector loaded by helper code we are about to refactor.

This script reads-only. It does not modify any Config, does not run any
simulation, and does not write to ``outputs/`` except when ``--json`` is
passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from abmhp import Config  # noqa: E402
from abmhp.estimation.smm import (  # noqa: E402
    AUDIT_AT_OPTIMUM,
    AUDIT_BOUND_ONLY_DETACHED,
    AUDIT_FULLY_DETACHED,
    DEFAULT_OPTIMUM_PATH,
    ParamAuditReport,
    apply_params,
    apply_smm_optimum,
    audit_config_against_optimum,
    build_param_space,
    load_smm_optimum,
)


# ---------------------------------------------------------------------------
# Entry-point registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryPoint:
    name: str
    script_path: str
    description: str
    build: Callable[[], Config]


def _build_run_baseline() -> Config:
    """Mirror scripts/run_baseline.py::main post-3.3 patch:
    ``apply_smm_optimum(Config(seed=seed))``."""
    return apply_smm_optimum(Config(seed=73))


def _build_counterfactual_material_security() -> Config:
    """Mirror scripts/counterfactual_material_security.py::make_config
    for Scenario A (no policy) post-3.3 patch:
    ``apply_smm_optimum(Config(seed=seed, n_periods=...))``. Other
    scenarios (B, C, D, E) override voting.beta_0 to -3.5 on top of
    the optimum to activate policy; that intentional detachment from
    the SMM-calibrated beta_0 = -5.435 is the activation lever and is
    not a defect.
    """
    return apply_smm_optimum(Config(seed=73, n_periods=40))


def _build_counterfactual_equal_cost() -> Config:
    """Mirror scripts/counterfactual_equal_cost.py::make_baseline_A
    post-3.3 patch. Scenarios E and C-plus override voting.beta_0 to
    -3.5 on top of the optimum (intentional activation). We audit
    Scenario A as the un-perturbed reference.
    """
    return apply_smm_optimum(Config(seed=73, n_periods=40))


def _build_run_uk_validation() -> Config:
    """Mirror scripts/run_uk_validation.py: imports run_uk_validation
    from abmhp.validation.uk, which calls make_uk_config(). The UK
    Config copies parameter values from the hard-coded
    ``GERMAN_SMM_PARAMS`` dict (rounded to 4 decimal places) rather
    than from outputs/smm_optimum.json directly.
    """
    from abmhp.validation.uk import make_uk_config

    return make_uk_config(seed=73)


def _build_phase1_reproducibility_check() -> Config:
    """Mirror scripts/phase1_reproducibility_check.py: loads theta_hat
    from outputs/smm_optimum.json and calls apply_params. Should audit
    as AT OPTIMUM.
    """
    payload = json.loads((ROOT / "outputs" / "smm_optimum.json").read_text())
    theta_hat = np.array(payload["theta_hat"], dtype=float)
    return apply_params(Config(), theta_hat)


def _build_german_partial_correlation_placebo() -> Config:
    """Mirror scripts/german_partial_correlation_placebo.py: loads theta
    from outputs/smm_optimum.json and applies it.
    """
    payload = json.loads((ROOT / "outputs" / "smm_optimum.json").read_text())
    theta = np.array(payload["theta_hat"], dtype=float)
    return apply_params(Config(), theta)


def _build_run_with_policy() -> Config:
    """Mirror scripts/run_with_policy.py::main with default flags
    (no --populist, no --beta0) post-3.3 patch:
    ``apply_smm_optimum(Config(seed=...))``.
    """
    return apply_smm_optimum(Config(seed=73))


def _build_compare_abm_hank() -> Config:
    """Mirror scripts/compare_abm_hank.py::abm_baseline_moments
    post-3.3 patch. The HANK block in the same script uses HankConfig,
    which is outside the SMM parameter space and is not audited here.
    """
    return apply_smm_optimum(Config(seed=73))


def _build_cosmopolitan_pilot() -> Config:
    """Mirror scripts/cosmopolitan_pilot.py: loads theta_hat from
    smm_optimum.json and applies it, then layers a cosmopolitan-shift
    overlay. The overlay touches voting.cosmopolitan_shift_by_region,
    which is not a SMM-calibrated parameter, so the audit sees an
    AT-OPTIMUM Config.
    """
    payload = json.loads((ROOT / "outputs" / "smm_optimum.json").read_text())
    theta_hat = np.array(payload["theta_hat"], dtype=float)
    return apply_params(Config(), theta_hat)


def _build_o_construction_diagnostic() -> Config:
    """Mirror scripts/o_construction_diagnostic.py::main: loads theta_hat
    via apply_params(Config(), theta_hat).
    """
    payload = json.loads((ROOT / "outputs" / "smm_optimum.json").read_text())
    theta_hat = np.array(payload["theta_hat"], dtype=float)
    return apply_params(Config(), theta_hat)


def _build_phase3_smm_augmented_reduced() -> Config:
    """Mirror scripts/phase3_smm_augmented_reduced.py::_build_base_cfg
    (the pre-SMM driver config). The script enables three behavioral
    flags (assortative_help_enabled, estimate_beta_n,
    estimate_gamma_cosmopolitan) and sets voting.grad_share_data_path;
    none of those are members of PARAM_SPACE, so the audit sees the
    base 8-parameter vector. Pre-SMM, that vector is at dataclass
    defaults; the audit classifies this as detached, which is the
    expected pre-run state for an SMM driver.
    """
    cfg = Config()
    behavioral = replace(
        cfg.behavioral,
        assortative_help_enabled=True,
        estimate_beta_n=True,
        estimate_gamma_cosmopolitan=True,
    )
    return replace(cfg, behavioral=behavioral)


def _build_phase3_v2_smm_augmented_p50() -> Config:
    """Mirror scripts/phase3_v2_smm_augmented_p50.py::_build_base_cfg.
    Same augmented flags as phase3_smm_augmented_reduced plus the p50
    aspiration-reference quantile override on the voting block. None
    of these are PARAM_SPACE members; the audit again sees the base
    8-parameter vector at dataclass defaults.
    """
    cfg = Config()
    behavioral = replace(
        cfg.behavioral,
        assortative_help_enabled=True,
        estimate_beta_n=True,
        estimate_gamma_cosmopolitan=True,
    )
    voting = replace(cfg.voting, aspiration_reference_quantile=0.50)
    return replace(cfg, behavioral=behavioral, voting=voting)


def _build_phase4_counterfactual_augmented() -> Config:
    """Mirror scripts/phase4_counterfactual_augmented.py::make_config
    for Scenario A (no policy). The script enables three behavioral
    flags, sets voting.grad_share_data_path, and applies the augmented
    theta from outputs/smm_augmented_reduced.json. The other scenarios
    (B-E) layer policy overrides and a beta_0 nudge on top; only the
    beta_0 nudge intersects PARAM_SPACE, and only outside Scenario A.
    Auditing Scenario A surfaces the parameter vector that actually
    enters simulate() for the un-perturbed reference cell.
    """
    augmented_path = ROOT / "outputs" / "smm_augmented_reduced.json"
    cfg = Config(seed=73, n_periods=40)
    cfg = replace(
        cfg,
        behavioral=replace(
            cfg.behavioral,
            assortative_help_enabled=True,
            estimate_beta_n=True,
            estimate_gamma_cosmopolitan=True,
        ),
    )
    if not augmented_path.exists():
        # No augmented optimum on disk: report the pre-apply_params
        # Config so the audit at least classifies the base block.
        return cfg
    payload = json.loads(augmented_path.read_text())
    theta = np.array(payload["theta_hat"], dtype=float)
    param_space = build_param_space(cfg)
    return apply_params(cfg, theta, param_space=param_space)


def _build_phase4_uk_augmented() -> Config:
    """Mirror scripts/phase4_uk_augmented.py::main. The script starts
    from make_uk_config (which already applies the 4-decimal-truncated
    GERMAN_SMM_PARAMS to the UK Config), then applies the augmented
    theta via apply_params under the augmented param_space, then sets
    assortative_help_enabled on behavioral. For audit purposes we
    construct the same Config: base UK config, augmented flags on
    behavioral (to make build_param_space return the augmented space),
    apply augmented theta, then set assortative_help_enabled (which is
    not in PARAM_SPACE and does not affect the audited values).
    """
    from abmhp.validation.uk import make_uk_config

    cfg = make_uk_config(seed=73)
    augmented_path = ROOT / "outputs" / "smm_augmented_reduced.json"
    cfg = replace(
        cfg,
        behavioral=replace(
            cfg.behavioral,
            estimate_beta_n=True,
            estimate_gamma_cosmopolitan=True,
        ),
    )
    if not augmented_path.exists():
        return cfg
    payload = json.loads(augmented_path.read_text())
    theta = np.array(payload["theta_hat"], dtype=float)
    param_space = build_param_space(cfg)
    cfg = apply_params(cfg, theta, param_space=param_space)
    cfg = replace(
        cfg, behavioral=replace(cfg.behavioral, assortative_help_enabled=True)
    )
    return cfg


def _build_validate_distribution() -> Config:
    """Mirror scripts/validate_distribution.py default invocation
    post-3.3 patch: ``apply_smm_optimum(Config(seed=seed, demographic=...))``.
    """
    return apply_smm_optimum(Config(seed=73))


# Order: manuscript-producing entry points first, then SMM drivers, then
# augmented-model entry points. The list is the source of truth for the
# "--list" output and for "audit all".
ENTRY_POINTS: tuple[EntryPoint, ...] = (
    EntryPoint(
        "run_baseline",
        "scripts/run_baseline.py",
        "Section 5 baseline simulator-drift table",
        _build_run_baseline,
    ),
    EntryPoint(
        "counterfactual_material_security",
        "scripts/counterfactual_material_security.py",
        "Section 6 bundling scenarios (A, B, C, D, E)",
        _build_counterfactual_material_security,
    ),
    EntryPoint(
        "counterfactual_equal_cost",
        "scripts/counterfactual_equal_cost.py",
        "Section 6.x equal-cost counterfactual",
        _build_counterfactual_equal_cost,
    ),
    EntryPoint(
        "run_uk_validation",
        "scripts/run_uk_validation.py",
        "Section 8 UK external stress test (via abmhp.validation.uk)",
        _build_run_uk_validation,
    ),
    EntryPoint(
        "run_with_policy",
        "scripts/run_with_policy.py",
        "Section 6 policy-on driver (default flags)",
        _build_run_with_policy,
    ),
    EntryPoint(
        "phase1_reproducibility_check",
        "scripts/phase1_reproducibility_check.py",
        "Phase 1 reproducibility: applies theta_hat from JSON",
        _build_phase1_reproducibility_check,
    ),
    EntryPoint(
        "german_partial_correlation_placebo",
        "scripts/german_partial_correlation_placebo.py",
        "Section 9 partial-correlation placebo (German)",
        _build_german_partial_correlation_placebo,
    ),
    EntryPoint(
        "compare_abm_hank",
        "scripts/compare_abm_hank.py",
        "Section 7 ABM vs HANK comparison (ABM block only)",
        _build_compare_abm_hank,
    ),
    EntryPoint(
        "cosmopolitan_pilot",
        "scripts/cosmopolitan_pilot.py",
        "Section 9.x cosmopolitan-overlay robustness",
        _build_cosmopolitan_pilot,
    ),
    EntryPoint(
        "o_construction_diagnostic",
        "scripts/o_construction_diagnostic.py",
        "Section 9.x o-construction robustness diagnostic",
        _build_o_construction_diagnostic,
    ),
    EntryPoint(
        "phase3_smm_augmented_reduced",
        "scripts/phase3_smm_augmented_reduced.py",
        "SMM driver for augmented model (pre-SMM Config; expected detached)",
        _build_phase3_smm_augmented_reduced,
    ),
    EntryPoint(
        "phase3_v2_smm_augmented_p50",
        "scripts/phase3_v2_smm_augmented_p50.py",
        "SMM driver for augmented v2 (pre-SMM Config; expected detached)",
        _build_phase3_v2_smm_augmented_p50,
    ),
    EntryPoint(
        "phase4_counterfactual_augmented",
        "scripts/phase4_counterfactual_augmented.py",
        "Section 9 augmented-model counterfactual (uses augmented optimum)",
        _build_phase4_counterfactual_augmented,
    ),
    EntryPoint(
        "phase4_uk_augmented",
        "scripts/phase4_uk_augmented.py",
        "Section 9 augmented-model UK stress test (uses augmented optimum)",
        _build_phase4_uk_augmented,
    ),
    EntryPoint(
        "validate_distribution",
        "scripts/validate_distribution.py",
        "Distributional sanity check (default Config)",
        _build_validate_distribution,
    ),
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


AUDIT_FAILED_PREFIX = "AUDIT FAILED"

_STATUS_GLYPH = {
    AUDIT_AT_OPTIMUM: "OK   ",
    AUDIT_BOUND_ONLY_DETACHED: "BOUND",
    AUDIT_FULLY_DETACHED: "DETCH",
}
_FAILED_GLYPH = "FAIL "


def _print_report(name: str, ep: EntryPoint, report: ParamAuditReport) -> None:
    print(f"== {name} ({ep.script_path}) ==")
    print(f"   {ep.description}")
    print(f"   optimum: {report.optimum_source}")
    print(f"   status:  {report.status}")
    print(f"   {'param':30s}  {'optimum':>12s}  {'current':>12s}  {'|diff|':>10s}  bound  match")
    for r in report.rows:
        bound = " *  " if r.optimum_at_bound else "    "
        match = " ok " if r.matches else "MISS"
        print(
            f"   {r.name:30s}  {r.optimum_value:+12.6f}  {r.current_value:+12.6f}  "
            f"{r.abs_diff:10.2e}  {bound}   {match}"
        )
    print()


def _report_to_dict(name: str, ep: EntryPoint, report: ParamAuditReport) -> dict:
    return {
        "entry_point": name,
        "script_path": ep.script_path,
        "description": ep.description,
        "optimum_source": str(report.optimum_source),
        "status": report.status,
        "params": [
            {
                "name": r.name,
                "optimum_value": r.optimum_value,
                "current_value": r.current_value,
                "abs_diff": r.abs_diff,
                "optimum_at_bound": r.optimum_at_bound,
                "matches": r.matches,
            }
            for r in report.rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "entry",
        nargs="?",
        default=None,
        help="Entry-point name to audit. Omit to audit all entries.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered entry points and exit.",
    )
    parser.add_argument(
        "--optimum",
        type=Path,
        default=None,
        help="Path to SMM optimum JSON. Default: outputs/smm_optimum.json.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="If set, write full audit report as JSON to this path.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for AT-OPTIMUM classification (default 1e-6).",
    )
    args = parser.parse_args()

    if args.list:
        for ep in ENTRY_POINTS:
            print(f"  {ep.name:35s}  {ep.script_path}")
        return 0

    optimum = load_smm_optimum(args.optimum) if args.optimum else load_smm_optimum()

    if args.entry is not None:
        matching = [ep for ep in ENTRY_POINTS if ep.name == args.entry]
        if not matching:
            print(f"Unknown entry point: {args.entry}", file=sys.stderr)
            print("Available:", file=sys.stderr)
            for ep in ENTRY_POINTS:
                print(f"  {ep.name}", file=sys.stderr)
            return 2
        targets = matching
    else:
        targets = list(ENTRY_POINTS)

    # Each entry in `outcomes` is either (ep, report, None) for a
    # successful audit or (ep, None, error_string) for a build failure.
    # A build failure means the entry point could not be constructed,
    # so its parameter vector cannot be compared; that is itself a
    # finding, recorded with status "AUDIT FAILED: <reason>" rather
    # than silently dropped.
    outcomes: list[tuple[EntryPoint, ParamAuditReport | None, str | None]] = []
    for ep in targets:
        try:
            cfg = ep.build()
        except Exception as exc:  # pragma: no cover (audit driver is best-effort)
            err = f"{type(exc).__name__}: {exc}"
            print(f"== {ep.name} ({ep.script_path}) ==")
            print(f"   {ep.description}")
            print(f"   status:  {AUDIT_FAILED_PREFIX}: {err}")
            print()
            outcomes.append((ep, None, err))
            continue
        report = audit_config_against_optimum(cfg, optimum, tolerance=args.tolerance)
        outcomes.append((ep, report, None))
        _print_report(ep.name, ep, report)

    # Summary table.
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    counts = {
        AUDIT_AT_OPTIMUM: 0,
        AUDIT_BOUND_ONLY_DETACHED: 0,
        AUDIT_FULLY_DETACHED: 0,
        AUDIT_FAILED_PREFIX: 0,
    }
    for ep, report, err in outcomes:
        if report is not None:
            glyph = _STATUS_GLYPH[report.status]
            print(f"  [{glyph}]  {ep.name:35s}  {report.status}")
            counts[report.status] += 1
        else:
            print(f"  [{_FAILED_GLYPH}]  {ep.name:35s}  {AUDIT_FAILED_PREFIX}: {err}")
            counts[AUDIT_FAILED_PREFIX] += 1
    print()
    print(
        f"  AT OPTIMUM:           {counts[AUDIT_AT_OPTIMUM]}\n"
        f"  BOUND-ONLY DETACHED:  {counts[AUDIT_BOUND_ONLY_DETACHED]}\n"
        f"  FULLY DETACHED:       {counts[AUDIT_FULLY_DETACHED]}\n"
        f"  AUDIT FAILED:         {counts[AUDIT_FAILED_PREFIX]}"
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        entry_payloads: list[dict] = []
        for ep, report, err in outcomes:
            if report is not None:
                entry_payloads.append(_report_to_dict(ep.name, ep, report))
            else:
                entry_payloads.append(
                    {
                        "entry_point": ep.name,
                        "script_path": ep.script_path,
                        "description": ep.description,
                        "optimum_source": str(optimum.source_path),
                        "status": f"{AUDIT_FAILED_PREFIX}: {err}",
                        "error": err,
                        "params": None,
                    }
                )
        payload = {
            "optimum_source": str(optimum.source_path),
            "tolerance": args.tolerance,
            "entry_points": entry_payloads,
            "summary": {k: int(v) for k, v in counts.items()},
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")

    # Exit non-zero if any entry point is detached or failed to audit.
    # The audit is a diagnostic tool, so a non-zero exit is
    # informational rather than an error; CI gates can rely on it to
    # fail when an entry point drifts away from the SMM optimum or
    # cannot be constructed for inspection.
    if (
        counts[AUDIT_FULLY_DETACHED]
        or counts[AUDIT_BOUND_ONLY_DETACHED]
        or counts[AUDIT_FAILED_PREFIX]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
