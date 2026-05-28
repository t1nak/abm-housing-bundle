"""Tests for the calibration / validation moment table and SMM estimation.

The SMM optimum is read from `outputs/smm_optimum.json` produced by
`scripts/run_smm.py`. Tests are skipped if the artifact is missing (i.e.
the full SMM has not yet been run on this checkout); this keeps the
suite green during development without re-running the 30-minute SMM
for every CI invocation.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from abmhp import Config, PolicyRegime, simulate
from abmhp.estimation.moments import (
    CALIBRATION_MOMENTS,
    DIAGNOSTIC_MOMENTS,
    VALIDATION_MOMENTS,
    assert_weights_sum_to_one,
    evaluate_moments,
    simulate_seeds,
)
from abmhp.estimation.smm import (
    AUDIT_AT_OPTIMUM,
    AUDIT_BOUND_ONLY_DETACHED,
    AUDIT_FULLY_DETACHED,
    PARAM_NAMES,
    PARAM_SPACE,
    SMMOptimum,
    apply_params,
    apply_smm_optimum,
    audit_config_against_optimum,
    load_smm_optimum,
    moment_targets,
    simulated_moments,
)


ROOT = Path(__file__).resolve().parents[1]
OPTIMUM_PATH = ROOT / "outputs" / "smm_optimum.json"


# ---------------------------------------------------------------------------
# Moment table tests (run always)
# ---------------------------------------------------------------------------


def test_calibration_weights_sum_to_one():
    assert_weights_sum_to_one()


def test_moment_counts():
    assert len(CALIBRATION_MOMENTS) == 12
    assert len(VALIDATION_MOMENTS) == 5
    assert len(DIAGNOSTIC_MOMENTS) == 3


def test_validation_moments_are_not_in_calibration():
    cal_names = {m.name for m in CALIBRATION_MOMENTS}
    for m in VALIDATION_MOMENTS:
        assert m.name not in cal_names, \
            f"{m.name} is in both calibration and validation"


def test_within_region_renter_owner_gap_is_validation_only():
    """The within-region renter-owner gap is a validation moment; it MUST
    NOT appear in the SMM objective. The renter coefficient (BETA_R) is
    in the free-parameter space, so including this moment in calibration
    would be circular."""
    cal_names = {m.name for m in CALIBRATION_MOMENTS}
    val_names = {m.name for m in VALIDATION_MOMENTS}
    assert "within_region_renter_owner_vote_gap" in val_names
    assert "within_region_renter_owner_vote_gap" not in cal_names


def test_param_space_count():
    assert len(PARAM_SPACE) == 8
    # Overidentification degrees of freedom: K - P = 4.
    assert len(CALIBRATION_MOMENTS) - len(PARAM_SPACE) == 4


def test_param_space_bounds():
    """The 8 free parameters have the bounds specified in the prompt."""
    expected = {
        # beta_dissat upper bound widened from 9.0 to 12.0 after Phase 3
        # reduced-budget SMM had it pinned at the boundary.
        "beta_dissat": (3.0, 12.0),
        "beta_renter": (0.2, 1.0),
        "rho_aspiration": (0.40, 0.90),
        "alpha_local": (0.20, 0.70),
        "price_slope": (0.04, 0.15),
        "beta_0": (-7.0, -3.0),
        "assortative_exponent": (1.5, 3.0),
        "intergenerational_skill_corr": (0.30, 0.85),
    }
    actual = {p.name: (p.low, p.high) for p in PARAM_SPACE}
    assert actual == expected


def test_moment_targets_documented():
    """Each moment has an explicit source string."""
    for m in (*CALIBRATION_MOMENTS, *VALIDATION_MOMENTS, *DIAGNOSTIC_MOMENTS):
        assert m.source.strip() != "", f"empty source for {m.name}"


def test_moment_evaluators_callable_at_baseline():
    """Sanity: every calibration evaluator runs without exception at the
    default cfg. Uses a single seed for speed."""
    runs = simulate_seeds(Config(), seeds=[73])
    out = evaluate_moments(CALIBRATION_MOMENTS, runs)
    assert len(out) == 12
    for name, val in out.items():
        assert np.isfinite(val), f"{name} returned non-finite {val}"


# ---------------------------------------------------------------------------
# SMM-optimum-dependent tests (skipped if SMM has not been run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smm_optimum():
    if not OPTIMUM_PATH.exists():
        pytest.skip(f"{OPTIMUM_PATH.relative_to(ROOT)} not present; run scripts/run_smm.py")
    return json.loads(OPTIMUM_PATH.read_text())


def test_smm_converged(smm_optimum):
    """SMM completed and produced a finite optimum within bounds."""
    theta = np.array(smm_optimum["theta_hat"])
    assert theta.shape == (len(PARAM_SPACE),)
    assert np.isfinite(theta).all()
    for v, ps in zip(theta, PARAM_SPACE):
        assert ps.low - 1e-9 <= v <= ps.high + 1e-9, \
            f"{ps.name} = {v} outside [{ps.low}, {ps.high}]"


def test_smm_aggregate_extreme_share_within_one_pp(smm_optimum):
    """The binding identification target. At the SMM optimum the model's
    aggregate extreme-share vote must match 0,208 within 1 percentage
    point (0,01 absolute)."""
    names = smm_optimum["moment_names"]
    targets = smm_optimum["moment_targets"]
    model = smm_optimum["moment_at_optimum"]
    idx = names.index("aggregate_extreme_share_final")
    target = float(targets[idx])
    mod = float(model[idx])
    assert abs(mod - target) <= 0.01, \
        f"aggregate extreme-share at SMM optimum: target={target:.3f}, " \
        f"model={mod:.3f}, |err|={abs(mod-target):.3f} > 0.01"


def test_smm_parameter_standard_errors_below_25_percent(smm_optimum):
    """SMM-identified parameters have standard error less than 25 percent
    of point estimate magnitude.

    `beta_renter` is identified only through the within-region
    renter-owner vote gap, which by design lives in the validation block
    and is NOT in the SMM objective. The renter premium therefore has no
    direct moment to pin it; its SE will be larger than the 0,25
    threshold. This is documented in `outputs/smm_results.md` and is
    expected.

    All seven other parameters must meet the threshold."""
    theta = np.array(smm_optimum["theta_hat"])
    se = np.array(smm_optimum["se"])
    weak = []
    for name, th, s in zip(PARAM_NAMES, theta, se):
        if name == "beta_renter":
            continue  # weakly identified by design; see docstring
        ratio = float(s / max(abs(th), 1e-9))
        if ratio > 0.25:
            weak.append((name, th, s, ratio))
    assert not weak, \
        "Calibration-identified parameters with SE > 25 percent of |theta|: " + \
        "; ".join(f"{n}={t:.4f} se={s:.4f} ratio={r:.2f}" for n, t, s, r in weak)


def test_beta_renter_is_weakly_identified_by_design(smm_optimum):
    """`beta_renter` is weakly identified at the calibration optimum
    because its informative moment (within-region renter-owner gap) is
    in the validation block. The SE-to-estimate ratio should be larger
    than 0,25 to confirm this is operating as designed; if SMM somehow
    identifies it tightly via indirect channels, the test fails and the
    moment-block split should be reconsidered."""
    theta = np.array(smm_optimum["theta_hat"])
    se = np.array(smm_optimum["se"])
    idx = list(PARAM_NAMES).index("beta_renter")
    ratio = float(se[idx] / max(abs(theta[idx]), 1e-9))
    # Document but do not assert the strict expectation; the design
    # intent is that beta_renter is the held-out parameter.
    assert ratio > 0.25, \
        f"beta_renter unexpectedly tightly identified (SE/|theta| = {ratio:.3f}). " \
        f"Reconsider whether the renter-owner gap should be in calibration."


def test_scenario_e_peak_survives_at_smm_optimum(smm_optimum):
    """The structural finding must survive identification. At the SMM
    optimum parameterisation, the integrated material-security
    intervention (Scenario E) must still produce a vote-share peak
    followed by decay. Tested on a 5-seed mean to keep the test fast;
    the full 10-seed counterfactual is in scripts/."""
    theta = np.array(smm_optimum["theta_hat"])
    base = apply_params(Config(n_periods=25), theta)
    # Activate Scenario E configuration on top of the SMM optimum.
    cfg_e = replace(
        base,
        voting=replace(base.voting, beta_0=-3.5),
        policy=replace(
            base.policy,
            rent_cap_leakage=0.40,
            supply_leakage=0.30,
            friction_leakage=0.50,
            redistribution_active=True,
            capital_tax_rate=0.027,
        ),
    )
    votes = []
    for seed in [73, 74, 75, 76, 77]:
        cfg_s = replace(cfg_e, seed=seed)
        _, hist, _ = simulate(cfg_s)
        votes.append(hist.vote_aggregate)
    mean_vote = np.mean(votes, axis=0)
    peak_idx = int(np.argmax(mean_vote))
    horizon = mean_vote.shape[0] - 1
    # Peak must be before the final period and must be followed by decay.
    assert peak_idx < horizon, \
        f"Scenario E does not peak before T={horizon}: peak at t={peak_idx} " \
        f"(mean trajectory monotone-increasing)"
    final_vote = float(mean_vote[-1])
    peak_vote = float(mean_vote[peak_idx])
    assert final_vote < peak_vote, \
        f"Scenario E does not decay after peak: peak={peak_vote:.3f}, " \
        f"final={final_vote:.3f}"


# ---------------------------------------------------------------------------
# SMM optimum loader / applier / audit (Phase 1 infrastructure)
# ---------------------------------------------------------------------------


def _require_optimum_artefact() -> None:
    if not OPTIMUM_PATH.exists():
        pytest.skip(f"SMM optimum artefact not present at {OPTIMUM_PATH}")


def test_load_smm_optimum_reads_real_artefact():
    _require_optimum_artefact()
    opt = load_smm_optimum()
    assert isinstance(opt, SMMOptimum)
    assert opt.source_path == OPTIMUM_PATH
    assert opt.theta_hat.shape == (len(opt.param_names),)
    assert set(opt.param_names).issubset(set(PARAM_NAMES) | {"beta_network", "gamma_cosmopolitan"})
    # Sanity: the artefact's theta_hat agrees with the raw dict.
    raw = json.loads(OPTIMUM_PATH.read_text())
    np.testing.assert_allclose(opt.theta_hat, np.asarray(raw["theta_hat"]))
    assert opt.param_names == tuple(raw["param_names"])


def test_load_smm_optimum_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_smm_optimum(tmp_path / "does_not_exist.json")


def test_load_smm_optimum_missing_required_fields(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"theta_hat": [1.0]}))  # no param_names
    with pytest.raises(ValueError, match="param_names"):
        load_smm_optimum(bad)


def test_load_smm_optimum_shape_mismatch(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"theta_hat": [1.0, 2.0], "param_names": ["beta_dissat"]}))
    with pytest.raises(ValueError, match="shape"):
        load_smm_optimum(bad)


def test_load_smm_optimum_unknown_param_name(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"theta_hat": [1.0], "param_names": ["not_a_real_param"]}))
    with pytest.raises(ValueError, match="param space"):
        load_smm_optimum(bad)


def test_apply_smm_optimum_matches_apply_params():
    _require_optimum_artefact()
    opt = load_smm_optimum()
    cfg_via_apply_smm = apply_smm_optimum(Config(), opt)
    # apply_params is positional over the 8-parameter PARAM_SPACE; this
    # only matches when the artefact's param_names are the canonical
    # 8 in canonical order (which the production artefact is).
    if tuple(opt.param_names) == PARAM_NAMES:
        cfg_via_apply_params = apply_params(Config(), opt.theta_hat)
        # Both configs should be identical on every calibrated field.
        from abmhp.estimation.smm import _PARAM_GETTERS  # type: ignore[attr-defined]
        for name in PARAM_NAMES:
            a = _PARAM_GETTERS[name](cfg_via_apply_smm)
            b = _PARAM_GETTERS[name](cfg_via_apply_params)
            assert a == b, f"{name}: apply_smm_optimum={a}, apply_params={b}"


def test_apply_smm_optimum_is_name_keyed(tmp_path):
    """apply_smm_optimum must dispatch by parameter name, not position;
    a permuted artefact still produces the right Config."""
    _require_optimum_artefact()
    opt = load_smm_optimum()
    perm = list(range(len(opt.param_names)))
    # Reverse the order; name-keyed apply must still produce the same Config.
    perm.reverse()
    permuted = tmp_path / "permuted.json"
    permuted.write_text(
        json.dumps(
            {
                "theta_hat": [float(opt.theta_hat[i]) for i in perm],
                "param_names": [opt.param_names[i] for i in perm],
            }
        )
    )
    cfg_canonical = apply_smm_optimum(Config(), opt)
    cfg_permuted = apply_smm_optimum(Config(), permuted)
    from abmhp.estimation.smm import _PARAM_GETTERS  # type: ignore[attr-defined]
    for name in opt.param_names:
        assert _PARAM_GETTERS[name](cfg_canonical) == _PARAM_GETTERS[name](cfg_permuted)


def test_audit_default_config_is_fully_detached():
    """A fresh Config() differs from the SMM optimum on at least one
    parameter whose optimum is interior to the bounds (beta_dissat,
    price_slope, beta_0 are all interior at the production optimum).
    The audit must classify this as FULLY DETACHED."""
    _require_optimum_artefact()
    report = audit_config_against_optimum(Config())
    assert report.status == AUDIT_FULLY_DETACHED
    # And the report must include every calibrated parameter.
    assert len(report.rows) == len(load_smm_optimum().param_names)


def test_audit_applied_optimum_is_at_optimum():
    _require_optimum_artefact()
    cfg = apply_smm_optimum(Config())
    report = audit_config_against_optimum(cfg)
    assert report.status == AUDIT_AT_OPTIMUM
    assert all(r.matches for r in report.rows)
    assert all(r.abs_diff == 0.0 for r in report.rows)


def test_audit_bound_only_detached_classification(tmp_path):
    """Construct a synthetic optimum whose only detached parameter sits at
    a bound. The audit must classify the default Config as
    BOUND-ONLY DETACHED rather than FULLY DETACHED."""
    # Build a synthetic artefact: theta_hat agrees with the default
    # Config on every parameter except one, which is set to a bound value
    # the default Config does not have. Use beta_renter: optimum at low
    # bound 0.2, default Config has 0.5.
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_text(
        json.dumps(
            {
                "theta_hat": [0.2],
                "param_names": ["beta_renter"],
            }
        )
    )
    report = audit_config_against_optimum(Config(), synthetic)
    assert report.status == AUDIT_BOUND_ONLY_DETACHED
    assert len(report.rows) == 1
    assert report.rows[0].name == "beta_renter"
    assert report.rows[0].optimum_at_bound is True
    assert not report.rows[0].matches


def test_audit_status_constants_are_distinct():
    assert AUDIT_AT_OPTIMUM != AUDIT_BOUND_ONLY_DETACHED
    assert AUDIT_AT_OPTIMUM != AUDIT_FULLY_DETACHED
    assert AUDIT_BOUND_ONLY_DETACHED != AUDIT_FULLY_DETACHED
