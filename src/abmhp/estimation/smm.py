"""Two-stage SMM estimation of the bundled-mechanism ABM.

Standard GMM machinery applied to a simulation-based moment vector.

  Stage 0: estimate moment variance by independent multi-seed replication
           at the central configuration.
  Stage 1: diagonal weighting matrix W_1 = diag(weight_i / variance_i);
           Bayesian-optimisation search via scikit-optimize gp_minimize
           with Sobol initial points.
  Stage 2: optimal weighting matrix W_2 = Sigma^{-1} where Sigma is the
           covariance of moment errors at the stage-1 optimum; warm-
           started gp_minimize.
  J-statistic: J = g(theta_hat)' W_2 g(theta_hat); under correct
           specification distributed as chi-squared with K - P degrees
           of freedom where K is the number of moments and P the number
           of free parameters.

Standard errors via the asymptotic sandwich formula:
  cov(theta_hat) = (G' W G)^{-1} G' W Sigma W G (G' W G)^{-1} / N_eff
where G is the moment Jacobian computed by central differences and
N_eff is the number of independent simulation replications used to
estimate Sigma.

The 8-parameter free space and the moment table are documented in
`outputs/moments_table.md`. The results writeup is in
`outputs/smm_results.md`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..config import Config
from .moments import (
    CALIBRATION_MOMENTS,
    VALIDATION_MOMENTS,
    Moment,
    evaluate_moments,
    simulate_seeds,
)


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    apply: Callable[[Config, float], Config] = field(repr=False)


def _set_voting(field_name: str) -> Callable[[Config, float], Config]:
    def _apply(cfg: Config, v: float) -> Config:
        return replace(cfg, voting=replace(cfg.voting, **{field_name: v}))
    return _apply


def _set_regional(field_name: str) -> Callable[[Config, float], Config]:
    def _apply(cfg: Config, v: float) -> Config:
        return replace(cfg, regional=replace(cfg.regional, **{field_name: v}))
    return _apply


def _set_demographic(field_name: str) -> Callable[[Config, float], Config]:
    def _apply(cfg: Config, v: float) -> Config:
        return replace(cfg, demographic=replace(cfg.demographic, **{field_name: v}))
    return _apply


# Read counterparts of the setters above. Used by the audit machinery
# (load_smm_optimum / audit_config_against_optimum) to extract the
# current value of a calibrated parameter from a runtime Config without
# round-tripping through apply_params.

def _get_voting(field_name: str) -> Callable[[Config], float]:
    def _read(cfg: Config) -> float:
        return float(getattr(cfg.voting, field_name))
    return _read


def _get_regional(field_name: str) -> Callable[[Config], float]:
    def _read(cfg: Config) -> float:
        return float(getattr(cfg.regional, field_name))
    return _read


def _get_demographic(field_name: str) -> Callable[[Config], float]:
    def _read(cfg: Config) -> float:
        return float(getattr(cfg.demographic, field_name))
    return _read


# Map from SMM parameter name to a getter that reads its value from a
# Config. The keys are exactly the names in _BASE_PARAM_SPACE plus the
# augmented-model extensions. This is the read-side companion to the
# ParamSpec.apply closures below; keeping it as a separate table avoids
# changing the ParamSpec record type (which is consumed by external
# callers via PARAM_SPACE / PARAM_NAMES) and avoids fragile closure
# introspection in the audit code.
_PARAM_GETTERS: dict[str, Callable[[Config], float]] = {
    "beta_dissat": _get_voting("beta_dissat"),
    "beta_renter": _get_voting("beta_renter"),
    "rho_aspiration": _get_voting("rho_aspiration"),
    "alpha_local": _get_voting("alpha_local"),
    "price_slope": _get_regional("price_slope"),
    "beta_0": _get_voting("beta_0"),
    "assortative_exponent": _get_demographic("assortative_exponent"),
    "intergenerational_skill_corr": _get_demographic("intergenerational_skill_corr"),
    "beta_network": _get_voting("beta_network"),
    "gamma_cosmopolitan": _get_voting("gamma_cosmopolitan"),
}


_BASE_PARAM_SPACE: tuple[ParamSpec, ...] = (
    # beta_dissat upper bound widened from 9.0 to 12.0 after Phase 3 reduced
    # had it pinned at the boundary; the wider bound lets the SMM find a
    # less-constrained optimum if it wants one.
    ParamSpec("beta_dissat", 3.0, 12.0, _set_voting("beta_dissat")),
    ParamSpec("beta_renter", 0.2, 1.0, _set_voting("beta_renter")),
    ParamSpec("rho_aspiration", 0.40, 0.90, _set_voting("rho_aspiration")),
    ParamSpec("alpha_local", 0.20, 0.70, _set_voting("alpha_local")),
    ParamSpec("price_slope", 0.04, 0.15, _set_regional("price_slope")),
    ParamSpec("beta_0", -7.0, -3.0, _set_voting("beta_0")),
    ParamSpec("assortative_exponent", 1.5, 3.0, _set_demographic("assortative_exponent")),
    ParamSpec("intergenerational_skill_corr", 0.30, 0.85,
              _set_demographic("intergenerational_skill_corr")),
)

# Augmented-model extension parameters; included in the SMM free-parameter
# space only when the corresponding flag in BehavioralConfig is True. When
# the flags are False (the defaults), build_param_space(cfg) returns
# exactly _BASE_PARAM_SPACE and the SMM behaves as before.
_BETA_N_PARAM = ParamSpec("beta_network", 0.3, 1.5, _set_voting("beta_network"))
_GAMMA_COSMO_PARAM = ParamSpec(
    "gamma_cosmopolitan", -15.0, 5.0, _set_voting("gamma_cosmopolitan")
)


def build_param_space(cfg: Config) -> tuple[ParamSpec, ...]:
    """Return the SMM free-parameter space implied by the cfg flags."""
    extras: list[ParamSpec] = []
    if cfg.behavioral.estimate_beta_n:
        extras.append(_BETA_N_PARAM)
    if cfg.behavioral.estimate_gamma_cosmopolitan:
        extras.append(_GAMMA_COSMO_PARAM)
    return _BASE_PARAM_SPACE + tuple(extras)


# Module-level constants for the default 8-parameter space. Callers that
# explicitly want the augmented space call `build_param_space(cfg)`.
PARAM_SPACE: tuple[ParamSpec, ...] = _BASE_PARAM_SPACE
PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PARAM_SPACE)
PARAM_LOWS: np.ndarray = np.array([p.low for p in PARAM_SPACE])
PARAM_HIGHS: np.ndarray = np.array([p.high for p in PARAM_SPACE])


def apply_params(
    base_cfg: Config,
    theta: np.ndarray,
    param_space: tuple[ParamSpec, ...] | None = None,
) -> Config:
    """Apply a theta vector to a base config. When `param_space` is None
    the default 8-parameter PARAM_SPACE is used (preserves the current
    apply_params semantics for every existing caller)."""
    if param_space is None:
        param_space = PARAM_SPACE
    cfg = base_cfg
    for ps, v in zip(param_space, theta):
        cfg = ps.apply(cfg, float(v))
    return cfg


def project_to_bounds(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, PARAM_LOWS, PARAM_HIGHS)


# ---------------------------------------------------------------------------
# Moment vector and residuals
# ---------------------------------------------------------------------------


def simulated_moments(
    theta: np.ndarray,
    base_cfg: Config,
    seeds: list[int],
    param_space: tuple[ParamSpec, ...] | None = None,
) -> np.ndarray:
    cfg = apply_params(base_cfg, theta, param_space=param_space)
    runs = simulate_seeds(cfg, seeds)
    sim = evaluate_moments(CALIBRATION_MOMENTS, runs)
    return np.array([sim[m.name] for m in CALIBRATION_MOMENTS])


def moment_targets() -> np.ndarray:
    return np.array([m.value for m in CALIBRATION_MOMENTS])


def moment_residuals(
    theta: np.ndarray,
    base_cfg: Config,
    seeds: list[int],
    param_space: tuple[ParamSpec, ...] | None = None,
) -> np.ndarray:
    return simulated_moments(theta, base_cfg, seeds, param_space=param_space) - moment_targets()


def objective(
    theta: np.ndarray,
    base_cfg: Config,
    seeds: list[int],
    W: np.ndarray,
    param_space: tuple[ParamSpec, ...] | None = None,
) -> float:
    g = moment_residuals(theta, base_cfg, seeds, param_space=param_space)
    return float(g @ W @ g)


# ---------------------------------------------------------------------------
# Weighting matrices
# ---------------------------------------------------------------------------


def estimate_moment_covariance(
    cfg: Config,
    n_replications: int,
    seeds_per_replication: int = 5,
    seed_offset: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (variance, covariance) of the simulated moment vector at cfg.

    Runs `n_replications` independent multi-seed simulations and returns
    the per-moment variance and full moment covariance matrix."""
    rows = []
    for i in range(n_replications):
        seeds = list(range(seed_offset + i * seeds_per_replication,
                           seed_offset + (i + 1) * seeds_per_replication))
        runs = simulate_seeds(cfg, seeds)
        sim = evaluate_moments(CALIBRATION_MOMENTS, runs)
        rows.append(np.array([sim[m.name] for m in CALIBRATION_MOMENTS]))
    M = np.array(rows)
    return M.var(axis=0, ddof=1), np.cov(M, rowvar=False, ddof=1)


def diagonal_weighting_matrix(variance: np.ndarray) -> np.ndarray:
    weights = np.array([m.weight for m in CALIBRATION_MOMENTS])
    return np.diag(weights / np.maximum(variance, 1e-12))


def optimal_weighting_matrix(covariance: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(covariance)


# ---------------------------------------------------------------------------
# Jacobian (numerical central differences)
# ---------------------------------------------------------------------------


def moment_jacobian(
    theta: np.ndarray,
    base_cfg: Config,
    seeds: list[int],
    rel_step: float = 0.05,
    param_space: tuple[ParamSpec, ...] | None = None,
) -> np.ndarray:
    """Central-difference Jacobian of simulated moments with respect to
    parameters. Step size is `rel_step` times the parameter bound range.
    Returns array of shape (K_moments, P_params)."""
    if param_space is None:
        param_space = PARAM_SPACE
    K = len(CALIBRATION_MOMENTS)
    P = len(param_space)
    J = np.zeros((K, P))
    for j, ps in enumerate(param_space):
        h = rel_step * (ps.high - ps.low)
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] = min(theta[j] + h, ps.high)
        theta_minus[j] = max(theta[j] - h, ps.low)
        actual_h = theta_plus[j] - theta_minus[j]
        if actual_h <= 0:
            continue
        m_plus = simulated_moments(theta_plus, base_cfg, seeds, param_space=param_space)
        m_minus = simulated_moments(theta_minus, base_cfg, seeds, param_space=param_space)
        J[:, j] = (m_plus - m_minus) / actual_h
    return J


# ---------------------------------------------------------------------------
# SMM result container
# ---------------------------------------------------------------------------


@dataclass
class SMMResult:
    theta_hat: np.ndarray
    se: np.ndarray
    j_statistic: float
    dof: int
    p_value: float
    moment_targets: np.ndarray
    moment_at_optimum: np.ndarray
    moment_residuals_at_optimum: np.ndarray
    weighting_matrix_stage1: np.ndarray
    weighting_matrix_stage2: np.ndarray
    covariance_at_optimum: np.ndarray
    jacobian: np.ndarray
    parameter_covariance: np.ndarray
    stage1_path_x: list
    stage1_path_y: list
    stage2_path_x: list
    stage2_path_y: list
    n_seeds_per_eval: int
    n_first_stage: int
    n_second_stage: int


# ---------------------------------------------------------------------------
# Top-level SMM routine
# ---------------------------------------------------------------------------


def run_smm(
    base_cfg: Config | None = None,
    n_seeds_per_eval: int = 5,
    seeds_offset: int = 73,
    n_first_stage: int = 100,
    n_second_stage: int = 50,
    n_variance_replications: int = 20,
    random_state: int = 42,
    verbose: bool = True,
    param_space: tuple[ParamSpec, ...] | None = None,
    n_initial_points_stage1: int = 20,
    n_initial_points_stage2: int = 10,
    x0_stage1: list[float] | None = None,
) -> SMMResult:
    """Run the two-stage SMM. Returns an SMMResult with parameter estimates,
    standard errors, J-statistic, and intermediate matrices.

    `param_space` defaults to the module-level 8-parameter PARAM_SPACE, but
    callers can pass `build_param_space(cfg)` to enable the augmented
    estimation with beta_n and/or gamma_cosmopolitan. `x0_stage1` is an
    optional Stage 1 warm-start point (length = len(param_space)); it is
    evaluated in addition to the `n_initial_points_stage1` Sobol points."""
    from skopt import gp_minimize
    from skopt.space import Real
    from scipy import stats

    if base_cfg is None:
        base_cfg = Config()

    if param_space is None:
        param_space = PARAM_SPACE

    seeds = list(range(seeds_offset, seeds_offset + n_seeds_per_eval))

    if verbose:
        print(f"SMM starting. K = {len(CALIBRATION_MOMENTS)} moments, "
              f"P = {len(param_space)} parameters, seeds = {seeds}")
        print(f"Stage 0: estimating moment variance "
              f"({n_variance_replications} replications)")

    variances, _ = estimate_moment_covariance(
        base_cfg, n_replications=n_variance_replications,
        seeds_per_replication=n_seeds_per_eval, seed_offset=2000,
    )
    W1 = diagonal_weighting_matrix(variances)

    if verbose:
        print(f"Stage 1: gp_minimize, n_calls = {n_first_stage}, "
              f"Sobol init = {n_initial_points_stage1}"
              + (f", warm-start x0 = {x0_stage1}" if x0_stage1 is not None else ""))

    space = [Real(ps.low, ps.high, name=ps.name) for ps in param_space]
    stage1_x: list = []
    stage1_y: list = []

    def stage1_obj(theta_list):
        theta = np.array(theta_list)
        val = objective(theta, base_cfg, seeds, W1, param_space=param_space)
        stage1_x.append(list(theta_list))
        stage1_y.append(val)
        if verbose and len(stage1_y) % 10 == 0:
            print(f"  stage1 iter {len(stage1_y):>3d}/{n_first_stage}: "
                  f"obj = {val:.5e}, best = {min(stage1_y):.5e}")
        return val

    stage1_kwargs: dict = dict(
        n_calls=n_first_stage,
        n_initial_points=min(n_initial_points_stage1, n_first_stage - 1),
        initial_point_generator="sobol",
        random_state=random_state,
        verbose=False,
    )
    if x0_stage1 is not None:
        stage1_kwargs["x0"] = [list(x0_stage1)]
    res1 = gp_minimize(stage1_obj, space, **stage1_kwargs)
    theta1 = np.array(res1.x)

    if verbose:
        print(f"Stage 1 optimum: obj = {res1.fun:.5e}")
        for ps, v in zip(param_space, theta1):
            print(f"  {ps.name:30s} = {v:+.4f}")

    if verbose:
        print(f"Stage 2: estimating moment covariance at stage-1 optimum")
    cfg1 = apply_params(base_cfg, theta1, param_space=param_space)
    _, cov_at_theta1 = estimate_moment_covariance(
        cfg1, n_replications=n_variance_replications,
        seeds_per_replication=n_seeds_per_eval, seed_offset=3000,
    )
    W2 = optimal_weighting_matrix(cov_at_theta1)

    if verbose:
        print(f"Stage 2: gp_minimize, n_calls = {n_second_stage}, "
              f"warm start at stage-1 optimum")

    stage2_x: list = []
    stage2_y: list = []

    def stage2_obj(theta_list):
        theta = np.array(theta_list)
        val = objective(theta, base_cfg, seeds, W2, param_space=param_space)
        stage2_x.append(list(theta_list))
        stage2_y.append(val)
        if verbose and len(stage2_y) % 10 == 0:
            print(f"  stage2 iter {len(stage2_y):>3d}/{n_second_stage}: "
                  f"obj = {val:.5e}, best = {min(stage2_y):.5e}")
        return val

    res2 = gp_minimize(
        stage2_obj, space,
        n_calls=n_second_stage,
        n_initial_points=min(n_initial_points_stage2, n_second_stage - 1),
        initial_point_generator="sobol",
        random_state=random_state + 1,
        x0=list(theta1),
        verbose=False,
    )
    theta_hat = np.array(res2.x)

    if verbose:
        print(f"Stage 2 optimum: obj = {res2.fun:.5e}")

    # J-statistic.
    g_hat = moment_residuals(theta_hat, base_cfg, seeds, param_space=param_space)
    m_at_opt = simulated_moments(theta_hat, base_cfg, seeds, param_space=param_space)
    j_stat = float(g_hat @ W2 @ g_hat)
    dof = len(CALIBRATION_MOMENTS) - len(param_space)
    p_value = float(1.0 - stats.chi2.cdf(j_stat, dof))

    # Jacobian and standard errors.
    if verbose:
        print("Computing moment Jacobian (central differences)")
    G = moment_jacobian(theta_hat, base_cfg, seeds, param_space=param_space)

    bread = np.linalg.pinv(G.T @ W2 @ G)
    meat = G.T @ W2 @ cov_at_theta1 @ W2 @ G
    cov_theta = bread @ meat @ bread / max(n_variance_replications, 1)
    se = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))

    return SMMResult(
        theta_hat=theta_hat,
        se=se,
        j_statistic=j_stat,
        dof=dof,
        p_value=p_value,
        moment_targets=moment_targets(),
        moment_at_optimum=m_at_opt,
        moment_residuals_at_optimum=g_hat,
        weighting_matrix_stage1=W1,
        weighting_matrix_stage2=W2,
        covariance_at_optimum=cov_at_theta1,
        jacobian=G,
        parameter_covariance=cov_theta,
        stage1_path_x=stage1_x,
        stage1_path_y=stage1_y,
        stage2_path_x=stage2_x,
        stage2_path_y=stage2_y,
        n_seeds_per_eval=n_seeds_per_eval,
        n_first_stage=n_first_stage,
        n_second_stage=n_second_stage,
    )


# ---------------------------------------------------------------------------
# Post-estimation: validation moments
# ---------------------------------------------------------------------------


def score_validation_moments(
    theta: np.ndarray,
    base_cfg: Config,
    seeds: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate validation moments at the identified parameterisation.

    Returns a dict keyed by moment name with target, model, error, and
    a pass flag indicating whether the error is within target_tolerance."""
    if seeds is None:
        seeds = list(range(73, 78))
    cfg = apply_params(base_cfg, theta)
    runs = simulate_seeds(cfg, seeds)
    sim = evaluate_moments(VALIDATION_MOMENTS, runs)
    out: dict[str, dict[str, float]] = {}
    for m in VALIDATION_MOMENTS:
        model = sim[m.name]
        err = float("nan") if not np.isfinite(model) else float(model - m.value)
        passed = (
            bool(np.isfinite(err) and abs(err) <= m.target_tolerance)
            if np.isfinite(err) else False
        )
        out[m.name] = {
            "target": float(m.value),
            "model": float(model),
            "error": err,
            "tolerance": float(m.target_tolerance),
            "passed": passed,
        }
    return out


# ---------------------------------------------------------------------------
# Sensitivity Jacobian figure
# ---------------------------------------------------------------------------


def plot_sensitivity_jacobian(result: SMMResult, out_path: Path) -> None:
    """Heatmap of standardised moment sensitivities at the SMM optimum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Standardise the Jacobian: each column scaled by parameter range,
    # each row scaled by moment target tolerance. The result is unitless
    # and shows "how much moment k moves when parameter p moves by one
    # SE-scaled unit".
    J = result.jacobian.copy()
    param_ranges = PARAM_HIGHS - PARAM_LOWS
    moment_scales = np.array([m.target_tolerance for m in CALIBRATION_MOMENTS])

    J_std = (J * param_ranges[None, :]) / np.maximum(moment_scales[:, None], 1e-6)

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = float(np.percentile(np.abs(J_std), 95))
    im = ax.imshow(J_std, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(PARAM_SPACE)))
    ax.set_xticklabels([p.name for p in PARAM_SPACE], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(CALIBRATION_MOMENTS)))
    ax.set_yticklabels([m.name for m in CALIBRATION_MOMENTS], fontsize=8)
    ax.set_title("Sensitivity Jacobian: standardised d(moment) / d(parameter range)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("standardised sensitivity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SMM optimum loader / applier / audit
# ---------------------------------------------------------------------------


DEFAULT_OPTIMUM_PATH: Path = (
    Path(__file__).resolve().parents[3] / "outputs" / "smm_optimum.json"
)


@dataclass(frozen=True)
class SMMOptimum:
    """Subset of the `outputs/smm_optimum.json` artefact needed to drive a
    runtime Config to the SMM point estimate. Carries the param-name /
    theta pairing explicitly so that callers do not depend on positional
    alignment with module-level PARAM_SPACE (which may differ from the
    artefact's param_space when the artefact predates an augmented-model
    extension, or vice-versa).
    """

    theta_hat: np.ndarray
    param_names: tuple[str, ...]
    source_path: Path
    raw: Mapping[str, Any]

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.param_names, self.theta_hat)}


def load_smm_optimum(path: Path | str | None = None) -> SMMOptimum:
    """Read the SMM optimum artefact from disk.

    The artefact format is the one produced by `scripts/run_smm.py`:
    a JSON object with at least `theta_hat` and `param_names` arrays of
    equal length. Other fields (`se`, `j_statistic`, `moment_*`, ...) are
    preserved in `SMMOptimum.raw` for callers that want them.

    Raises FileNotFoundError if the artefact is absent, and ValueError if
    the artefact is structurally malformed (missing or mismatched
    theta_hat / param_names).
    """
    p = Path(path) if path is not None else DEFAULT_OPTIMUM_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"SMM optimum artefact not found at {p}. Run scripts/run_smm.py "
            f"to produce it, or pass an explicit path."
        )
    with p.open() as f:
        data = json.load(f)
    if "theta_hat" not in data or "param_names" not in data:
        raise ValueError(
            f"SMM optimum at {p} is missing required fields "
            f"theta_hat / param_names."
        )
    theta = np.asarray(data["theta_hat"], dtype=float)
    names = tuple(data["param_names"])
    if theta.shape != (len(names),):
        raise ValueError(
            f"SMM optimum at {p}: theta_hat has shape {theta.shape}, "
            f"expected ({len(names)},) to match param_names."
        )
    unknown = [n for n in names if n not in _PARAM_GETTERS]
    if unknown:
        raise ValueError(
            f"SMM optimum at {p} references parameter names not present "
            f"in the SMM param space: {unknown}. Update _PARAM_GETTERS and "
            f"_BASE_PARAM_SPACE / augmented extensions in smm.py."
        )
    return SMMOptimum(theta_hat=theta, param_names=names, source_path=p, raw=data)


def _build_apply_lookup() -> dict[str, Callable[[Config, float], Config]]:
    """Map parameter name to the apply closure defined in _BASE_PARAM_SPACE
    plus the augmented-model extensions. This is the write-side companion
    to _PARAM_GETTERS and is used by apply_smm_optimum to apply a theta
    by name rather than by position."""
    table: dict[str, Callable[[Config, float], Config]] = {}
    for ps in _BASE_PARAM_SPACE:
        table[ps.name] = ps.apply
    table[_BETA_N_PARAM.name] = _BETA_N_PARAM.apply
    table[_GAMMA_COSMO_PARAM.name] = _GAMMA_COSMO_PARAM.apply
    return table


_APPLY_BY_NAME: dict[str, Callable[[Config, float], Config]] = _build_apply_lookup()


def apply_smm_optimum(
    base_cfg: Config | None = None,
    optimum: SMMOptimum | Path | str | None = None,
) -> Config:
    """Return a Config with the SMM optimum applied to the given base.

    If `base_cfg` is None, a fresh `Config()` is used. If `optimum` is
    None, the artefact at DEFAULT_OPTIMUM_PATH is loaded. The two-step
    form (load_smm_optimum then apply_smm_optimum) is preferred when the
    same optimum drives several entry points; the one-shot form
    (apply_smm_optimum() with both arguments defaulted) is supported as
    a convenience for short scripts.

    Application is name-keyed, not position-keyed, so an artefact that
    omits an augmented-model parameter leaves that parameter at its
    base_cfg default rather than silently mis-applying a theta entry.
    """
    if base_cfg is None:
        base_cfg = Config()
    if optimum is None or isinstance(optimum, (str, Path)):
        optimum = load_smm_optimum(optimum)
    cfg = base_cfg
    for name, value in optimum.as_dict().items():
        cfg = _APPLY_BY_NAME[name](cfg, value)
    return cfg


# Status codes for the parameter-vector audit. See
# `scripts/audit_parameter_vector.py` for the user-facing driver.
AUDIT_AT_OPTIMUM = "AT OPTIMUM"
AUDIT_BOUND_ONLY_DETACHED = "BOUND-ONLY DETACHED"
AUDIT_FULLY_DETACHED = "FULLY DETACHED"


@dataclass(frozen=True)
class ParamAuditRow:
    name: str
    optimum_value: float
    current_value: float
    abs_diff: float
    optimum_at_bound: bool  # whether optimum sits at the SMM low/high bound
    matches: bool


@dataclass(frozen=True)
class ParamAuditReport:
    rows: tuple[ParamAuditRow, ...]
    status: str  # one of AUDIT_AT_OPTIMUM / _BOUND_ONLY_DETACHED / _FULLY_DETACHED
    optimum_source: Path


def audit_config_against_optimum(
    cfg: Config,
    optimum: SMMOptimum | Path | str | None = None,
    tolerance: float = 1e-6,
    bound_tolerance: float = 1e-9,
) -> ParamAuditReport:
    """Compare a runtime Config against the SMM optimum and classify.

    For each calibrated parameter, the audit reports
        optimum_value      — value in outputs/smm_optimum.json
        current_value      — value currently held by `cfg`
        abs_diff           — |optimum - current|
        optimum_at_bound   — whether the optimum sits at the SMM low or
                             high bound (some SMM components do; this is
                             a property of the optimum, not of cfg)
        matches            — abs_diff < tolerance

    The overall status follows:
        AT OPTIMUM           — every parameter matches.
        BOUND-ONLY DETACHED  — at least one parameter mismatches, but
                               every mismatching parameter has its
                               optimum value sitting at a bound. The
                               entry point is still arguably broken
                               (it does not load the optimum) but the
                               numerical impact is at most a small
                               bound-clipping error.
        FULLY DETACHED       — at least one mismatching parameter has
                               its optimum interior to the bounds. The
                               entry point is definitely not loading
                               the SMM optimum.
    """
    if optimum is None or isinstance(optimum, (str, Path)):
        optimum = load_smm_optimum(optimum)
    # Build a name -> (low, high) table from the augmented param space so
    # we can ask "is the optimum at a bound?" without re-importing.
    bounds: dict[str, tuple[float, float]] = {ps.name: (ps.low, ps.high) for ps in _BASE_PARAM_SPACE}
    bounds[_BETA_N_PARAM.name] = (_BETA_N_PARAM.low, _BETA_N_PARAM.high)
    bounds[_GAMMA_COSMO_PARAM.name] = (_GAMMA_COSMO_PARAM.low, _GAMMA_COSMO_PARAM.high)

    rows: list[ParamAuditRow] = []
    for name, theta in optimum.as_dict().items():
        current = _PARAM_GETTERS[name](cfg)
        diff = abs(theta - current)
        low, high = bounds[name]
        at_bound = (
            abs(theta - low) <= bound_tolerance or abs(theta - high) <= bound_tolerance
        )
        rows.append(
            ParamAuditRow(
                name=name,
                optimum_value=theta,
                current_value=current,
                abs_diff=diff,
                optimum_at_bound=at_bound,
                matches=diff < tolerance,
            )
        )

    mismatched = [r for r in rows if not r.matches]
    if not mismatched:
        status = AUDIT_AT_OPTIMUM
    elif all(r.optimum_at_bound for r in mismatched):
        status = AUDIT_BOUND_ONLY_DETACHED
    else:
        status = AUDIT_FULLY_DETACHED

    return ParamAuditReport(
        rows=tuple(rows), status=status, optimum_source=optimum.source_path
    )
