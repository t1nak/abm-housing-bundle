"""Material-security counterfactual: the paper's central housing-policy result.

Five scenarios, 25 periods, 10-seed replication:

  A. baseline (no extreme-share activation; incumbency_threshold=1.0)
  B. housing-only mainstream response, leak-free
  C. housing-only mainstream response, leaky (central Barcelona-approx)
  D. rhetorical extreme-share governance (no policy intensity)
  E. integrated material-security intervention (housing + redistribution)

To force regime activation in B, C, D, E, beta_0 is shifted to -3.5 (the
calibration used in test_extreme_share_regime_activates_above_threshold).
The counterfactual is conditional on the extreme-share activation regime
actually occurring. Scenario A keeps the default beta_0 = -5.2 as the
pre-activation baseline.

Outputs:
  outputs/fig_material_security_scenarios.png
  outputs/fig_robustness_leakage.png
  outputs/material_security_results.md
"""
from __future__ import annotations

import sys
from dataclasses import replace, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abmhp import Config, PolicyRegime, simulate
from abmhp.config import PolicyConfig, VotingConfig
from abmhp.estimation.smm import apply_smm_optimum

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

T_HORIZON = 25
SEEDS = list(range(73, 83))
SCENARIO_ORDER = ["A", "B", "C", "D", "E"]
SCENARIO_LABEL = {
    "A": "A. Baseline (no extreme-share activation)",
    "B": "B. Housing-only mainstream response, leak-free",
    "C": "C. Housing-only mainstream response, leaky (central)",
    "D": "D. Rhetorical extreme-share governance (no policy)",
    "E": "E. Integrated material-security intervention",
}
SCENARIO_COLOR = {
    "A": "#444444",
    "B": "#1a5f7a",
    "C": "#a8271c",
    "D": "#8b6914",
    "E": "#2a6e5a",
}

TAU_K = 0.027  # calibrated to 5.0 percent of aggregate income, 10-seed mean

DEFAULT_LEAKAGES = {"rent_cap_leakage": 0.4, "supply_leakage": 0.3, "friction_leakage": 0.5}


@dataclass(frozen=True)
class LeakageProfile:
    rent_cap: float
    supply: float
    friction: float
    label: str

LOW = LeakageProfile(0.20, 0.15, 0.25, "low")
MEDIUM = LeakageProfile(0.40, 0.30, 0.50, "medium")
HIGH = LeakageProfile(0.60, 0.50, 0.70, "high")


def make_config(
    scenario: str,
    seed: int,
    leakage: LeakageProfile = MEDIUM,
    n_periods: int = T_HORIZON,
) -> Config:
    """Construct one of the five scenarios at a given seed and leakage."""
    cfg = apply_smm_optimum(Config(seed=seed, n_periods=n_periods))
    if scenario == "A":
        # Pre-activation baseline. Threshold = 1.0 prevents activation even
        # if the extreme-share vote were to climb past 0.3.
        cfg = replace(cfg, policy=replace(cfg.policy, incumbency_threshold=1.0))
        return cfg
    # B, C, D, E: shift beta_0 so the extreme-share vote crosses the
    # activation threshold.
    cfg = replace(cfg, voting=replace(cfg.voting, beta_0=-3.5))
    if scenario == "B":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_leakage=0.0,
            supply_leakage=0.0,
            friction_leakage=0.0,
        ))
    elif scenario == "C":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_leakage=leakage.rent_cap,
            supply_leakage=leakage.supply,
            friction_leakage=leakage.friction,
        ))
    elif scenario == "D":
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_intensity=0.0,
            supply_restriction_intensity=0.0,
            transaction_friction=0.0,
        ))
    elif scenario == "E":
        # Integrated material-security intervention: housing-only mainstream
        # response at Scenario C's leakage profile, plus a capital tax of
        # tau_K with proceeds distributed lump-sum to the bottom 50 percent
        # of the wealth distribution. The combination addresses consumption
        # stress (rent), asset exclusion (cap_gain access), and the
        # broader material-security gap via transfers.
        cfg = replace(cfg, policy=replace(
            cfg.policy,
            rent_cap_leakage=leakage.rent_cap,
            supply_leakage=leakage.supply,
            friction_leakage=leakage.friction,
            redistribution_active=True,
            capital_tax_rate=TAU_K,
        ))
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return cfg


def aggregate_rent_burden(hist, cfg: Config) -> np.ndarray:
    """Population-weighted aggregate rent burden by period.

    rent burden = rent_per_renter / mean_regional_income, weighted by the
    renter headcount in each region. Real-equivalent because both numerator
    and denominator are nominal levels."""
    n_periods_plus_one = hist.price.shape[0]
    rent_yield = cfg.behavioral.rent_yield
    burden_share = cfg.behavioral.rent_burden_share
    initial_price = hist.price[0]
    renter_share = 1.0 - hist.ownership
    pop_share = cfg.regional.pop_share
    burden_series = np.zeros(n_periods_plus_one)
    for t in range(n_periods_plus_one):
        rent_per_renter = initial_price * rent_yield * hist.rent_index[t] * burden_share
        income_mean = np.where(hist.mean_income[t] > 0, hist.mean_income[t], 1.0)
        per_region_burden = rent_per_renter / income_mean
        renter_count_share = pop_share * renter_share[t]
        denom = renter_count_share.sum()
        if denom > 0:
            burden_series[t] = float((per_region_burden * renter_count_share).sum() / denom)
        else:
            burden_series[t] = float(per_region_burden.mean())
    return burden_series


def aggregate_dissat(hist, cfg: Config) -> np.ndarray:
    pop_share = cfg.regional.pop_share
    return (hist.dissat * pop_share[None, :]).sum(axis=1)


def bottom50_resource_trajectory(hist, cfg: Config) -> np.ndarray:
    """Approximation: mean transfer received per agent across the bottom
    50 percent of the wealth distribution, indexed against starting aggregate
    income per capita. Zero before any transfer is paid and positive once
    the integrated material-security intervention is active."""
    n_total = cfg.n_agents
    start_income_per_capita = hist.income_aggregate[0] / n_total if hist.income_aggregate[0] > 0 else 1.0
    if start_income_per_capita == 1.0:
        start_income_per_capita = float(hist.mean_income[0].mean())
    rec_share = cfg.policy.transfer_recipient_share
    per_recipient = hist.transfer_aggregate / max(n_total * rec_share, 1.0)
    return per_recipient / max(start_income_per_capita, 1.0)


def years_extreme_share(hist) -> int:
    """Count of periods in which the extreme-share activation regime governs."""
    return int(sum(1 for r in hist.regime[1:] if r is PolicyRegime.POPULIST))


def tenure_dissat_gap(hist, cfg: Config) -> float:
    """Final-period within-region tenure gap, approximated via the
    extreme-share vote-by-tenure gap at year T."""
    T = cfg.n_periods
    rent_v = hist.vote_by_tenure[T, :, 0].mean()
    own_v = hist.vote_by_tenure[T, :, 1].mean()
    return float(rent_v - own_v)


def run_scenario(scenario: str, leakage: LeakageProfile = MEDIUM, seeds: Iterable[int] = SEEDS):
    runs = []
    for s in seeds:
        cfg = make_config(scenario, s, leakage)
        _, hist, _ = simulate(cfg)
        runs.append((cfg, hist))
    return runs


def stack_metric(runs, fn) -> np.ndarray:
    arrays = [fn(h, c) if fn.__code__.co_argcount == 2 else fn(h) for c, h in runs]
    return np.stack(arrays, axis=0)


def compute_half_life(scenario_votes: np.ndarray, baseline_votes: np.ndarray) -> float | None:
    """Periods from the scenario's peak extreme-share vote until it returns
    to within 25 percent above the baseline vote share at that period.

    scenario_votes, baseline_votes: shape (T+1,). Returns None if not reached
    within the horizon (censored)."""
    peak_t = int(np.argmax(scenario_votes))
    target = baseline_votes * 1.25
    for t in range(peak_t, len(scenario_votes)):
        if scenario_votes[t] <= target[t]:
            return float(t - peak_t)
    return None


def half_life_summary(scenario_runs, baseline_runs) -> dict:
    """Per-seed half-life, summarised as (median, share_censored, mean_uncensored)."""
    half_lives = []
    censored = 0
    for (cb, hb), (cs, hs) in zip(baseline_runs, scenario_runs):
        baseline_votes = hb.vote_aggregate
        scenario_votes = hs.vote_aggregate
        hl = compute_half_life(scenario_votes, baseline_votes)
        if hl is None:
            censored += 1
            half_lives.append(np.nan)
        else:
            half_lives.append(hl)
    arr = np.array(half_lives, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "median": float(np.nanmedian(arr)) if finite.size else float("nan"),
        "mean_uncensored": float(finite.mean()) if finite.size else float("nan"),
        "sd_uncensored": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "share_censored": censored / len(half_lives),
        "n_censored": censored,
        "n_total": len(half_lives),
        "raw": half_lives,
    }


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })


def plot_band(ax, t, mean, lo, hi, color, label):
    ax.plot(t, mean, color=color, linewidth=2.2, label=label)
    ax.fill_between(t, lo, hi, color=color, alpha=0.15, linewidth=0)


def make_headline_figure(scenario_data: dict, out_path: Path) -> None:
    """Three-panel headline. Scenario E is the visual focus: bold green,
    annotated peak-and-decay, threshold lines un-occluded. Other scenarios
    are de-emphasised in grey so the reader sees the one trajectory that
    reverses. Panel D (years-of-activation, trivially uniform at 11) is
    dropped; the information lives in the table."""
    setup_plot_style()
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(12, 15))

    t = np.arange(T_HORIZON + 1)

    # Visual hierarchy: E is the headline, A is the no-policy baseline, others
    # are de-emphasised in shades of grey.
    emphasis_style = {
        "A": {"color": "#444444", "lw": 1.4, "alpha": 1.0, "ls": "-",  "z": 4},
        "B": {"color": "#aaaaaa", "lw": 1.2, "alpha": 0.9, "ls": "-",  "z": 2},
        "C": {"color": "#888888", "lw": 1.2, "alpha": 0.9, "ls": "-",  "z": 2},
        "D": {"color": "#666666", "lw": 1.2, "alpha": 0.9, "ls": ":",  "z": 2},
        "E": {"color": "#2a6e5a", "lw": 2.6, "alpha": 1.0, "ls": "-",  "z": 5},
    }
    label_for = {
        "A": "A. Baseline",
        "B": "B. Housing-only, leak-free",
        "C": "C. Housing-only, leaky",
        "D": "D. Rhetoric, no policy",
        "E": "E. Bundled intervention",
    }

    def line(ax, s, series_key):
        d = scenario_data[s][series_key]
        st = emphasis_style[s]
        ax.plot(t, d["mean"], color=st["color"], lw=st["lw"], alpha=st["alpha"],
                ls=st["ls"], zorder=st["z"], label=label_for[s])

    for s in SCENARIO_ORDER:
        line(axA, s, "vote")
        line(axB, s, "rent_burden")
        line(axC, s, "dissat")

    # Panel A: extreme-share trajectory with peak annotation on E.
    e_vote = scenario_data["E"]["vote"]["mean"]
    e_peak_t = int(np.argmax(e_vote))
    e_peak_v = float(e_vote[e_peak_t])
    e_final_v = float(e_vote[-1])
    axA.axhline(0.30, color="#cccccc", lw=0.8, ls="--", zorder=0)
    axA.axhline(0.20, color="#cccccc", lw=0.8, ls="--", zorder=0)
    axA.text(0.3, 0.305, "activation (0.30)", fontsize=10, ha="left", color="#777777")
    axA.text(0.3, 0.207, "deactivation (0.20)", fontsize=10, ha="left", color="#777777")
    axA.scatter([e_peak_t], [e_peak_v], color=emphasis_style["E"]["color"],
                s=36, zorder=6, edgecolor="white", linewidth=1.2)
    axA.annotate(
        f"E peaks at t={e_peak_t}, {e_peak_v:.2f}\nthen declines to {e_final_v:.2f}",
        xy=(e_peak_t, e_peak_v), xytext=(e_peak_t - 9, e_peak_v + 0.07),
        fontsize=11, color="#2a6e5a", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2a6e5a", lw=1.0, shrinkA=2, shrinkB=4),
    )
    axA.set_title("A. Extreme-share vote trajectory", fontsize=14)
    axA.set_xlabel("year")
    axA.set_ylabel("extreme / anti-system vote share")
    axA.set_ylim(0, max(0.46, max(scenario_data[s]["vote"]["mean"].max() for s in SCENARIO_ORDER) * 1.10))

    # Panel B: rent burden. Annotate the wedge between E (and C) and B.
    axB.set_title("B. Rent burden (renters)", fontsize=14)
    axB.set_xlabel("year")
    axB.set_ylabel("rent / income for renters")

    # Panel C: dissatisfaction. The reversal in E should be visible here.
    axC.set_title("C. Material-security dissatisfaction", fontsize=14)
    axC.set_xlabel("year")
    axC.set_ylabel("share of aspiration gap")

    # Single legend below the figure, outside the plot area, single row.
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_scenario_e_figure(scenario_data: dict, out_path: Path) -> None:
    """Three-panel diagnostic comparing E against A / B / C / D on the
    extreme-share vote, rent burden, and the bottom-50 transfer trajectory."""
    setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    axV, axR, axT = axes
    t = np.arange(T_HORIZON + 1)

    for s in SCENARIO_ORDER:
        d = scenario_data[s]
        plot_band(axV, t, d["vote"]["mean"], d["vote"]["lo"], d["vote"]["hi"],
                  SCENARIO_COLOR[s], SCENARIO_LABEL[s])
        plot_band(axR, t, d["rent_burden"]["mean"], d["rent_burden"]["lo"], d["rent_burden"]["hi"],
                  SCENARIO_COLOR[s], SCENARIO_LABEL[s])
        if "transfer" in d:
            plot_band(axT, t, d["transfer"]["mean"], d["transfer"]["lo"], d["transfer"]["hi"],
                      SCENARIO_COLOR[s], SCENARIO_LABEL[s])

    axV.set_title("Extreme/anti-system vote share")
    axV.set_xlabel("year")
    axV.set_ylabel("share")
    axV.axhline(0.30, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
    axV.axhline(0.20, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)
    axV.legend(loc="upper left", fontsize=10, frameon=False)

    axR.set_title("Rent burden (consumption-stress channel)")
    axR.set_xlabel("year")
    axR.set_ylabel("rent / income for renters")

    axT.set_title("Bottom-50 transfer per capita (material-security channel)")
    axT.set_xlabel("year")
    axT.set_ylabel("transfer per recipient, share of initial p.c. income")

    fig.suptitle(
        "Scenario E: housing-only interventions cannot reach the material-security channel",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_robustness_figure(rob_table: pd.DataFrame, out_path: Path) -> None:
    """Half-life across leakage profiles. Includes Scenario E (the
    informative line) alongside B, C, D. Censored points are drawn as
    open markers at the horizon; uncensored points are filled with a
    numeric label."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    leakage_levels = ["low", "medium", "high"]
    leakage_label = {
        "low":    "Low\n(0.20 / 0.15 / 0.25)",
        "medium": "Medium\n(0.40 / 0.30 / 0.50)",
        "high":   "High\n(0.60 / 0.50 / 0.70)",
    }
    xs = np.arange(len(leakage_levels))

    scenarios = ["B", "C", "D", "E"]
    label_for = {
        "B": "B. Housing-only, leak-free anchor",
        "C": "C. Housing-only, central leakage",
        "D": "D. Rhetoric, no policy",
        "E": "E. Bundled housing + redistribution",
    }
    jitter = {"B": -0.06, "C": -0.02, "D": 0.02, "E": 0.06}

    for s in scenarios:
        ys = []
        censored = []
        labels = []
        for lk in leakage_levels:
            row = rob_table[(rob_table["scenario"] == s) & (rob_table["leakage"] == lk)].iloc[0]
            if row["share_censored"] >= 0.5:
                ys.append(T_HORIZON)
                censored.append(True)
                labels.append(f"$\\geq${T_HORIZON}")
            else:
                ys.append(row["median"])
                censored.append(False)
                labels.append(f"{row['median']:.0f}")
        xs_j = xs + jitter[s]
        ax.plot(xs_j, ys, color=SCENARIO_COLOR[s], linewidth=2.2, zorder=3,
                label=label_for[s], alpha=0.95)
        for x, y, lab, cens in zip(xs_j, ys, labels, censored):
            if cens:
                ax.scatter([x], [y], s=110, facecolors="white",
                           edgecolors=SCENARIO_COLOR[s], linewidths=2.2, zorder=4)
            else:
                ax.scatter([x], [y], s=110, color=SCENARIO_COLOR[s],
                           edgecolors="white", linewidths=1.2, zorder=4)
                ax.text(x, y - 1.6, lab, ha="center", fontsize=12,
                        color=SCENARIO_COLOR[s], fontweight="bold")

    ax.axhline(T_HORIZON, color="#aaaaaa", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(2.42, T_HORIZON - 0.5, "Censoring boundary (horizon = 25)",
            fontsize=10, ha="right", va="top", color="#666666", style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([leakage_label[lk] for lk in leakage_levels], fontsize=11)
    ax.set_xlabel("Leakage profile (rent cap / supply / transaction friction)", fontsize=12)
    ax.set_ylabel("Median half-life from peak (periods)", fontsize=12)
    ax.set_title("Only the bundled intervention reaches a finite half-life under low leakage",
                 fontsize=13.5, loc="left", pad=10)
    ax.set_ylim(0, T_HORIZON + 4)
    ax.set_xlim(-0.45, 2.45)

    leg = ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.92,
                    edgecolor="#dddddd")
    leg.get_frame().set_linewidth(0.5)

    ax.text(0.005, -0.20,
            "Filled circles: finite median half-life (label = periods). "
            "Open circles: $\\geq$50% of seeds censored at the 25-period horizon.",
            transform=ax.transAxes, fontsize=9.5, color="#444444", style="italic")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarise_scenario(runs, cfg_for_pop) -> dict:
    votes = np.stack([h.vote_aggregate for _, h in runs])
    rent = np.stack([aggregate_rent_burden(h, c) for c, h in runs])
    diss = np.stack([aggregate_dissat(h, c) for c, h in runs])
    transfer = np.stack([bottom50_resource_trajectory(h, c) for c, h in runs])
    years = np.array([years_extreme_share(h) for _, h in runs])
    gap = np.array([tenure_dissat_gap(h, c) for c, h in runs])

    # Cumulative transfer / cumulative income while extreme-share governance active.
    pol_share = []
    for c, h in runs:
        active = np.array([r is PolicyRegime.POPULIST for r in h.regime])
        inc = h.income_aggregate[active]
        tr = h.transfer_aggregate[active]
        pol_share.append(float(tr.sum() / inc.sum()) if inc.sum() > 0 else 0.0)
    transfer_to_income = float(np.mean(pol_share)) if pol_share else 0.0

    def band(arr):
        return {
            "mean": arr.mean(axis=0),
            "lo": np.percentile(arr, 5, axis=0),
            "hi": np.percentile(arr, 95, axis=0),
            "sd": arr.std(axis=0, ddof=1),
        }

    return {
        "vote": band(votes),
        "rent_burden": band(rent),
        "dissat": band(diss),
        "transfer": band(transfer),
        "years_extreme_share": {"mean": float(years.mean()), "sd": float(years.std(ddof=1)),
                                "min": int(years.min()), "max": int(years.max())},
        "tenure_gap_final": {"mean": float(gap.mean()), "sd": float(gap.std(ddof=1))},
        "vote_peak_period_mean": float(np.mean([int(np.argmax(h.vote_aggregate)) for _, h in runs])),
        "vote_peak_mean": float(np.mean([h.vote_aggregate.max() for _, h in runs])),
        "vote_final_mean": float(votes[:, -1].mean()),
        "rent_burden_final_mean": float(rent[:, -1].mean()),
        "transfer_to_income_share": transfer_to_income,
    }


def main() -> None:
    print(f"Running headline counterfactual (medium leakage, "
          f"{len(SCENARIO_ORDER)} scenarios x {len(SEEDS)} seeds, T={T_HORIZON})")
    scenario_runs = {s: run_scenario(s, MEDIUM) for s in SCENARIO_ORDER}
    summaries = {s: summarise_scenario(scenario_runs[s], None) for s in SCENARIO_ORDER}

    half_lives = {}
    for s in ["B", "C", "D"]:
        half_lives[s] = half_life_summary(scenario_runs[s], scenario_runs["A"])
    half_lives["A"] = {"median": 0.0, "share_censored": 0.0, "n_censored": 0,
                       "n_total": len(SEEDS), "mean_uncensored": 0.0, "sd_uncensored": 0.0}

    # Add Scenario E half-life vs Scenario A.
    half_lives["E"] = half_life_summary(scenario_runs["E"], scenario_runs["A"])

    print()
    print("=" * 78)
    print("HEADLINE SUMMARY (medium leakage)")
    print("=" * 78)
    print(f"{'scenario':>4s}  {'years_ext':>10s}  {'vote_peak':>9s}  "
          f"{'vote_final':>10s}  {'rent_T':>7s}  {'transfer/Y':>11s}  "
          f"{'half_life_med':>13s}  {'censored':>9s}")
    for s in SCENARIO_ORDER:
        sm = summaries[s]
        hl = half_lives[s]
        hl_str = (f"{hl['median']:.1f}" if hl['share_censored'] < 0.5
                  else f">={T_HORIZON}")
        print(
            f"{s:>4s}  {sm['years_extreme_share']['mean']:>10.1f}  "
            f"{sm['vote_peak_mean']:>9.3f}  {sm['vote_final_mean']:>10.3f}  "
            f"{sm['rent_burden_final_mean']:>7.3f}  "
            f"{sm['transfer_to_income_share']:>11.4f}  "
            f"{hl_str:>13s}  "
            f"{hl['n_censored']:>3d}/{hl['n_total']:>3d}"
        )

    # Scenario-E-versus-C headline metric.
    vote_e = summaries["E"]["vote_final_mean"]
    vote_c = summaries["C"]["vote_final_mean"]
    vote_a = summaries["A"]["vote_final_mean"]
    print()
    print("=" * 78)
    print("SCENARIO E HEADLINE COMPARISON")
    print("=" * 78)
    print(f"  Extreme-share vote at T = 25  E: {vote_e:.3f}   C: {vote_c:.3f}   A: {vote_a:.3f}")
    print(f"  E vs C absolute reduction: {vote_c - vote_e:+.3f} ({(vote_c - vote_e) / vote_c:+.1%})")
    print(f"  E vs C reduction target  : >= 0.30 relative (i.e., {0.30 * vote_c:.3f} absolute)")
    e_uncensored = half_lives["E"]["share_censored"] < 0.5
    if e_uncensored:
        print(f"  Scenario E half-life: median {half_lives['E']['median']:.1f} periods, "
              f"{half_lives['E']['n_censored']}/{half_lives['E']['n_total']} censored")
    else:
        print(f"  Scenario E half-life: censored (>= {T_HORIZON} in "
              f"{half_lives['E']['n_censored']}/{half_lives['E']['n_total']} seeds)")

    sys.stdout.flush()
    print()
    print("Generating outputs/fig_material_security_scenarios.png")
    sys.stdout.flush()
    figure_payload = {s: {
        "vote": summaries[s]["vote"],
        "rent_burden": summaries[s]["rent_burden"],
        "dissat": summaries[s]["dissat"],
        "transfer": summaries[s]["transfer"],
        "years_extreme_share": summaries[s]["years_extreme_share"],
    } for s in SCENARIO_ORDER}
    make_headline_figure(figure_payload, OUTPUTS / "fig_material_security_scenarios.png")
    print("Generating outputs/fig_scenario_e.png")
    sys.stdout.flush()
    make_scenario_e_figure(figure_payload, OUTPUTS / "fig_scenario_e.png")

    # Robustness battery.
    print()
    print("Running robustness battery (low / medium / high leakage)")
    sys.stdout.flush()
    rob_rows = []
    for profile in [LOW, MEDIUM, HIGH]:
        runs_A = scenario_runs["A"]  # baseline unaffected by leakage
        for s in ["B", "C", "D", "E"]:
            # B and D are leakage-invariant; reuse medium.
            if profile is MEDIUM:
                runs = scenario_runs[s]
            elif s in ("B", "D"):
                runs = scenario_runs[s]  # leakage-invariant
            else:
                # C and E both vary their housing-policy leakage profile.
                runs = run_scenario(s, profile)
            hl = half_life_summary(runs, runs_A)
            rob_rows.append({
                "scenario": s,
                "leakage": profile.label,
                "rent_leakage": profile.rent_cap,
                "supply_leakage": profile.supply,
                "friction_leakage": profile.friction,
                "median": hl["median"],
                "mean_uncensored": hl["mean_uncensored"],
                "share_censored": hl["share_censored"],
                "n_censored": hl["n_censored"],
                "n_total": hl["n_total"],
            })
    rob_table = pd.DataFrame(rob_rows)
    print(rob_table.to_string(index=False, float_format="%.3f"))

    print()
    print("Generating outputs/fig_robustness_leakage.png")
    make_robustness_figure(rob_table, OUTPUTS / "fig_robustness_leakage.png")

    # Persist artefacts for the writeup step.
    np.savez(
        OUTPUTS / "counterfactual_data.npz",
        **{f"{s}_vote_mean": summaries[s]["vote"]["mean"] for s in SCENARIO_ORDER},
        **{f"{s}_vote_lo": summaries[s]["vote"]["lo"] for s in SCENARIO_ORDER},
        **{f"{s}_vote_hi": summaries[s]["vote"]["hi"] for s in SCENARIO_ORDER},
        **{f"{s}_rent_mean": summaries[s]["rent_burden"]["mean"] for s in SCENARIO_ORDER},
        **{f"{s}_dissat_mean": summaries[s]["dissat"]["mean"] for s in SCENARIO_ORDER},
        **{f"{s}_years": np.array([years_extreme_share(h) for _, h in scenario_runs[s]]) for s in SCENARIO_ORDER},
    )
    rob_table.to_csv(OUTPUTS / "robustness_table.csv", index=False)

    return summaries, half_lives, rob_table


if __name__ == "__main__":
    main()
