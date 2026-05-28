# Housing as a Bundled Political Shock: Rent, Wealth, and Voting

Replication archive for the paper **"Housing as a Bundled Political Shock:
Rent, Wealth, and Voting"** (Koziol 2026).

This repository contains the agent-based simulation engine, the Simulated
Method of Moments (SMM) estimation pipeline, the counterfactual policy
battery, the robustness suite, and the UK Brexit external stress test used in
the paper. Every quantitative element in the manuscript — the calibration
tables, the parameter estimates, the counterfactual scenarios, the robustness
diagnostics, and the UK stress test — is reproducible from the scripts here
under a fixed, seed-controlled protocol.

---

## 1. Repository layout

| Path | Contents |
|------|----------|
| `src/abmhp/` | The model package: agent population, regional housing market, bequest and wealth-accumulation processes, aspiration block, voting block, policy block, and the SMM estimation routines. |
| `scripts/` | Entry points that run each exercise and write artefacts to `outputs/`. |
| `tests/` | Regression tests, including `test_reference_rule.py`, which pins the baseline Akerlof–Yellen reference rule as byte-identical to the pre-flag implementation. |
| `data/` | Small input data files (regional graduate-share inputs for the cosmopolitan-channel pilot). |
| `outputs/` | Pre-computed artefacts (JSON payloads, CSV/Markdown tables, LaTeX snippets, and PNG figures). The manuscript's table- and figure-rendering routines read from this directory, so the paper can be regenerated either from these artefacts or from a clean simulation run. |
| `pyproject.toml`, `uv.lock` | Pinned Python environment. |

---

## 2. Requirements

- **Python ≥ 3.12**
- Dependencies (pinned in `pyproject.toml` / `uv.lock`): `numpy`, `pandas`,
  `scipy`, `matplotlib`, `scikit-learn`, `scikit-optimize`, `statsmodels`.
- Recommended runner: [`uv`](https://github.com/astral-sh/uv).

## 3. Setup

```bash
git clone https://github.com/t1nak/abm-housing-bundle.git
cd abm-housing-bundle

# with uv (recommended) — creates a pinned virtual environment
uv sync

# or with pip
pip install -e .
```

---

## 4. Reproducing the results

Each script writes its artefacts to `outputs/`. Prefix commands with
`uv run` (or activate the environment and run `python` directly). All
exercises are deterministic given the seed protocol in Section 5.

```bash
# (a) Baseline German calibration — distributional, housing, and political
#     moments at the SMM Stage 1 optimum.
uv run scripts/run_baseline.py

# (b) Two-stage SMM estimation of the eight behavioural parameters.
#     Writes outputs/smm_optimum.json and the sensitivity Jacobian figure.
uv run scripts/run_smm.py
uv run scripts/recover_stage1_optimum.py
uv run scripts/write_smm_markdown.py

# (c) Counterfactual battery — Scenarios A–E plus the cross-leakage
#     half-life diagnostics, and the equal-cost E-light / C-plus scenarios.
uv run scripts/counterfactual_material_security.py
uv run scripts/counterfactual_equal_cost.py

# (d) 50-seed central-leakage reliability diagnostic for Scenario E
#     (and the T=40 secondary diagnostic).
uv run scripts/scenario_e_robustness_n50.py

# (e) Reference-point specification stress test (Kőszegi–Rabin alternative).
uv run scripts/run_kr_robustness.py

# (f) UK Brexit external stress test at NUTS-1 resolution.
uv run scripts/run_uk_validation.py

# (g) UK / German partial-correlation residual and cosmopolitan-channel pilot.
uv run scripts/german_partial_correlation_placebo.py
uv run scripts/cosmopolitan_pilot.py

# (h) Exploratory mechanism-enrichment exercise (augmented parameter space).
uv run scripts/phase2_v2_validation.py
uv run scripts/phase3_v2_smm_augmented_p50.py
uv run scripts/phase4_counterfactual_augmented.py
uv run scripts/phase4_uk_augmented.py

# (i) Identification-tradeoff diagnostics: o_{i,t}-construction variants and
#     one-at-a-time / joint bound widenings.
uv run scripts/o_construction_diagnostic.py
uv run scripts/o_construction_addendum.py
uv run scripts/widen_bounds_diagnostic.py
```

### Manuscript element → script → artefact

| Manuscript element | Script | Key artefact(s) in `outputs/` |
|--------------------|--------|-------------------------------|
| Calibration moment fit; parameter estimates; sensitivity Jacobian | `run_smm.py`, `recover_stage1_optimum.py`, `write_smm_markdown.py` | `smm_optimum.json`, `moments_table.md`, `sensitivity_jacobian.png` |
| Baseline distributional / political moments | `run_baseline.py` | `phase1_master_baseline.json` |
| Counterfactual scenario summary & leakage diagnostics; scenario figures | `counterfactual_material_security.py` | `material_security_results.md`, `fig_material_security_scenarios.png`, `fig_scenario_e.png`, `fig_robustness_leakage.png` |
| Equal-cost E-light / C-plus columns | `counterfactual_equal_cost.py` | `equal_cost_counterfactual.json` |
| 50-seed reliability / T=40 diagnostic | `scenario_e_robustness_n50.py` | `scenario_e_robustness_n50.json`, `scenario_e_robustness_n50_table.tex` |
| Reference-point (Kőszegi–Rabin) robustness | `run_kr_robustness.py` | `kr_robustness.json` |
| UK stress test (level, ordering, cleavage) | `run_uk_validation.py` | `fig_uk_validation.png`, `fig_germany_uk_comparison.png` |
| Partial-correlation residual; cosmopolitan pilot | `german_partial_correlation_placebo.py`, `cosmopolitan_pilot.py` | `german_partial_correlation_placebo.json`, `cosmopolitan_pilot.json` |
| Augmented mechanism enrichment | `phase2_v2_validation.py`, `phase3_v2_smm_augmented_p50.py`, `phase4_counterfactual_augmented.py`, `phase4_uk_augmented.py` | `augmented_phase2_v2_results.json`, `augmented_phase4_counterfactual.json`, `augmented_phase4_uk.json` |
| o-construction / bound-widening diagnostics | `o_construction_diagnostic.py`, `o_construction_addendum.py`, `widen_bounds_diagnostic.py` | `o_construction_diagnostic.json`, `o_construction_addendum.json` |

The `compare_abm_hank.py` script is a development artefact implementing a
representative-cell heterogeneous-agent comparator. It is retained for archival
reference and is not invoked by the main paper.

---

## 5. Seed protocol

Every exercise uses a deterministic integer seed range passed to the
simulator's master RNG. The ranges are:

| Exercise | Seeds |
|----------|-------|
| Variance estimation / held-out validation moments | `{73, 74, 75, 76, 77}` (5) |
| Two-stage SMM (per parameter evaluation) | `{73, 74, 75, 76, 77}` (5; offset 73) |
| Counterfactual scenarios (headline) | `{73, …, 82}` (10; shared across scenarios and leakage profiles for paired comparisons) |
| Central-leakage 50-seed extension / T=40 diagnostic | `{73, …, 122}` (50; a strict superset of the headline 10) |
| UK stress test | `{73, 74, 75, 76, 77}` (5) |

Because the 50-seed range begins at the same seed as the 10-seed headline, the
headline values are reproduced exactly within the larger run.

---

## 6. Tests

```bash
uv run pytest
```

The suite includes a regression test (`tests/test_reference_rule.py`) that
verifies the default Akerlof–Yellen reference rule is byte-identical to the
pre-flag implementation, so that the Kőszegi–Rabin robustness flag does not
perturb the headline pipeline.

---

## 7. Citation

If you use this code, please cite:

> Koziol, T. (2026). *Housing as a Bundled Political Shock: Rent, Wealth, and
> Voting.*

---

## 8. AI-tooling disclosure

This codebase and the accompanying manuscript were developed with assistance
from Anthropic Claude, used as an iterative coding assistant for the simulation
engine, the SMM estimation, the robustness and UK validation scripts. The
author specified all model structure, calibration choices, analytical
decisions, and acceptance criteria; reviewed, edited, and verified the
resulting code and prose; and takes full responsibility for the contents of
both the codebase and the publication.

---

## 9. License

License terms available on request. Correspondence: tinak.contact@gmail.com.
