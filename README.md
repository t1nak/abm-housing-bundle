# Priced Out and Locked Out — Replication Package

Public replication archive for **"Priced Out and Locked Out: A Calibrated
Heterogeneous-Agent Model of Housing Pressure and Anti-Establishment
Voting"** (Koziol 2026).

This repository contains **only** the replication materials: the simulation
package, the reproduction scripts, the test suite, the generated outputs, and
the calibration inputs. The manuscript and online appendix are distributed
through the journal; this archive reproduces every number, table, and figure
they report.

The paper decomposes housing-related political dissatisfaction into three
separable mechanisms — rent stress, asset exclusion, and ownership-access
exclusion — and embeds them in a calibrated heterogeneous-agent model of
Germany, 2010–2025. The model is *calibrated, not estimated*: no political
coefficient is statistically identified, and the renter–owner voting gap is a
literature-informed scenario range, not an observed estimate.

## Environment

Python is managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync          # create the environment from pyproject.toml / uv.lock
uv run pytest    # 63 passing tests, 4 documented xfails
```

All results are deterministic given the seeds fixed inside each script. No
network access or confidential microdata are required to reproduce any
reported number.

## Reproduce every table and figure

```bash
bash reproduce_all.sh
```

or run individual scripts (all at the single calibrated configuration pinned
in `src/abmhp/margin_calibration.py`; outputs written to `outputs/`):

| Paper object | Script |
|---|---|
| Table 3 (calibration diagnostics), §6 numbers, housing-channel ceiling | `scripts/baseline_diagnostics.py` |
| Table 1 (mechanism distinctness) | `scripts/mechanism_diagnostics.py` |
| Table 6 (policy decomposition), Table 7 (dose), `fig-decomp.png` | `scripts/decomposition_headline.py`, `scripts/make_fig_decomp.py` |
| Table 4 (§5 untargeted moments), OA.4 variant table | `scripts/validation_moments.py` |
| Table 8 (three gap scenarios) | `scripts/adjusted_anchor_band.py` |
| OA.3 coefficient anchor-matching grid | `scripts/gamma_manifold.py` |
| OA.3 aspiration-locality table, `fig-alpha-loc.png` | `scripts/alpha_loc_robustness.py`, `scripts/make_fig_alpha_loc.py` |
| OA.3 alternative mechanism definitions | `scripts/mechanism_definitions_robustness.py` |
| OA.3 robustness sweep (28 variants × 6 × 5 = 840 sims) | `scripts/robustness_sweep.py` |

A clean test-suite log is in `replication/test_run_log.txt`.

## Empirical roadmap (Online Appendix OA.6)

The renter–owner gap and the three political-response coefficients are **not**
estimated in this paper; OA.6 pre-specifies the design for a future estimation
on SOEP-Core v41 (DIW Berlin, available under a standard data-use contract,
not redistributable). The frozen pipeline is self-testing without any SOEP
access:

```bash
uv run python scripts/soep_anchor_pipeline.py --selftest
```

## Repository layout

- `src/abmhp/` — the simulation package (`config.py` holds all parameters;
  `margin_calibration.py` pins the calibrated configuration).
- `scripts/` — the reproduction scripts in the table above, plus auxiliary
  development tools. The scripts in the table reproduce every number in the
  paper. Other scripts (`run_smm.py`, `compare_abm_hank.py`,
  `run_uk_validation.py`, and related SMM / HANK-benchmark / UK-validation
  routines) are development-stage artifacts retained for provenance; they are
  **not** used to produce any result reported in this paper.
- `tests/` — the test suite (63 pass, 4 documented xfails).
- `outputs/` — generated tables, figures, and JSON result files.
- `data/` — small non-confidential calibration inputs.

## Citation

Koziol, T. (2026). *Priced Out and Locked Out: A Calibrated
Heterogeneous-Agent Model of Housing Pressure and Anti-Establishment Voting.*
Working paper. (Update with journal reference / DOI on publication.)

## License

MIT License (see `LICENSE`).

## AI tooling disclosure

The simulation codebase and analysis scripts were developed with assistance
from Anthropic Claude models (via the Claude Code assistant, 2025–2026
releases) used as an iterative programming tool. The author specified all
model structure, calibration choices, and acceptance criteria; reviewed and
verified the resulting code; and takes full responsibility for its contents.
All reported numbers are regenerated from the seed-controlled scripts above.
