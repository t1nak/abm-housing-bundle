#!/usr/bin/env bash
# Reproduce every table and figure in the paper and online appendix from the
# single calibrated configuration in src/abmhp/margin_calibration.py.
# Usage:  bash reproduce_all.sh
# All results are deterministic given the seeds fixed inside each script.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/11 calibration diagnostics (Table 5, section 6 numbers) =="
uv run python scripts/baseline_diagnostics.py

echo "== 2/11 mechanism distinctness (Table 4) =="
uv run python scripts/mechanism_diagnostics.py

echo "== 3/11 policy decomposition + dose (Tables 6-7) =="
uv run python scripts/decomposition_headline.py

echo "== 4/11 decomposition figure (fig-decomp.png) =="
uv run python scripts/make_fig_decomp.py

echo "== 5/11 untargeted moments + OA.4 variant table =="
uv run python scripts/validation_moments.py

echo "== 6/11 three gap scenarios / adjusted-anchor band (Table 8) =="
uv run python scripts/adjusted_anchor_band.py

echo "== 7/11 coefficient anchor-matching grid (OA.3) =="
uv run python scripts/gamma_manifold.py

echo "== 8/11 aspiration-locality robustness (OA.3) + figure =="
uv run python scripts/alpha_loc_robustness.py
uv run python scripts/make_fig_alpha_loc.py

echo "== 9/11 alternative mechanism definitions (OA.3) =="
uv run python scripts/mechanism_definitions_robustness.py

echo "== 10/11 systematic robustness sweep, 28 variants x 6 x 5 = 840 sims (OA.3) =="
uv run python scripts/robustness_sweep.py

echo "== 11/11 SOEP roadmap pipeline self-test (OA.6; no SOEP access needed) =="
uv run python scripts/soep_anchor_pipeline.py --selftest

echo
echo "All outputs written to outputs/. Run 'uv run pytest' for the test suite."
