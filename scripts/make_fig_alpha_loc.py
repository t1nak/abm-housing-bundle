"""Figure: policy vote-effects across the aspiration-locality parameter.

Reads outputs/alpha_loc_robustness.json and plots, for each single
instrument and the bundle, the final-period change in the aggregate vote as
a function of alpha_local. A flat set of lines demonstrates that the policy
ranking and magnitudes are stable across alpha_local. Writes
paper/rewrite_assets/fig-alpha-loc.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "outputs" / "alpha_loc_robustness.json").read_text())

alphas = sorted(float(k) for k in d)
instruments = [("rent", "Rent relief", "#1b6b6b"),
               ("supply", "Supply expansion", "#9e2b25"),
               ("access", "Ownership support", "#b8860b"),
               ("transfer", "Transfer", "#2e6b4f"),
               ("bundle", "Bundle", "#333333")]

fig, ax = plt.subplots(figsize=(7.6, 4.4))
for key, label, color in instruments:
    ys = [d[f"{a:.2f}"]["policy"]["deltas"][key]["vote"] for a in alphas]
    style = "--" if key == "bundle" else "-"
    ax.plot(alphas, ys, style, marker="o", color=color, label=label, linewidth=1.6)
ax.axvline(0.45, color="grey", linewidth=0.8, linestyle=":")
ax.text(0.45, ax.get_ylim()[1], " baseline", va="top", ha="left",
        fontsize=8, color="grey")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel(r"aspiration-locality parameter $\alpha_{\mathrm{loc}}$")
ax.set_ylabel("change in aggregate vote (relief = negative)")
ax.legend(loc="center right", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = ROOT / "paper" / "rewrite_assets" / "fig-alpha-loc.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
