"""Regenerate paper/fig-decomp.png from the Table 4 (20-seed) decomposition.

Bar heights are the final-period Delta d^margin values reported in
tab:decomposition (means over 20 seeds), so the figure and the table agree
exactly. Run: uv run python scripts/make_fig_decomp.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

instruments = ["Rent\nrelief", "Supply\nexpansion", "Transfer", "Ownership\nsupport", "Bundle"]
# (d_rent, d_asset, d_access), 20-seed means, matching Table 4
# Gated-baseline values from outputs/decomposition_headline.json
# (scripts/decomposition_headline.py; instrument order: rent relief,
# supply expansion, transfer, ownership support, bundle).
vals = {
    "rent":   [-0.033, -0.013, -0.016, -0.006, -0.071],
    "asset":  [+0.003, -0.020, -0.011, -0.005, -0.046],
    "access": [-0.004, -0.017, -0.150, -0.009, -0.180],
}
colors = {"rent": "#1b6b6b", "asset": "#9e2b25", "access": "#2e6b4f"}
labels = {"rent": r"$\Delta d^{rent}$", "asset": r"$\Delta d^{asset}$", "access": r"$\Delta d^{access}$"}

x = np.arange(len(instruments)); w = 0.26
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for i, k in enumerate(("rent", "asset", "access")):
    ax.bar(x + (i - 1) * w, vals[k], w, label=labels[k], color=colors[k])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(instruments)
ax.set_ylabel("change in margin (relief = negative)")
ax.legend(loc="lower left", frameon=False, ncol=3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("paper/fig-decomp.png", dpi=150)
print("wrote paper/fig-decomp.png")
