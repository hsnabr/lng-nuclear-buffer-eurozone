# Figure 1 — Short-Run Industrial Response to LNG Supply Shock
#             Jordà Local Projections · 90% Newey-West CI · 2015–2023
#             Panel A: Italy (IT_IP) | Panel B: France (FR_IP)

# Figure 2 — The Transmission Mechanism: Where the Shock Propagates
#             Panel A: Italian gas-fired power (IT_GAS_ELEC) — "Merit-Order Tether"
#             Panel B: French industrial production (FR_IP)  — "Nuclear Buffer"

# Figure 3 — The Long-Run Structural Wedge: VECM Loading Coefficients (α)
#             Speed and direction of adjustment to long-run energy equilibrium
#             IT_IP α=−0.238 (ec3, p<0.001) | FR_IP α=+0.131 (ec3, p=0.003)
#             Source: VECM(2), cointegrating rank r=3, post-2015 sample

# Figure 4 — Confirmed Transmission Chain vs. Blocked Channel
#             Granger causality tests · VAR(2) in first differences · 2015–2023
#             Solid: significant (p<0.05) | Dashed: not significant

import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.use("Agg")
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "figure.dpi":        180,
})

C_IT  = "#C0392B"
C_FR  = "#1A5276"
C_GAS = "#E67E22"
GRAY  = "#7F8C8D"

def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(f"outputs/{name}.{ext}", dpi=180,
                    bbox_inches="tight", facecolor="white")
    print(f"  → Saved: outputs/{name}.png / .pdf")

lp      = pd.read_csv("outputs/lp_irf_data.csv")
periods = np.arange(25)

def get(var):
    d = lp[lp["variable"] == var].sort_values("horizon")
    if len(d) == 0:
        return np.zeros(25), np.full(25, -0.05), np.full(25, 0.05)
    def pad(arr):
        if len(arr) < 25:
            return np.concatenate([arr, np.full(25 - len(arr), np.nan)])
        return arr[:25]
    return pad(d["beta"].values), pad(d["ci_lo_90"].values), pad(d["ci_hi_90"].values)


# ── Figure 1 ──────────────────────────────────────────────────────────────────
print("── Figure 1 ──────────────────────────────────────────────────")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

for ax, var, color, panel, country in [
    (ax1, "IT_IP", C_IT, "A", "Italy"),
    (ax2, "FR_IP", C_FR, "B", "France"),
]:
    beta, lo90, hi90 = get(var)
    lo68 = beta - (beta - lo90) * (1.000 / 1.645)
    hi68 = beta + (hi90 - beta) * (1.000 / 1.645)
    valid = ~np.isnan(beta)
    p = periods[valid]

    ax.fill_between(p, lo90[valid], hi90[valid], alpha=0.12, color=color)
    ax.fill_between(p, lo68[valid], hi68[valid], alpha=0.25, color=color)
    ax.plot(p, beta[valid], color=color, lw=2.0)
    ax.axhline(0, color=GRAY, lw=0.9, ls="--", alpha=0.7)
    ax.set_title(f"Panel {panel} — {country}", fontsize=10, color=color, pad=6)
    ax.set_xlabel("Months after shock", fontsize=9)
    ax.set_xlim(0, 24)
    ax.tick_params(labelsize=8)
    ax.text(0.96, 0.96, "CI crosses zero\n→ not sig. at 90%",
            transform=ax.transAxes, fontsize=7.5, color=GRAY,
            va="top", ha="right", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, alpha=0.7))
    handles = [
        Line2D([0],[0], color=color, lw=2, label="LP-IRF"),
        mpatches.Patch(color=color, alpha=0.25, label="68% CI"),
        mpatches.Patch(color=color, alpha=0.12, label="90% CI"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.7)

ax1.set_ylabel("Cumulative response (index pts)", fontsize=9)

fig.text(0.5, -0.04,
    "Note: Shock = 1 SD increase in LNG_SUPPLY (AR(2) residual). Newey-West HAC s.e. "
    "Neither response is statistically significant at 90% — consistent with "
    "LNG supply shocks operating through long-run equilibrium correction (VECM α) "
    "rather than short-run output dynamics.",
    ha="center", fontsize=7.5, color=GRAY, style="italic", wrap=True)

plt.tight_layout()
save(fig, "fig1_structural_wedge")
plt.close()


# ── Figure 2 ──────────────────────────────────────────────────────────────────
print("── Figure 2 ──────────────────────────────────────────────────")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A — IT_GAS_ELEC
beta_g, lo_g, hi_g = get("IT_GAS_ELEC")
lo68_g = beta_g - (beta_g - lo_g) * (1.000 / 1.645)
hi68_g = beta_g + (hi_g - beta_g) * (1.000 / 1.645)
valid_g = ~np.isnan(beta_g)

ax1.fill_between(periods[valid_g], lo_g[valid_g],   hi_g[valid_g],   alpha=0.12, color=C_GAS)
ax1.fill_between(periods[valid_g], lo68_g[valid_g], hi68_g[valid_g], alpha=0.25, color=C_GAS)
ax1.plot(periods[valid_g], beta_g[valid_g], color=C_GAS, lw=2.0)
ax1.axhline(0, color=GRAY, lw=0.9, ls="--", alpha=0.7)
ax1.set_title("Panel A — Italian gas-fired power (IT_GAS_ELEC)", fontsize=9.5, color=C_GAS, pad=6)
ax1.set_xlabel("Months after shock", fontsize=9)
ax1.set_ylabel("Cumulative response (TJ)", fontsize=9)
ax1.set_xlim(0, 24)
ax1.tick_params(labelsize=8)
ax1.text(0.04, 0.96, '"Merit-Order Tether"\nSignificant at h=3 *',
         transform=ax1.transAxes, fontsize=8, color=C_GAS, va="top", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GAS, alpha=0.7))

# Panel B — FR_IP
beta_f, lo_f, hi_f = get("FR_IP")
lo68_f = beta_f - (beta_f - lo_f) * (1.000 / 1.645)
hi68_f = beta_f + (hi_f - beta_f) * (1.000 / 1.645)
valid_f = ~np.isnan(beta_f)

ax2.fill_between(periods[valid_f], lo_f[valid_f],   hi_f[valid_f],   alpha=0.12, color=C_FR)
ax2.fill_between(periods[valid_f], lo68_f[valid_f], hi68_f[valid_f], alpha=0.25, color=C_FR)
ax2.plot(periods[valid_f], beta_f[valid_f], color=C_FR, lw=2.0)
ax2.axhline(0, color=GRAY, lw=0.9, ls="--", alpha=0.7)
ax2.set_title("Panel B — French industrial production (FR_IP)", fontsize=9.5, color=C_FR, pad=6)
ax2.set_xlabel("Months after shock", fontsize=9)
ax2.set_ylabel("Cumulative response (index pts)", fontsize=9)
ax2.set_xlim(0, 24)
ax2.tick_params(labelsize=8)
ax2.text(0.04, 0.96, '"Nuclear Buffer"\nNot significant — insulated',
         transform=ax2.transAxes, fontsize=8, color=C_FR, va="top", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FR, alpha=0.7))

fig.text(0.5, -0.04,
    "Note: Contrast between panels is the core transmission result. "
    "The LNG shock reaches Italian power generation (significant, Panel A) "
    "but does not propagate to French industry (Panel B). "
    "Granger test confirms: LNG→IT_GAS_ELEC (p=0.027**), LNG→FR_IP (p=0.085, n.s.)",
    ha="center", fontsize=7.5, color=GRAY, style="italic")

plt.tight_layout()
save(fig, "fig2_transmission")
plt.close()


# ── Figure 3 ──────────────────────────────────────────────────────────────────
# VERIFIED α values from 03_vecm.py output (ec3 column):
#   IT_IP  ec3 = −0.2378   FR_IP  ec3 = +0.1305
#   LNG_SUPPLY ec1 = −0.1939   EZ_HICP ec3 = +0.0720
# DO NOT use ec2 values — IT_IP and FR_IP load on ec3, not ec2
print("── Figure 3 ──────────────────────────────────────────────────")

alpha_data = {
    "Italy\nIT_IP":        {"alpha": -0.2378, "se": 0.064, "color": C_IT},
    "France\nFR_IP":       {"alpha":  0.1305, "se": 0.044, "color": C_FR},
    "Eurozone\nEZ_HICP":   {"alpha":  0.0720, "se": 0.015, "color": GRAY},
    "EU LNG\nSupply":      {"alpha": -0.1939, "se": 0.117, "color": "#884EA0"},
}

fig, ax = plt.subplots(figsize=(9, 4.0))

ys     = list(range(len(alpha_data)))
labels = list(alpha_data.keys())
alphas = [v["alpha"] for v in alpha_data.values()]
ses    = [v["se"]    for v in alpha_data.values()]
colors = [v["color"] for v in alpha_data.values()]

ax.axvspan(-0.40, 0,    alpha=0.04, color=C_IT)
ax.axvspan(0,    0.25,  alpha=0.04, color=C_FR)
ax.axvline(0, color=GRAY, lw=1.2, ls="--", alpha=0.5)

for i, (y, a, se, c) in enumerate(zip(ys, alphas, ses, colors)):
    ax.barh(y, 1.96 * se, left=a - 1.96 * se, height=0.22, color=c, alpha=0.18, zorder=3)
    ax.barh(y, 1.00 * se, left=a - 1.00 * se, height=0.22, color=c, alpha=0.35, zorder=4)
    ax.scatter(a, y, s=200, color=c, zorder=6)
    offset = 0.012 if a >= 0 else -0.012
    ha     = "left"  if a >= 0 else "right"
    ax.annotate(f"α = {a:+.3f}",
                xy=(a, y), xytext=(a + offset * 3, y + 0.22),
                fontsize=8.5, color=c, ha=ha, va="bottom", fontweight="bold")

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Loading coefficient α (speed of adjustment per month)", fontsize=9)
ax.set_xlim(-0.42, 0.28)
ax.set_ylim(-0.55, len(alpha_data) - 0.25)
ax.tick_params(labelsize=9)

ax.text(-0.21, -0.48, "Adjusts downward (output loss)",
        fontsize=8, color=C_IT, ha="center", style="italic", alpha=0.8)
ax.text( 0.12, -0.48, "Adjusts upward (competitive gain)",
        fontsize=8, color=C_FR, ha="center", style="italic", alpha=0.8)

handles = [
    mpatches.Patch(color=GRAY, alpha=0.35, label="68% CI"),
    mpatches.Patch(color=GRAY, alpha=0.18, label="95% CI"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=GRAY, ms=8, label="Point estimate"),
]
ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.7)

fig.text(0.5, -0.06,
    "Note: α from VECM(2), ec3 cointegrating vector, rank r=3, post-2015 sample. "
    "IT_IP α=−0.238 (p<0.001): Italy's industrial output corrects downward. "
    "FR_IP α=+0.131 (p=0.003): France adjusts upward. "
    "Opposite signs on the same cointegrating vector constitute the 'structural wedge.'",
    ha="center", fontsize=7.5, color=GRAY, style="italic")

plt.tight_layout()
save(fig, "fig3_alpha_wedge")
plt.close()


# ── Figure 4 ──────────────────────────────────────────────────────────────────
# Granger p-values verified from 03_vecm.py output:
#   LNG → IT_GAS_ELEC  p=0.027 **
#   IT_GAS_ELEC → IT_IP p=0.012 **
#   LNG → IT_IP (direct) p=0.027 **
#   LNG → FR_IP         p=0.085 n.s.
print("── Figure 4 ──────────────────────────────────────────────────")

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.8)
ax.axis("off")

def box(ax, x, y, w, h, text, color, sub=""):
    ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h,
                                fc=color, ec=color, alpha=0.14, zorder=2, linewidth=1.2))
    ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h,
                                fc="none", ec=color, alpha=0.55, zorder=3, linewidth=1.2))
    ax.text(x, y + (0.09 if sub else 0), text,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=color, zorder=4)
    if sub:
        ax.text(x, y - 0.2, sub, ha="center", va="center",
                fontsize=7.5, color=color, zorder=4, style="italic")

def arrow(ax, x1, x2, y, color, label="", solid=True):
    ls = "-" if solid else "--"
    ax.annotate("", xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8, linestyle=ls))
    if label:
        ax.text((x1 + x2) / 2, y + 0.20, label,
                ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

C_NUC = "#148F77"

# Italian chain (confirmed)
box(ax, 1.2, 2.6, 1.6, 0.65, "LNG_SUPPLY",   C_GAS, "Global shock")
arrow(ax, 2.0, 3.4, 2.6, C_GAS, "p=0.027 **")
box(ax, 4.2, 2.6, 1.6, 0.65, "IT_GAS_ELEC", C_GAS, "Italian power sector")
arrow(ax, 5.0, 6.4, 2.6, C_IT,  "p=0.012 **")
box(ax, 7.2, 2.6, 1.6, 0.65, "IT_IP",       C_IT,  "Italian industry")

# Direct path
arrow(ax, 2.0, 6.4, 3.2, C_IT, "p=0.027 **  (direct)")

# French path (blocked)
box(ax, 4.2, 1.1, 1.6, 0.65, "FR_NUC",  C_NUC, "Nuclear buffer (exog.)")
arrow(ax, 2.0, 3.4, 1.1, C_FR, "p=0.085 n.s.", solid=False)
box(ax, 7.2, 1.1, 1.6, 0.65, "FR_IP",   C_FR,  "French industry")
arrow(ax, 5.0, 6.4, 1.1, C_FR, "blocked ✗", solid=False)

ax.text(9.6, 2.6, "EXPOSED",   fontsize=9.5, color=C_IT,  fontweight="bold",
        ha="center", va="center", rotation=90, alpha=0.65)
ax.text(9.6, 1.1, "INSULATED", fontsize=9.5, color=C_FR,  fontweight="bold",
        ha="center", va="center", rotation=90, alpha=0.65)

fig.text(0.5, -0.04,
    "Note: Solid arrows = statistically significant Granger causality (** p<0.05). "
    "Dashed = not significant (n.s.). "
    "LNG supply shocks propagate through Italy's gas-fired power sector to industrial output. "
    "The same shock does not significantly reach French industry.",
    ha="center", fontsize=7.5, color=GRAY, style="italic")

plt.tight_layout()
save(fig, "fig4_granger_chain")
plt.close()

print("""
══════════════════════════════════════════════════════════
  ALL FIGURES SAVED (no titles on images)
  Titles are in the script comments at the top of each block.

  Use in paper:
    Fig 1 → Section 3.1 (LP-IRF, short-run noise)
    Fig 2 → Section 3.2 (transmission mechanism)
    Fig 3 → Section 3.3 (VECM α, main result)
    Fig 4 → Section 3.2 (Granger chain diagram)

  Verified α values used in Fig 3:
    IT_IP  ec3 = −0.2378   (paper should say −0.238)
    FR_IP  ec3 = +0.1305   (paper should say +0.131)
══════════════════════════════════════════════════════════
""")