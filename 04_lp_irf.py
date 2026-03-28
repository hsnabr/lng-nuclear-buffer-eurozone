"""
04_lp_irf.py
────────────
Stage 2: Jordà Local Projections
  - IRF of IT_IP and FR_IP to a 1 SD shock in LNG_SUPPLY
  - Newey-West HAC standard errors (robust to serial correlation)
  - 90% confidence intervals
  - FR_NUC treated as exogenous control
  - COVID + outlier dummies included

Why LP instead of VAR-based IRF:
  - Does not compound misspecification across horizons
  - More robust to extreme observations (COVID, 2022 spike)
  - Allows state-dependent extensions (future work)
  - HAC s.e. valid even under moderate serial correlation

Reference: Jordà (2005), AER

Output: outputs/lp_irf.png, outputs/lp_irf_data.csv
"""
import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

os.makedirs("outputs", exist_ok=True)

# ── data prep ─────────────────────────────────────────────────────────────────
def prep(path="svar_data.csv"):
    df = pd.read_csv(path, index_col="date", parse_dates=True).dropna()
    df = df[df.index >= "2015-01-01"]
    covid = ["2020-03","2020-04","2020-05"]
    df["D_COVID"] = df.index.to_period("M").astype(str).isin(covid).astype(float)
    df_d = df[["IT_IP","FR_IP"]].diff()
    for v in ["IT_IP","FR_IP"]:
        z = (df_d[v] - df_d[v].mean()) / df_d[v].std()
        for dt in z[z.abs() > 3].index:
            key = f"D_{v}_{str(dt)[:7].replace('-','')}"
            df[key] = (df.index == dt).astype(float)
    return df


df   = prep()
ENDO = ["LNG_SUPPLY", "IT_GAS_ELEC", "IT_IP", "FR_IP"]
EXOG = [c for c in df.columns if c.startswith("D_")] + ["FR_NUC"]

P_LAGS   = 2
HORIZONS = 24
RESP     = ["IT_IP", "FR_IP", "IT_GAS_ELEC"]

print(f"Sample : {df.index[0].date()} → {df.index[-1].date()}")
print(f"Obs    : {len(df)}")
print(f"Dummies: {[c for c in df.columns if c.startswith('D_')]}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTRUCT LNG SHOCK — residual from AR(2) on ΔLNG_SUPPLY, standardised
# ══════════════════════════════════════════════════════════════════════════════
df_d = df[ENDO].diff().dropna()
lng  = df_d["LNG_SUPPLY"]

X_ar   = add_constant(pd.concat([lng.shift(1), lng.shift(2)], axis=1).dropna())
y_ar   = lng.loc[X_ar.index]
ar_fit = OLS(y_ar, X_ar).fit()
shock  = ar_fit.resid / ar_fit.resid.std()
shock.name = "LNG_SHOCK"

print(f"\nShock constructed: AR(2) residual on ΔLNG_SUPPLY")
print(f"  Mean: {shock.mean():.4f}  Std: {shock.std():.4f}  (should be ~0 and 1)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCAL PROJECTIONS LOOP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nComputing LP-IRFs for h = 0 to {HORIZONS}...")

results = {v: {"beta": [], "ci_lo_90": [], "ci_hi_90": [],
               "ci_lo_68": [], "ci_hi_68": []}
           for v in RESP}

exog_ctrl = df[[c for c in EXOG if c in df.columns]].reindex(df_d.index)

for h in range(HORIZONS + 1):
    for rv in RESP:
        # Cumulative change: sum of Δy from t to t+h
        y_h = df_d[rv].rolling(h+1).sum().shift(-h) if h > 0 else df_d[rv]

        reg = pd.concat([
            y_h.rename("y"),
            shock,
            *[df_d[v].shift(j).rename(f"{v}_L{j}")
              for v in ENDO for j in range(1, P_LAGS+1)],
            exog_ctrl
        ], axis=1).dropna()

        if len(reg) < 20:
            for k in results[rv]:
                results[rv][k].append(np.nan)
            continue

        X   = add_constant(reg.drop("y", axis=1))
        ols = OLS(reg["y"], X).fit()
        hac = np.sqrt(np.diag(cov_hac(ols, nlags=max(h+1, 4))))
        idx = list(X.columns).index("LNG_SHOCK")

        b  = ols.params["LNG_SHOCK"]
        se = hac[idx]

        results[rv]["beta"].append(b)
        results[rv]["ci_lo_90"].append(b - 1.645 * se)
        results[rv]["ci_hi_90"].append(b + 1.645 * se)
        results[rv]["ci_lo_68"].append(b - 1.000 * se)
        results[rv]["ci_hi_68"].append(b + 1.000 * se)

print("  Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {"IT_IP": "#D85A30", "FR_IP": "#185FA5", "IT_GAS_ELEC": "#1D9E75"}
TITLES = {
    "IT_IP":       "Italian industrial production",
    "FR_IP":       "French industrial production",
    "IT_GAS_ELEC": "Italian gas-fired electricity",
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
periods   = np.arange(HORIZONS + 1)

for ax, rv in zip(axes, RESP):
    c     = COLORS[rv]
    beta  = np.array(results[rv]["beta"])
    lo90  = np.array(results[rv]["ci_lo_90"])
    hi90  = np.array(results[rv]["ci_hi_90"])
    lo68  = np.array(results[rv]["ci_lo_68"])
    hi68  = np.array(results[rv]["ci_hi_68"])

    ax.fill_between(periods, lo90, hi90, alpha=0.12, color=c, label="90% CI")
    ax.fill_between(periods, lo68, hi68, alpha=0.22, color=c, label="68% CI")
    ax.plot(periods, beta, color=c, lw=1.8, label="LP estimate")
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_title(TITLES[rv], fontsize=10, pad=8)
    ax.set_xlabel("Months after shock", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    if rv == RESP[0]:
        ax.set_ylabel("Cumulative response", fontsize=9)

fig.suptitle(
    "Local projection IRFs: response to 1 SD shock in LNG supply\n"
    "Post-2015 sample · COVID + outlier dummies · Newey-West HAC s.e.",
    fontsize=9, y=1.02
)
plt.tight_layout()
plt.savefig("outputs/lp_irf.png", dpi=180, bbox_inches="tight")
print("\n  → Saved: outputs/lp_irf.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── LP-IRF summary ────────────────────────────────────────────")
print(f"  {'Variable':<22} {'Peak β':>8} {'Horizon':>8} {'Sig at peak?':>14}")
print("  " + "-"*56)

export_rows = []
for rv in RESP:
    beta  = np.array(results[rv]["beta"])
    lo90  = np.array(results[rv]["ci_lo_90"])
    hi90  = np.array(results[rv]["ci_hi_90"])
    valid = ~np.isnan(beta)
    if not valid.any():
        continue
    # horizon of maximum absolute response
    peak_h = int(np.nanargmax(np.abs(beta)))
    peak_b = beta[peak_h]
    sig    = "yes *" if not (lo90[peak_h] < 0 < hi90[peak_h]) else "no"
    print(f"  {rv:<22} {peak_b:>8.4f} {'h='+str(peak_h):>8} {sig:>14}")
    for h in range(HORIZONS+1):
        export_rows.append({
            "variable": rv, "horizon": h,
            "beta": results[rv]["beta"][h],
            "ci_lo_90": results[rv]["ci_lo_90"][h],
            "ci_hi_90": results[rv]["ci_hi_90"][h],
        })

pd.DataFrame(export_rows).to_csv("outputs/lp_irf_data.csv", index=False)
print("\n  → Saved: outputs/lp_irf_data.csv")

print("""
  ★ INTERPRETATION:
    If IT_IP peak β < 0 and CI does not cross 0 → Italy significantly exposed
    If FR_IP response is flat and CI crosses 0   → France statistically insulated
    IT_GAS_ELEC response confirms transmission channel
""")
