"""
02_diagnostics.py
─────────────────
Pre-estimation diagnostics:
  - ADF + KPSS stationarity tests (all variables)
  - Bai-Perron structural break detection
  - Johansen cointegration trace test
  - Pairwise Engle-Granger cointegration (key pairs)
  - Outlier detection (|z-score| > 3 in first differences)

Output: stationarity_results.csv, prints full diagnostic report
"""
import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ── helpers ───────────────────────────────────────────────────────────────────
def _prep(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply COVID dummies + restrict to post-2015 sample."""
    df = df_raw[df_raw.index >= "2015-01-01"].copy()
    covid = ["2020-03", "2020-04", "2020-05"]
    df["D_COVID"] = df.index.to_period("M").astype(str).isin(covid).astype(float)
    return df.dropna(subset=["LNG_SUPPLY","IT_GAS_ELEC","IT_IP","FR_IP","FR_NUC","EZ_HICP"])


def stationarity_table(df: pd.DataFrame, vars_: list) -> pd.DataFrame:
    rows = []
    for v in vars_:
        s = df[v].dropna()
        # levels
        adf_l  = adfuller(s, autolag="AIC", regression="ct")
        kpss_l = kpss(s, regression="ct", nlags="auto")
        # first diff
        adf_d  = adfuller(s.diff().dropna(), autolag="AIC", regression="ct")
        # integration order
        if adf_l[1] < 0.05 and kpss_l[1] > 0.05:
            order = "I(0)"
        elif adf_l[1] >= 0.05 and kpss_l[1] < 0.05:
            order = "I(1)"
        else:
            order = "unclear"
        rows.append({
            "variable":      v,
            "adf_p_level":   round(adf_l[1], 3),
            "kpss_p_level":  round(kpss_l[1], 3),
            "adf_p_diff":    round(adf_d[1], 3),
            "integration":   order,
        })
    return pd.DataFrame(rows).set_index("variable")


def johansen_test(df: pd.DataFrame, vars_: list) -> None:
    print("\n── Johansen trace test ───────────────────────────────────────")
    print("  H₀: at most r cointegrating vectors | reject if trace > cv(5%)\n")
    j = coint_johansen(df[vars_], det_order=0, k_ar_diff=2)
    for r in range(len(vars_)):
        sig = "  *** reject" if j.lr1[r] > j.cvt[r, 1] else ""
        print(f"  r ≤ {r}  trace={j.lr1[r]:>8.3f}  cv5%={j.cvt[r,1]:>7.3f}{sig}")
    # infer rank
    rank = 0
    for r in range(len(vars_)):
        if j.lr1[r] > j.cvt[r, 1]:
            rank = r + 1
    print(f"\n  → Confirmed rank: r = {rank}")


def pairwise_coint(df: pd.DataFrame) -> None:
    print("\n── Pairwise cointegration (Engle-Granger) ────────────────────")
    pairs = [
        ("LNG_SUPPLY", "IT_IP",       "supply → Italian industry"),
        ("LNG_SUPPLY", "FR_IP",       "supply → French industry"),
        ("LNG_SUPPLY", "IT_GAS_ELEC", "supply → IT gas power"),
        ("IT_IP",      "FR_IP",       "Italy vs France co-movement"),
    ]
    for v1, v2, label in pairs:
        stat, p, _ = coint(df[v1], df[v2])
        res = "cointegrated *" if p < 0.05 else "no cointegration"
        print(f"  {v1} ~ {v2:<20} p={p:.3f}  {res}   [{label}]")


def outlier_report(df: pd.DataFrame, vars_: list) -> list:
    print("\n── Outlier detection (|z| > 3 in Δ series) ──────────────────")
    dummies = []
    df_d = df[vars_].diff().dropna()
    for v in vars_:
        z = (df_d[v] - df_d[v].mean()) / df_d[v].std()
        hits = z[z.abs() > 3]
        if len(hits):
            for dt in hits.index:
                label = f"D_{v}_{str(dt)[:7].replace('-','')}"
                dummies.append(label)
                print(f"  {v:<20} {str(dt)[:10]}  z={z[dt]:.2f}  → dummy: {label}")
    return dummies


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = pd.read_csv("svar_data.csv", index_col="date", parse_dates=True)
    df     = _prep(df_raw)
    VARS   = ["LNG_SUPPLY", "IT_GAS_ELEC", "IT_IP", "FR_IP", "FR_NUC", "EZ_HICP"]

    print("=" * 62)
    print("  DIAGNOSTIC REPORT")
    print(f"  Sample: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} obs)")
    print("=" * 62)

    # 1. Stationarity
    print("\n── Stationarity (ADF + KPSS) ─────────────────────────────────")
    tbl = stationarity_table(df, VARS)
    print(tbl.to_string())
    os.makedirs("outputs", exist_ok=True)
    tbl.to_csv("outputs/stationarity_results.csv")
    print("\n  Saved → outputs/stationarity_results.csv")

    # 2. Structural breaks
    print("\n── Structural breaks (Bai-Perron) ────────────────────────────")
    try:
        import ruptures as rpt
        for v in ["IT_IP", "FR_IP", "LNG_SUPPLY"]:
            algo   = rpt.Binseg(model="rbf").fit(df[v].values)
            breaks = algo.predict(n_bkps=2)
            dates  = [str(df.index[b-1].date()) for b in breaks[:-1]]
            print(f"  {v:<20} breaks at: {dates}")
    except ImportError:
        print("  ruptures not installed — pip install ruptures")

    # 3. Johansen
    johansen_test(df, VARS)

    # 4. Pairwise cointegration
    pairwise_coint(df)

    # 5. Outliers
    outlier_report(df, ["IT_IP", "FR_IP"])