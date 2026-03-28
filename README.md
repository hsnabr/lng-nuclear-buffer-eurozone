# lng-nuclear-buffer-eurozone

**Quantifying the nuclear buffer effect on EU industrial resilience to LNG supply shocks — VECM & Jordà LP · Italy vs France · Eurostat/ECB data**

Replication package for: *Asymmetric Industrial Pass-through of LNG Supply Shocks in the Eurozone*

---

## Overview

This repository contains the full replication code for the econometric analysis. The study uses a two-stage strategy:

1. **VECM** — identifies long-run adjustment coefficients (α) for Italian and French industrial production following LNG supply shocks
2. **Jordà Local Projections** — estimates impulse response functions robust to outliers and serial correlation, with Newey-West HAC standard errors

### Core finding
Italy's industrial production adjusts **downward** (α = −0.238, p < 0.001) while France's adjusts **upward** (α = +0.131, p = 0.003) on the same cointegrating vector — a structural divergence consistent with the nuclear insulation hypothesis. The Granger chain `LNG_SUPPLY → IT_GAS_ELEC → IT_IP` (p = 0.012) is confirmed; the equivalent chain for France is not significant (p = 0.085).

---

## Repository structure

```
lng-nuclear-buffer-eurozone/
├── 01_download_data.py      Download all variables from Eurostat + ECB
├── 02_diagnostics.py        Stationarity, cointegration, structural breaks, outliers
├── 03_vecm.py               VECM estimation — α coefficients (main result)
├── 04_lp_irf.py             Jordà Local Projections — IRFs (robustness)
├── 05_figures.py         Generate all 4 publication figures
├── requirements.txt
└── README.md
```

---

## Variables

| Variable | Description | Source | Dataset | Unit |
|---|---|---|---|---|
| `LNG_SUPPLY` | EU27 natural gas imports | Eurostat | `nrg_cb_gasm` | TJ_GCV |
| `IT_GAS_ELEC` | Gas transformed into electricity — Italy | Eurostat | `nrg_cb_gasm` | TJ_GCV |
| `IT_IP` | Industrial production index — Italy (SCA) | Eurostat | `sts_inpr_m` | Index 2015=100 |
| `FR_IP` | Industrial production index — France (SCA) | Eurostat | `sts_inpr_m` | Index 2015=100 |
| `FR_NUC` | Nuclear electricity generation — France | Eurostat | `nrg_cb_pem` | GWh |
| `EZ_HICP` | Eurozone HICP inflation | ECB SDW | `ICP` | Annual % |

**Endogenous system:** `LNG_SUPPLY`, `IT_GAS_ELEC`, `IT_IP`, `FR_IP`  
**Exogenous controls:** `FR_NUC`, COVID dummies (2020m3–2020m5), outlier dummies (|z| > 3)  
**Raw sample:** January 2010 – March 2025  
**Estimation sample:** January 2015 – December 2023 (108 monthly observations)

---

## How to run

```bash
git clone https://github.com/[your-username]/lng-nuclear-buffer-eurozone
cd lng-nuclear-buffer-eurozone
pip install -r requirements.txt

# Run scripts in order
python 01_download_data.py     # downloads svar_data.csv
python 02_diagnostics.py       # stationarity + cointegration + outliers
python 03_vecm.py              # VECM estimation + Granger tests
python 04_lp_irf.py            # Jordà LP-IRFs
python 05_figures.py        # publication figures
```

---

## Outputs

```
outputs/
├── svar_data.csv                 Raw dataset
├── stationarity_results.csv      ADF + KPSS table
├── vecm_summary.txt              Full VECM estimation output
├── granger_results.csv           Granger causality p-values
├── lp_irf.png                    LP-IRF diagnostic plot
├── lp_irf_data.csv               LP-IRF point estimates + 90% CI
├── fig1_structural_wedge.png/pdf Section 3.1 — short-run LP-IRFs
├── fig2_transmission.png/pdf     Section 3.2 — transmission mechanism
├── fig3_alpha_wedge.png/pdf      Section 3.3 — VECM α coefficients (main result)
└── fig4_granger_chain.png/pdf    Section 3.2 — Granger chain diagram
```

---

## Methodology

### Stage 1 — VECM
- Johansen trace test confirms cointegrating rank r = 3
- Loading coefficients (α) quantify speed and direction of adjustment
- **Key result:** IT_IP α = −0.238 (p < 0.001), FR_IP α = +0.131 (p = 0.003) on ec3
- Opposite signs on the same cointegrating vector = structural divergence

### Stage 2 — Jordà Local Projections
- OLS estimated independently at each horizon h = 0, …, 24
- Shock: standardised AR(2) residual on ΔLNG_SUPPLY
- Newey-West HAC standard errors (robust to serial correlation)
- 68% and 90% confidence intervals reported

### Outlier treatment
Additive impulse dummies for all observations where |z-score| > 3 in ΔIT_IP or ΔFR_IP, plus COVID dummies for 2020m3–2020m5 and 2020m6 (France). All dummy dates are listed in `02_diagnostics.py` output.

---

## Data sources

- **Eurostat API:** `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data`
- **ECB Data Portal:** `https://data-api.ecb.europa.eu/service/data`
- No API keys required for either source
