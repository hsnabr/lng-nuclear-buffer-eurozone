"""
01_download_data.py
───────────────────
Download all 6 variables from Eurostat and ECB.
All endpoints verified working as of 2025.
No API keys required.

Output: svar_data.csv
"""
import requests
import pandas as pd
from io import StringIO

START = "2010-01"
END   = "2025-03"
OUT   = "svar_data.csv"

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ECB  = "https://data-api.ecb.europa.eu/service/data"
SP   = f"startPeriod={START}"
EP   = f"endPeriod={END}"

SERIES = {
    # Global shock: EU27 natural gas imports (TJ, gross calorific value)
    "LNG_SUPPLY":   f"{BASE}/nrg_cb_gasm?format=JSON&lang=EN&{SP}&{EP}&freq=M&geo=EU27_2020&unit=TJ_GCV&nrg_bal=IMP",
    # Transmission channel: gas transformed into electricity — Italy
    "IT_GAS_ELEC":  f"{BASE}/nrg_cb_gasm?format=JSON&lang=EN&{SP}&{EP}&freq=M&geo=IT&unit=TJ_GCV&nrg_bal=TI_EHG_MAP",
    # Response variable: industrial production — Italy (seasonally adjusted)
    "IT_IP":        f"{BASE}/sts_inpr_m?format=JSON&lang=EN&{SP}&{EP}&nace_r2=B-D&s_adj=SCA&unit=I15&geo=IT",
    # Response variable: industrial production — France (seasonally adjusted)
    "FR_IP":        f"{BASE}/sts_inpr_m?format=JSON&lang=EN&{SP}&{EP}&nace_r2=B-D&s_adj=SCA&unit=I15&geo=FR",
    # Exogenous control: nuclear electricity generation — France
    "FR_NUC":       f"{BASE}/nrg_cb_pem?format=JSON&lang=EN&{SP}&{EP}&geo=FR&siec=N9000&unit=GWH",
    # Anchor: Eurozone HICP annual rate of change
    "EZ_HICP":      f"{ECB}/ICP/M.U2.N.000000.4.ANR?{SP}&{EP}&format=csvdata",
}


def parse_eurostat(r: requests.Response) -> pd.Series:
    data = r.json()
    time_labels = list(data["dimension"]["time"]["category"]["label"].values())
    idx_map = {int(k): float(v) if v is not None else float("nan")
               for k, v in data["value"].items()}
    values = [idx_map.get(i, float("nan")) for i in range(len(time_labels))]
    index  = pd.to_datetime([t + "-01" for t in time_labels])
    return pd.Series(values, index=index, dtype=float)


def parse_ecb(r: requests.Response) -> pd.Series:
    df = pd.read_csv(StringIO(r.text))
    df["date"] = pd.to_datetime(df["TIME_PERIOD"].str[:7] + "-01")
    return df.set_index("date")["OBS_VALUE"].astype(float)


def download() -> pd.DataFrame:
    print(f"Downloading {START} → {END}\n")
    series = {}
    for name, url in SERIES.items():
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        s = parse_eurostat(r) if "JSON" in url else parse_ecb(r)
        s.name = name
        series[name] = s
        print(f"  ✓ {name:<20} {s.notna().sum()} obs")

    df = pd.concat(series, axis=1).sort_index()
    df.index.name = "date"
    return df


if __name__ == "__main__":
    import os
    df = download()
    print(f"\nShape  : {df.shape}")
    print(f"Missing:\n{df.isna().sum().to_string()}")
    df.to_csv(OUT)
    print(f"\nSaved  → {os.path.abspath(OUT)}")
