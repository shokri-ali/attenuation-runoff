"""
01_process_cml.py
-------------------
Turn the raw 15-min commercial microwave link (CML) received-power record into a
clean HOURLY ATTENUATION FORCING series for the direct attenuation->runoff model.

Design rule for this project: attenuation is the forcing. We do NOT convert it to
rainfall and we never call any quantity here "rainfall". Output is attenuation in
dB (and specific attenuation in dB/km).

Input : data_ML_paper_new_version/CML_data_ML.dat
        (single link, 22.715 GHz, 3.79 km, V-pol, tx 18 dBm; 1 Nov 2017 - 19 Dec 2018)
Output: outputs/cml_attenuation_hourly.csv
        outputs/cml_attenuation_diagnostic.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data_ML_paper_new_version" / "CML_data_ML.dat"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PATH_LENGTH_KM = 3.79          # nominal link length (paper); file PathLength col = 3.69
BASELINE_WINDOW = "6h"         # de-baselining window. Sweep (07b) showed the rain
                               # 'signal' = local mean - Pavg gives best runoff skill
                               # at ~5-6 h (KGE 0.44 vs 0.25 at 24 h): a short window
                               # removes the slow base attenuation (water-vapour/temp
                               # drift at 22.7 GHz) and isolates the sharp rain dips.
WET_THRESHOLD_DB = 1.0         # attenuation above this (after baseline) flagged "wet"

# --- load ------------------------------------------------------------------
df = pd.read_csv(SRC)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
df = df.sort_values("Date")
n_dup = int(df["Date"].duplicated().sum())
# Collapse duplicate timestamps (a handful exist) by averaging the numeric fields.
df = df.groupby("Date", as_index=True).mean(numeric_only=True)

# Received power columns are in dBm. Pmax = least attenuated (closest to dry),
# Pmin = most attenuated within the 15-min interval, Pavg = mean.
for c in ["Pmax", "Pmin", "Pavg", "tx"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Regularise onto a strict 15-min grid (exposes gaps; does not fill them).
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
df = df.reindex(full_idx)
df.index.name = "Date"
n_gap = int(df["Pavg"].isna().sum())

# --- de-baselining: short moving-average reference -------------------------
# Reference = short (6 h) centred moving average of Pavg. The rain 'signal' is the
# dip of Pavg below this local baseline; the slowly-varying base attenuation
# (water-vapour/temperature/hardware drift) is removed. Window optimised in 07b.
p_ref = df["Pavg"].rolling(BASELINE_WINDOW, center=True, min_periods=2).mean()
df["P_ref"] = p_ref

# --- attenuation (dB) ------------------------------------------------------
# Attenuation = how far received power has dropped below the dry reference.
df["A_avg"] = (df["P_ref"] - df["Pavg"]).clip(lower=0)   # mean over 15 min
df["A_max"] = (df["P_ref"] - df["Pmin"]).clip(lower=0)   # deepest in 15 min
df["wet"] = df["A_avg"] > WET_THRESHOLD_DB

# --- aggregate to hourly forcing -------------------------------------------
hourly = pd.DataFrame({
    "A_avg_dB": df["A_avg"].resample("1h").mean(),
    "A_max_dB": df["A_max"].resample("1h").max(),
    "wet_frac": df["wet"].resample("1h").mean(),
    "n_samples": df["Pavg"].resample("1h").count(),
})
hourly["k_avg_dB_per_km"] = hourly["A_avg_dB"] / PATH_LENGTH_KM
# Hours with no usable 15-min samples -> missing, not zero.
hourly.loc[hourly["n_samples"] == 0, ["A_avg_dB", "A_max_dB", "k_avg_dB_per_km"]] = np.nan
hourly.to_csv(OUT / "cml_attenuation_hourly.csv")

# --- diagnostics -----------------------------------------------------------
print("=== CML attenuation processing ===")
print(f"15-min records loaded     : {df['Pavg'].notna().sum():,}")
print(f"Duplicate timestamps      : {n_dup:,} (averaged)")
print(f"Missing 15-min slots (gap): {n_gap:,}")
print(f"Span                      : {df.index.min()}  ->  {df.index.max()}")
print(f"Dry reference power P_ref  : {df['P_ref'].median():.1f} dBm (median)")
print(f"Hourly steps written      : {len(hourly):,}")
print(f"Wet hours (A_avg>0.1 dB)  : {(hourly['A_avg_dB']>0.1).sum():,} "
      f"({100*(hourly['A_avg_dB']>0.1).mean():.1f}%)")
print(f"Max hourly A_avg          : {hourly['A_avg_dB'].max():.1f} dB")
print(f"Max 15-min A_max          : {df['A_max'].max():.1f} dB")
print(f"Output -> {OUT/'cml_attenuation_hourly.csv'}")

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
ax[0].plot(df.index, df["Pavg"], lw=0.3, color="0.5", label="Pavg (received)")
ax[0].plot(df.index, df["P_ref"], lw=0.8, color="C3", label="P_ref (dry baseline)")
ax[0].set_ylabel("dBm"); ax[0].legend(loc="lower left"); ax[0].set_title("Received power and dry-weather reference")

ax[1].plot(hourly.index, hourly["A_avg_dB"], lw=0.4, color="C0")
ax[1].set_ylabel("A_avg (dB)"); ax[1].set_title("Hourly attenuation forcing (full record)")

# one-month zoom to show event structure
zoom = hourly.loc["2018-04-01":"2018-04-30"]
ax[2].plot(zoom.index, zoom["A_avg_dB"], lw=0.9, color="C0")
ax[2].set_ylabel("A_avg (dB)"); ax[2].set_title("Zoom: April 2018")
fig.tight_layout()
fig.savefig(OUT / "cml_attenuation_diagnostic.png", dpi=130)
print(f"Plot   -> {OUT/'cml_attenuation_diagnostic.png'}")
