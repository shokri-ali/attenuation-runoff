"""
02_attenuation_flow_screening.py
--------------------------------
GO/NO-GO screening for the direct attenuation->runoff idea:
does path-averaged CML attenuation relate to measured discharge in the
catchments around the link, BEFORE building any model?

For each of the 15 study-area gauges we compute, at DAILY resolution
(robust first pass; the model will later run hourly):
  - Pearson r / R^2 between daily-mean attenuation and daily-mean total flow
  - same against quick flow (Lyne-Hollick baseflow separation)
  - distance from the gauge to the link midpoint
and rank by distance. This says whether the signal is there and where.

Inputs : outputs/cml_attenuation_hourly.csv  (from 01)
         melbourne_water_flow_2017-10-01_to_2018-12-31/study_area_15_combined_river-flow_hourly.csv
Outputs: outputs/screening_attenuation_vs_flow.csv
         outputs/screening_scatter_best.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FLOW = ROOT / "melbourne_water_flow_2017-10-01_to_2018-12-31" / "study_area_15_combined_river-flow_hourly.csv"
LINK_LAT, LINK_LON = -37.876, 145.169   # link midpoint (Glen Waverley)

def haversine_km(lat, lon, lat0, lon0):
    R = 6371.0
    p = np.pi / 180
    a = (np.sin((lat0-lat)*p/2)**2
         + np.cos(lat*p)*np.cos(lat0*p)*np.sin((lon0-lon)*p/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))

def lyne_hollick_quickflow(q, alpha=0.925, passes=3):
    """Recursive digital baseflow filter; returns quick flow (>=0)."""
    q = q.to_numpy(dtype=float)
    qf = q.copy()
    for _ in range(passes):
        f = np.zeros_like(qf)
        for i in range(1, len(qf)):
            f[i] = alpha*f[i-1] + (1+alpha)/2*(qf[i]-qf[i-1])
        f = np.clip(f, 0, qf)   # quick flow can't exceed total or go negative
        qf = f
    return qf

# --- attenuation: hourly -> daily mean -------------------------------------
att = pd.read_csv(OUT/"cml_attenuation_hourly.csv", index_col=0, parse_dates=True)
att_daily = att["A_avg_dB"].resample("1D").mean()

# --- flow: load, pivot, hourly -> daily ------------------------------------
flow = pd.read_csv(FLOW)
flow["dateTime"] = pd.to_datetime(flow["dateTime"])
meta = (flow.groupby("siteId")
        .agg(name=("siteName","first"), lat=("latitude","first"), lon=("longitude","first"))
        .reset_index())

rows = []
best = None
for _, m in meta.iterrows():
    sid = m["siteId"]
    s = flow.loc[flow["siteId"]==sid, ["dateTime","meanRiverFlow_m3_per_s"]].copy()
    s = s.sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    s = s[~s.index.duplicated()]
    qf_h = pd.Series(lyne_hollick_quickflow(s), index=s.index)
    tot_d = s.resample("1D").mean()
    qf_d = qf_h.resample("1D").mean()

    df = pd.concat([att_daily.rename("A"), tot_d.rename("Q"), qf_d.rename("QF")], axis=1).dropna()
    if len(df) < 60:
        continue
    r_tot = df["A"].corr(df["Q"])
    r_qf = df["A"].corr(df["QF"])
    dist = haversine_km(float(m["lat"]), float(m["lon"]), LINK_LAT, LINK_LON)
    rows.append({"siteId": sid, "name": m["name"], "dist_km": round(dist,1),
                 "n_days": len(df),
                 "r_total": round(r_tot,3), "R2_total": round(r_tot**2,3),
                 "r_quick": round(r_qf,3),  "R2_quick": round(r_qf**2,3)})
    if best is None or r_qf**2 > best[1]:
        best = (m["name"], r_qf**2, df.copy())

res = pd.DataFrame(rows).sort_values("dist_km").reset_index(drop=True)
res.to_csv(OUT/"screening_attenuation_vs_flow.csv", index=False)
print("=== Attenuation vs flow screening (daily) ===")
print(res.to_string(index=False))
print(f"\nBest quick-flow R^2: {best[0]} (R2={best[1]:.3f})")
print(f"Mean R2_quick over study gauges: {res['R2_quick'].mean():.3f} | "
      f"max R2_total: {res['R2_total'].max():.3f}")

# --- scatter for the best gauge --------------------------------------------
name, r2, df = best
fig, ax = plt.subplots(1, 2, figsize=(11,4.5))
ax[0].scatter(df["A"], df["Q"], s=12, alpha=.5)
ax[0].set_xlabel("daily mean attenuation (dB)"); ax[0].set_ylabel("daily mean total flow (m3/s)")
ax[0].set_title(f"{name}: A vs total flow")
ax[1].scatter(df["A"], df["QF"], s=12, alpha=.5, color="C1")
ax[1].set_xlabel("daily mean attenuation (dB)"); ax[1].set_ylabel("daily mean quick flow (m3/s)")
ax[1].set_title(f"{name}: A vs quick flow (R2={r2:.2f})")
fig.tight_layout(); fig.savefig(OUT/"screening_scatter_best.png", dpi=130)
print(f"Plot -> {OUT/'screening_scatter_best.png'}")
