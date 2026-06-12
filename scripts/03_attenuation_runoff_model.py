"""
03_attenuation_runoff_model.py
------------------------------
Direct attenuation->runoff model + rainfall and rainfall+attenuation FUSION.

Goal (Ali's framing): attenuation is a NEW, independent information source for
runoff. We do NOT estimate rainfall from attenuation. We build ONE parsimonious
IHACRES-type transfer model and drive it three ways, comparing OUT OF SAMPLE:
    (R)  rainfall only          -- benchmark
    (A)  attenuation only       -- the new model
    (RA) rainfall + attenuation -- fusion (the headline question)

Model (all vectorised with scipy.signal.lfilter):
    forcing   x_t = th_r * rain_t + th_a * max(0, att_t - A0)      (each pre-scaled)
    wetness   s_t = a_w * s_{t-1} + (1-a_w) * x_t                  (antecedent index)
    eff input u_t = s_t * x_t                                      (nonlinear)
    routing   Q_t = [b1/(1-a1 z^-1)] u  +  [b2/(1-a2 z^-1)] u      (quick + slow stores)
Parameters fit by differential evolution to maximise KGE on the CALIBRATION
period; VALIDATION metrics are fully out of sample. Single-source modes fix the
unused input weight to zero.

Inputs : outputs/cml_attenuation_hourly.csv
         melbourne_water_flow_2017-10-01_to_2018-12-31/study_area_15_combined_river-flow_hourly.csv
         melbourne_water_rainfall_2017-10-01_to_2018-12-31/station_csv/*.csv  (+ manifest)
Outputs: outputs/model_metrics.csv
         outputs/model_hydrographs_<catchment>.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FLOWDIR = ROOT / "melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR = ROOT / "melbourne_water_rainfall_2017-10-01_to_2018-12-31"

WARMUP_END = "2017-11-30"
CAL = ("2017-12-01", "2018-07-31")
VAL = ("2018-08-01", "2018-12-19")

# Responsive catchments to model + two regulated counter-examples.
TARGETS = ["229624A","229625A","228366A","228393A","228351B","229638A","229640A",
           "229639A","228368A"]
NAMES = {"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
         "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
         "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}

def haversine_km(lat, lon, lat0, lon0):
    R, p = 6371.0, np.pi/180
    a = (np.sin((lat0-lat)*p/2)**2 + np.cos(lat*p)*np.cos(lat0*p)*np.sin((lon0-lon)*p/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))

def kge(obs, sim):
    if sim.std()==0 or obs.std()==0: return -9
    r = np.corrcoef(obs, sim)[0,1]
    return 1 - np.sqrt((r-1)**2 + (sim.std()/obs.std()-1)**2 + (sim.mean()/obs.mean()-1)**2)

def nse(obs, sim):
    return 1 - np.sum((obs-sim)**2)/np.sum((obs-obs.mean())**2)

# ---- load attenuation ------------------------------------------------------
att = pd.read_csv(OUT/"cml_attenuation_hourly.csv", index_col=0, parse_dates=True)["A_avg_dB"]

# ---- load flow (study area, m3/s) -----------------------------------------
flow = pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv")
flow["dateTime"] = pd.to_datetime(flow["dateTime"])
flow_meta = flow.groupby("siteId").agg(lat=("latitude","first"), lon=("longitude","first"))

# ---- rain gauge inventory (local station_csv) ------------------------------
rain_man = pd.read_csv(RAINDIR/"manifest.csv")
rain_man = rain_man[rain_man["rows"].astype(int) > 0].copy()   # drop empty (Blackburn)
rain_files = {f.name.split("_")[0]: f for f in (RAINDIR/"station_csv").glob("*.csv")}

RAIN_CAP_MM_H = 50.0   # physical hourly cap; removes telemetry accumulation-dump
                       # artefacts (e.g. a day's tips logged as 316 mm in one hour)

def load_rain(site_id):
    f = rain_files[site_id]
    s = pd.read_csv(f)
    s["Date/Time"] = pd.to_datetime(s["Date/Time"])
    # 'Current rainfall (mm)' is the true hourly series (NaN = dry); cap artefacts.
    s = (s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"]
         .astype(float).fillna(0.0).clip(upper=RAIN_CAP_MM_H))
    return s[~s.index.duplicated()]

RAIN_NEAREST_K = 3      # areal rainfall = mean of K nearest gauges (fair analogue
                        # to a path-averaged link; smoother than a single point)

def rain_for(site_id, lat, lon):
    """Areal rainfall: mean of the K nearest rain gauges (cleaned)."""
    d = rain_man.assign(dist=haversine_km(rain_man["latitude"].astype(float),
                                          rain_man["longitude"].astype(float), lat, lon))
    ids = d.sort_values("dist")["siteId"].head(RAIN_NEAREST_K).tolist()
    areal = pd.concat([load_rain(i) for i in ids], axis=1).mean(axis=1)
    return "+".join(ids), areal

# ---- model -----------------------------------------------------------------
def simulate(p, rain, att_eff_raw, mode):
    a_w, cref, c0, aq, as_, f, k, th_r, th_a, A0 = p
    att_eff = np.clip(att_eff_raw - A0, 0, None)
    if mode == "R":  x = th_r*rain
    elif mode == "A": x = th_a*att_eff
    else:            x = th_r*rain + th_a*att_eff
    w = lfilter([1-a_w], [1, -a_w], x)          # antecedent wetness (DC gain 1)
    cr = c0 + (1-c0) * w/(w + cref)              # runoff coeff in [c0,1): c0 = floor
    u = cr * x                                   # effective input (immediate for c0>0)
    # two parallel stores, each unit DC gain -> volume conserved; k = runoff coeff
    quick = lfilter([1-aq], [1, -aq], u)
    slow  = lfilter([1-as_], [1, -as_], u)
    return k * (f*quick + (1-f)*slow)

#         a_w          cref       c0        aq         a_slow       f        k       th_r     th_a      A0
BOUNDS = [(0.5,0.999),(0.01,10.),(0.,1.),(0.0,0.95),(0.95,0.998),(0.0,1.0),(0.0,50.),(0.0,3.0),(0.0,3.0),(0.0,3.0)]

def fit(rain, att_eff, obs, cal_mask, mode):
    o = obs[cal_mask]
    so = np.sqrt(np.clip(o, 0, None))
    def loss(p):
        sim = simulate(p, rain, att_eff, mode)[cal_mask]
        # Balanced NSE on raw + sqrt flow: raw rewards peaks, sqrt rewards the
        # bulk/low flows -> prevents peak-overfitting that diverges out of sample.
        # Same objective for every forcing mode keeps the comparison fair.
        return -(0.5*nse(o, sim) + 0.5*nse(so, np.sqrt(np.clip(sim, 0, None))))
    res = differential_evolution(loss, BOUNDS, seed=1, maxiter=120, popsize=15,
                                 tol=1e-6, polish=True)
    return res.x

# ---- run -------------------------------------------------------------------
rows = []
hydro = {}
for sid in TARGETS:
    lat, lon = float(flow_meta.loc[sid,"lat"]), float(flow_meta.loc[sid,"lon"])
    q = (flow.loc[flow["siteId"]==sid].sort_values("dateTime")
         .set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float))
    q = q[~q.index.duplicated()]
    rgid, rain = rain_for(sid, lat, lon)

    df = pd.concat([q.rename("Q"), att.rename("A"), rain.rename("P")], axis=1)
    df = df.loc[att.index.min():att.index.max()]
    # fill gaps: attenuation/rain missing -> 0 (dry); drop rows w/o flow
    df["A"] = df["A"].interpolate(limit=6).fillna(0.0)
    df["P"] = df["P"].fillna(0.0)
    df = df.dropna(subset=["Q"])
    idx = df.index

    # pre-scale forcings by calibration-period std (keeps weights comparable)
    cal_mask = (idx >= CAL[0]) & (idx <= CAL[1])
    val_mask = (idx >= VAL[0]) & (idx <= VAL[1])
    P = (df["P"]/ (df["P"][cal_mask].std() or 1)).to_numpy()
    A = (df["A"]/ (df["A"][cal_mask].std() or 1)).to_numpy()
    obs = df["Q"].to_numpy()

    days = len(df)/24.0
    print(f"[{NAMES[sid]:<20}] rain~{df['P'].sum()/days*365:.0f} mm/yr  "
          f"att_mean={df['A'].mean():.2f} dB  Qmean={df['Q'].mean():.3f} m3/s  "
          f"rain_gauge={rgid}")

    sims = {}
    for mode in ["R","A","RA"]:
        p = fit(P, A, obs, cal_mask, mode)
        sim = simulate(p, P, A, mode)
        sims[mode] = sim
        rows.append({"siteId": sid, "catchment": NAMES[sid], "rain_gauge": rgid,
                     "dist_link_km": round(haversine_km(lat,lon,-37.876,145.169),1),
                     "mode": mode,
                     "NSE_cal": round(nse(obs[cal_mask], sim[cal_mask]),3),
                     "KGE_cal": round(kge(obs[cal_mask], sim[cal_mask]),3),
                     "NSE_val": round(nse(obs[val_mask], sim[val_mask]),3),
                     "KGE_val": round(kge(obs[val_mask], sim[val_mask]),3)})
    hydro[sid] = (idx, obs, sims, val_mask)

res = pd.DataFrame(rows)
res.to_csv(OUT/"model_metrics.csv", index=False)

# ---- report ----------------------------------------------------------------
pd.set_option("display.width", 160)
print("=== Validation NSE/KGE by catchment and forcing ===")
piv = res.pivot_table(index=["catchment","dist_link_km"], columns="mode",
                      values=["NSE_val","KGE_val"])
print(piv.to_string())
print("\n=== Mean over responsive catchments (excl. regulated) ===")
resp = res[~res["catchment"].str.contains("reg")]
print(resp.groupby("mode")[["NSE_cal","NSE_val","KGE_val"]].mean().round(3).to_string())

# fusion vs best single source, out of sample
print("\n=== Does fusion help? (NSE_val) ===")
for sid in TARGETS:
    sub = res[res["siteId"]==sid].set_index("mode")
    best_single = max(sub.loc["R","NSE_val"], sub.loc["A","NSE_val"])
    gain = sub.loc["RA","NSE_val"] - best_single
    print(f"  {NAMES[sid]:<22} R={sub.loc['R','NSE_val']:.2f} A={sub.loc['A','NSE_val']:.2f} "
          f"RA={sub.loc['RA','NSE_val']:.2f}  fusion gain={gain:+.2f}")

# ---- hydrograph plots (validation) for 3 illustrative catchments ----------
for sid in ["229624A","228393A","229639A"]:
    idx, obs, sims, vmask = hydro[sid]
    fig, ax = plt.subplots(figsize=(12,3.6))
    ax.plot(idx[vmask], obs[vmask], color="k", lw=1.1, label="observed")
    ax.plot(idx[vmask], sims["A"][vmask], color="C0", lw=.9, alpha=.9, label="attenuation-only")
    ax.plot(idx[vmask], sims["RA"][vmask], color="C3", lw=.9, alpha=.9, label="rain+attenuation")
    ax.set_title(f"{NAMES[sid]} — validation"); ax.set_ylabel("flow (m3/s)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/f"model_hydro_{sid}.png", dpi=130); plt.close(fig)
print("\nPlots -> outputs/model_hydro_*.png")
