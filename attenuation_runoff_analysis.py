"""
attenuation_runoff_analysis.py
==============================
Complete data analysis for:

  "Do runoff models need rainfall? Direct attenuation-runoff modelling with a
   commercial microwave link" (Esmaeil Nia & Shokri)

One self-contained script, three stages:

  1. FORCING   raw 15-min CML received power -> hourly de-baselined attenuation
               A(t) = max(B6h - Pavg, 0)  (6 h centred moving-average baseline)
  2. DAILY     per catchment, 5-seed multistart differential evolution:
               (R)  GR4J rainfall benchmark (areal 3-gauge rain + sinusoidal PET)
               (A)  direct attenuation transfer model (no rainfall variable)
               (RA) output-level fusion  Q = w*QR + (1-w)*QA
               -> results_daily.csv (paper Table 3), multistart_daily_seeds.csv
  3. HOURLY    attenuation-only model at hourly resolution, same protocol
               -> results_hourly.csv, multistart_hourly_seeds.csv

Calibration 1 Dec 2017 - 31 Jul 2018, validation 1 Aug - 19 Dec 2018 (out of
sample), warm-up Nov 2017. Identical objective, optimiser, and flow cap for
every model. All seeds fixed: every number is exactly reproducible.

Expected data layout (see README.md for download sources):
  ./data_ML_paper_new_version/CML_data_ML.dat
  ./melbourne_water_flow_2017-10-01_to_2018-12-31/study_area_15_combined_river-flow_hourly.csv
  ./melbourne_water_rainfall_2017-10-01_to_2018-12-31/manifest.csv + station_csv/*.csv

Run:  python attenuation_runoff_analysis.py     (outputs -> ./outputs/)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from numba import njit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
SRC = ROOT / "data_ML_paper_new_version" / "CML_data_ML.dat"
FLOWDIR = ROOT / "melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR = ROOT / "melbourne_water_rainfall_2017-10-01_to_2018-12-31"

CAL = ("2017-12-01", "2018-07-31")
VAL = ("2018-08-01", "2018-12-19")
SEEDS = [1, 2, 3, 4, 5]
TARGETS = ["229624A", "229640A", "229625A", "228366A", "228393A",
           "228351B", "229638A", "229639A", "228368A"]
NAMES = {"229624A": "Gardiner", "229625A": "Ashwood", "228366A": "Knox",
         "228393A": "Scoresby", "228351B": "Wantirna South",
         "229638A": "Burwood East", "229640A": "Mt Waverley",
         "229639A": "Glen Waverley (reg.)", "228368A": "Rowville (reg.)"}

# --------------------------------------------------------------------------
# metrics and objective
# --------------------------------------------------------------------------
def nse(o, s):
    return 1 - np.sum((o - s)**2) / np.sum((o - o.mean())**2)

def kge(o, s):
    if s.std() == 0 or o.std() == 0:
        return -9.0
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

def bal(o, so, s):
    """Balanced calibration objective: 0.5*NSE(Q) + 0.5*NSE(sqrt(Q))."""
    return 0.5*nse(o, s) + 0.5*nse(so, np.sqrt(np.clip(s, 0, None)))

def hav(la, lo, la0, lo0):
    p = np.pi/180
    a = np.sin((la0-la)*p/2)**2 + np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2
    return 2*6371*np.arcsin(np.sqrt(a))

# --------------------------------------------------------------------------
# stage 1: CML forcing
# --------------------------------------------------------------------------
def build_forcing():
    df = pd.read_csv(SRC)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("Date").groupby("Date").mean(numeric_only=True)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="15min"))
    p_ref = df["Pavg"].rolling("6h", center=True, min_periods=2).mean()
    a = (p_ref - df["Pavg"]).clip(lower=0)
    hourly = pd.DataFrame({"A_avg_dB": a.resample("1h").mean(),
                           "n": df["Pavg"].resample("1h").count()})
    hourly.loc[hourly["n"] == 0, "A_avg_dB"] = np.nan
    att = hourly["A_avg_dB"]
    att.to_csv(OUT / "attenuation_hourly.csv")
    print(f"forcing: {len(att):,} hourly steps, "
          f"{100*(att > 0.1).mean():.0f}% wet (>0.1 dB), "
          f"{att.isna().sum():,} missing hours")
    return att

# --------------------------------------------------------------------------
# data loaders
# --------------------------------------------------------------------------
def load_flow():
    f = pd.read_csv(FLOWDIR / "study_area_15_combined_river-flow_hourly.csv")
    f["dateTime"] = pd.to_datetime(f["dateTime"])
    meta = f.groupby("siteId").agg(lat=("latitude", "first"), lon=("longitude", "first"))
    return f, meta

def load_rain_daily(rfiles, sid):
    """Daily rain-day total from the cumulative register (9am-9am window max).
    The real-time hourly field of the public export is unreliable during
    high-intensity rain, so daily totals are reconstructed from the register."""
    s = pd.read_csv(rfiles[sid]); s["Date/Time"] = pd.to_datetime(s["Date/Time"])
    cum = s.sort_values("Date/Time").set_index("Date/Time")["Cumulative rainfall (mm)"].astype(float)
    rainday = (cum.index - pd.Timedelta(hours=9, seconds=1)).floor("D")
    tot = cum.groupby(rainday).max().clip(lower=0)
    tot.index = tot.index + pd.Timedelta(days=1)
    return tot

def areal_rain_daily(rman, rfiles, lat, lon, k=3):
    d = rman.assign(dist=hav(rman.latitude.astype(float), rman.longitude.astype(float), lat, lon))
    return pd.concat([load_rain_daily(rfiles, i) for i in d.sort_values("dist")["siteId"].head(k)],
                     axis=1).mean(axis=1)

# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
@njit
def gr4j(P, E, x1, x2, x3, x4):
    """Standard GR4J (Perrin et al., 2003), daily."""
    n = len(P)
    nUH1 = max(int(np.ceil(x4)), 1); nUH2 = max(int(np.ceil(2.0*x4)), 1)
    UH1 = np.zeros(nUH1); UH2 = np.zeros(nUH2)
    for j in range(1, nUH1+1):
        a1 = 1.0 if j >= x4 else (j/x4)**2.5
        jm = j-1; a0 = 0.0 if jm <= 0 else (1.0 if jm >= x4 else (jm/x4)**2.5)
        UH1[j-1] = a1 - a0
    for j in range(1, nUH2+1):
        t = float(j)
        if t <= x4: b1 = 0.5*(t/x4)**2.5
        elif t < 2*x4: b1 = 1.0 - 0.5*(2.0-t/x4)**2.5
        else: b1 = 1.0
        tm = t - 1.0
        if tm <= 0: b0 = 0.0
        elif tm <= x4: b0 = 0.5*(tm/x4)**2.5
        elif tm < 2*x4: b0 = 1.0 - 0.5*(2.0-tm/x4)**2.5
        else: b0 = 1.0
        UH2[j-1] = b1 - b0
    StUH1 = np.zeros(nUH1); StUH2 = np.zeros(nUH2)
    S = 0.3*x1; R = 0.5*x3; Q = np.empty(n)
    for t in range(n):
        Pt = P[t]; Et = E[t]
        if Pt >= Et:
            Pn = Pt - Et; tw = np.tanh(Pn/x1); sr = S/x1
            Ps = x1*(1-sr*sr)*tw/(1+sr*tw); S = S + Ps; Pr = Pn - Ps
        else:
            En = Et - Pt; tw = np.tanh(En/x1); sr = S/x1
            Es = S*(2-sr)*tw/(1+(1-sr)*tw); S = S - Es; Pr = 0.0
        Perc = S*(1-(1+(4.0/9.0*S/x1)**4)**(-0.25)); S = S - Perc; Pr = Pr + Perc
        p1 = 0.9*Pr; p2 = 0.1*Pr
        for k in range(nUH1-1): StUH1[k] = StUH1[k+1] + UH1[k]*p1
        StUH1[nUH1-1] = UH1[nUH1-1]*p1; Q9 = StUH1[0]
        for k in range(nUH2-1): StUH2[k] = StUH2[k+1] + UH2[k]*p2
        StUH2[nUH2-1] = UH2[nUH2-1]*p2; Q1 = StUH2[0]
        F = x2*(R/x3)**3.5
        R = R + Q9 + F
        if R < 0: R = 0.0
        Qr = R*(1-(1+(R/x3)**4)**(-0.25)); R = R - Qr
        Qd = Q1 + F
        if Qd < 0: Qd = 0.0
        Q[t] = Qr + Qd
    return Q

BA = [(0.5, 0.999), (0.01, 10.), (0., 1.), (0., 0.95), (0.90, 0.99), (0., 1.), (0., 15.)]
def simA(p, x, qcap):
    """Direct attenuation transfer model: antecedent-wetness runoff coefficient
    (floor c0 + saturating term) feeding two parallel unit-gain linear stores."""
    a_w, cref, c0, aq, as_, f, k = p
    w = lfilter([1-a_w], [1, -a_w], x)
    cr = c0 + (1-c0)*w/(w+cref)
    u = cr*x
    return np.clip(k*(f*lfilter([1-aq], [1, -aq], u) + (1-f)*lfilter([1-as_], [1, -as_], u)), 0, qcap)

BR = [(50., 2000.), (-5., 5.), (10., 500.), (0.5, 10.), (0.5, 5.), (0., 10.)]
def simR(p, P, E, qcap):
    """GR4J + rainfall multiplier ps + output scalar c (mm -> m3/s)."""
    x1, x2, x3, x4, pscale, c = p
    return np.clip(c*gr4j(pscale*P, E, x1, x2, x3, x4), 0, qcap)

# --------------------------------------------------------------------------
# stage 2: daily multistart (R / A / RA)
# --------------------------------------------------------------------------
def daily_analysis(att, flow, fmeta, rman, rfiles):
    seed_rows, table = [], []
    for sid in TARGETS:
        name = NAMES[sid]
        lat, lon = float(fmeta.loc[sid, "lat"]), float(fmeta.loc[sid, "lon"])
        q = flow[flow.siteId == sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
        q = q[~q.index.duplicated()]
        rainD = areal_rain_daily(rman, rfiles, lat, lon)
        df = pd.concat([q.rename("Q"), att.rename("A")], axis=1).loc[att.index.min():att.index.max()]
        df["A"] = df["A"].interpolate(limit=6); df = df.dropna(subset=["Q"])
        d = pd.DataFrame({"Q": df.Q.resample("1D").mean(),
                          "A": df.A.resample("1D").mean()}).join(rainD.rename("P"), how="inner").dropna()
        idx = d.index
        cal = (idx >= CAL[0]) & (idx <= CAL[1]); val = (idx >= VAL[0]) & (idx <= VAL[1])
        E = 3.0 + 2.3*np.cos(2*np.pi*(idx.dayofyear.to_numpy()-20)/365.0)
        P = d.P.to_numpy(); A = (d.A/(d.A[cal].std() or 1)).to_numpy(); obs = d.Q.to_numpy()
        qcap = 1.5*obs.max(); o = obs[cal]; so = np.sqrt(np.clip(o, 0, None))
        per_seed = {}
        for seed in SEEDS:
            rR = differential_evolution(lambda p: -bal(o, so, simR(p, P, E, qcap)[cal]),
                                        BR, seed=seed, maxiter=120, popsize=18, tol=1e-6, polish=True)
            QR = simR(rR.x, P, E, qcap)
            rA = differential_evolution(lambda p: -bal(o, so, simA(p, A, qcap)[cal]),
                                        BA, seed=seed, maxiter=100, popsize=15, tol=1e-6, polish=True)
            QA = simA(rA.x, A, qcap)
            rW = differential_evolution(lambda w: -bal(o, so, (w[0]*QR+(1-w[0])*QA)[cal]),
                                        [(0., 1.)], seed=seed, maxiter=40, popsize=10, tol=1e-7, polish=True)
            w = rW.x[0]; QRA = w*QR + (1-w)*QA
            per_seed[seed] = {"R": (QR, -rR.fun), "A": (QA, -rA.fun), "RA": (QRA, -rW.fun), "w": w}
            for mode in ["R", "A", "RA"]:
                s, objc = per_seed[seed][mode]
                seed_rows.append({"catchment": name, "mode": mode, "seed": seed,
                                  "obj_cal": round(objc, 4),
                                  "NSE_cal": round(nse(o, s[cal]), 3), "KGE_cal": round(kge(o, s[cal]), 3),
                                  "NSE_val": round(nse(obs[val], s[val]), 3), "KGE_val": round(kge(obs[val], s[val]), 3),
                                  "fusion_w": round(w, 2) if mode == "RA" else np.nan})
        row = {"catchment": name}
        for mode in ["R", "A", "RA"]:
            best = max(SEEDS, key=lambda s: per_seed[s][mode][1])
            for r in seed_rows:
                if r["catchment"] == name and r["mode"] == mode:
                    r["chosen"] = int(r["seed"] == best)
            s, _ = per_seed[best][mode]
            row[f"NSEcal_{mode}"] = round(nse(o, s[cal]), 3)
            row[f"NSE_{mode}"] = round(nse(obs[val], s[val]), 3)
            row[f"KGE_{mode}"] = round(kge(obs[val], s[val]), 3)
            if mode == "RA":
                row["fusion_w"] = round(per_seed[best]["w"], 2)
        table.append(row)
        print(f"daily done: {name}", flush=True)
    pd.DataFrame(seed_rows).to_csv(OUT / "multistart_daily_seeds.csv", index=False)
    res = pd.DataFrame(table)
    res.to_csv(OUT / "results_daily.csv", index=False)
    return res

# --------------------------------------------------------------------------
# stage 3: hourly multistart (attenuation only)
# --------------------------------------------------------------------------
def hourly_analysis(att, flow):
    seed_rows, table = [], []
    for sid in TARGETS:
        name = NAMES[sid]
        q = flow[flow.siteId == sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
        q = q[~q.index.duplicated()]
        df = pd.concat([q.rename("Q"), att.rename("A")], axis=1).loc[att.index.min():att.index.max()]
        df["A"] = df["A"].interpolate(limit=6).fillna(0); df = df.dropna(subset=["Q"])
        idx = df.index
        cal = (idx >= CAL[0]) & (idx <= CAL[1]); val = (idx >= VAL[0]) & (idx <= VAL[1])
        A = (df.A/(df.A[cal].std() or 1)).to_numpy(); obs = df.Q.to_numpy()
        qcap = 1.5*obs.max(); o = obs[cal]; so = np.sqrt(np.clip(o, 0, None))
        best_obj, best = -9e9, None
        for seed in SEEDS:
            r = differential_evolution(lambda p: -bal(o, so, simA(p, A, qcap)[cal]),
                                       BA, seed=seed, maxiter=120, popsize=15, tol=1e-6, polish=True)
            s = simA(r.x, A, qcap)
            seed_rows.append({"catchment": name, "seed": seed, "obj_cal": round(-r.fun, 4),
                              "NSE_cal": round(nse(o, s[cal]), 3), "KGE_cal": round(kge(o, s[cal]), 3),
                              "NSE_val": round(nse(obs[val], s[val]), 3), "KGE_val": round(kge(obs[val], s[val]), 3)})
            if -r.fun > best_obj:
                best_obj, best = -r.fun, seed_rows[-1]
        for r in seed_rows:
            if r["catchment"] == name:
                r["chosen"] = int(r is best)
        table.append({"catchment": name, **{k: best[k] for k in
                      ["NSE_cal", "KGE_cal", "NSE_val", "KGE_val"]}})
        print(f"hourly done: {name}", flush=True)
    pd.DataFrame(seed_rows).to_csv(OUT / "multistart_hourly_seeds.csv", index=False)
    res = pd.DataFrame(table)
    res.to_csv(OUT / "results_hourly.csv", index=False)
    return res

# --------------------------------------------------------------------------
if __name__ == "__main__":
    att = build_forcing()
    flow, fmeta = load_flow()
    rman = pd.read_csv(RAINDIR / "manifest.csv")
    rman = rman[rman["rows"].astype(int) > 0]
    rfiles = {f.name.split("_")[0]: f for f in (RAINDIR / "station_csv").glob("*.csv")}

    pd.set_option("display.width", 200)
    daily = daily_analysis(att, flow, fmeta, rman, rfiles)
    print("\n=== DAILY (best of 5 seeds, validation Aug-Dec 2018) ===")
    print(daily.to_string(index=False))

    hourly = hourly_analysis(att, flow)
    print("\n=== HOURLY, attenuation only (best of 5 seeds, validation) ===")
    print(hourly.to_string(index=False))
    print("\noutputs ->", OUT)
