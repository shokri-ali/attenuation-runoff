"""Diagnose why rainfall-forced mode fails: correlations + fitted sim stats."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"outputs"
FLOWDIR = ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR = ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"

att = pd.read_csv(OUT/"cml_attenuation_hourly.csv", index_col=0, parse_dates=True)["A_avg_dB"]
flow = pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv")
flow["dateTime"]=pd.to_datetime(flow["dateTime"])

def load_rain(sid):
    f=list((RAINDIR/"station_csv").glob(f"{sid}_*.csv"))[0]
    s=pd.read_csv(f); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
    return s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"].astype(float)

for sid,rgid in [("229624A","229624A"),("228393A","228368A")]:
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    rain=load_rain(rgid); rain=rain[~rain.index.duplicated()]
    df=pd.concat([q.rename("Q"),att.rename("A"),rain.rename("P")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6).fillna(0); df["P"]=df["P"].fillna(0); df=df.dropna(subset=["Q"])
    print(f"\n==== {sid} (rain {rgid}) n={len(df)} ====")
    print(f"  rain: sum={df.P.sum():.0f}mm  max={df.P.max():.1f}  nonzero={100*(df.P>0).mean():.1f}%")
    print(f"  flow: mean={df.Q.mean():.3f} max={df.Q.max():.2f}")
    # hourly corr
    print(f"  corr(P,Q) hourly = {df.P.corr(df.Q):.3f}   corr(A,Q) hourly = {df.A.corr(df.Q):.3f}")
    # antecedent-smoothed rain (API) vs flow at several decay rates
    for aw in [0.9,0.95,0.98,0.99]:
        api=lfilter([1-aw],[1,-aw],df.P.to_numpy())
        print(f"    API(a={aw}) corr with Q = {np.corrcoef(api,df.Q)[0,1]:.3f}")
    # daily
    d=df.resample("1D").agg({"Q":"mean","P":"sum","A":"mean"})
    print(f"  daily corr(P,Q)={d.P.corr(d.Q):.3f}  corr(A,Q)={d.A.corr(d.Q):.3f}")
    # lag scan hourly: does flow lag rain?
    best=(0,0)
    for L in range(0,25):
        c=df.P.shift(L).corr(df.Q)
        if c>best[1]: best=(L,c)
    print(f"  best rain->flow lag = {best[0]} h (corr {best[1]:.3f})")
