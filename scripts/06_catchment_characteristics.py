"""
06_catchment_characteristics.py
-------------------------------
WHY does attenuation/fusion work in some catchments and not others?
Relate per-catchment physical character to model skill (hourly metrics from 05).

Per catchment: mean/max flow, flashiness (Qmax/Qmean), baseflow index (BFI),
zero-flow fraction, distance to link, and the raw hourly/daily correlations of
attenuation and (areal) rainfall with flow. Merge with fair_metrics_hourly.csv.
Output: outputs/catchment_characteristics.csv  + console interpretation.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}
def hav(la,lo,la0,lo0):
    p=np.pi/180; a=(np.sin((la0-la)*p/2)**2+np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2)
    return 2*6371*np.arcsin(np.sqrt(a))
def bfi(q, alpha=0.925, passes=3):
    q=q.to_numpy(float); b=q.copy()
    for _ in range(passes):
        f=np.zeros_like(b)
        for i in range(1,len(b)): f[i]=alpha*f[i-1]+(1+alpha)/2*(b[i]-b[i-1])
        f=np.clip(f,0,b); b=b-f  # baseflow = total - quickflow
    return np.nan_to_num(b).sum()/max(q.sum(),1e-9)

att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
fmeta=flow.groupby("siteId").agg(lat=("latitude","first"),lon=("longitude","first"))
rain_man=pd.read_csv(RAINDIR/"manifest.csv"); rain_man=rain_man[rain_man["rows"].astype(int)>0]
rfiles={f.name.split("_")[0]:f for f in (RAINDIR/"station_csv").glob("*.csv")}
def load_rain(sid):
    s=pd.read_csv(rfiles[sid]); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
    return s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"].astype(float).fillna(0).clip(upper=50)
def areal_rain(lat,lon,k=3):
    d=rain_man.assign(dist=hav(rain_man.latitude.astype(float),rain_man.longitude.astype(float),lat,lon))
    return pd.concat([load_rain(i) for i in d.sort_values("dist")["siteId"].head(k)],axis=1).mean(axis=1)

rows=[]
for sid,name in NAMES.items():
    lat,lon=float(fmeta.loc[sid,"lat"]),float(fmeta.loc[sid,"lon"])
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    rain=areal_rain(lat,lon)
    df=pd.concat([q.rename("Q"),att.rename("A"),rain.rename("P")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6).fillna(0); df["P"]=df["P"].fillna(0); df=df.dropna(subset=["Q"])
    d=pd.DataFrame({"Q":df.Q.resample("1D").mean(),"P":df.P.resample("1D").sum(),"A":df.A.resample("1D").mean()}).dropna()
    rows.append({"catchment":name,"dist_km":round(hav(lat,lon,-37.876,145.169),1),
                 "Qmean":round(df.Q.mean(),3),"Qmax":round(df.Q.max(),1),
                 "flashiness":round(df.Q.max()/max(df.Q.mean(),1e-9),0),
                 "BFI":round(bfi(df.Q),2),"zeroflow_%":round(100*(df.Q<=0.001).mean(),1),
                 "corrA_hr":round(df.A.corr(df.Q),2),"corrP_hr":round(df.P.corr(df.Q),2),
                 "corrA_day":round(d.A.corr(d.Q),2),"corrP_day":round(d.P.corr(d.Q),2)})
char=pd.DataFrame(rows)

m=pd.read_csv(OUT/"fair_metrics_hourly.csv")
piv=m.pivot_table(index="catchment",columns="mode",values="NSE_val")
piv.columns=[f"NSE_{c}" for c in piv.columns]
out=char.merge(piv.reset_index(),on="catchment",how="left")
out["fusion_gain"]=out["NSE_RA"]-out[["NSE_R","NSE_A"]].max(axis=1)
out.to_csv(OUT/"catchment_characteristics.csv",index=False)
pd.set_option("display.width",200); pd.set_option("display.max_columns",30)
print(out.to_string(index=False))
print("\n--- how does attenuation skill (NSE_A) relate to catchment character? ---")
for col in ["dist_km","Qmean","flashiness","BFI","corrA_day","corrP_day"]:
    print(f"  corr(NSE_A, {col:<11}) = {out['NSE_A'].corr(out[col]):+.2f}")
