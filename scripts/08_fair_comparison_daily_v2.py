"""
08_fair_comparison_daily_v2.py
------------------------------
DAILY counterpart of script 05 (same model: production store for rain + smooth-
transfer for attenuation + shared routing + flow cap), using the IMPROVED 6h-baseline
attenuation. Confirms whether FUSION still helps at daily once attenuation is good.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from numba import njit
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
TARGETS=["229624A","229625A","228366A","228393A","228351B","229638A","229640A","229639A","228368A"]
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}
def hav(la,lo,la0,lo0):
    p=np.pi/180; a=(np.sin((la0-la)*p/2)**2+np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2); return 2*6371*np.arcsin(np.sqrt(a))
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)
@njit
def production(x,pet,Smax,etk,drk):
    S=0.5*Smax; out=np.empty_like(x)
    for t in range(len(x)):
        S+=x[t]; e=pet[t]*etk*(S/Smax)
        if e>S: e=S
        S-=e; q=drk*S; S-=q
        if S>Smax: q+=S-Smax; S=Smax
        out[t]=q
    return out
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
B=[(1.,400.),(0.,2.),(0.,0.5),(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.),(0.,5.),(0.,5.)]
def simulate(p,rain,att_,pet,mode,qcap):
    Smax,etk,drk,a_w,cref,c0,aq,as_,f,k,th_r,th_a=p
    if mode=="A": th_r=0.0
    if mode=="R": th_a=0.0
    u=np.zeros_like(rain)
    if th_r>0: u=u+production((th_r*rain).astype(np.float64),pet,Smax,etk,drk)
    if th_a>0:
        xa=th_a*att_; w=lfilter([1-a_w],[1,-a_w],xa); cr=c0+(1-c0)*w/(w+cref); u=u+cr*xa
    q=k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u))
    return np.clip(q,0,qcap)
rows=[]
for sid in TARGETS:
    lat,lon=float(fmeta.loc[sid,"lat"]),float(fmeta.loc[sid,"lon"])
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float); q=q[~q.index.duplicated()]
    rain=areal_rain(lat,lon)
    df=pd.concat([q.rename("Q"),att.rename("A"),rain.rename("P")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6); df=df.dropna(subset=["Q"])
    d=pd.DataFrame({"Q":df.Q.resample("1D").mean(),"P":df.P.resample("1D").sum(),"A":df.A.resample("1D").mean()}).dropna()
    idx=d.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    pet=(3.0+2.3*np.cos(2*np.pi*(idx.dayofyear.to_numpy()-20)/365.0))
    P=(d.P/(d.P[cal].std() or 1)).to_numpy(); A=(d.A/(d.A[cal].std() or 1)).to_numpy(); obs=d.Q.to_numpy()
    qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    sub={}
    for mode in ["R","A","RA"]:
        def loss(p):
            s=simulate(p,P,A,pet,mode,qcap)[cal]; return -(0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None))))
        r=differential_evolution(loss,B,seed=1,maxiter=100,popsize=15,tol=1e-6,polish=True)
        s=simulate(r.x,P,A,pet,mode,qcap); sub[mode]=nse(obs[val],s[val])
        rows.append({"catchment":NAMES[sid],"mode":mode,"NSE_val":round(nse(obs[val],s[val]),3),"KGE_val":round(kge(obs[val],s[val]),3)})
res=pd.DataFrame(rows); res.to_csv(OUT/"fair_metrics_daily_v2.csv",index=False)
pd.set_option("display.width",160)
print("=== DAILY fair comparison (improved 6h attenuation; same model as 05) ===")
print(res.pivot_table(index="catchment",columns="mode",values=["KGE_val","NSE_val"]).to_string())
resp=res[~res.catchment.str.contains("reg")]
print("\n=== mean over responsive catchments ==="); print(resp.groupby("mode")[["NSE_val","KGE_val"]].mean().round(3).to_string())
print("\n=== fusion vs best single source (NSE_val) ===")
for sid in TARGETS:
    sb=res[res.catchment==NAMES[sid]].set_index("mode")
    g=sb.loc["RA","NSE_val"]-max(sb.loc["R","NSE_val"],sb.loc["A","NSE_val"])
    print(f"  {NAMES[sid]:<22} R={sb.loc['R','NSE_val']:+.2f} A={sb.loc['A','NSE_val']:+.2f} RA={sb.loc['RA','NSE_val']:+.2f}  gain={g:+.2f}")
