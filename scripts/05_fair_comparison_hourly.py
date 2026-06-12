"""
05_fair_comparison_hourly.py
----------------------------
FINAL fair comparison at HOURLY resolution (preserves the CML's sub-daily
advantage -- the whole point of using attenuation, and what the rejected paper
threw away by going daily).

One coherent model, each source through its APPROPRIATE front-end, shared routing:
  rainfall   -> GR4J-style soil PRODUCTION STORE (+ climatological PET)  --\
                                                                            >-- u --> [quick + slow linear routing] x k --> Q
  attenuation-> smooth-transfer wetness loss (no production store needed) --/
Fusion is additive (u = u_rain + u_att); single-source modes set the other
input weight to zero (nested -> fully fair). Production loop is numba-JIT'd so
hourly global calibration is fast.

Calibrate (balanced NSE) 2017-12..2018-07 ; validate 2018-08..2018-12 (hourly).
Outputs: outputs/fair_metrics_hourly.csv ; outputs/fairh_hydro_<id>.png
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from numba import njit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
TARGETS=["229624A","229625A","228366A","228393A","228351B","229638A","229640A","229639A","228368A"]
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}

def hav(la,lo,la0,lo0):
    p=np.pi/180; a=(np.sin((la0-la)*p/2)**2+np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2)
    return 2*6371*np.arcsin(np.sqrt(a))
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)

@njit
def production(x, pet, Smax, etk, drk):
    S=0.5*Smax; out=np.empty_like(x)
    for t in range(len(x)):
        S+=x[t]
        e=pet[t]*etk*(S/Smax)
        if e>S: e=S
        S-=e
        q=drk*S; S-=q
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
    return (s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"].astype(float).fillna(0).clip(upper=50))
def areal_rain(lat,lon,k=3):
    d=rain_man.assign(dist=hav(rain_man.latitude.astype(float),rain_man.longitude.astype(float),lat,lon))
    ids=d.sort_values("dist")["siteId"].head(k).tolist()
    return pd.concat([load_rain(i) for i in ids],axis=1).mean(axis=1)

# params: Smax,etk,drk, a_w,cref,c0, aq,as_,f,k, th_r,th_a
# Regularised bounds: slow store capped at 0.99 (tau~100h, no near-unit-root drift)
# and gain k capped at 15 -> removes the explosive-amplification optima that made
# the rainfall benchmark diverge out of sample.
B=[(1.,400.),(0.,2.),(0.,0.5),(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.),(0.,5.),(0.,5.)]
def simulate(p, rain, att_, pet, mode, qcap=np.inf):
    Smax,etk,drk,a_w,cref,c0,aq,as_,f,k,th_r,th_a=p
    if mode=="A": th_r=0.0
    if mode=="R": th_a=0.0
    u=np.zeros_like(rain)
    if th_r>0: u=u+production((th_r*rain).astype(np.float64), pet, Smax,etk,drk)
    if th_a>0:
        xa=th_a*att_; w=lfilter([1-a_w],[1,-a_w],xa); cr=c0+(1-c0)*w/(w+cref); u=u+cr*xa
    q=k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u))
    # physical plausibility cap: no model may predict flow beyond 1.5x the gauge's
    # record max. Applied in calibration too, so exploding fits score badly and the
    # optimiser avoids them (self-regularising). Same cap for every mode -> fair.
    return np.clip(q, 0.0, qcap)

def pet_hourly(idx):
    doy=idx.dayofyear.to_numpy(); return (3.0+2.3*np.cos(2*np.pi*(doy-20)/365.0))/24.0

rows=[]; hyd={}
for sid in TARGETS:
    lat,lon=float(fmeta.loc[sid,"lat"]),float(fmeta.loc[sid,"lon"])
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    rain=areal_rain(lat,lon)
    df=pd.concat([q.rename("Q"),att.rename("A"),rain.rename("P")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6).fillna(0); df["P"]=df["P"].fillna(0); df=df.dropna(subset=["Q"])
    idx=df.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    pet=pet_hourly(idx)
    P=(df.P/df.P[cal].std()).to_numpy(); A=(df.A/df.A[cal].std()).to_numpy(); obs=df.Q.to_numpy()
    qcap=1.5*obs.max()
    o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    sims={}
    for mode in ["R","A","RA"]:
        def loss(p):
            s=simulate(p,P,A,pet,mode,qcap)[cal]
            return -(0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None))))
        r=differential_evolution(loss,B,seed=1,maxiter=120,popsize=15,tol=1e-6,polish=True)
        s=simulate(r.x,P,A,pet,mode,qcap); sims[mode]=s
        rows.append({"catchment":NAMES[sid],"dist_km":round(hav(lat,lon,-37.876,145.169),1),"mode":mode,
                     "NSE_cal":round(nse(o,s[cal]),3),"NSE_val":round(nse(obs[val],s[val]),3),
                     "KGE_val":round(kge(obs[val],s[val]),3)})
    hyd[sid]=(idx,obs,sims,val)

res=pd.DataFrame(rows); res.to_csv(OUT/"fair_metrics_hourly.csv",index=False)
pd.set_option("display.width",160)
print("=== HOURLY fair comparison (appropriate front-end per source; shared routing) ===")
print(res.pivot_table(index="catchment",columns="mode",values=["NSE_val","KGE_val"]).to_string())
resp=res[~res.catchment.str.contains("reg")]
print("\n=== mean over responsive catchments ==="); print(resp.groupby("mode")[["NSE_cal","NSE_val","KGE_val"]].mean().round(3).to_string())
print("\n=== fusion vs best single source (NSE_val) ===")
for sid in TARGETS:
    sub=res[res.catchment==NAMES[sid]].set_index("mode")
    g=sub.loc["RA","NSE_val"]-max(sub.loc["R","NSE_val"],sub.loc["A","NSE_val"])
    print(f"  {NAMES[sid]:<22} R={sub.loc['R','NSE_val']:+.2f} A={sub.loc['A','NSE_val']:+.2f} RA={sub.loc['RA','NSE_val']:+.2f}  fusion gain={g:+.2f}")
for sid in ["229624A","228393A","228366A"]:
    idx,obs,sims,val=hyd[sid]
    fig,ax=plt.subplots(figsize=(12,3.4))
    ax.plot(idx[val],obs[val],"k",lw=1.0,label="observed")
    for m,c in [("R","C2"),("A","C0"),("RA","C3")]: ax.plot(idx[val],sims[m][val],c,lw=.8,alpha=.85,label=m)
    ax.set_title(f"{NAMES[sid]} hourly — validation"); ax.legend(); ax.set_ylabel("flow m3/s")
    fig.tight_layout(); fig.savefig(OUT/f"fairh_hydro_{sid}.png",dpi=130); plt.close(fig)
print("\nplots -> outputs/fairh_hydro_*.png")
