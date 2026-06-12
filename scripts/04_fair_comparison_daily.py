"""
04_fair_comparison_daily.py
---------------------------
FAIR daily comparison of three runoff information sources, with ONE identical
model structure (so no source is advantaged):
    (R)  rainfall            (A)  attenuation            (RA) rainfall + attenuation

Model = GR4J-style soil-moisture PRODUCTION STORE (with climatological Melbourne
PET) -> two parallel linear routing stores -> runoff coefficient k. The production
store is the piece the simple transfer model lacked: it converts concentrated
rainfall into runoff via storage + evapotranspiration, giving rainfall a fair shot.
The SAME bucket is applied to every forcing (params calibrated per source).

Calibrate (NSE-balanced) on 2017-12..2018-07, validate 2018-08..2018-12.

Inputs : outputs/cml_attenuation_hourly.csv ; MW flow & rainfall folders
Outputs: outputs/fair_metrics_daily.csv ; outputs/fair_hydro_<id>.png
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
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

att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
fmeta=flow.groupby("siteId").agg(lat=("latitude","first"),lon=("longitude","first"))
rain_man=pd.read_csv(RAINDIR/"manifest.csv"); rain_man=rain_man[rain_man["rows"].astype(int)>0]
rfiles={f.name.split("_")[0]:f for f in (RAINDIR/"station_csv").glob("*.csv")}
def load_rain(sid):
    s=pd.read_csv(rfiles[sid]); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
    return (s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"].astype(float)
            .fillna(0).clip(upper=50))
def areal_rain(lat,lon,k=3):
    d=rain_man.assign(dist=hav(rain_man.latitude.astype(float),rain_man.longitude.astype(float),lat,lon))
    ids=d.sort_values("dist")["siteId"].head(k).tolist()
    return pd.concat([load_rain(i) for i in ids],axis=1).mean(axis=1)

def pet_daily(idx):  # climatological Melbourne PET (mm/day), peak late Jan
    doy=idx.dayofyear.to_numpy()
    return 3.0 + 2.3*np.cos(2*np.pi*(doy-20)/365.0)

def production_route(x, pet, p):
    Smax,etk,drk,aq,as_,f,k = p
    S=0.5*Smax; qgen=np.empty_like(x)
    for t in range(len(x)):
        S+=x[t]
        S-=min(pet[t]*etk*(S/Smax), S)          # evapotranspiration
        q=drk*S; S-=q                            # slow drainage (baseflow gen)
        if S>Smax: q+=S-Smax; S=Smax             # saturation excess (quick gen)
        qgen[t]=q
    return k*(f*lfilter([1-aq],[1,-aq],qgen)+(1-f)*lfilter([1-as_],[1,-as_],qgen))

# params: Smax,etk,drk,aq,as_,f,k, th_r, th_a
B=[(1,400),(0,2),(0,0.5),(0,0.8),(0.8,0.99),(0,1),(0,30),(0,5),(0,5)]
def split_inputs(p,rain,att,mode):
    *core, th_r, th_a = p
    if mode=="R": x=th_r*rain
    elif mode=="A": x=th_a*att
    else: x=th_r*rain+th_a*att
    return np.asarray(core), x

rows=[]; hyd={}
for sid in TARGETS:
    lat,lon=float(fmeta.loc[sid,"lat"]),float(fmeta.loc[sid,"lon"])
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    rain=areal_rain(lat,lon);
    df=pd.concat([q.rename("Q"),att.rename("A"),rain.rename("P")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6); df=df.dropna(subset=["Q"])
    # aggregate to DAILY: flow mean, rain sum, attenuation mean
    d=pd.DataFrame({"Q":df.Q.resample("1D").mean(),"P":df.P.resample("1D").sum(),
                    "A":df.A.resample("1D").mean()}).dropna()
    idx=d.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    pet=pet_daily(idx)
    P=(d.P/d.P[cal].std()).to_numpy(); A=(d.A/d.A[cal].std()).to_numpy(); obs=d.Q.to_numpy()
    o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    sims={}
    for mode in ["R","A","RA"]:
        def loss(p):
            core,x=split_inputs(p,P,A,mode); s=production_route(x,pet,core)[cal]
            return -(0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None))))
        r=differential_evolution(loss,B,seed=1,maxiter=80,popsize=15,tol=1e-6,polish=True)
        core,x=split_inputs(r.x,P,A,mode); s=production_route(x,pet,core); sims[mode]=s
        rows.append({"catchment":NAMES[sid],"dist_km":round(hav(lat,lon,-37.876,145.169),1),"mode":mode,
                     "NSE_cal":round(nse(o,s[cal]),3),"NSE_val":round(nse(obs[val],s[val]),3),
                     "KGE_val":round(kge(obs[val],s[val]),3)})
    hyd[sid]=(idx,obs,sims,val)

res=pd.DataFrame(rows); res.to_csv(OUT/"fair_metrics_daily.csv",index=False)
pd.set_option("display.width",160)
print("=== DAILY fair comparison (same GR4J-style model, 3 forcings) ===")
print(res.pivot_table(index="catchment",columns="mode",values=["NSE_val","KGE_val"]).to_string())
resp=res[~res.catchment.str.contains("reg")]
print("\n=== mean over responsive catchments ===")
print(resp.groupby("mode")[["NSE_cal","NSE_val","KGE_val"]].mean().round(3).to_string())
print("\n=== fusion vs best single source (NSE_val) ===")
for sid in TARGETS:
    sub=res[res.catchment==NAMES[sid]].set_index("mode")
    g=sub.loc["RA","NSE_val"]-max(sub.loc["R","NSE_val"],sub.loc["A","NSE_val"])
    print(f"  {NAMES[sid]:<22} R={sub.loc['R','NSE_val']:+.2f} A={sub.loc['A','NSE_val']:+.2f} RA={sub.loc['RA','NSE_val']:+.2f}  fusion gain={g:+.2f}")
for sid in ["229624A","228393A"]:
    idx,obs,sims,val=hyd[sid]
    fig,ax=plt.subplots(figsize=(11,3.4))
    ax.plot(idx[val],obs[val],"k",lw=1.2,label="observed")
    for m,c in [("R","C2"),("A","C0"),("RA","C3")]: ax.plot(idx[val],sims[m][val],c,lw=1,alpha=.85,label=m)
    ax.set_title(f"{NAMES[sid]} daily — validation"); ax.legend(); ax.set_ylabel("flow m3/s")
    fig.tight_layout(); fig.savefig(OUT/f"fair_hydro_{sid}.png",dpi=130); plt.close(fig)
print("\nplots -> outputs/fair_hydro_*.png")
