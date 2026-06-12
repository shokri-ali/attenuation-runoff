"""
07_attenuation_decomposition.py
-------------------------------
Test Ali's two ideas about the attenuation signal:
 Q2  Does removing the slowly-varying BASE attenuation (A - moving average = the
     rain "signal") improve runoff estimation from attenuation alone?
 Q1  Does the BASE attenuation itself carry useful (moisture/humidity) info for
     runoff? (Our link is 22.715 GHz, on the 22.235 GHz water-vapour line, so the
     base ~ humidity.)  -> test base-only and signal+base together.

Decompose hourly received power Pavg:
   B_W      = rolling-mean baseline over window W
   A_signal = clip(B_W - Pavg, 0)             # fast rain-induced dips (base removed)
   A_base   = clip(P0  - B_W , 0)             # slow base (drift ~ humidity/temp)
   A_total  = clip(P0  - Pavg, 0)  ~ A_base + A_signal
Run the SAME attenuation-only transfer model (wetness loss + shared routing, capped)
on each forcing variant; compare validation KGE/NSE.

Outputs: outputs/attenuation_decomposition.csv + console summary.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
SRC=ROOT/"data_ML_paper_new_version"/"CML_data_ML.dat"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
RESP=["229624A","229625A","228366A","228393A","228351B","229640A"]  # responsive (excl. ephemeral Burwood, regulated)
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229640A":"Mt Waverley"}
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)

# --- rebuild hourly Pavg from raw link data ---
d=pd.read_csv(SRC); d["Date"]=pd.to_datetime(d["Date"],format="%d/%m/%Y %H:%M")
d=d.sort_values("Date").groupby("Date").mean(numeric_only=True)
d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="15min"))
Pavg=d["Pavg"].resample("1h").mean().interpolate(limit=6)
P0=Pavg.quantile(0.98)                                   # global dry (least-attenuated) level
def variants(W):
    B=Pavg.rolling(W,center=True,min_periods=4).mean()
    return dict(A_signal=(B-Pavg).clip(lower=0), A_base=(P0-B).clip(lower=0))
v24=variants("24h"); v6=variants("6h")
FORCING={
    "A_total (base+signal)": (P0-Pavg).clip(lower=0),
    "A_signal_24h (rain)"  : v24["A_signal"],
    "A_signal_6h"          : v6["A_signal"],
    "A_base_24h (moisture)": v24["A_base"],
}
att_idx=Pavg.dropna().index

# --- attenuation-only transfer model (capped) ---
B1=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]  # a_w,cref,c0,aq,as_,f,k
def sim1(p,x,qcap):
    a_w,cref,c0,aq,as_,f,k=p
    w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
    q=k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u))
    return np.clip(q,0,qcap)
# two-input model: signal as rain, base as moisture modulator of runoff coeff
B2=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.),(0.,5.)]  # +mb
def sim2(p,xs,xb,qcap):
    a_w,cref,c0,aq,as_,f,k,mb=p
    moist=lfilter([1-a_w],[1,-a_w],xb)          # smoothed base ~ antecedent moisture
    cr=c0+(1-c0)*moist/(moist+cref+1e-9)
    u=cr*xs*(1+mb*0)  # signal gated by moisture
    u=cr*xs
    q=k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u))
    return np.clip(q,0,qcap)

flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
def get_q(sid):
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    return q[~q.index.duplicated()]

rows=[]
for sid in RESP:
    q=get_q(sid)
    base_df=pd.concat([q.rename("Q")]+[s.rename(n) for n,s in FORCING.items()],axis=1).loc[att_idx.min():att_idx.max()]
    base_df=base_df.dropna(subset=["Q"]).fillna(0.0)
    idx=base_df.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    obs=base_df["Q"].to_numpy(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None)); qcap=1.5*obs.max()
    # single-input variants
    for name,series in FORCING.items():
        x=(base_df[name]/ (base_df[name][cal].std() or 1)).to_numpy()
        def L(p): s=sim1(p,x,qcap)[cal]; return -(0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None))))
        r=differential_evolution(L,B1,seed=1,maxiter=80,popsize=15,tol=1e-6,polish=True)
        s=sim1(r.x,x,qcap)
        rows.append({"catchment":NAMES[sid],"forcing":name,"KGE_val":round(kge(obs[val],s[val]),3),"NSE_val":round(nse(obs[val],s[val]),3)})
    # two-input: signal (rain) modulated by base (moisture)
    xs=(base_df["A_signal_24h (rain)"]/(base_df["A_signal_24h (rain)"][cal].std() or 1)).to_numpy()
    xb=(base_df["A_base_24h (moisture)"]/(base_df["A_base_24h (moisture)"][cal].std() or 1)).to_numpy()
    def L2(p): s=sim2(p,xs,xb,qcap)[cal]; return -(0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None))))
    r=differential_evolution(L2,B2,seed=1,maxiter=80,popsize=15,tol=1e-6,polish=True)
    s=sim2(r.x,xs,xb,qcap)
    rows.append({"catchment":NAMES[sid],"forcing":"signal x base (moisture-gated)","KGE_val":round(kge(obs[val],s[val]),3),"NSE_val":round(nse(obs[val],s[val]),3)})

res=pd.DataFrame(rows); res.to_csv(OUT/"attenuation_decomposition.csv",index=False)
pd.set_option("display.width",160)
print("=== Attenuation forcing variants — validation skill ===")
print(res.pivot_table(index="catchment",columns="forcing",values="KGE_val").to_string())
print("\n=== mean over responsive catchments ===")
print(res.groupby("forcing")[["KGE_val","NSE_val"]].mean().round(3).sort_values("KGE_val",ascending=False).to_string())
