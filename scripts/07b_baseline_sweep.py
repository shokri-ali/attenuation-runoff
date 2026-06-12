"""Sweep the de-baselining (moving-average) window to find the window that best
isolates the rain 'signal' for runoff. A_signal_W = clip(rollmean(Pavg,W) - Pavg,0).
Mean validation skill over responsive catchments vs W."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
SRC=ROOT/"data_ML_paper_new_version"/"CML_data_ML.dat"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
RESP=["229624A","229625A","228366A","228393A","228351B","229640A"]
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)
d=pd.read_csv(SRC); d["Date"]=pd.to_datetime(d["Date"],format="%d/%m/%Y %H:%M")
d=d.sort_values("Date").groupby("Date").mean(numeric_only=True)
d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="15min"))
Pavg=d["Pavg"].resample("1h").mean().interpolate(limit=6)
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
def get_q(sid):
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    return q[~q.index.duplicated()]
B1=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]
def sim1(p,x,qcap):
    a_w,cref,c0,aq,as_,f,k=p
    w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
    q=k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u))
    return np.clip(q,0,qcap)
qs={sid:get_q(sid) for sid in RESP}
print(f"{'window':>7} {'mean_KGE':>9} {'mean_NSE':>9}")
for W in [2,3,4,5,6,8,10,12,16,24]:
    B=Pavg.rolling(f"{W}h",center=True,min_periods=2).mean()
    sig=(B-Pavg).clip(lower=0)
    kk=[]; nn=[]
    for sid in RESP:
        df=pd.concat([qs[sid].rename("Q"),sig.rename("S")],axis=1).loc[Pavg.index.min():Pavg.index.max()].dropna(subset=["Q"]).fillna(0)
        idx=df.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
        obs=df.Q.to_numpy(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None)); qcap=1.5*obs.max()
        x=(df.S/(df.S[cal].std() or 1)).to_numpy()
        def L(p): s=sim1(p,x,qcap)[cal]; return -(0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None))))
        r=differential_evolution(L,B1,seed=1,maxiter=70,popsize=15,tol=1e-6,polish=True)
        s=sim1(r.x,x,qcap); kk.append(kge(obs[val],s[val])); nn.append(nse(obs[val],s[val]))
    print(f"{W:>6}h {np.mean(kk):>9.3f} {np.mean(nn):>9.3f}")
