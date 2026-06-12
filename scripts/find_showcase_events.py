"""
find_showcase_events.py — scan validation-period storm events for the best
showcase windows for F7 (b)/(c): hourly attenuation-only sim vs observed.
Refits the script-05 A-mode model (same bounds/objective/seed/cap) for the top
hourly catchments, finds observed events (peaks > 5x median absolute level),
and ranks +/-36 h windows by event NSE and peak ratio.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter, find_peaks
from scipy.optimize import differential_evolution

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
def nse(o,s):
    d=np.sum((o-o.mean())**2)
    return 1-np.sum((o-s)**2)/d if d>0 else np.nan

att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv")
flow["dateTime"]=pd.to_datetime(flow["dateTime"])
B=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]

for sid,name in [("228393A","Scoresby"),("229624A","Gardiner"),
                 ("229625A","Ashwood"),("228366A","Knox")]:
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6).fillna(0); df=df.dropna(subset=["Q"])
    idx=df.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    A=(df.A/(df.A[cal].std() or 1)).to_numpy(); obs=df.Q.to_numpy()
    qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    def sim(p,x):
        a_w,cref,c0,aq,as_,f,k=p
        w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
        return np.clip(k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u)),0,qcap)
    r=differential_evolution(lambda p:-(0.5*nse(o,sim(p,A)[cal])+0.5*nse(so,np.sqrt(np.clip(sim(p,A)[cal],0,None)))),
                             B,seed=1,maxiter=120,popsize=15,tol=1e-6,polish=True)
    s=sim(r.x,A)
    os_=pd.Series(obs,index=idx); ss=pd.Series(s,index=idx)
    ov,sv=os_[val],ss[val]
    pk,_=find_peaks(ov.values,height=max(5*np.median(ov[ov>0]),ov.max()*0.12),distance=48)
    rows=[]
    for p in pk:
        t=ov.index[p]; w0,w1=t-pd.Timedelta("36h"),t+pd.Timedelta("36h")
        oe,se=ov.loc[w0:w1],sv.loc[w0:w1]
        if len(oe)<24: continue
        rows.append((str(t)[:16],round(oe.max(),1),round(se.max(),1),
                     round(se.max()/oe.max(),2),round(nse(oe.values,se.values),2)))
    t=pd.DataFrame(rows,columns=["peak_time","obs_pk","sim_pk","pk_ratio","event_NSE"]).sort_values("event_NSE",ascending=False)
    print(f"=== {name} (full-val NSE {nse(ov.values,sv.values):.2f}) ===")
    print(t.to_string(index=False))
