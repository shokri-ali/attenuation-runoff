"""Why can't rainfall fit even in calibration? Inspect fitted params + sim stats,
and test whether the issue is baseflow (add a constant) or routing."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
def load_rain(sid):
    f=list((RAINDIR/"station_csv").glob(f"{sid}_*.csv"))[0]; s=pd.read_csv(f); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
    return (s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"].astype(float).fillna(0).clip(upper=50))
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)

sid="229624A"  # Gardiner
q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float); q=q[~q.index.duplicated()]
rain=load_rain(sid)
df=pd.concat([q.rename("Q"),att.rename("A"),rain.rename("P")],axis=1).loc[att.index.min():att.index.max()]
df["A"]=df["A"].interpolate(limit=6).fillna(0); df["P"]=df["P"].fillna(0); df=df.dropna(subset=["Q"])
idx=df.index; cal=(idx>="2017-12-01")&(idx<="2018-07-31")
P=(df.P/df.P[cal].std()).to_numpy(); A=(df.A/df.A[cal].std()).to_numpy(); obs=df.Q.to_numpy()
print(f"obs flow: min={obs[cal].min():.3f} mean={obs[cal].mean():.3f} max={obs[cal].max():.2f}")
print(f"  baseflow proxy (10th pct)={np.percentile(obs[cal],10):.3f}  (frac of mean={np.percentile(obs[cal],10)/obs[cal].mean():.2f})")

def sim_base(p,x,with_qb):
    a_w,cref,c0,aq,as_,f,k,qb=p
    w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
    Q=k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u))
    return (qb+Q) if with_qb else Q
B=[(.5,.999),(.01,10),(0,1),(0,.95),(.95,.998),(0,1),(0,50),(0,2)]
for label,x,qb in [("RAIN no-Qb",P,False),("RAIN +Qb",P,True),("ATT no-Qb",A,False),("ATT +Qb",A,True)]:
    o=obs[cal]
    def L(p):
        s=sim_base(p,x,qb)[cal]; return -(0.5*nse(o,s)+0.5*nse(np.sqrt(np.clip(o,0,None)),np.sqrt(np.clip(s,0,None))))
    r=differential_evolution(L,B,seed=1,maxiter=120,popsize=15,tol=1e-6,polish=True)
    p=r.x; s=sim_base(p,x,qb)
    print(f"\n{label}: NSE_cal={nse(o,s[cal]):.3f}")
    print(f"  params a_w={p[0]:.3f} cref={p[1]:.2f} c0={p[2]:.2f} aq={p[3]:.3f} as={p[4]:.4f} f={p[5]:.2f} k={p[6]:.2f} qb={p[7]:.3f}")
    print(f"  sim: min={s[cal].min():.3f} mean={s[cal].mean():.3f} max={s[cal].max():.2f}")
