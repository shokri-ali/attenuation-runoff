"""
10c_F7_hourly.py — F7: the hourly evidence for the attenuation-runoff model.
(a) attenuation-only validation KGE per catchment, hourly vs daily
    (hourly from outputs/fair_metrics_hourly.csv; daily from outputs/gr4j_metrics.csv)
(b)/(c) single-event zooms (Gardiner, validation): hourly observed + hourly
    attenuation-only sim + observed daily mean as steps — shows what daily
    resolution loses (13–14 Dec peak 40.7 m3/s vs daily mean 8.7).
    Sim refit exactly as in script 05 (same forcing, bounds, objective, seed, cap).
Output: outputs/figures/F7_hourly_attenuation.png
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"; FIG=OUT/"figures"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
ORDER=["Gardiner","Mt Waverley","Ashwood","Knox","Scoresby","Wantirna South",
       "Burwood East","Glen Waverley (reg.)","Rowville (reg.)"]
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)

# (a) hourly vs daily attenuation-only KGE
h=pd.read_csv(OUT/"fair_metrics_hourly.csv"); h=h[h["mode"]=="A"][["catchment","KGE_val"]].rename(columns={"KGE_val":"hourly"})
d=pd.read_csv(OUT/"gr4j_metrics.csv")[["catchment","KGE_A"]].rename(columns={"KGE_A":"daily"})
m=h.merge(d,on="catchment"); m["o"]=m.catchment.map({c:i for i,c in enumerate(ORDER)}); m=m.sort_values("o")

# (b) Scoresby hourly attenuation-only sim (reproduce script-05 A-mode exactly)
att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
sid="228393A"   # Scoresby: best hourly KGE in panel (a), and the event scan
                # (find_showcase_events.py) ranks its 6 Nov 2018 storm (event
                # NSE 0.77, peak ratio 0.80) and 11 Aug winter event (NSE 0.74,
                # peak ratio 0.97) as the best honest showcases
q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
q=q[~q.index.duplicated()]
df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
df["A"]=df["A"].interpolate(limit=6).fillna(0); df=df.dropna(subset=["Q"])
idx=df.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
A=(df.A/(df.A[cal].std() or 1)).to_numpy(); obs=df.Q.to_numpy()
qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
B=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]
def sim(p,x):
    a_w,cref,c0,aq,as_,f,k=p
    w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
    return np.clip(k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u)),0,qcap)
r=differential_evolution(lambda p:-(0.5*nse(o,sim(p,A)[cal])+0.5*nse(so,np.sqrt(np.clip(sim(p,A)[cal],0,None)))),
                         B,seed=1,maxiter=120,popsize=15,tol=1e-6,polish=True)
s=sim(r.x,A)
kv,nv=kge(obs[val],s[val]),nse(obs[val],s[val])

sim_s=pd.Series(s,index=idx); obs_s=pd.Series(obs,index=idx)
fig=plt.figure(figsize=(9.5,9))
ax1=fig.add_subplot(3,1,1)
x=np.arange(len(m)); w=0.36
ax1.bar(x-w/2,m["hourly"],w,color="steelblue",label="hourly")
ax1.bar(x+w/2,m["daily"],w,color="lightsteelblue",edgecolor="steelblue",label="daily")
ax1.axhline(0,color="k",lw=.6)
ax1.set_xticks(x); ax1.set_xticklabels(m.catchment,rotation=30,ha="right",fontsize=8)
ax1.set_ylabel("KGE (validation)"); ax1.legend()
ax1.set_title("(a) Attenuation-only runoff skill at hourly and daily resolution")

# single-event zooms: hourly obs + hourly sim + the daily view of the same event
# (observed daily mean as grey steps) — what daily resolution throws away
obs_d=obs_s.resample("1D").mean()
ZOOMS=[("2018-11-05 00:00","2018-11-08 00:00",
        "(b) Scoresby — 6 Nov 2018 storm: the 10.2 m$^3$/s hourly peak becomes 3.8 in the daily mean"),
       ("2018-08-10 00:00","2018-08-13 00:00",
        "(c) Scoresby — 11 Aug 2018 winter event (validation), same comparison")]
for i,(z0,z1,ttl) in enumerate(ZOOMS):
    ax=fig.add_subplot(3,1,2+i)
    ax.plot(obs_s.loc[z0:z1],"k-",lw=1.1,label="observed (hourly)")
    ax.plot(sim_s.loc[z0:z1],color="steelblue",lw=1.1,alpha=.95,
            label=f"attenuation-only, hourly (full-validation KGE {kv:.2f}, NSE {nv:.2f})")
    dm=obs_d.loc[pd.Timestamp(z0).floor("D"):z1]
    ax.step(dm.index,dm.values,where="post",color="0.45",lw=1.4,ls="--",
            label="observed, daily mean (the daily-resolution view)")
    ax.set_xlim(pd.Timestamp(z0),pd.Timestamp(z1))
    ax.set_ylabel("flow (m$^3$/s)"); ax.set_title(ttl)
    if i==0: ax.legend(loc="upper right",fontsize=8)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout(); fig.savefig(FIG/"F7_hourly_attenuation.png",dpi=300)
print(f"F7 -> {FIG/'F7_hourly_attenuation.png'}  (Scoresby hourly KGE={kv:.3f}, NSE={nv:.3f})")
