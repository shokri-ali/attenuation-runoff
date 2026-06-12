"""
10e_F8_scatter.py — F8: hourly observed vs simulated 1:1 scatter, validation
period, attenuation-only model, one dot per hour, one colour per catchment.
Sims refit exactly as in script 05 A-mode (same forcing, bounds, balanced
objective, seed, flow cap). Log-log axes; flows below FLOOR m3/s are clipped
to the axis floor (dry hours pile up on the edges instead of being hidden).
Legend quotes each catchment's hourly validation KGE from the refit.
Output: outputs/figures/F8_obs_sim_scatter_hourly.png
        outputs/f8_scatter_kge_check.csv (refit KGE vs script-05 stored KGE)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"; FIG=OUT/"figures"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
TARGETS=[("229624A","Gardiner"),("229640A","Mt Waverley"),("229625A","Ashwood"),
         ("228366A","Knox"),("228393A","Scoresby"),("228351B","Wantirna South"),
         ("229638A","Burwood East"),("229639A","Glen Waverley (reg.)"),
         ("228368A","Rowville (reg.)")]
FLOOR=5e-3
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":110})

def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)

att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv")
flow["dateTime"]=pd.to_datetime(flow["dateTime"])
B=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]

fig,ax=plt.subplots(figsize=(7.2,7.2))
cmap=plt.get_cmap("tab10")
rows=[]
for ci,(sid,name) in enumerate(TARGETS):
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6)
    df["gap"]=df["A"].isna()          # link outage hours: forcing imposed as dry
    df["A"]=df["A"].fillna(0); df=df.dropna(subset=["Q"])
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
    kv=kge(obs[val],s[val])
    rows.append((name,round(kv,3)))
    gap=df["gap"].to_numpy()
    ok=val&~gap; ing=val&gap
    ov=np.clip(obs[ok],FLOOR,None); sv=np.clip(s[ok],FLOOR,None)
    ax.scatter(ov,sv,s=4,alpha=.25,color=cmap(ci),edgecolors="none",
               label=f"{name} (KGE {kv:.2f})",rasterized=True)
    ax.scatter(np.clip(obs[ing],FLOOR,None),np.clip(s[ing],FLOOR,None),s=4,alpha=.3,
               color="0.6",edgecolors="none",rasterized=True,zorder=1,
               label="link gap (forcing = dry)" if ci==0 else None)
    print(f"{name:22s} refit hourly KGE_val {kv:.3f}")

lim=(FLOOR,300)
ax.plot(lim,lim,"k-",lw=1,zorder=5,label="1:1")
ax.plot(lim,[5*l for l in lim],"k--",lw=.6,zorder=5)
ax.plot(lim,[l/5 for l in lim],"k--",lw=.6,zorder=5,label="5:1 / 1:5")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("observed flow (m$^3$/s)"); ax.set_ylabel("simulated flow (m$^3$/s)")
ax.set_title("Hourly observed vs attenuation-only simulated flow — validation (Aug–Dec 2018)\n"
             f"one dot per hour; flows < {FLOOR} m$^3$/s clipped to the axis floor")
leg=ax.legend(loc="upper left",fontsize=7.5,markerscale=3,framealpha=.95)
for lh in leg.legend_handles:
    try: lh.set_alpha(1)
    except Exception: pass
fig.tight_layout(); fig.savefig(FIG/"F8_obs_sim_scatter_hourly.png",dpi=300); plt.close(fig)

chk=pd.DataFrame(rows,columns=["catchment","KGE_refit"])
try:
    h=pd.read_csv(OUT/"fair_metrics_hourly.csv"); h=h[h["mode"]=="A"][["catchment","KGE_val"]]
    chk=chk.merge(h.rename(columns={"KGE_val":"KGE_script05"}),on="catchment",how="left")
except Exception: pass
chk.to_csv(OUT/"f8_scatter_kge_check.csv",index=False)
print(chk.to_string(index=False))
print("F8 ->",FIG/"F8_obs_sim_scatter_hourly.png")
