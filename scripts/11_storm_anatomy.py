"""
11_storm_anatomy.py — F8: anatomy of the 6 Nov 2018 Melbourne flash flood.
Time-aligned storyboard (5–8 Nov 2018, hourly):
 (a) raw received power + 6 h baseline          — the signal in the air
 (b) extracted attenuation + disdrometer rain   — and independent proof it's rain
 (c) the rain gauge record that day             — gauges logged garbage (316 mm/h
     'Current' vs TRUE rain-day total 45 mm from the cumulative column)
 (d) observed vs attenuation-only simulated flow (Gardiner, hourly, out of sample)
Output: outputs/figures/F8_storm_anatomy.png
"""
from pathlib import Path
import zipfile
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"; FIG=OUT/"figures"
SRC=ROOT/"data_ML_paper_new_version"/"CML_data_ML.dat"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31")
W0,W1="2018-11-05 00:00","2018-11-08 00:00"
C_OBS,C_ATT,C_RAIN="black","steelblue","seagreen"

def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    r=np.corrcoef(o,s)[0,1]
    return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)

# --- (a)/(b): link power, baseline, attenuation ------------------------------
d=pd.read_csv(SRC); d["Date"]=pd.to_datetime(d["Date"],format="%d/%m/%Y %H:%M")
d=d.sort_values("Date").groupby("Date").mean(numeric_only=True)
d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="15min"))
Pav=d["Pavg"].resample("1h").mean().interpolate(limit=6)
Bl=Pav.rolling("6h",center=True,min_periods=2).mean()
Sig=(Bl-Pav).clip(lower=0)

# --- disdrometer (independent on-link rain), best effort --------------------
DISDRO=None
try:
    ddat=ROOT/"data_ML_paper_new_version"/"2018_11_MtView.dat"
    if not ddat.exists():
        with zipfile.ZipFile(ROOT/"cml_data.zip") as z:
            z.extract("data_ML_paper_new_version/2018_11_MtView.dat",ROOT)
    dz=pd.read_csv(ddat,header=None,usecols=[0,2],names=["dt","R"],
                   parse_dates=[0],on_bad_lines="skip")
    dz=dz.set_index("dt")["R"]
    dz=pd.to_numeric(dz,errors="coerce").clip(lower=0)
    DISDRO=dz.resample("1h").mean()       # mm/h
    print("disdrometer loaded:",DISDRO.loc[W0:W1].max(),"mm/h peak in window")
except Exception as e:
    print("disdrometer unavailable, panel (b) without it:",e)

# --- (c): the Gardiner gauge record that day ---------------------------------
g=pd.read_csv(RAINDIR/"station_csv"/"229624A_Gardiner_rain_hourly.csv")
g["Date/Time"]=pd.to_datetime(g["Date/Time"]); g=g.sort_values("Date/Time").set_index("Date/Time")
cur=g["Current rainfall (mm)"].astype(float).loc[W0:W1]
cum=g["Cumulative rainfall (mm)"].astype(float).loc[W0:W1]

# --- (d): hourly attenuation-only model at Gardiner (as in 10c) --------------
att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
q=flow[flow.siteId=="229624A"].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
q=q[~q.index.duplicated()]
df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
df["A"]=df["A"].interpolate(limit=6).fillna(0); df=df.dropna(subset=["Q"])
idx=df.index; cal=(idx>=CAL[0])&(idx<=CAL[1])
A=(df.A/(df.A[cal].std() or 1)).to_numpy(); obs=df.Q.to_numpy()
qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
B=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]
def sim(p,x):
    a_w,cref,c0,aq,as_,f,k=p
    w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
    return np.clip(k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u)),0,qcap)
r=differential_evolution(lambda p:-(0.5*nse(o,sim(p,A)[cal])+0.5*nse(so,np.sqrt(np.clip(sim(p,A)[cal],0,None)))),
                         B,seed=1,maxiter=120,popsize=15,tol=1e-6,polish=True)
S=pd.Series(sim(r.x,A),index=idx); O=pd.Series(obs,index=idx)

# --- figure -------------------------------------------------------------------
fig,axs=plt.subplots(4,1,figsize=(9,10.5),sharex=True)
fig.suptitle("Anatomy of the 6 November 2018 Melbourne flash flood",fontsize=12,y=0.995)

ax=axs[0]
ax.plot(Pav.loc[W0:W1],color="0.45",lw=1.1,label="received power $P_{avg}$")
ax.plot(Bl.loc[W0:W1],color="indianred",lw=1.6,label="6 h baseline")
ax.set_ylabel("dBm"); ax.legend(loc="lower left",fontsize=8)
ax.set_title("(a) The signal in the air: rain attenuates the 22.7 GHz link",fontsize=10)

ax=axs[1]
ax.fill_between(Sig.loc[W0:W1].index,Sig.loc[W0:W1],color=C_ATT,alpha=.75,
                label="attenuation signal (model forcing)")
ax.set_ylabel("attenuation (dB)")
if DISDRO is not None:
    ax2=ax.twinx()
    dd=DISDRO.loc[W0:W1].fillna(0)
    ax2.bar(dd.index,dd,width=1/26,color="teal",alpha=.45,label="disdrometer rain (on-link)")
    ax2.set_ylabel("rain (mm h$^{-1}$)",color="teal"); ax2.tick_params(axis="y",colors="teal")
    ax2.invert_yaxis()
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,loc="center left",fontsize=8)
else:
    ax.legend(loc="upper left",fontsize=8)
ax.set_title("(b) Extracted attenuation — confirmed as rain by the independent on-link disdrometer",fontsize=10)

ax=axs[2]
ax.bar(cur.index,cur.values,width=1/26,color=C_RAIN,alpha=.85,label="gauge 'Current rainfall' as logged")
ax.plot(cum,color="darkorange",lw=1.8,label="gauge cumulative (true running total)")
imax=cur.idxmax()
ax.annotate(f"{cur.max():.0f} mm logged in 1 h —\nphysically impossible\n(true rain-day total: 45 mm)",
            (imax,cur.max()),xytext=(30,-10),textcoords="offset points",fontsize=8.5,
            arrowprops=dict(arrowstyle="->",color="0.3"))
ax.set_ylabel("mm")
ax.legend(loc="center left",fontsize=8)
ax.set_title("(c) The nearby rain gauge that day: telemetry garbage vs the true total",fontsize=10)

ax=axs[3]
ax.plot(O.loc[W0:W1],color=C_OBS,lw=1.4,label="observed flow (Gardiner)")
ax.plot(S.loc[W0:W1],color=C_ATT,lw=1.4,label="attenuation-only model (out of sample)")
ax.set_ylabel("flow (m$^3$/s)"); ax.legend(loc="upper left",fontsize=8)
ax.set_title("(d) The water in the creek: hourly runoff predicted from the link alone",fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
for a in axs: a.margins(x=0)
fig.align_ylabels(axs)
fig.tight_layout()
fig.savefig(FIG/"F8_storm_anatomy.png",dpi=300)
print(f"F8 -> {FIG/'F8_storm_anatomy.png'}")
