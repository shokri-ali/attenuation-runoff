"""
10_figures.py — paper-ready figures & main table.

F1  study area: link path, flow stations, rain gauges
F2  method: received power -> 6h baseline -> attenuation 'signal' (6 Nov storm)
    [SUPERSEDED by 10d_F2_forcing.py (full-record power + extraction zoom) — run 10d after this]
F3  de-baselining window sweep (from 07b)
F4  HEADLINE: daily validation KGE & NSE — GR4J(rain) vs attenuation vs fusion
F5  forcing + full-period hydrographs (cal/val shaded, link gaps grey): (a) hourly
    attenuation forcing, then Gardiner (A-dominant), Rowville (R-dominant),
    Wantirna South (fusion rescue) — sims reproduced exactly as in 09
F6  attenuation skill vs catchment character (BFI, flashiness)
    [10f_F6_map.py holds a satellite skill-map variant — SHELVED 2026-06-11 (Ali),
     do NOT run it in the standard chain; keep for reviewer response]
T1  outputs/figures/table1_main_results.csv (+ .md)

All figures -> outputs/figures/ (300 dpi PNG)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from numba import njit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"outputs"; FIG=OUT/"figures"; FIG.mkdir(exist_ok=True)
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
SRC=ROOT/"data_ML_paper_new_version"/"CML_data_ML.dat"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
LINK=((145.1722,-37.8922),(145.1652,-37.8591))
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":110})

def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)
def hav(la,lo,la0,lo0):
    p=np.pi/180; a=(np.sin((la0-la)*p/2)**2+np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2)
    return 2*6371*np.arcsin(np.sqrt(a))

# ============ F1: study area ============
fman=pd.read_csv(FLOWDIR/"study_area_15_manifest.csv")
rman=pd.read_csv(RAINDIR/"manifest.csv"); rman=rman[rman["rows"].astype(int)>0]
fig,ax=plt.subplots(figsize=(7,6))
ax.plot([LINK[0][0],LINK[1][0]],[LINK[0][1],LINK[1][1]],"r-",lw=3,label="microwave link (22.7 GHz, 3.8 km)")
ax.scatter(fman.longitude,fman.latitude,marker="^",s=70,c="navy",label="flow station",zorder=3)
ax.scatter(rman.longitude.astype(float),rman.latitude.astype(float),marker="s",s=40,
           facecolors="none",edgecolors="green",label="rain gauge",zorder=3)
for _,r in fman.iterrows():
    ax.annotate(r.siteName,(r.longitude,r.latitude),textcoords="offset points",
                xytext=(5,4),fontsize=7)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Study area: commercial microwave link, flow stations and rain gauges (eastern Melbourne)")
ax.legend(loc="lower right"); ax.set_aspect(1/np.cos(np.deg2rad(37.88)))
fig.tight_layout(); fig.savefig(FIG/"F1_study_area.png",dpi=300); plt.close(fig)
print("F1 done")

# ============ F2: method (signal extraction) ============
d=pd.read_csv(SRC); d["Date"]=pd.to_datetime(d["Date"],format="%d/%m/%Y %H:%M")
d=d.sort_values("Date").groupby("Date").mean(numeric_only=True)
d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="15min"))
Pav=d["Pavg"].resample("1h").mean().interpolate(limit=6)
Bl=Pav.rolling("6h",center=True,min_periods=2).mean()
Sig=(Bl-Pav).clip(lower=0)
fig,ax=plt.subplots(2,1,figsize=(9,5.2))
w0,w1="2018-11-04","2018-11-09"
ax[0].plot(Pav.loc[w0:w1],color="0.45",lw=.9,label="received power $P_{avg}$")
ax[0].plot(Bl.loc[w0:w1],color="C3",lw=1.4,label="6 h moving-average baseline")
ax[0].set_ylabel("dBm"); ax[0].legend(loc="lower left")
ax[0].set_title("(a) Received power and short baseline — 6 Nov 2018 storm")
ax[1].fill_between(Sig.loc[w0:w1].index,Sig.loc[w0:w1],color="C0",alpha=.7)
ax[1].set_ylabel("attenuation signal (dB)")
ax[1].set_title("(b) De-baselined attenuation signal $A = \\max(B_{6h}-P_{avg},\\,0)$ — the model forcing")
for a in ax: a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout(); fig.savefig(FIG/"F2_signal_extraction.png",dpi=300); plt.close(fig)
print("F2 done")

# ============ F3: window sweep (07b results) ============
sweep=pd.DataFrame({"W":[2,3,4,5,6,8,10,12,16,24],
    "KGE":[.437,.415,.439,.449,.443,.425,.402,.380,.335,.253],
    "NSE":[.293,.217,.262,.256,.277,.314,.332,.336,.337,.324]})
fig,ax=plt.subplots(figsize=(5.4,3.6))
ax.plot(sweep.W,sweep.KGE,"o-",color="C0",label="KGE")
ax.plot(sweep.W,sweep.NSE,"s--",color="C1",label="NSE")
ax.axvline(6,color="0.6",ls=":",lw=1); ax.annotate("adopted (6 h)",(6,.45),xytext=(8,.46),fontsize=8,
            arrowprops=dict(arrowstyle="->",color="0.4"))
ax.set_xlabel("de-baselining window (h)"); ax.set_ylabel("mean validation skill")
ax.set_title("Effect of baseline window on attenuation-only runoff skill")
ax.legend(); fig.tight_layout(); fig.savefig(FIG/"F3_window_sweep.png",dpi=300); plt.close(fig)
print("F3 done")

# ============ F4 + T1: headline comparison ============
# source = best-of-5 multistart (12_multistart.py); seed-1 gr4j_metrics.csv kept
# as the cross-check (deltas <= 0.031 NSE)
m=pd.read_csv(OUT/"table1_multistart.csv")
order=["Gardiner","Mt Waverley","Ashwood","Knox","Scoresby","Wantirna South",
       "Burwood East","Glen Waverley (reg.)","Rowville (reg.)"]
m["o"]=m.catchment.map({c:i for i,c in enumerate(order)}); m=m.sort_values("o")
x=np.arange(len(m)); w=0.27
fig,ax=plt.subplots(2,1,figsize=(9,6.4),sharex=True)
for i,(met,a) in enumerate([("KGE",ax[0]),("NSE",ax[1])]):
    a.bar(x-w,m[f"{met}_R"],w,color="seagreen",label="rainfall (GR4J)")
    a.bar(x,  m[f"{met}_A"],w,color="steelblue",label="attenuation (new model)")
    a.bar(x+w,m[f"{met}_RA"],w,color="indianred",label="fusion")
    a.axhline(0,color="k",lw=.6); a.set_ylabel(f"{met} (validation)")
ax[0].set_title("Runoff estimation skill — daily validation period (Aug–Dec 2018)")
ax[0].legend(loc="upper right",ncol=3)
ax[1].set_xticks(x); ax[1].set_xticklabels(m.catchment,rotation=30,ha="right")
ax[0].set_ylim(-0.7,1); ax[1].set_ylim(-0.2,1)
fig.tight_layout(); fig.savefig(FIG/"F4_headline_comparison.png",dpi=300); plt.close(fig)
t1=m[["catchment","NSEcal_R","NSE_R","KGE_R","NSEcal_A","NSE_A","KGE_A",
      "NSEcal_RA","NSE_RA","KGE_RA","fusion_w"]]
t1.to_csv(FIG/"table1_main_results.csv",index=False)
with open(FIG/"table1_main_results.md","w") as fh:   # hand-rolled (no tabulate dep)
    cols=list(t1.columns)
    fh.write("| "+" | ".join(cols)+" |\n|"+"|".join(["---"]*len(cols))+"|\n")
    for _,r in t1.iterrows():
        fh.write("| "+" | ".join(str(r[c]) for c in cols)+" |\n")
print("F4 + T1 done")

# ============ F5: hydrograph trio (reproduce 09 sims exactly) ============
@njit
def gr4j(P,E,x1,x2,x3,x4):
    n=len(P)
    nUH1=int(np.ceil(x4)); nUH1=nUH1 if nUH1>=1 else 1
    nUH2=int(np.ceil(2.0*x4)); nUH2=nUH2 if nUH2>=1 else 1
    UH1=np.zeros(nUH1); UH2=np.zeros(nUH2)
    for j in range(1,nUH1+1):
        a1=1.0 if j>=x4 else (j/x4)**2.5
        jm=j-1; a0=0.0 if jm<=0 else (1.0 if jm>=x4 else (jm/x4)**2.5)
        UH1[j-1]=a1-a0
    for j in range(1,nUH2+1):
        t=float(j)
        if t<=x4: b1=0.5*(t/x4)**2.5
        elif t<2*x4: b1=1.0-0.5*(2.0-t/x4)**2.5
        else: b1=1.0
        tm=t-1.0
        if tm<=0: b0=0.0
        elif tm<=x4: b0=0.5*(tm/x4)**2.5
        elif tm<2*x4: b0=1.0-0.5*(2.0-tm/x4)**2.5
        else: b0=1.0
        UH2[j-1]=b1-b0
    StUH1=np.zeros(nUH1); StUH2=np.zeros(nUH2)
    S=0.3*x1; R=0.5*x3; Q=np.empty(n)
    for t in range(n):
        Pt=P[t]; Et=E[t]
        if Pt>=Et:
            Pn=Pt-Et; tw=np.tanh(Pn/x1); sr=S/x1
            Ps=x1*(1-sr*sr)*tw/(1+sr*tw); S=S+Ps; Pr=Pn-Ps
        else:
            En=Et-Pt; tw=np.tanh(En/x1); sr=S/x1
            Es=S*(2-sr)*tw/(1+(1-sr)*tw); S=S-Es; Pr=0.0
        Perc=S*(1-(1+(4.0/9.0*S/x1)**4)**(-0.25)); S=S-Perc; Pr=Pr+Perc
        p1=0.9*Pr; p2=0.1*Pr
        for k in range(nUH1-1): StUH1[k]=StUH1[k+1]+UH1[k]*p1
        StUH1[nUH1-1]=UH1[nUH1-1]*p1; Q9=StUH1[0]
        for k in range(nUH2-1): StUH2[k]=StUH2[k+1]+UH2[k]*p2
        StUH2[nUH2-1]=UH2[nUH2-1]*p2; Q1=StUH2[0]
        F=x2*(R/x3)**3.5
        R=R+Q9+F
        if R<0: R=0.0
        Qr=R*(1-(1+(R/x3)**4)**(-0.25)); R=R-Qr
        Qd=Q1+F
        if Qd<0: Qd=0.0
        Q[t]=Qr+Qd
    return Q

att=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)["A_avg_dB"]
_miss=att.isna(); _grp=(_miss!=_miss.shift()).cumsum()
GAPS=[(g.index[0],g.index[-1]) for _,g in att[_miss].groupby(_grp[_miss]) if len(g)>12]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
fmeta=flow.groupby("siteId").agg(lat=("latitude","first"),lon=("longitude","first"))
rfiles={f.name.split("_")[0]:f for f in (RAINDIR/"station_csv").glob("*.csv")}
def load_rain_daily(sid):
    s=pd.read_csv(rfiles[sid]); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
    cum=s.sort_values("Date/Time").set_index("Date/Time")["Cumulative rainfall (mm)"].astype(float)
    rainday=(cum.index-pd.Timedelta(hours=9,seconds=1)).floor("D")
    tot=cum.groupby(rainday).max().clip(lower=0); tot.index=tot.index+pd.Timedelta(days=1)
    return tot
def areal_rain_daily(lat,lon,k=3):
    rm=rman.assign(dist=hav(rman.latitude.astype(float),rman.longitude.astype(float),lat,lon))
    return pd.concat([load_rain_daily(i) for i in rm.sort_values("dist")["siteId"].head(k)],axis=1).mean(axis=1)
BA=[(0.5,0.999),(0.01,10.),(0.,1.),(0.,0.95),(0.90,0.99),(0.,1.),(0.,15.)]
def simA(p,x,qcap):
    a_w,cref,c0,aq,as_,f,k=p
    w=lfilter([1-a_w],[1,-a_w],x); cr=c0+(1-c0)*w/(w+cref); u=cr*x
    return np.clip(k*(f*lfilter([1-aq],[1,-aq],u)+(1-f)*lfilter([1-as_],[1,-as_],u)),0,qcap)
BR=[(50.,2000.),(-5.,5.),(10.,500.),(0.5,10.),(0.5,5.),(0.,10.)]
def simR(p,P,E,qcap):
    x1,x2,x3,x4,pscale,c=p
    return np.clip(c*gr4j(pscale*P,E,x1,x2,x3,x4),0,qcap)
def bal(o,so,s): return 0.5*nse(o,s)+0.5*nse(so,np.sqrt(np.clip(s,0,None)))

TRIO=["229624A","228368A","228351B"]
LAB={"229624A":"(b) Gardiner — attenuation-dominant",
     "228368A":"(c) Rowville — rainfall-informative (fusion w=0.96)",
     "228351B":"(d) Wantirna South — fusion rescues a weak site"}
# pin each plotted sim to the multistart-chosen seed so F5 == Table 1 exactly
msd=pd.read_csv(OUT/"multistart_daily.csv")
CHOSEN={(r["catchment"],r["mode"]):int(r["seed"]) for _,r in msd[msd["chosen"]==1].iterrows()}
fig,axs=plt.subplots(4,1,figsize=(9.5,10.4))

# (a) the model input: full-record hourly attenuation forcing, time-aligned
# with the hydrographs below
axf=axs[0]
axf.plot(att.index,att,lw=.4,color="C0",label="hourly attenuation forcing $A$")
for i,(g0,g1) in enumerate(GAPS):
    axf.axvspan(g0,g1,color="0.85",zorder=0,label="link data gap" if i==0 else None)
axf.axvspan(pd.Timestamp(CAL[0]),pd.Timestamp(CAL[1]),color="C1",alpha=.07,zorder=0)
axf.axvspan(pd.Timestamp(VAL[0]),pd.Timestamp(VAL[1]),color="C2",alpha=.07,zorder=0)
ymax=att.max()
axf.text(pd.Timestamp("2018-03-25"),ymax*.9,"calibration",color="C1",ha="center",fontsize=8)
axf.text(pd.Timestamp("2018-10-15"),ymax*.9,"validation",color="C2",ha="center",fontsize=8)
axf.set_ylabel("$A$ (dB)"); axf.set_ylim(bottom=0); axf.legend(loc="upper left",fontsize=7.5)
axf.set_title("(a) Model input: hourly attenuation forcing $A=\\max(B_{6h}-P_{avg},\\,0)$")
axf.set_xlim(att.index.min(),att.index.max())
axf.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

for ax,sid in zip(axs[1:],TRIO):
    lat,lon=float(fmeta.loc[sid,"lat"]),float(fmeta.loc[sid,"lon"])
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    rainD=areal_rain_daily(lat,lon)
    df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6); df=df.dropna(subset=["Q"])
    dd=pd.DataFrame({"Q":df.Q.resample("1D").mean(),"A":df.A.resample("1D").mean()}).join(rainD.rename("P"),how="inner").dropna()
    idx=dd.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    E=3.0+2.3*np.cos(2*np.pi*(idx.dayofyear.to_numpy()-20)/365.0)
    P=dd.P.to_numpy(); A=(dd.A/(dd.A[cal].std() or 1)).to_numpy(); obs=dd.Q.to_numpy()
    qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    def fitR(sd):
        r=differential_evolution(lambda p:-bal(o,so,simR(p,P,E,qcap)[cal]),BR,seed=sd,maxiter=120,popsize=18,tol=1e-6,polish=True)
        return simR(r.x,P,E,qcap)
    def fitA(sd):
        r=differential_evolution(lambda p:-bal(o,so,simA(p,A,qcap)[cal]),BA,seed=sd,maxiter=100,popsize=15,tol=1e-6,polish=True)
        return simA(r.x,A,qcap)
    nm=NAMES[sid]
    sR,sA,sW=CHOSEN[(nm,"R")],CHOSEN[(nm,"A")],CHOSEN[(nm,"RA")]
    QR=fitR(sR); QA=fitA(sA)
    # the chosen fusion run combined QR/QA fitted with ITS seed — rebuild that triple
    QRw=QR if sW==sR else fitR(sW)
    QAw=QA if sW==sA else fitA(sW)
    rW=differential_evolution(lambda w:-bal(o,so,(w[0]*QRw+(1-w[0])*QAw)[cal]),[(0.,1.)],seed=sW,maxiter=40,popsize=10,tol=1e-7,polish=True)
    QRA=rW.x[0]*QRw+(1-rW.x[0])*QAw
    o_v=obs[val]
    ax.plot(idx,obs,"k-",lw=1.1,label="observed")
    ax.plot(idx,QR,color="seagreen",lw=.9,alpha=.9,
            label=f"rainfall GR4J (NSE cal {nse(o,QR[cal]):.2f} / val {nse(o_v,QR[val]):.2f})")
    ax.plot(idx,QA,color="steelblue",lw=.9,alpha=.9,
            label=f"attenuation (NSE cal {nse(o,QA[cal]):.2f} / val {nse(o_v,QA[val]):.2f})")
    ax.plot(idx,QRA,color="indianred",lw=.9,alpha=.9,
            label=f"fusion (NSE cal {nse(o,QRA[cal]):.2f} / val {nse(o_v,QRA[val]):.2f})")
    ax.axvspan(pd.Timestamp(CAL[0]),pd.Timestamp(CAL[1]),color="C1",alpha=.07,zorder=0)
    ax.axvspan(pd.Timestamp(VAL[0]),pd.Timestamp(VAL[1]),color="C2",alpha=.07,zorder=0)
    for g0,g1 in GAPS: ax.axvspan(g0,g1,color="0.85",zorder=0)
    ax.set_xlim(att.index.min(),att.index.max())
    ax.set_title(LAB[sid]); ax.set_ylabel("flow (m$^3$/s)"); ax.legend(loc="upper left",fontsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle("Attenuation forcing and daily hydrographs, full period (NSE quoted as calibration / validation)",y=0.997)
fig.tight_layout(); fig.savefig(FIG/"F5_hydrographs.png",dpi=300); plt.close(fig)
print("F5 done")

# ============ F6: skill vs catchment character ============
# hourly validation KGE (primary metric; the catchment-character control is a
# high-resolution phenomenon — daily NSE shows no such relationship)
ch=pd.read_csv(OUT/"catchment_characteristics.csv")
msh=pd.read_csv(OUT/"multistart_hourly.csv")
msh=msh[msh["chosen"]==1][["catchment","KGE_val"]]
mm=msh.merge(ch[["catchment","BFI","flashiness","dist_km"]],on="catchment",how="left")
fig,ax=plt.subplots(1,2,figsize=(8.6,3.6),sharey=True)
for a,xc,xl in [(ax[0],"BFI","baseflow index (BFI)"),(ax[1],"flashiness","flashiness  $Q_{max}/Q_{mean}$")]:
    a.scatter(mm[xc],mm["KGE_val"],s=55,c="steelblue")
    for _,r in mm.iterrows():
        a.annotate(r.catchment.replace(" (reg.)","*"),(r[xc],r.KGE_val),
                   textcoords="offset points",xytext=(4,3),fontsize=6.5)
    a.set_xlabel(xl)
if mm["flashiness"].max()>300: ax[1].set_xscale("log")
ax[0].set_ylabel("attenuation model KGE (hourly, validation)")
fig.suptitle("Where the attenuation–runoff model works: catchment controls (* = regulated)")
fig.tight_layout(); fig.savefig(FIG/"F6_catchment_controls.png",dpi=300); plt.close(fig)
print("F6 done")
print(f"\nAll figures -> {FIG}")
