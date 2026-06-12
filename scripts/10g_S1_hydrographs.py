"""
10g_S1_hydrographs.py — Supplement Figures S1–S3: full-period daily hydrographs
(observed, GR4J rainfall benchmark, direct attenuation model, fusion) for ALL
nine catchments, three panels per figure, reproduced exactly as in 09/12 and
pinned to the multistart-chosen seeds (same convention as F5).
Output: outputs/figures/S1_hydrographs_1of3.png / S2_..._2of3 / S3_..._3of3
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from numba import njit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"; FIG=OUT/"figures"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
TARGETS=["229624A","229640A","229625A","228366A","228393A","228351B","229638A","229639A","228368A"]
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":110})

def hav(la,lo,la0,lo0):
    p=np.pi/180; a=(np.sin((la0-la)*p/2)**2+np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2); return 2*6371*np.arcsin(np.sqrt(a))
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)

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
_m=att.isna(); _g=(_m!=_m.shift()).cumsum()
GAPS=[(g.index[0],g.index[-1]) for _,g in att[_m].groupby(_g[_m]) if len(g)>12]
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
fmeta=flow.groupby("siteId").agg(lat=("latitude","first"),lon=("longitude","first"))
rman=pd.read_csv(RAINDIR/"manifest.csv"); rman=rman[rman["rows"].astype(int)>0]
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

msd=pd.read_csv(OUT/"multistart_daily.csv")
CHOSEN={(r["catchment"],r["mode"]):int(r["seed"]) for _,r in msd[msd["chosen"]==1].iterrows()}

GROUPS=[TARGETS[0:3],TARGETS[3:6],TARGETS[6:9]]
letters="abc"
for gi,group in enumerate(GROUPS):
  fig,axs=plt.subplots(3,1,figsize=(9.5,7.8))
  for pi,(ax,sid) in enumerate(zip(axs,group)):
    nm=NAMES[sid]
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
    sR,sA,sW=CHOSEN[(nm,"R")],CHOSEN[(nm,"A")],CHOSEN[(nm,"RA")]
    QR=fitR(sR); QA=fitA(sA)
    QRw=QR if sW==sR else fitR(sW)
    QAw=QA if sW==sA else fitA(sW)
    rW=differential_evolution(lambda w:-bal(o,so,(w[0]*QRw+(1-w[0])*QAw)[cal]),[(0.,1.)],seed=sW,maxiter=40,popsize=10,tol=1e-7,polish=True)
    QRA=rW.x[0]*QRw+(1-rW.x[0])*QAw
    o_v=obs[val]
    ax.plot(idx,obs,"k-",lw=1.0,label="observed")
    ax.plot(idx,QR,color="seagreen",lw=.8,alpha=.9,label=f"rainfall GR4J (NSE cal {nse(o,QR[cal]):.2f} / val {nse(o_v,QR[val]):.2f})")
    ax.plot(idx,QA,color="steelblue",lw=.8,alpha=.9,label=f"attenuation (NSE cal {nse(o,QA[cal]):.2f} / val {nse(o_v,QA[val]):.2f})")
    ax.plot(idx,QRA,color="indianred",lw=.8,alpha=.9,label=f"fusion (NSE cal {nse(o,QRA[cal]):.2f} / val {nse(o_v,QRA[val]):.2f})")
    ax.axvspan(pd.Timestamp(CAL[0]),pd.Timestamp(CAL[1]),color="C1",alpha=.07,zorder=0)
    ax.axvspan(pd.Timestamp(VAL[0]),pd.Timestamp(VAL[1]),color="C2",alpha=.07,zorder=0)
    for g0,g1 in GAPS: ax.axvspan(g0,g1,color="0.85",zorder=0)
    ax.set_xlim(att.index.min(),att.index.max())
    ax.set_title(f"({letters[pi]}) {nm}")
    ax.set_ylabel("flow (m$^3$/s)"); ax.legend(loc="upper left",fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    print(f"S{gi+1} panel done:",nm,flush=True)
  fig.suptitle(f"Figure S{gi+1}: daily simulations, full period, catchments {3*gi+1}–{3*gi+3} of 9 "
               "(orange = calibration, green = validation, grey = link outages)",y=0.995)
  fig.tight_layout()
  fn=FIG/f"S{gi+1}_hydrographs_{gi+1}of3.png"
  fig.savefig(fn,dpi=300); plt.close(fig)
  print(f"S{gi+1} ->",fn)
old=FIG/"S1_hydrographs_all.png"
if old.exists(): old.unlink()
