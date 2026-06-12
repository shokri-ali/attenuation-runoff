"""
12_multistart.py — calibration robustness: multistart differential evolution.
Re-runs the paper's calibrations with 5 DE seeds instead of one:
  DAILY  (exactly as 09): GR4J rain (R), attenuation transfer (A), fusion (RA)
  HOURLY (exactly as 05/10c): attenuation-only (A)
Per catchment/mode the reported run = best CALIBRATION objective among seeds
(standard best-of-N multistart); validation metrics of every seed are kept so
the supplement can show the spread.
Outputs:
  outputs/multistart_daily.csv    (long: catchment, mode, seed, cal/val metrics, chosen)
  outputs/multistart_hourly.csv   (long: catchment, seed, cal/val metrics, chosen)
  outputs/table1_multistart.csv   (Table-1 layout from the chosen daily runs,
                                   now incl. NSE_cal for ALL modes)
  prints: chosen-vs-seed1 deltas and per-mode validation spread
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from numba import njit

ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
TARGETS=["229624A","229625A","228366A","228393A","228351B","229638A","229640A","229639A","228368A"]
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}
SEEDS=[1,2,3,4,5]

def hav(la,lo,la0,lo0):
    p=np.pi/180; a=(np.sin((la0-la)*p/2)**2+np.cos(la*p)*np.cos(la0*p)*np.sin((lo0-lo)*p/2)**2); return 2*6371*np.arcsin(np.sqrt(a))
def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)
def kge(o,s):
    if s.std()==0 or o.std()==0: return -9
    r=np.corrcoef(o,s)[0,1]; return 1-np.sqrt((r-1)**2+(s.std()/o.std()-1)**2+(s.mean()/o.mean()-1)**2)

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
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv")
flow["dateTime"]=pd.to_datetime(flow["dateTime"])
fmeta=flow.groupby("siteId").agg(lat=("latitude","first"),lon=("longitude","first"))
rain_man=pd.read_csv(RAINDIR/"manifest.csv"); rain_man=rain_man[rain_man["rows"].astype(int)>0]
rfiles={f.name.split("_")[0]:f for f in (RAINDIR/"station_csv").glob("*.csv")}
def load_rain_daily(sid):
    s=pd.read_csv(rfiles[sid]); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
    cum=s.sort_values("Date/Time").set_index("Date/Time")["Cumulative rainfall (mm)"].astype(float)
    rainday=(cum.index-pd.Timedelta(hours=9,seconds=1)).floor("D")
    tot=cum.groupby(rainday).max().clip(lower=0); tot.index=tot.index+pd.Timedelta(days=1)
    return tot
def areal_rain_daily(lat,lon,k=3):
    d=rain_man.assign(dist=hav(rain_man.latitude.astype(float),rain_man.longitude.astype(float),lat,lon))
    return pd.concat([load_rain_daily(i) for i in d.sort_values("dist")["siteId"].head(k)],axis=1).mean(axis=1)

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

# ---------------- DAILY multistart (as 09) ----------------
drows=[]; t1rows=[]
for sid in TARGETS:
    name=NAMES[sid]
    lat,lon=float(fmeta.loc[sid,"lat"]),float(fmeta.loc[sid,"lon"])
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    rainD=areal_rain_daily(lat,lon)
    df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6); df=df.dropna(subset=["Q"])
    d=pd.DataFrame({"Q":df.Q.resample("1D").mean(),"A":df.A.resample("1D").mean()}).join(rainD.rename("P"),how="inner").dropna()
    idx=d.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    E=(3.0+2.3*np.cos(2*np.pi*(idx.dayofyear.to_numpy()-20)/365.0))
    P=d.P.to_numpy(); A=(d.A/(d.A[cal].std() or 1)).to_numpy(); obs=d.Q.to_numpy()
    qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    per_seed={}
    for seed in SEEDS:
        rR=differential_evolution(lambda p:-bal(o,so,simR(p,P,E,qcap)[cal]),BR,seed=seed,maxiter=120,popsize=18,tol=1e-6,polish=True)
        QR=simR(rR.x,P,E,qcap)
        rA=differential_evolution(lambda p:-bal(o,so,simA(p,A,qcap)[cal]),BA,seed=seed,maxiter=100,popsize=15,tol=1e-6,polish=True)
        QA=simA(rA.x,A,qcap)
        rW=differential_evolution(lambda w:-bal(o,so,(w[0]*QR+(1-w[0])*QA)[cal]),[(0.,1.)],seed=seed,maxiter=40,popsize=10,tol=1e-7,polish=True)
        w=rW.x[0]; QRA=w*QR+(1-w)*QA
        per_seed[seed]={"R":(QR,-rR.fun),"A":(QA,-rA.fun),"RA":(QRA,-rW.fun),"w":w}
        for mode in ["R","A","RA"]:
            s,objc=per_seed[seed][mode]
            drows.append({"catchment":name,"mode":mode,"seed":seed,"obj_cal":round(objc,4),
                          "NSE_cal":round(nse(o,s[cal]),3),"KGE_cal":round(kge(o,s[cal]),3),
                          "NSE_val":round(nse(obs[val],s[val]),3),"KGE_val":round(kge(obs[val],s[val]),3),
                          "fusion_w":round(w,2) if mode=="RA" else np.nan})
    # chosen = best calibration objective per mode
    t1={"catchment":name}
    for mode in ["R","A","RA"]:
        best=max(SEEDS,key=lambda s:per_seed[s][mode][1])
        s,_=per_seed[best][mode]
        for r in drows:
            if r["catchment"]==name and r["mode"]==mode: r["chosen"]=int(r["seed"]==best)
        t1[f"NSE_{mode}"]=round(nse(obs[val],s[val]),3); t1[f"KGE_{mode}"]=round(kge(obs[val],s[val]),3)
        t1[f"NSEcal_{mode}"]=round(nse(o,s[cal]),3)
        if mode=="RA": t1["fusion_w"]=round(per_seed[best]["w"],2)
    t1rows.append(t1)
    print(f"daily done: {name}",flush=True)

dd=pd.DataFrame(drows); dd.to_csv(OUT/"multistart_daily.csv",index=False)
t1=pd.DataFrame(t1rows); t1.to_csv(OUT/"table1_multistart.csv",index=False)

# ---------------- HOURLY multistart, A-mode (as 05/10c) ----------------
hrows=[]
for sid in TARGETS:
    name=NAMES[sid]
    q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
    q=q[~q.index.duplicated()]
    df=pd.concat([q.rename("Q"),att.rename("A")],axis=1).loc[att.index.min():att.index.max()]
    df["A"]=df["A"].interpolate(limit=6).fillna(0); df=df.dropna(subset=["Q"])
    idx=df.index; cal=(idx>=CAL[0])&(idx<=CAL[1]); val=(idx>=VAL[0])&(idx<=VAL[1])
    A=(df.A/(df.A[cal].std() or 1)).to_numpy(); obs=df.Q.to_numpy()
    qcap=1.5*obs.max(); o=obs[cal]; so=np.sqrt(np.clip(o,0,None))
    best_obj=-9e9
    for seed in SEEDS:
        r=differential_evolution(lambda p:-bal(o,so,simA(p,A,qcap)[cal]),BA,seed=seed,maxiter=120,popsize=15,tol=1e-6,polish=True)
        s=simA(r.x,A,qcap)
        hrows.append({"catchment":name,"seed":seed,"obj_cal":round(-r.fun,4),
                      "NSE_cal":round(nse(o,s[cal]),3),"KGE_cal":round(kge(o,s[cal]),3),
                      "NSE_val":round(nse(obs[val],s[val]),3),"KGE_val":round(kge(obs[val],s[val]),3)})
        if -r.fun>best_obj: best_obj=-r.fun; best_seed=seed
    for r in hrows:
        if r["catchment"]==name: r["chosen"]=int(r["seed"]==best_seed)
    print(f"hourly done: {name}",flush=True)
hh=pd.DataFrame(hrows); hh.to_csv(OUT/"multistart_hourly.csv",index=False)

# ---------------- summaries ----------------
pd.set_option("display.width",200)
print("\n=== Table 1 (multistart best-of-5, daily validation; NSEcal_* added) ===")
print(t1.to_string(index=False))
old=pd.read_csv(OUT/"gr4j_metrics.csv")
cmp=t1.merge(old,on="catchment",suffixes=("_m5","_s1"))
for mode in ["R","A","RA"]:
    cmp[f"dNSE_{mode}"]=(cmp[f"NSE_{mode}_m5"]-cmp[f"NSE_{mode}_s1"]).round(3)
print("\n=== delta vs published seed-1 numbers (NSE_val, multistart - seed1) ===")
print(cmp[["catchment","dNSE_R","dNSE_A","dNSE_RA"]].to_string(index=False))
print("\n=== validation spread across seeds (daily) ===")
sp=dd.groupby(["catchment","mode"])["NSE_val"].agg(["min","max"]); sp["range"]=(sp["max"]-sp["min"]).round(3)
print(sp.to_string())
print("\n=== validation spread across seeds (hourly A, KGE_val) ===")
sph=hh.groupby("catchment")["KGE_val"].agg(["min","max"]); sph["range"]=(sph["max"]-sph["min"]).round(3)
print(sph.to_string())
print("\ndone.")
