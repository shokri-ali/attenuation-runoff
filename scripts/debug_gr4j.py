"""Find why GR4J can't fit (NSE_cal ~0.08).
Hypotheses:
  H1: bug in my GR4J implementation  -> GR4J output won't even correlate with flow
      while smoothed rain does.
  H2: rainfall DATA is volume-deficient/sparse -> GR4J (water-balance model)
      starved, while correlation-based skill survives.
Tests on Ashwood (best rain-flow corr) + data forensics on the rain series.
"""
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent.parent
FLOWDIR = ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR = ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"

# --- import the exact gr4j used in 09 (same file, same code path) -----------
import importlib.util
spec = importlib.util.spec_from_file_location("bench", ROOT/"scripts"/"09_gr4j_benchmark.py")
# can't import 09 directly (it runs the whole study); copy of the function instead:
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
            Pn=Pt-Et
            tw=np.tanh(Pn/x1); sr=S/x1
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

sid="229625A"  # Ashwood
flow=pd.read_csv(FLOWDIR/"study_area_15_combined_river-flow_hourly.csv"); flow["dateTime"]=pd.to_datetime(flow["dateTime"])
q=flow[flow.siteId==sid].sort_values("dateTime").set_index("dateTime")["meanRiverFlow_m3_per_s"].astype(float)
q=q[~q.index.duplicated()]
f=list((RAINDIR/"station_csv").glob(f"{sid}_*.csv"))[0]
s=pd.read_csv(f); s["Date/Time"]=pd.to_datetime(s["Date/Time"])
rain=(s.sort_values("Date/Time").set_index("Date/Time")["Current rainfall (mm)"].astype(float)
      .fillna(0).clip(upper=50))
d=pd.DataFrame({"Q":q.resample("1D").mean(),"P":rain.resample("1D").sum()}).dropna()
E=3.0+2.3*np.cos(2*np.pi*(d.index.dayofyear.to_numpy()-20)/365.0)
P=d.P.to_numpy(); obs=d.Q.to_numpy()

print("=== DATA FORENSICS (Ashwood daily) ===")
print(f"days={len(d)}  rain: mean={P.mean():.2f} mm/d (={P.mean()*365:.0f} mm/yr)  PET mean={E.mean():.2f} mm/d (={E.mean()*365:.0f} mm/yr)")
print(f"wet days (P>0.2mm): {(P>0.2).sum()} ({100*(P>0.2).mean():.0f}%)   [Melbourne climatology ~130-150/yr ~ 40%]")
print(f"hourly wet fraction: {100*(rain>0).mean():.1f}%   [realistic ~5-8%]")
print(f"P annual / PET annual = {P.mean()*365/(E.mean()*365):.2f}  -> aridity index; GR4J starves if <<1")
print(f"corr(P_daily, Q_daily) = {np.corrcoef(P,obs)[0,1]:.3f}")

print("\n=== H1: does GR4J preserve the rain-flow correlation? ===")
for x1,x2,x3,x4,ps in [(350.,0.,90.,1.7,1.0),(350.,0.,90.,1.7,2.0),(100.,0.,40.,1.2,2.5),(60.,1.,30.,1.0,3.0)]:
    sim=gr4j(ps*P,E,x1,x2,x3,x4)
    r=np.corrcoef(sim,obs)[0,1]
    rc=sim.sum()/max((ps*P).sum(),1e-9)
    print(f"  x1={x1:5.0f} x3={x3:4.0f} x4={x4:.1f} pscale={ps:.1f}:  corr(sim,Q)={r:.3f}  sim_mean={sim.mean():.3f} mm/d  runoff_coef={rc:.2f}")
# reference: simple smoothed rain (no model)
sm=lfilter([0.3],[1,-0.7],P)
print(f"  reference smoothed-rain corr(sm,Q) = {np.corrcoef(sm,obs)[0,1]:.3f}")

print("\n=== monthly volumes: does rain track flow? ===")
m=d.resample("MS").agg({"Q":"mean","P":"sum"})
m["P_mm"]=m.P.round(0)
print(m.assign(Q=lambda x:x.Q.round(3))[["Q","P_mm"]].to_string())

print("\n=== H2 FIX TEST: reconstruct rain from the Cumulative column (9am rain-day) ===")
raw=pd.read_csv(f); raw["Date/Time"]=pd.to_datetime(raw["Date/Time"])
raw=raw.sort_values("Date/Time").set_index("Date/Time")
cum=raw["Cumulative rainfall (mm)"].astype(float)
# rain-day = 9am-9am (verified: cum holds overnight, resets just after the 09:00
# reading). 09:00 belongs to the OLD day -> shift by 9h+1s before flooring.
rainday=(cum.index - pd.Timedelta(hours=9, seconds=1)).floor("D")
# robust daily total = max of cum within the window (immune to reset-time noise)
daily_tot=cum.groupby(rainday).max()
daily_tot.index=daily_tot.index+pd.Timedelta(days=1)   # label by the calendar day the 9am window ENDS in
d2=pd.DataFrame({"Q":q.resample("1D").mean(),"P2":daily_tot}).dropna()
P2=d2.P2.to_numpy(); obs2=d2.Q.to_numpy()
print(f"reconstructed: {P2.mean()*365:.0f} mm/yr  wet days={(P2>0.2).mean()*100:.0f}%  "
      f"corr(P2,Q)={np.corrcoef(P2,obs2)[0,1]:.3f}   [vs Current-col: {P.mean()*365:.0f} mm/yr, corr {np.corrcoef(P,obs)[0,1]:.3f}]")
E2=3.0+2.3*np.cos(2*np.pi*(d2.index.dayofyear.to_numpy()-20)/365.0)
for ps in (1.0,1.5):
    sim=gr4j(ps*P2,E2,350.,0.,90.,1.7)
    print(f"GR4J on reconstructed rain (pscale={ps}): corr(sim,Q)={np.corrcoef(sim,obs2)[0,1]:.3f}  sim_mean={sim.mean():.3f} mm/d")
