"""
10d_F2_forcing.py — F2 (methods): signal extraction from raw link power.
Supersedes the F2 section of 10_figures.py:
 (a) full 14-month received power + 6 h baseline   — seasonal base-attenuation
     drift and data gaps (why de-baselining is needed)
 (b) one-week zoom (4–9 Nov 2018): power + baseline (left axis) and the
     extracted signal A = max(B6h − Pavg, 0) (right axis) — the mechanism
The full-record FORCING panel now lives on top of F5 (results), so the model
input sits time-aligned above the hydrographs it produces.
Output: outputs/figures/F2_signal_extraction.png (overwrites)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"outputs"; FIG=OUT/"figures"
SRC=ROOT/"data_ML_paper_new_version"/"CML_data_ML.dat"
CAL=("2017-12-01","2018-07-31"); VAL=("2018-08-01","2018-12-19")
W0,W1="2018-11-04","2018-11-09"
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":110})

# raw 15-min power + 6 h baseline, exactly as in 01_process_cml.py
d=pd.read_csv(SRC); d["Date"]=pd.to_datetime(d["Date"],format="%d/%m/%Y %H:%M")
d=d.sort_values("Date").groupby("Date").mean(numeric_only=True)
d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="15min"))
Pref=d["Pavg"].rolling("6h",center=True,min_periods=2).mean()

# the actual model forcing written by 01
h=pd.read_csv(OUT/"cml_attenuation_hourly.csv",index_col=0,parse_dates=True)
A=h["A_avg_dB"]

# contiguous gaps (>12 h) in the hourly forcing, for shading
miss=A.isna(); grp=(miss!=miss.shift()).cumsum()
gaps=[(g.index[0],g.index[-1]) for _,g in A[miss].groupby(grp[miss]) if len(g)>12]

fig,ax=plt.subplots(2,1,figsize=(9,5.6))

# (a) full-record received power + baseline
ax[0].plot(d.index,d["Pavg"],lw=.25,color="0.55",label="received power $P_{avg}$ (15 min)")
ax[0].plot(Pref.index,Pref,lw=.6,color="C3",label="6 h moving-average baseline $B_{6h}$")
for i,(g0,g1) in enumerate(gaps):
    ax[0].axvspan(g0,g1,color="0.85",zorder=0,label="link data gap" if i==0 else None)
ax[0].axvspan(*[pd.Timestamp(t) for t in CAL],color="C1",alpha=.07,zorder=0)
ax[0].axvspan(*[pd.Timestamp(t) for t in VAL],color="C2",alpha=.07,zorder=0)
ax[0].set_ylabel("dBm"); ax[0].legend(loc="lower left",ncol=3)
ax[0].set_title("(a) Received power, Nov 2017 – Dec 2018 — rain dips on a slowly drifting base level")
ax[0].set_xlim(d.index.min(),d.index.max())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# (b) one-week zoom: mechanism (hourly, as used by the model)
Pav=d["Pavg"].resample("1h").mean().interpolate(limit=6)
Bl=Pav.rolling("6h",center=True,min_periods=2).mean()
Sig=(Bl-Pav).clip(lower=0)
ax[1].plot(Pav.loc[W0:W1],color="0.45",lw=.9,label="$P_{avg}$ (hourly)")
ax[1].plot(Bl.loc[W0:W1],color="C3",lw=1.4,label="baseline $B_{6h}$")
ax[1].set_ylabel("dBm"); ax[1].legend(loc="lower left")
ax2=ax[1].twinx()
ax2.fill_between(Sig.loc[W0:W1].index,Sig.loc[W0:W1],color="C0",alpha=.55,label="signal $A$")
ax2.set_ylabel("$A$ (dB)",color="C0"); ax2.tick_params(axis="y",labelcolor="C0")
ax2.set_ylim(bottom=0); ax2.legend(loc="lower right")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
ax[1].set_title("(b) Signal extraction, 4–9 Nov 2018: dips below the local baseline become the forcing $A=\\max(B_{6h}-P_{avg},\\,0)$")

fig.tight_layout(); fig.savefig(FIG/"F2_signal_extraction.png",dpi=300); plt.close(fig)
print("gaps >12 h:",[(str(a)[:16],str(b)[:16]) for a,b in gaps])
print("F2 (combined) ->",FIG/"F2_signal_extraction.png")
