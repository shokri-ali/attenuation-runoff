"""
10f_F6_map.py — [SHELVED 2026-06-11, Ali's call: paper uses the plain 2-scatter
F6 from 10_figures.py; keep this for reviewer response] satellite skill-map F6:
 (a) satellite map (Esri World Imagery, same tile engine/cache as 10b):
     flow stations coloured by daily validation NSE of the attenuation-only
     model, link path in red — shows skill is NOT governed by distance to
     the link (the far, baseflow-rich catchments do best)
 (b) NSE vs baseflow index (BFI)        (c) NSE vs flashiness (log x)
Output: outputs/figures/F6_catchment_controls.png (overwrites)
"""
from pathlib import Path
from io import BytesIO
import math, urllib.request
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec, cm
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"outputs"; FIG=OUT/"figures"
CACHE=ROOT/"data"/"tiles"; CACHE.mkdir(parents=True,exist_ok=True)
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
LINK=((145.1722,-37.8922),(145.1652,-37.8591))
SAT="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
HDRS={"User-Agent":"hydrology-research-figure/1.0 (University of Waikato)"}
plt.rcParams.update({"font.size":9,"axes.titlesize":10,"figure.dpi":110})

def deg2num(lat,lon,z):
    n=2**z; x=(lon+180.0)/360.0*n
    lr=math.radians(lat)
    y=(1.0-math.log(math.tan(lr)+1/math.cos(lr))/math.pi)/2.0*n
    return x,y
def fetch(z,x,y):
    f=CACHE/f"esri_{z}_{x}_{y}.jpg"
    if f.exists(): return Image.open(f).convert("RGB")
    req=urllib.request.Request(SAT.format(z=z,x=x,y=y),headers=HDRS)
    data=urllib.request.urlopen(req,timeout=30).read()
    f.write_bytes(data)
    return Image.open(BytesIO(data)).convert("RGB")
class TileMap:
    def __init__(self,lon0,lon1,lat0,lat1,z):
        self.z=z
        xW,yN=deg2num(lat1,lon0,z); xE,yS=deg2num(lat0,lon1,z)
        self.xa,self.ya=int(xW),int(yN)
        xb,yb=int(xE),int(yS)
        rows=[]
        for ty in range(self.ya,yb+1):
            rows.append(np.hstack([np.asarray(fetch(z,tx,ty)) for tx in range(self.xa,xb+1)]))
        self.img=np.vstack(rows)
    def px(self,lat,lon):
        x,y=deg2num(lat,lon,self.z)
        return (x-self.xa)*256.0,(y-self.ya)*256.0
    def m_per_px(self,lat):
        return 156543.03392*math.cos(math.radians(lat))/2**self.z

# --- data ---------------------------------------------------------------
m=pd.read_csv(OUT/"gr4j_metrics.csv")
ch=pd.read_csv(OUT/"catchment_characteristics.csv")
mm=m.merge(ch[["catchment","BFI","flashiness"]],on="catchment",how="left")
fman=pd.read_csv(FLOWDIR/"study_area_15_manifest.csv")
NAMES={"229624A":"Gardiner","229625A":"Ashwood","228366A":"Knox","228393A":"Scoresby",
       "228351B":"Wantirna South","229638A":"Burwood East","229640A":"Mt Waverley",
       "229639A":"Glen Waverley (reg.)","228368A":"Rowville (reg.)"}
fman["catchment"]=fman["siteId"].map(NAMES)
mm=mm.merge(fman[["catchment","latitude","longitude"]],on="catchment",how="left")

# --- figure -------------------------------------------------------------
fig=plt.figure(figsize=(9,9.6))
gs=gridspec.GridSpec(2,2,height_ratios=[1.85,1],hspace=.22,wspace=.22)
axm=fig.add_subplot(gs[0,:]); axb=fig.add_subplot(gs[1,0]); axc=fig.add_subplot(gs[1,1])

# (a) satellite skill map
W,E,S,N=145.02,145.28,-37.975,-37.78
tm=TileMap(W,E,S,N,13)
axm.imshow(tm.img,origin="upper")
x0,y0=tm.px(N,W); x1,y1=tm.px(S,E)
axm.set_xlim(x0,x1); axm.set_ylim(y1,y0); axm.set_aspect("equal")
halo=[pe.withStroke(linewidth=2.4,foreground="black")]
lx,ly=zip(*[tm.px(p[1],p[0]) for p in LINK])
axm.plot(lx,ly,color="red",lw=4.5,solid_capstyle="round",zorder=6,label="microwave link")
axm.annotate("CML",(np.mean(lx),np.mean(ly)),textcoords="offset points",xytext=(-12,4),
             ha="right",fontsize=10,fontweight="bold",color="red",path_effects=halo,zorder=7)
norm=Normalize(vmin=0,vmax=0.8); cmap=plt.get_cmap("RdYlGn")
sx,sy=zip(*[tm.px(la,lo) for la,lo in zip(mm.latitude,mm.longitude)])
axm.scatter(sx,sy,s=240,c=mm["NSE_A"],cmap=cmap,norm=norm,edgecolors="black",
            linewidths=1.2,zorder=5)
OFF={"Glen Waverley (reg.)":(9,-26)}   # avoid covering the link / CML label
for X,Y,nm,v in zip(sx,sy,mm.catchment,mm["NSE_A"]):
    axm.annotate(f"{nm.replace(' (reg.)','*')}\nNSE {v:.2f}",(X,Y),textcoords="offset points",
                 xytext=OFF.get(nm,(9,7)),fontsize=7.5,fontweight="bold",color="white",
                 path_effects=halo,zorder=7)
px5=5000.0/tm.m_per_px(-37.88)
bx,by=x0+40,y1-40
axm.plot([bx,bx+px5],[by,by],color="white",lw=4,zorder=7,path_effects=halo)
axm.annotate("5 km",(bx+px5/2,by),textcoords="offset points",xytext=(0,8),ha="center",
             fontsize=9,fontweight="bold",color="white",path_effects=halo,zorder=7)
axm.set_xticks([]); axm.set_yticks([])
cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=axm,shrink=.8,pad=.01)
cb.set_label("attenuation-only NSE (daily, validation)")
axm.set_title("(a) Validation skill in space — distance to the link does not control performance (* = regulated)")
axm.annotate("Imagery © Esri, Maxar, Earthstar Geographics",(0.995,0.006),
             xycoords="axes fraction",ha="right",fontsize=6.5,color="white",path_effects=halo)

# (b)/(c) controls
for a,xc,xl,ttl in [(axb,"BFI","baseflow index (BFI)","(b) skill vs baseflow"),
                    (axc,"flashiness","flashiness  $Q_{max}/Q_{mean}$","(c) skill vs flashiness")]:
    a.scatter(mm[xc],mm["NSE_A"],s=55,c="steelblue")
    for _,r in mm.iterrows():
        a.annotate(r.catchment.replace(" (reg.)","*"),(r[xc],r.NSE_A),
                   textcoords="offset points",xytext=(4,3),fontsize=6.5)
    a.set_xlabel(xl); a.set_title(ttl)
if mm["flashiness"].max()>300: axc.set_xscale("log")
axb.set_ylabel("attenuation model NSE (validation)")
fig.savefig(FIG/"F6_catchment_controls.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("F6 (with map) ->",FIG/"F6_catchment_controls.png")
