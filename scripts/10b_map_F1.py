"""
10b_map_F1.py — study-area map on SATELLITE imagery (replaces F1).
Esri World Imagery tiles, plotted in native Web-Mercator tile-pixel space (correct
georeferencing at all scales), with link path, flow stations, rain gauges, scale
bar and an Australia inset locating Melbourne. Tiles cached in data/tiles/.
Output: outputs/figures/F1_study_area.png
"""
from pathlib import Path
from io import BytesIO
import math, urllib.request
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT=Path(__file__).resolve().parent.parent
FIG=ROOT/"outputs"/"figures"; FIG.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/"data"/"tiles"; CACHE.mkdir(parents=True,exist_ok=True)
FLOWDIR=ROOT/"melbourne_water_flow_2017-10-01_to_2018-12-31"
RAINDIR=ROOT/"melbourne_water_rainfall_2017-10-01_to_2018-12-31"
LINK=((145.1722,-37.8922),(145.1652,-37.8591))
SAT="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
HDRS={"User-Agent":"hydrology-research-figure/1.0 (University of Waikato)"}

def deg2num(lat,lon,z):
    n=2**z
    x=(lon+180.0)/360.0*n
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
    """Stitched tile basemap; converts lon/lat -> pixel coords of the mosaic."""
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

fman=pd.read_csv(FLOWDIR/"study_area_15_manifest.csv")
rman=pd.read_csv(RAINDIR/"manifest.csv"); rman=rman[rman["rows"].astype(int)>0]

W,E,S,N=145.02,145.28,-37.975,-37.78
tm=TileMap(W,E,S,N,13)
fig,ax=plt.subplots(figsize=(8.6,7.4))
ax.imshow(tm.img,origin="upper")
# crop to the requested box
x0,y0=tm.px(N,W); x1,y1=tm.px(S,E)
ax.set_xlim(x0,x1); ax.set_ylim(y1,y0)   # y inverted (pixel space)
ax.set_aspect("equal")

halo_w=[pe.withStroke(linewidth=2.6,foreground="black")]
# link
lx,ly=zip(*[tm.px(p[1],p[0]) for p in LINK])
ax.plot(lx,ly,color="red",lw=4.5,solid_capstyle="round",zorder=6,
        label="microwave link (22.7 GHz, 3.8 km)")
mx,my=np.mean(lx),np.mean(ly)
ax.annotate("CML",(mx,my),textcoords="offset points",xytext=(10,0),fontsize=10,
            fontweight="bold",color="red",path_effects=halo_w,zorder=7)
# flow stations
fx,fy=zip(*[tm.px(la,lo) for la,lo in zip(fman.latitude,fman.longitude)])
ax.scatter(fx,fy,marker="^",s=110,c="cyan",edgecolors="black",linewidths=.9,
           zorder=5,label="flow station")
for X,Y,nm in zip(fx,fy,fman.siteName):
    ax.annotate(nm,(X,Y),textcoords="offset points",xytext=(7,6),fontsize=8,
                fontweight="bold",color="white",path_effects=halo_w,zorder=7)
# rain gauges
rx,ry=zip(*[tm.px(la,lo) for la,lo in zip(rman.latitude.astype(float),rman.longitude.astype(float))])
ax.scatter(rx,ry,marker="s",s=60,facecolors="yellow",edgecolors="black",
           linewidths=.9,zorder=4,label="rain gauge")
# scale bar 5 km
px5=5000.0/tm.m_per_px(-37.88)
sx,sy=x0+40,y1-40
ax.plot([sx,sx+px5],[sy,sy],color="white",lw=4,zorder=7,path_effects=halo_w)
ax.annotate("5 km",(sx+px5/2,sy),textcoords="offset points",xytext=(0,8),ha="center",
            fontsize=9,fontweight="bold",color="white",path_effects=halo_w,zorder=7)
# graticule ticks (round 0.05-deg)
lon_ticks=np.arange(145.05,145.28,0.05); lat_ticks=np.arange(-37.95,-37.78,0.05)
ax.set_xticks([tm.px(S,t)[0] for t in lon_ticks]); ax.set_xticklabels([f"{t:.2f}°E" for t in lon_ticks],fontsize=8)
ax.set_yticks([tm.px(t,W)[1] for t in lat_ticks]); ax.set_yticklabels([f"{abs(t):.2f}°S" for t in lat_ticks],fontsize=8)
leg=ax.legend(loc="upper left",framealpha=0.95,fontsize=9)
ax.set_title("Study area — eastern Melbourne: microwave link, flow stations and rain gauges")

# Australia inset (same Mercator-pixel treatment -> star lands exactly on Melbourne)
tmi=TileMap(110,156,-44,-9,4)
axi=ax.inset_axes([0.778,0.012,0.21,0.27])
axi.imshow(tmi.img,origin="upper")
ix0,iy0=tmi.px(-9,110); ix1,iy1=tmi.px(-44,156)
axi.set_xlim(ix0,ix1); axi.set_ylim(iy1,iy0); axi.set_aspect("equal")
mxp,myp=tmi.px(-37.81,144.96)
axi.plot(mxp,myp,"*",ms=13,color="red",mec="white",mew=.8)
axi.annotate("Melbourne",(mxp,myp),textcoords="offset points",xytext=(-5,7),ha="right",
             fontsize=7,fontweight="bold",color="white",path_effects=halo_w)
axi.set_xticks([]); axi.set_yticks([])
for s in axi.spines.values(): s.set_color("white")

ax.annotate("Imagery © Esri, Maxar, Earthstar Geographics",(0.995,0.004),
            xycoords="axes fraction",ha="right",fontsize=6.5,color="white",
            path_effects=halo_w)
fig.tight_layout()
fig.savefig(FIG/"F1_study_area.png",dpi=300)
# JPG for submission/manuscript embedding: HESS caps individual figures at 5 MB,
# and the satellite PNG is ~8 MB; quality-85 JPG is visually identical at ~1.5 MB
fig.savefig(FIG/"F1_study_area.jpg",dpi=300,pil_kwargs={"quality":85})
print(f"F1 rebuilt -> {FIG/'F1_study_area.png'} (+ .jpg for submission)")
