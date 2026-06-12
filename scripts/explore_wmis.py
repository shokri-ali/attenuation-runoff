"""Explore the Victoria WMIS Hydstra web service: find flow gauges for the
5 target catchments near the CML link (Glen Waverley ~ -37.88, 145.17)."""
import json, urllib.parse, urllib.request, math

BASE = "https://data.water.vic.gov.au/cgi/webservice.exe"

def call(req):
    url = BASE + "?" + urllib.parse.quote(json.dumps(req))
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))

# Pull all sites with name + coordinates.
req = {"function": "get_site_list", "version": 1,
       "params": {"site_list": "MERGE(GROUP(MW,SITE))"}}
# Fallback: get_db_info on the site table for names/coords.
info = {"function": "get_db_info", "version": 3,
        "params": {"table_name": "site",
                   "return_type": "array",
                   "field_list": ["station", "stname", "latitude", "longitude"]}}

res = call(info)
print("keys:", list(res.keys()))
rows = res.get("return", {}).get("rows", res.get("_return", {}).get("rows", []))
print("n sites:", len(rows))

LINK_LAT, LINK_LON = -37.876, 145.169
def dist(la, lo):
    try:
        return math.hypot((float(la)-LINK_LAT)*111, (float(lo)-LINK_LON)*88)
    except Exception:
        return 9e9

names = ["waverley", "scoresby", "wantirna", "rowville", "dandenong", "scotchman", "gardiner", "blackburn"]
hits = []
for row in rows:
    stn = str(row.get("station", ""))
    nm = str(row.get("stname", "")).lower()
    d = dist(row.get("latitude"), row.get("longitude"))
    if any(k in nm for k in names) or d < 20:
        hits.append((round(d, 1), stn, row.get("stname"), row.get("latitude"), row.get("longitude")))

hits.sort()
print(f"\n{'dist_km':>7}  {'station':<10} {'name':<40} {'lat':>9} {'lon':>9}")
for d, stn, nm, la, lo in hits[:60]:
    print(f"{d:>7}  {stn:<10} {str(nm)[:40]:<40} {str(la):>9} {str(lo):>9}")
