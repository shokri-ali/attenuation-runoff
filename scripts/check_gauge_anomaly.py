"""
check_gauge_anomaly.py — is the 6 Nov 2018 'Current rainfall' spike real rain
or a data error? Cross-check EVERY rain gauge in the export.

For each gauge, on the 2018-11-06 rain-day (9am-9am):
  cur_peak  = max hourly 'Current rainfall (mm)'      <- the suspect column
  cur_sum   = sum of 'Current rainfall (mm)'           <- implied day total
  cum_total = true rain-day total from the 'Cumulative rainfall (mm)'
              register (max within the 9am-9am window, minus previous day max
              if the register doesn't reset — here it resets daily, so the
              window max IS the day total)
  ratio     = cur_sum / cum_total  (~1 if 'Current' is honest, ~10 if inflated)

Physical sanity references:
  - Australian 1-hour rainfall record  ~ 330 mm (Dutton 1911, cyclonic, QLD)
  - Melbourne all-time DAILY record    ~ 108 mm (1972)
  - BOM-observed 6 Nov 2018 Melbourne storm totals: ~20-50 mm for the day
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAIN = ROOT / "melbourne_water_rainfall_2017-10-01_to_2018-12-31" / "station_csv"

rows = []
for f in sorted(RAIN.glob("*_rain_hourly.csv")):
    name = f.stem.replace("_rain_hourly", "")
    try:
        d = pd.read_csv(f, parse_dates=["Date/Time"]).sort_values("Date/Time").set_index("Date/Time")
    except ValueError:
        rows.append((name, np.nan, np.nan, np.nan, np.nan, "empty export"))
        continue
    cur = pd.to_numeric(d["Current rainfall (mm)"], errors="coerce")
    cum = pd.to_numeric(d["Cumulative rainfall (mm)"], errors="coerce")

    # 9am 6 Nov -> 9am 7 Nov (Australian rain-day labelled 6 Nov... the storm
    # hit ~11:00-12:00 on the 6th, so it falls in the day starting 9am on the 6th)
    w0, w1 = "2018-11-06 09:00", "2018-11-07 09:00"
    cw, mw = cur.loc[w0:w1], cum.loc[w0:w1]
    if cw.notna().sum() == 0:
        rows.append((name, np.nan, np.nan, np.nan, np.nan, "no data"))
        continue

    cur_peak = cw.max()
    cur_sum  = cw.sum()
    cum_total = mw.max()          # register resets just after 09:00 -> window max = day total
    peak_time = cw.idxmax()
    ratio = cur_sum / cum_total if cum_total and cum_total > 0 else np.nan
    rows.append((name, cur_peak, cur_sum, cum_total, ratio,
                 str(peak_time)[5:16] if pd.notna(cur_peak) and cur_peak > 0 else "-"))

t = pd.DataFrame(rows, columns=["gauge", "cur_peak_mm_per_h", "cur_day_sum_mm",
                                "cum_day_total_mm", "ratio_cur/cum", "peak_at"])
pd.set_option("display.width", 160)
print(t.to_string(index=False, float_format=lambda x: f"{x:8.1f}"))

print("\n--- context: Melbourne all-time daily rainfall record ~108 mm;")
print("    Australian 1-hour record ~330 mm (tropical cyclone).")

# also: how does the 'Current' column behave on an ordinary rain day? pick a
# moderate rain-day and repeat, to show the column is fine in light rain.
print("\n=== same check on a LIGHT-rain day (2018-05-11) ===")
rows = []
for f in sorted(RAIN.glob("*_rain_hourly.csv")):
    name = f.stem.replace("_rain_hourly", "")
    try:
        d = pd.read_csv(f, parse_dates=["Date/Time"]).sort_values("Date/Time").set_index("Date/Time")
    except ValueError:
        continue
    cur = pd.to_numeric(d["Current rainfall (mm)"], errors="coerce")
    cum = pd.to_numeric(d["Cumulative rainfall (mm)"], errors="coerce")
    w0, w1 = "2018-05-11 09:00", "2018-05-12 09:00"
    cw, mw = cur.loc[w0:w1], cum.loc[w0:w1]
    if cw.notna().sum() == 0:
        continue
    cum_total = mw.max()
    ratio = cw.sum() / cum_total if cum_total and cum_total > 0 else np.nan
    rows.append((name, cw.max(), cw.sum(), cum_total, ratio))
t2 = pd.DataFrame(rows, columns=["gauge", "cur_peak", "cur_day_sum", "cum_day_total", "ratio"])
print(t2.to_string(index=False, float_format=lambda x: f"{x:8.1f}"))
