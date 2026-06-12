# attenuation-runoff

Data analysis for **"Do runoff models need rainfall? Direct attenuation–runoff
modelling with a commercial microwave link"** (Esmaeil Nia & Shokri).

A runoff model forced directly by the de-baselined attenuation of a single
commercial microwave link, with no rainfall variable constructed at any stage,
benchmarked against a gauge-driven GR4J and an output-level fusion across nine
urban catchments in eastern Melbourne, Australia.

## Contents

One self-contained script, `attenuation_runoff_analysis.py`, runs the complete
analysis in three stages:

1. **Forcing** — raw 15-min CML received power → hourly de-baselined
   attenuation A(t) = max(B6h − Pavg, 0).
2. **Daily comparison** — per catchment, 5-seed multistart differential
   evolution for the GR4J rainfall benchmark (R), the direct attenuation
   transfer model (A), and the output-level fusion (RA). Produces the paper's
   main results table.
3. **Hourly** — the attenuation-only model at hourly resolution, same protocol.

All optimiser seeds are fixed; every reported number is exactly reproducible.

## Input data (all open)

| Data | Source | Place at |
|---|---|---|
| CML received power (22.715 GHz, 3.79 km link, 15-min, Nov 2017 – Dec 2018) | Pudashine et al. (2020), Zenodo, https://doi.org/10.5281/zenodo.3629929 (CC-BY 4.0) | `data_ML_paper_new_version/CML_data_ML.dat` |
| Hourly discharge, 9 stations | Melbourne Water, https://www.melbournewater.com.au/water-and-environment/water-management/rainfall-and-river-levels | `melbourne_water_flow_2017-10-01_to_2018-12-31/` |
| Rain-gauge records, 15 telemetered gauges | Melbourne Water (same portal) | `melbourne_water_rainfall_2017-10-01_to_2018-12-31/station_csv/` |

The exact flow and rainfall extracts used in the paper are archived at the
Zenodo record accompanying this repository, so the analysis is reproducible
without re-downloading.

## Run

```
pip install -r requirements.txt
python attenuation_runoff_analysis.py
```

Outputs (written to `./outputs/`):

| File | Content |
|---|---|
| `attenuation_hourly.csv` | the hourly attenuation forcing |
| `results_daily.csv` | daily validation results, R / A / RA (paper main table) |
| `multistart_daily_seeds.csv` | every calibration seed (supplement) |
| `results_hourly.csv` | hourly attenuation-only results |
| `multistart_hourly_seeds.csv` | every calibration seed (supplement) |

Tested with Python 3.13 and the pinned versions in `requirements.txt`.
Runtime is roughly 20–30 minutes on a desktop CPU (135 daily + 45 hourly
differential-evolution calibrations).
