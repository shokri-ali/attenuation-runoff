# Code for "Do runoff models need rainfall? Direct attenuation–runoff modelling with a commercial microwave link"

Reproduces every result, table and figure of the paper. Tested with Python
3.13 and the pinned versions in `../requirements.txt`.

## Input data (all open)

| Data | Source | Where the scripts expect it |
|---|---|---|
| CML received power (22.715 GHz, 3.79 km link, 15-min, Nov 2017 – Dec 2018) | Pudashine et al. (2020), Zenodo, https://doi.org/10.5281/zenodo.3629929 (CC-BY 4.0) | `data_ML_paper_new_version/CML_data_ML.dat` (from `cml_data.zip`) |
| Hourly discharge, 9 stations | Melbourne Water, https://www.melbournewater.com.au/water-and-environment/water-management/rainfall-and-river-levels | `melbourne_water_flow_2017-10-01_to_2018-12-31/` |
| Rain-gauge records, 15 telemetered gauges | Melbourne Water (same portal) | `melbourne_water_rainfall_2017-10-01_to_2018-12-31/station_csv/` |

The exact flow and rainfall extracts used in the paper are archived together
with this code so the analysis is reproducible without re-downloading.

## Pipeline (run in this order)

| Step | Script | Produces |
|---|---|---|
| 1 | `01_process_cml.py` | hourly attenuation forcing (`outputs/cml_attenuation_hourly.csv`) |
| 2 | `09_gr4j_benchmark.py` | daily R/A/RA comparison (`outputs/gr4j_metrics.csv`, seed-1 cross-check) |
| 3 | `05_fair_comparison_hourly.py` | hourly attenuation results (`outputs/fair_metrics_hourly.csv`) |
| 4 | `06_catchment_characteristics.py` | BFI, flashiness, etc. (`outputs/catchment_characteristics.csv`) |
| 5 | `12_multistart.py` | 5-seed multistart; **paper Table 3** (`outputs/table1_multistart.csv`) + per-seed supplement CSVs |
| 6 | `10_figures.py` | F2–F6 (superseded panels later overwritten), Table 3 export |
| 7 | `10b_map_F1.py` | F1 study-area map (satellite) |
| 8 | `10c_F7_hourly.py` | F7 hourly evidence |
| 9 | `10d_F2_forcing.py` | F2 final version |
| 10 | `10e_F8_scatter.py` | F8 hourly 1:1 scatter |
| 11 | `10g_S1_hydrographs.py` | Supplement Fig. S1 (all-catchment hydrographs) |

Supporting / exploratory scripts kept for transparency: `02`–`04`, `07`, `07b`,
`08` (model-development history), `11` + `check_gauge_anomaly.py` (rain-gauge
product diagnostics), `find_showcase_events.py` (event selection for F7),
`debug_*`/`explore_*` (development aids). `make_docx.py` assembles the
review manuscript.

All calibrations use differential evolution with fixed seeds (1–5); every
reported number is exactly reproducible.
