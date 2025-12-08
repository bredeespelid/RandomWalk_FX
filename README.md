# RandomWalk_FX

A lightweight baseline project that evaluates a **random-walk forecasting model** for the EUR/NOK exchange rate. The repository is designed as a reproducible benchmark for level forecasting and reports standard accuracy metrics such as RMSE, MAE, and directional accuracy.

## Purpose
- Provide a transparent **baseline** for EUR/NOK forecasting.
- Demonstrate an end-to-end random-walk workflow: data retrieval, forecast construction, and evaluation.
- Serve as a reference point for comparing more complex ML and econometric models in the thesis.

## Repository Contents
- `RandomwalkM.py` — random-walk benchmark using **monthly aggregation**.
- `Randomwalk.py` — random-walk benchmark using **quarterly aggregation**.
- `requirements.txt` — pinned dependencies for reproducibility.
- `LICENSE` — MIT License.

## Scripts
### Price-Only Random-Walk (PO)
- **RandomwalkM.py** — monthly random-walk benchmark for EUR/NOK.  
  Link: [`RandomwalkM.py`](RandomwalkM.py)

- **Randomwalk.py** — quarterly random-walk benchmark for EUR/NOK.  
  Link: [`Randomwalk.py`](Randomwalk.py)

## What the Scripts Do
- Download a CSV time series of EUR/NOK rates from a configured HTTPS source.
- Construct random-walk forecasts at monthly or quarterly frequency.
- Compute and print evaluation metrics:
  - **RMSE (level)**
  - **MAE (level)**
  - **Directional accuracy**
- Print basic diagnostics on sample size, dropped observations, and date range.

Example output:

```text
=== Random Walk performance (monthly mean, EUR/NOK) ===
Observations: 310
RMSE (level): 0.136012
MAE  (level): 0.093644
Directional accuracy: 220/309 (71.2%)
