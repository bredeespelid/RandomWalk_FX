# Randomwalk.M — descriptive guide

Randomwalk.M is a lightweight analysis script that evaluates a simple random-walk forecasting model for the EUR/NOK exchange rate. It is intended as a reproducible demonstration of a baseline forecasting technique and related performance metrics (RMSE, MAE, directional accuracy).

This README explains what the script does, what inputs and outputs to expect, how to run it safely (including common SSL and interpreter issues), and how to package it for sharing on GitHub.

## What the script does (high level)

- Download a CSV time series of EUR/NOK rates from a remote source (HTTPS).
- Compute simple random-walk forecasts aggregated at a chosen frequency (monthly/quarterly depending on the script variant).
- Calculate and print simple evaluation metrics: RMSE, MAE, and directional accuracy.
- Optionally print short tables of first/last rows and dropped observations with reasons.

## Inputs and outputs

- Inputs: the script fetches a CSV from a configured URL (no local data required). It expects standard CSV columns used by the script (dates and the EUR_NOK price column).
- Outputs: human-readable statistics printed to stdout. Example excerpt from a successful run:

```
=== Random Walk performance (monthly mean, EUR/NOK) ===
Observations: 310
RMSE (level): 0.136012
MAE  (level): 0.093644
Directional accuracy: 220/309 (71.2%)
```

The script also prints diagnostics about dropped rows and sample ranges.

## Quick, reproducible setup (recommended)

Use a virtual environment so your system packages aren't mixed with project dependencies. These steps assume you have Python 3.12 available as `python3.12`.

```bash
# create and activate a venv (from repo root)
python3.12 -m venv .venv
source .venv/bin/activate

# install the pinned dependencies
pip install -r requirements.txt
```

Run the script:

```bash
python3.12 RandomwalkM.py
```

If you still see SSL certificate verification errors when the script downloads the remote CSV, use the `certifi` CA bundle as a quick fix (this is safe and portable):

```bash
SSL_CERT_FILE=$(python3.12 -c "import certifi; print(certifi.where())") python3.12 RandomwalkM.py
```

If you run the script with an explicit Python binary (for example `/usr/local/bin/python3.12`), substitute that binary everywhere above to ensure you use the same interpreter for installing packages and running the script.

## Common issues and troubleshooting

- ModuleNotFoundError (e.g. `No module named 'numpy'`): you likely installed packages with a different `pip` than the `python` you run. Fix by running the installer with the same interpreter:

```bash
/path/to/python3.12 -m pip install --user <package>
```

- SSL errors (`CERTIFICATE_VERIFY_FAILED`): either run with `SSL_CERT_FILE` set to `certifi.where()` (shown above) or install `certifi` into the interpreter and set the env var. If you installed Python from python.org on macOS, you can also run the shipped `Install Certificates.command`.

- PATH warnings about scripts installed to `~/Library/Python/3.12/bin`: these are helper scripts installed during `--user` installs; add that directory to your `PATH` if you want to run them directly:

```bash
export PATH="$HOME/Library/Python/3.12/bin:$PATH"
```

## Development & reproducibility

- A `requirements.txt` is included with pinned versions used during development. For reproducible installs, use `pip install -r requirements.txt` inside a venv.
- To reproduce the exact environment used during testing, you can create the venv, install the requirements, and run the script as shown in the Quick Setup section.


