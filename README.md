# Randomwalk.M

This repository contains `RandomwalkM.py`, a small analysis script that evaluates a random-walk forecasting model for EUR/NOK exchange rates.

## What this repo contains

- `RandomwalkM.py` — main script that downloads the data, computes simple random-walk forecasts, and prints performance statistics.
- `requirements.txt` — pinned Python packages needed to run the script.
- `README.md` — this file.
- `LICENSE` — MIT license.
- `.gitignore` — common Python ignores.

## Quick start

Recommended: create and activate a virtual environment so packages are isolated.

```bash
# from the repository root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the script (the script downloads a CSV over HTTPS). To avoid SSL certificate issues when using system Pythons, set the `SSL_CERT_FILE` environment variable to the `certifi` CA bundle, for example:

```bash
SSL_CERT_FILE=$(python3.12 -c "import certifi; print(certifi.where())") python3.12 RandomwalkM.py
```

If you use the Python binary directly (e.g. `/usr/local/bin/python3.12`), substitute that path in the commands above.

## Notes

- Use `python -m pip install --user ...` or `python -m pip install -r requirements.txt` to ensure packages are installed for the correct interpreter.
- Consider using a virtual environment to keep dependencies isolated.

## Recommended GitHub Upload Steps

```bash
git init
git add .
git commit -m "Initial commit: add script and deps"
# create a new repo on GitHub and push
git remote add origin <your-repo-url>
git push -u origin main
```

## License

This project is provided under the MIT license — see the `LICENSE` file.
