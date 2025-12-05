# -*- coding: utf-8 -*-
"""
Random Walk (RW) – EUR/NOK walk-forward (monthly) without confidence intervals
Source: variables_daily.csv (All_Variables-link)

- Dataset: variables_daily.csv (daily wide panel)
- Only EUR_NOK column is used
- Evaluation frequency: monthly
- Cut: last business day in previous month (based on B-calendar with ffill)
- Minimum history before cut: min_hist_days
- Target: monthly mean of daily business-day levels (S_b)
- Evaluates over ALL available months where requirements are satisfied
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    url: str = (
        "https://raw.githubusercontent.com/bredeespelid/"
        "Data_MasterOppgave/refs/heads/main/Variables/All_Variables/variables_daily.csv"
    )
    m_freq: str = "M"      # monthly evaluation
    min_hist_days: int = 40
    verbose: bool = True
    fig_png: str = "EUR_NOK_RW_vs_Actual_Monthly.png"
    fig_pdf: str = "EUR_NOK_RW_vs_Actual_Monthly.pdf"

CFG = Config()
TARGET_SERIES = "EUR_NOK"

# -----------------------------
# Data
# -----------------------------
def load_series_from_allvariables(url: str) -> Tuple[pd.Series, pd.Series]:
    """
    Read variables_daily.csv (wide daily panel).

    Expected columns:
        Date, EUR_NOK, ... (others ignored)

    Returns:
      S_b: business-day (B) series with ffill (for cut and monthly target)
      S_d: daily (D) series with ffill (for history checks)
    """
    raw = pd.read_csv(url)

    required_cols = {"Date", TARGET_SERIES}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Got: {list(raw.columns)}")

    df = (
        raw[["Date", TARGET_SERIES]]
        .rename(columns={"Date": "DATE"})
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"], errors="coerce"))
        .dropna(subset=["DATE", TARGET_SERIES])
        .sort_values("DATE")
        .set_index("DATE")
    )

    # numeric coercion
    df[TARGET_SERIES] = pd.to_numeric(df[TARGET_SERIES], errors="coerce")
    df = df.dropna(subset=[TARGET_SERIES])

    # B-series (target / aggregation base)
    S_b = df[TARGET_SERIES].asfreq("B").ffill().astype(float)
    S_b.name = TARGET_SERIES

    # D-series (calendar days)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    S_d = df[TARGET_SERIES].reindex(full_idx).ffill().astype(float)
    S_d.index.name = "DATE"
    S_d.name = TARGET_SERIES

    return S_b, S_d


def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return the last available business day in [start, end] for S_b (B-calendar).
    """
    sl = S_b.loc[start:end]
    if sl.empty:
        return None
    return sl.index[-1]

# -----------------------------
# Walk-forward (monthly) – Random Walk
# -----------------------------
def walk_forward_rw_monthly(S_b: pd.Series, S_d: pd.Series) -> pd.DataFrame:
    """
    Monthly walk-forward evaluation using a Random Walk benchmark:

    - For each month m, use the last business day in m-1 as the cut.
    - Forecast for month m is the EUR/NOK level at the cut (RW).
    - Target is the monthly mean of S_b within m (business days).
    - History requirement is checked on daily data (S_d) up to the cut.
    """
    first_m = pd.Period(S_b.index.min(), freq=CFG.m_freq)
    last_m  = pd.Period(S_b.index.max(), freq=CFG.m_freq)
    months = pd.period_range(first_m, last_m, freq=CFG.m_freq)

    rows: Dict[str, Dict] = {}
    dropped: Dict[str, str] = {}

    for m in months:
        prev_m = m - 1
        m_start, m_end = m.start_time, m.end_time
        prev_start, prev_end = prev_m.start_time, prev_m.end_time

        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(m)] = "no_cut_in_prev_month"
            continue

        hist_d = S_d.loc[:cut]
        if hist_d.size < CFG.min_hist_days:
            dropped[str(m)] = f"hist<{CFG.min_hist_days}"
            continue

        idx_m_b = S_b.index[(S_b.index >= m_start) & (S_b.index <= m_end)]
        if idx_m_b.size < 1:
            dropped[str(m)] = "no_bdays_in_month"
            continue
        y_true = float(S_b.loc[idx_m_b].mean())

        y_pred = float(S_b.loc[cut])

        rows[str(m)] = {
            "month": m,
            "cut": cut,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.set_index("month").sort_index()

    if CFG.verbose and dropped:
        miss = [str(m) for m in months if m not in df.index]
        if miss:
            print("\nDropped months and reasons:")
            for mm in miss:
                print(f"  {mm}: {dropped.get(mm, 'unknown')}")

    return df

# -----------------------------
# Evaluation (level + direction)
# -----------------------------
def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RMSE, MAE and directional accuracy for the monthly RW benchmark.
    """
    df = df.copy()
    df["err"] = df["y_true"] - df["y_pred"]
    eval_df = df.dropna(subset=["y_true", "y_pred"]).copy()

    n_obs = int(len(eval_df))
    rmse = float(np.sqrt(mean_squared_error(eval_df["y_true"], eval_df["y_pred"]))) if n_obs else np.nan
    mae  = float(mean_absolute_error(eval_df["y_true"], eval_df["y_pred"])) if n_obs else np.nan

    eval_df["y_prev"] = eval_df["y_true"].shift(1)
    mask = eval_df["y_prev"].notna()
    dir_true = np.sign(eval_df.loc[mask, "y_true"] - eval_df.loc[mask, "y_prev"])
    dir_pred = np.sign(eval_df.loc[mask, "y_pred"] - eval_df.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    hit_rate = (hits / total) if total else np.nan

    print("\n=== Random Walk performance (monthly mean, EUR/NOK) ===")
    print(f"Observations: {n_obs}")
    print(f"RMSE (level): {rmse:.6f}")
    print(f"MAE  (level): {mae:.6f}")
    if total:
        print(f"Directional accuracy: {hits}/{total} ({hit_rate*100:.1f}%)")

    return eval_df

# -----------------------------
# Plot (no bands)
# -----------------------------
def plot_monthly_rw(eval_df: pd.DataFrame, png_path: str, pdf_path: str):
    """
    Plot actual monthly means vs. Random Walk forecasts.
    """
    if eval_df.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    plt.plot(x, eval_df["y_true"], color="black", label="Actual (monthly mean, B-days)")
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--",
             label="Forecast (Random Walk)")

    plt.title("Random Walk Forecast vs Actual (Monthly Mean, EUR/NOK)")
    plt.xlabel("Month")
    plt.ylabel("Level (EUR/NOK)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    S_b, S_d = load_series_from_allvariables(CFG.url)
    if CFG.verbose:
        print(f"Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")
        print(f"Data (D): {S_d.index.min().date()} → {S_d.index.max().date()} | n={len(S_d)}")

    df_eval = walk_forward_rw_monthly(S_b, S_d)
    eval_df = evaluate(df_eval)
    plot_monthly_rw(eval_df, CFG.fig_png, CFG.fig_pdf)

if __name__ == "__main__":
    main()
