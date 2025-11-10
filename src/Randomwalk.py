# -*- coding: utf-8 -*-
"""
Random Walk (RW) walk-forward (next quarter) on Norges Bank EUR/NOK data:

- Dataset: EUR/NOK spot from Norges Bank (daily levels), provided via GitHub CSV
- Cut: last business day in the previous quarter (based on B-calendar with ffill)
- Minimum history before cut: min_hist_days
- Target: quarterly mean of daily business-day levels
- Evaluates over ALL available quarters where the requirements are satisfied
"""

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    # Source: GitHub CSV (Norges Bank EUR/NOK, semicolon-separated, decimal comma)
    url: str = (
        "https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/"
        "refs/heads/main/EURNOK/EUR_NOK_NorgesBank.csv"
    )
    business_freq: str = "B"     # business days
    q_freq: str = "Q-DEC"        # quarterly periods with December year-end
    min_hist_days: int = 40      # TimesFM-like minimum history requirement
    verbose: bool = True
    fig_png: str = "EUR_NOK_RW_vs_Actual.png"
    fig_pdf: str = "EUR_NOK_RW_vs_Actual.pdf"


CFG = Config()

# -----------------------------
# Helpers
# -----------------------------
def last_trading_day(series: pd.Series,
                     start: pd.Timestamp,
                     end: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Return the last available business day in [start, end] for a pre-filled B-calendar series.
    """
    window = series.loc[start:end]
    if window.empty:
        return None
    return window.index[-1]


def load_business_series(url: str) -> pd.Series:
    """
    Load EUR/NOK series from GitHub CSV (semicolon + decimal comma),
    convert to a business-day (B) frequency with forward fill.

    Returns:
        S_b: pd.Series with B-frequency and EUR/NOK levels.
    """
    df = pd.read_csv(
        url,
        sep=";",
        encoding="utf-8-sig",
        decimal=","
    )

    required_cols = {"TIME_PERIOD", "OBS_VALUE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Got: {list(df.columns)}")

    df = (
        df[["TIME_PERIOD", "OBS_VALUE"]]
        .rename(columns={"OBS_VALUE": "EUR_NOK"})
        .assign(TIME_PERIOD=lambda x: pd.to_datetime(x["TIME_PERIOD"], errors="coerce"))
        .dropna(subset=["TIME_PERIOD", "EUR_NOK"])
        .sort_values("TIME_PERIOD")
        .set_index("TIME_PERIOD")
    )

    # Business-day series with forward fill (no weekend observations in source)
    S_b = df["EUR_NOK"].asfreq(CFG.business_freq).ffill().astype(float)
    S_b.name = "EUR_NOK"
    return S_b

# -----------------------------
# Walk-forward Random Walk (quarterly)
# -----------------------------
def walk_forward_rw(S_b: pd.Series) -> pd.DataFrame:
    """
    Quarterly walk-forward evaluation using a Random Walk benchmark:

    - For each quarter q, use the last business day in q-1 as the cut.
    - Forecast for q is the EUR/NOK level at the cut (RW).
    - Target is the quarterly mean of S_b within q.
    """
    first_q = pd.Period(S_b.index.min(), freq=CFG.q_freq)
    last_q = pd.Period(S_b.index.max(), freq=CFG.q_freq)
    quarters = pd.period_range(first_q, last_q, freq=CFG.q_freq)

    rows: Dict[str, Dict] = {}
    dropped: Dict[str, str] = {}

    for q in quarters:
        prev_q = q - 1
        q_start, q_end = q.start_time, q.end_time
        prev_start, prev_end = prev_q.start_time, prev_q.end_time

        # Cut in the previous quarter
        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(q)] = "no_cut_in_prev_q"
            continue

        # History requirement before cut
        hist = S_b.loc[:cut]
        if hist.size < CFG.min_hist_days:
            dropped[str(q)] = f"hist<{CFG.min_hist_days}"
            continue

        # Business days inside the quarter
        idx_q = S_b.index[(S_b.index >= q_start) & (S_b.index <= q_end)]
        if idx_q.size < 1:
            dropped[str(q)] = "no_bdays_in_q"
            continue

        # Target: quarterly mean (business days)
        y_true = float(S_b.loc[idx_q].mean())

        # RW forecast: last observed level at cut
        y_pred = float(S_b.loc[cut])

        rows[str(q)] = {
            "quarter": q,
            "cut": cut,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.set_index("quarter").sort_index()

    if CFG.verbose and dropped:
        missing = [str(q) for q in quarters if q not in df.index]
        if missing:
            print("\nDropped quarters and reasons:")
            for q in missing:
                print(f"  {q}: {dropped.get(q, 'unknown')}")

    return df

# -----------------------------
# Evaluation (level + direction)
# -----------------------------
def evaluate(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RMSE, MAE and directional accuracy for the RW benchmark.
    """
    df = eval_df.copy()
    df["err"] = df["y_true"] - df["y_pred"]
    df = df.dropna(subset=["y_true", "y_pred"]).copy()

    n_obs = int(len(df))
    rmse = float(np.sqrt(np.mean(np.square(df["err"])))) if n_obs else np.nan
    mae = float(mean_absolute_error(df["y_true"], df["y_pred"])) if n_obs else np.nan

    # Directional accuracy: sign of change from previous quarter
    df["y_prev"] = df["y_true"].shift(1)
    mask = df["y_prev"].notna()
    dir_true = np.sign(df.loc[mask, "y_true"] - df.loc[mask, "y_prev"])
    dir_pred = np.sign(df.loc[mask, "y_pred"] - df.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    hit_rate = (hits / total) if total else np.nan

    print("\n=== Random Walk performance (quarterly mean, EUR/NOK) ===")
    print(f"Observations: {n_obs}")
    print(f"RMSE (level): {rmse:.6f}")
    print(f"MAE  (level): {mae:.6f}")
    if total:
        print(f"Directional accuracy: {hits}/{total} ({hit_rate*100:.1f}%)")

    # Show first/last rows for inspection
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\nFirst 5 rows:")
        print(df[["cut", "y_true", "y_pred"]].head(5))
        print("\nLast 5 rows:")
        print(df[["cut", "y_true", "y_pred"]].tail(5))

    return df

# -----------------------------
# Plot – same style as TimesFM example (no bands)
# -----------------------------
def plot_quarterly_rw(eval_df: pd.DataFrame,
                      png_path: str,
                      pdf_path: str) -> None:
    """
    Plot actual quarterly means vs. Random Walk forecasts
    using the same simple style as in the TimesFM example.
    """
    if eval_df.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    # Actual: black line
    plt.plot(x, eval_df["y_true"], color="black", label="Actual (quarterly mean)")

    # RW forecast: blue dashed line
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--",
             label="Forecast (Random Walk)")

    plt.title("Random Walk Forecast vs Actual (Quarterly Mean, EUR/NOK)")
    plt.xlabel("Quarter")
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
    # 1) Load business-day series
    S_b = load_business_series(CFG.url)
    if CFG.verbose:
        print(f"Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")
        print(S_b.describe())

    # 2) Walk-forward Random Walk benchmark
    df_eval = walk_forward_rw(S_b)

    # 3) Evaluation
    eval_df = evaluate(df_eval)

    # 4) Plot in the same style as the TimesFM example (no confidence bands)
    plot_quarterly_rw(eval_df, CFG.fig_png, CFG.fig_pdf)


if __name__ == "__main__":
    main()
