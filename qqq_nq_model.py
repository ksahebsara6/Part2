#!/usr/bin/env python3
"""
QQQ -> NQ daily RTH high/low research model (no paid API required).

What it does
------------
1. Downloads recent 5-minute QQQ and NQ futures bars from Yahoo Finance.
2. Builds one row per New York regular session, using information available
   at the forecast time (default 10:00 ET).
3. Predicts the *remaining* upside/downside extension from the running NQ
   high/low to the final RTH high/low, with quantile models.
4. Runs a chronological holdout backtest and saves predictions/metrics.

Important limitations
---------------------
- Yahoo Finance intraday history is limited and can change. This is a prototype,
  not a robust multi-year backtest.
- NQ=F Yahoo bars may have provider/session quirks. Inspect the downloaded data.
  For reliable RTH labels, use verified 1m/5m NQ futures data and provide it as
  CSV (see README instructions printed by --help).
- This first version does NOT include historical GEX/vanna unless you provide
  an options snapshot CSV. Exposure features must be timestamped and available
  by the forecast time; end-of-day values must not be used for earlier forecasts.
- Forecasts are statistical estimates, not trading advice.

Install:
    pip install yfinance pandas numpy scikit-learn

Run:
    python qqq_nq_model.py
    python qqq_nq_model.py --period 60d --interval 5m --forecast-time 10:00
    python qqq_nq_model.py --nq-csv nq_5m.csv --qqq-csv qqq_5m.csv
    python qqq_nq_model.py --options-csv options_snapshots.csv

CSV price format:
    timestamp,open,high,low,close,volume
Timestamp can be UTC or timezone-aware. Naive timestamps are assumed to be
America/New_York. CSV should contain intraday bars, ideally 1m or 5m.

Optional options CSV:
    session_date,forecast_time,qqq_atm_iv,qqq_iv_change,net_gex_norm,
    net_vanna_norm,net_dex_norm,gex_near_spot,call_wall_distance_pct,
    put_wall_distance_pct,gamma_flip_distance_pct

Options columns are optional; absent columns are ignored. Dates should be
YYYY-MM-DD and forecast_time like 10:00. Values must reflect a snapshot taken
at or before that forecast time.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install packages with:\n"
        "pip install yfinance pandas numpy scikit-learn"
    ) from exc

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss


ET = "America/New_York"
TARGETS = ["upside_extension", "downside_extension"]

BASE_FEATURES = [
    "qqq_overnight_return",
    "qqq_opening_gap_pct",
    "qqq_opening_range_pct",
    "qqq_return_since_open_pct",
    "qqq_range_so_far_pct",
    "nq_opening_gap_pct",
    "nq_opening_range_pts",
    "nq_range_so_far_pts",
    "nq_return_since_open_pts",
    "prior_nq_range_pts",
    "prior_qqq_range_pct",
    "qqq_atr_pct",
    "nq_atr_pts",
    "minutes_remaining",
]

OPTION_FEATURES = [
    "qqq_atm_iv",
    "qqq_iv_change",
    "net_gex_norm",
    "net_vanna_norm",
    "net_dex_norm",
    "gex_near_spot",
    "call_wall_distance_pct",
    "put_wall_distance_pct",
    "gamma_flip_distance_pct",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--period", default="60d", help="Yahoo download period, e.g. 30d or 60d (default: 60d)")
    p.add_argument("--interval", default="5m", help="Yahoo intraday interval, e.g. 1m or 5m (default: 5m)")
    p.add_argument("--forecast-time", default="10:00", help="New York forecast time, HH:MM (default: 10:00)")
    p.add_argument("--nq-ticker", default="NQ=F", help="Yahoo futures ticker (default: NQ=F)")
    p.add_argument("--qqq-ticker", default="QQQ", help="Yahoo ETF ticker (default: QQQ)")
    p.add_argument("--nq-csv", default=None, help="Optional local NQ intraday CSV")
    p.add_argument("--qqq-csv", default=None, help="Optional local QQQ intraday CSV")
    p.add_argument("--options-csv", default=None, help="Optional timestamped options feature CSV")
    p.add_argument("--outdir", default="nq_model_output", help="Output directory")
    p.add_argument("--min-train-sessions", type=int, default=25, help="Minimum sessions before test (default: 25)")
    p.add_argument("--test-fraction", type=float, default=0.25, help="Final chronological test fraction (default: 0.25)")
    return p.parse_args()


def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower_map = {c.lower(): c for c in df.columns}
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in lower_map]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}; expected timestamp,open,high,low,close[,volume]")
    df = df.rename(columns={lower_map[c]: c for c in required if c in lower_map})
    if "volume" not in df.columns:
        vol_col = lower_map.get("volume")
        df["volume"] = df[vol_col] if vol_col else 0
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{path}: some timestamps could not be parsed")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ET, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(ET)
    df["timestamp"] = ts
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp")
    return df.set_index("timestamp")


def download_intraday(ticker: str, period: str, interval: str) -> pd.DataFrame:
    print(f"Downloading {ticker} ({period}, {interval}) from Yahoo Finance...")
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Try a shorter period/other interval, "
            "or supply a local CSV with --nq-csv / --qqq-csv."
        )
    # yfinance sometimes returns multi-level column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename_axis("timestamp").reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    if "datetime" in raw.columns and "timestamp" not in raw.columns:
        raw = raw.rename(columns={"datetime": "timestamp"})
    if "date" in raw.columns and "timestamp" not in raw.columns:
        raw = raw.rename(columns={"date": "timestamp"})
    if "timestamp" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "timestamp"})
    # Normalize names expected by CSV loader.
    return load_price_csv_from_frame(raw)


def load_price_csv_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    aliases = {
        "timestamp": ["timestamp", "datetime", "date"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close"],
        "volume": ["volume"],
    }
    rename = {}
    for target, names in aliases.items():
        found = next((lower_map[n] for n in names if n in lower_map), None)
        if found is not None:
            rename[found] = target
    df = df.rename(columns=rename)
    return load_price_frame(df)


def load_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Price data must contain timestamp column")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Some timestamps could not be parsed")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ET, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(ET)
    df = df.copy()
    df["timestamp"] = ts
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp")
    return df.set_index("timestamp")


def load_or_download(args):
    nq = load_price_csv(args.nq_csv) if args.nq_csv else download_intraday(args.nq_ticker, args.period, args.interval)
    qqq = load_price_csv(args.qqq_csv) if args.qqq_csv else download_intraday(args.qqq_ticker, args.period, args.interval)
    return nq, qqq


def session_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Regular session bars only: 09:30 <= local time < 16:00 ET."""
    local = df.tz_convert(ET)
    mask = (local.index.time >= pd.Timestamp("09:30").time()) & (local.index.time < pd.Timestamp("16:00").time())
    return local.loc[mask].copy()


def by_date(df: pd.DataFrame):
    return {str(date): group.sort_index() for date, group in df.groupby(df.index.date)}


def pct(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return 100.0 * (a / b - 1.0)


def true_range(high, low, prev_close):
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def make_features(nq_all: pd.DataFrame, qqq_all: pd.DataFrame, forecast_time: str) -> pd.DataFrame:
    nq_rth = session_bars(nq_all)
    qqq_rth = session_bars(qqq_all)
    nq_days = by_date(nq_rth)
    qqq_days = by_date(qqq_rth)
    nq_all_days = by_date(nq_all)
    qqq_all_days = by_date(qqq_all)

    dates = sorted(set(nq_days).intersection(qqq_days))
    if len(dates) < 15:
        raise RuntimeError(
            f"Only {len(dates)} overlapping RTH sessions found. Need more data. "
            "Check ticker symbols, timestamps, and CSV format."
        )

    forecast_h, forecast_m = map(int, forecast_time.split(":"))
    forecast_minute = forecast_h * 60 + forecast_m
    rows = []
    prior_nq_ranges, prior_qqq_ranges = [], []
    prior_nq_closes, prior_qqq_closes = [], []

    for d in dates:
        nq = nq_days[d]
        qqq = qqq_days[d]
        nq_prior = prior_nq_closes[-1] if prior_nq_closes else np.nan
        qqq_prior = prior_qqq_closes[-1] if prior_qqq_closes else np.nan

        # Use only bars that started before the forecast time; the bar stamped exactly at the forecast time is not yet complete.
        nq_times = nq.index.hour * 60 + nq.index.minute
        qqq_times = qqq.index.hour * 60 + qqq.index.minute
        nq_seen = nq.loc[nq_times < forecast_minute]
        qqq_seen = qqq.loc[qqq_times < forecast_minute]
        if nq_seen.empty or qqq_seen.empty:
            # No forecast snapshot for this date.
            # Still update history below.
            pass
        else:
            nq_open = float(nq.iloc[0]["open"])
            qqq_open = float(qqq.iloc[0]["open"])
            nq_seen_high = float(nq_seen["high"].max())
            nq_seen_low = float(nq_seen["low"].min())
            qqq_seen_high = float(qqq_seen["high"].max())
            qqq_seen_low = float(qqq_seen["low"].min())
            nq_last = float(nq_seen.iloc[-1]["close"])
            qqq_last = float(qqq_seen.iloc[-1]["close"])
            nq_final_high = float(nq["high"].max())
            nq_final_low = float(nq["low"].min())

            # Overnight QQQ return uses latest premarket close if available.
            qqq_all_day = qqq_all_days.get(d, pd.DataFrame())
            if not qqq_all_day.empty:
                pre = qqq_all_day.loc[qqq_all_day.index.time < pd.Timestamp("09:30").time()]
                qqq_pre_close = float(pre.iloc[-1]["close"]) if not pre.empty else np.nan
            else:
                qqq_pre_close = np.nan

            qqq_prev_close = qqq_prior
            qqq_overnight_return = pct(qqq_pre_close, qqq_prev_close) if pd.notna(qqq_pre_close) and pd.notna(qqq_prev_close) else np.nan
            qqq_gap = pct(qqq_open, qqq_prev_close) if pd.notna(qqq_prev_close) else np.nan
            nq_gap = pct(nq_open, nq_prior) if pd.notna(nq_prior) else np.nan

            prior_nq_range = float(np.mean(prior_nq_ranges[-14:])) if prior_nq_ranges else np.nan
            prior_qqq_range = float(np.mean(prior_qqq_ranges[-14:])) if prior_qqq_ranges else np.nan
            nq_atr = prior_nq_range
            qqq_atr_pct = (prior_qqq_range / qqq_prev_close * 100.0) if pd.notna(prior_qqq_range) and pd.notna(qqq_prev_close) else np.nan

            rows.append({
                "session_date": pd.Timestamp(d),
                "forecast_time": forecast_time,
                "nq_spot": nq_last,
                "nq_running_high": nq_seen_high,
                "nq_running_low": nq_seen_low,
                "nq_open": nq_open,
                "qqq_open": qqq_open,
                "qqq_overnight_return": qqq_overnight_return,
                "qqq_opening_gap_pct": qqq_gap,
                "qqq_opening_range_pct": pct(qqq_seen_high, qqq_open) + pct(qqq_open, qqq_seen_low) if pd.notna(qqq_open) else np.nan,
                "qqq_return_since_open_pct": pct(qqq_last, qqq_open),
                "qqq_range_so_far_pct": (qqq_seen_high - qqq_seen_low) / qqq_open * 100.0,
                "nq_opening_gap_pct": nq_gap,
                "nq_opening_range_pts": nq_seen_high - nq_seen_low,
                "nq_range_so_far_pts": nq_seen_high - nq_seen_low,
                "nq_return_since_open_pts": nq_last - nq_open,
                "prior_nq_range_pts": prior_nq_range,
                "prior_qqq_range_pct": (prior_qqq_range / qqq_prev_close * 100.0) if pd.notna(prior_qqq_range) and pd.notna(qqq_prev_close) else np.nan,
                "qqq_atr_pct": qqq_atr_pct,
                "nq_atr_pts": nq_atr,
                "minutes_remaining": max(0, 16 * 60 - forecast_minute),
                "upside_extension": max(0.0, nq_final_high - nq_seen_high),
                "downside_extension": max(0.0, nq_seen_low - nq_final_low),
            })

        # Update history only after today's feature row is created.
        prior_nq_ranges.append(float(nq["high"].max() - nq["low"].min()))
        prior_qqq_ranges.append(float(qqq["high"].max() - qqq["low"].min()))
        prior_nq_closes.append(float(nq.iloc[-1]["close"]))
        prior_qqq_closes.append(float(qqq.iloc[-1]["close"]))

    return pd.DataFrame(rows)


def add_options_features(features: pd.DataFrame, options_csv: str | None) -> pd.DataFrame:
    if not options_csv:
        for col in OPTION_FEATURES:
            features[col] = np.nan
        return features
    opt = pd.read_csv(options_csv)
    required = {"session_date", "forecast_time"}
    if not required.issubset(opt.columns):
        raise ValueError("Options CSV must contain session_date and forecast_time")
    opt["session_date"] = pd.to_datetime(opt["session_date"]).dt.normalize()
    opt["forecast_time"] = opt["forecast_time"].astype(str).str.slice(0, 5)
    keep = ["session_date", "forecast_time"] + [c for c in OPTION_FEATURES if c in opt.columns]
    features["session_date"] = pd.to_datetime(features["session_date"]).dt.normalize()
    features = features.merge(
        opt[keep].drop_duplicates(["session_date", "forecast_time"]),
        on=["session_date", "forecast_time"],
        how="left",
    )
    for col in OPTION_FEATURES:
        if col not in features.columns:
            features[col] = np.nan
    return features


def evaluate_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, quantile: float):
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(train[features], train[target])
    pred = np.maximum(model.predict(test[features]), 0.0)
    return model, pred


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nq, qqq = load_or_download(args)
    features = make_features(nq, qqq, args.forecast_time)
    features = add_options_features(features, args.options_csv)

    feature_cols = BASE_FEATURES.copy()
    if args.options_csv:
        feature_cols += [c for c in OPTION_FEATURES if c in features.columns and features[c].notna().sum() > 0]

    needed = feature_cols + TARGETS
    data = features.dropna(subset=needed).sort_values("session_date").reset_index(drop=True)
    if len(data) < args.min_train_sessions + 10:
        raise RuntimeError(
            f"Only {len(data)} usable sessions after missing-data filtering. "
            f"Need at least {args.min_train_sessions + 10}. Yahoo's intraday history may be too short; "
            "provide longer local CSV files."
        )

    split = int(len(data) * (1.0 - args.test_fraction))
    split = max(split, args.min_train_sessions)
    if split >= len(data) - 5:
        raise RuntimeError("Too few test sessions. Lower --min-train-sessions or provide more history.")
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()

    all_predictions = test[
        ["session_date", "forecast_time", "nq_spot", "nq_running_high", "nq_running_low"]
    ].copy()
    metric_rows = []

    for target in TARGETS:
        for q in [0.50, 0.75, 0.90]:
            model, pred = evaluate_model(train, test, feature_cols, target, q)
            all_predictions[f"{target}_q{int(q*100)}"] = pred
            metric_rows.append({
                "target": target,
                "quantile": q,
                "pinball_loss": mean_pinball_loss(test[target], pred, alpha=q),
                "mae_extension_points": mean_absolute_error(test[target], pred),
                "n_train": len(train),
                "n_test": len(test),
            })

    # Price boundaries based on the current running extreme at forecast time.
    for q in [50, 75, 90]:
        all_predictions[f"pred_high_q{q}"] = (
            all_predictions["nq_running_high"] + all_predictions[f"upside_extension_q{q}"]
        )
        all_predictions[f"pred_low_q{q}"] = (
            all_predictions["nq_running_low"] - all_predictions[f"downside_extension_q{q}"]
        )

    # Add actual targets for inspection.
    actual = test[["session_date", "upside_extension", "downside_extension"]]
    all_predictions = all_predictions.merge(actual, on="session_date", how="left")
    all_predictions.to_csv(outdir / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(outdir / "metrics.csv", index=False)
    data.to_parquet(outdir / "features_targets.parquet", index=False)

    print("\n=== Backtest complete ===")
    print(f"Forecast time: {args.forecast_time} ET")
    print(f"Usable sessions: {len(data)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Features used: {', '.join(feature_cols)}")
    print("\nMetrics:")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"\nSaved files to: {outdir.resolve()}")
    print("  predictions.csv       Forecasts vs actual outcomes")
    print("  metrics.csv           Quantile loss and extension MAE")
    print("  features_targets.parquet  Model-ready dataset")
    print("\nInterpretation: with short Yahoo intraday history, results are exploratory only.")
    print("For stronger validation, provide at least several hundred RTH sessions of verified intraday bars.")


if __name__ == "__main__":
    main()
