from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from qqq_nq_model import (
    BASE_FEATURES,
    OPTION_FEATURES,
    TARGETS,
    add_options_features,
    download_intraday,
    evaluate_model,
    load_price_csv,
    make_features,
)
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

st.set_page_config(page_title="QQQ → NQ High/Low Model", page_icon="📈", layout="wide")
st.title("QQQ → NQ RTH High/Low Model")
st.caption("Research dashboard • Forecasts remaining NQ regular-session high/low extension")
st.warning(
    "Research prototype only—not trading advice. Yahoo intraday history is limited, and the quality of the forecasts depends on clean historical data. "
    "Options features are only used when you upload timestamped snapshots available by the selected forecast time."
)

with st.sidebar:
    st.header("Forecast settings")
    forecast_time = st.selectbox("Forecast time (New York)", ["09:45", "10:00", "10:15", "10:30", "11:00", "12:00", "13:00", "14:00"], index=1)
    data_source = st.radio("Price data source", ["Download from Yahoo Finance", "Upload CSV files"], index=0)
    period = st.selectbox("Yahoo history window", ["30d", "60d"], index=1, help="Yahoo Finance limits intraday history. For a serious backtest, upload a longer verified dataset.")
    interval = st.selectbox("Yahoo bar interval", ["5m", "15m", "30m"], index=0)
    min_train = st.slider("Minimum training sessions", min_value=15, max_value=80, value=25)
    test_fraction = st.slider("Test fraction (latest sessions)", min_value=0.15, max_value=0.40, value=0.25, step=0.05)

nq_upload = qqq_upload = options_upload = None
if data_source == "Upload CSV files":
    c1, c2 = st.columns(2)
    with c1:
        nq_upload = st.file_uploader("NQ intraday CSV", type=["csv"], key="nq")
    with c2:
        qqq_upload = st.file_uploader("QQQ intraday CSV", type=["csv"], key="qqq")
options_upload = st.file_uploader("Optional options snapshot CSV (GEX / vanna / IV / walls)", type=["csv"], help="Required columns: session_date, forecast_time. Optional feature columns are listed below.")
with st.expander("CSV formats and supported options fields"):
    st.markdown("**Price CSV columns:** `timestamp, open, high, low, close, volume`. Timestamps may be timezone-aware; naive timestamps are assumed to be New York time.")
    st.markdown("**Options snapshot columns:** `session_date, forecast_time` plus any of the following:")
    st.code(", ".join(OPTION_FEATURES))
    st.caption("Each options row must represent a snapshot taken at or before that day's selected forecast time. Do not use end-of-day GEX/vanna to predict earlier in the session.")


def uploaded_csv_to_temp(uploaded, suffix=".csv"):
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return tmp.name


def train_and_backtest(nq, qqq, options_path):
    features = make_features(nq, qqq, forecast_time)
    features = add_options_features(features, options_path)
    feature_cols = BASE_FEATURES.copy()
    if options_path:
        feature_cols += [c for c in OPTION_FEATURES if c in features.columns and features[c].notna().sum() > 0]
    data = features.dropna(subset=feature_cols + TARGETS).sort_values("session_date").reset_index(drop=True)
    if len(data) < min_train + 10:
        raise ValueError(
            f"Only {len(data)} usable sessions remain. Need at least {min_train + 10}. "
            "Try longer verified CSV history, fewer required training sessions, or check timestamps/data coverage."
        )
    split = int(len(data) * (1.0 - test_fraction))
    split = max(split, min_train)
    if split >= len(data) - 5:
        raise ValueError("Too few test sessions. Lower the minimum training sessions or upload more data.")
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()
    predictions = test[["session_date", "forecast_time", "nq_spot", "nq_running_high", "nq_running_low"]].copy()
    metrics = []
    models = {}
    for target in TARGETS:
        for q in [0.50, 0.75, 0.90]:
            model, pred = evaluate_model(train, test, feature_cols, target, q)
            models[(target, q)] = model
            predictions[f"{target}_q{int(q*100)}"] = pred
            metrics.append({
                "target": target,
                "quantile": f"{int(q*100)}th percentile",
                "pinball_loss": mean_pinball_loss(test[target], pred, alpha=q),
                "extension_MAE_points": mean_absolute_error(test[target], pred),
                "train_sessions": len(train),
                "test_sessions": len(test),
            })
    for q in [50, 75, 90]:
        predictions[f"pred_high_q{q}"] = predictions["nq_running_high"] + predictions[f"upside_extension_q{q}"]
        predictions[f"pred_low_q{q}"] = predictions["nq_running_low"] - predictions[f"downside_extension_q{q}"]
    actual = test[["session_date", "upside_extension", "downside_extension"]]
    predictions = predictions.merge(actual, on="session_date", how="left")
    return data, train, test, predictions, pd.DataFrame(metrics), feature_cols, models

if st.button("Run model and walk-forward backtest", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading prices and fitting quantile models…"):
            options_path = uploaded_csv_to_temp(options_upload)
            if data_source == "Upload CSV files":
                if nq_upload is None or qqq_upload is None:
                    st.error("Upload both NQ and QQQ price CSV files first.")
                    st.stop()
                nq_path = uploaded_csv_to_temp(nq_upload)
                qqq_path = uploaded_csv_to_temp(qqq_upload)
                nq = load_price_csv(nq_path)
                qqq = load_price_csv(qqq_path)
            else:
                nq = download_intraday("NQ=F", period, interval)
                qqq = download_intraday("QQQ", period, interval)
            data, train, test, predictions, metrics, feature_cols, models = train_and_backtest(nq, qqq, options_path)
        st.session_state["model_results"] = {
            "data": data, "train": train, "test": test, "predictions": predictions,
            "metrics": metrics, "features": feature_cols,
        }
    except Exception as exc:
        st.error(f"Could not run model: {exc}")

results = st.session_state.get("model_results")
if results:
    data = results["data"]
    predictions = results["predictions"]
    metrics = results["metrics"]
    latest = predictions.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Usable sessions", f"{len(data)}")
    m2.metric("Training sessions", f"{len(results['train'])}")
    m3.metric("Test sessions", f"{len(results['test'])}")
    m4.metric("Forecast timestamp", str(forecast_time) + " ET")

    st.subheader("Latest held-out session forecast")
    st.caption(f"Session: {pd.to_datetime(latest['session_date']).date()} • This is the last date in the backtest, not a live forecast for today.")
    a, b, c = st.columns(3)
    a.metric("NQ high estimate · median", f"{latest['pred_high_q50']:,.2f}")
    b.metric("NQ low estimate · median", f"{latest['pred_low_q50']:,.2f}")
    c.metric("NQ at forecast time", f"{latest['nq_spot']:,.2f}")
    st.markdown("**Forecast bands**")
    band = pd.DataFrame({
        "Estimate": ["High · 50th percentile", "High · 75th percentile", "High · 90th percentile", "Low · 50th percentile", "Low · 75th percentile", "Low · 90th percentile"],
        "NQ level": [latest[f"pred_high_q{q}"] for q in [50, 75, 90]] + [latest[f"pred_low_q{q}"] for q in [50, 75, 90]],
    })
    st.dataframe(band.style.format({"NQ level": "{:,.2f}"}), use_container_width=True, hide_index=True)
    st.caption("These are quantile estimates of the final session high/low, conditional on the selected forecast time and the available features—not guaranteed boundaries.")

    st.subheader("Backtest performance")
    st.dataframe(metrics.style.format({"pinball_loss": "{:.3f}", "extension_MAE_points": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.subheader("Predicted vs actual high/low")
    chart_df = predictions.copy()
    chart_df["session_date"] = pd.to_datetime(chart_df["session_date"])
    chart_df = chart_df.set_index("session_date")
    high_cols = ["pred_high_q50", "pred_high_q75", "pred_high_q90"]
    low_cols = ["pred_low_q50", "pred_low_q75", "pred_low_q90"]
    st.markdown("**High estimates**")
    st.line_chart(chart_df[high_cols])
    st.markdown("**Low estimates**")
    st.line_chart(chart_df[low_cols])
    st.subheader("Backtest predictions")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button("Download predictions CSV", predictions.to_csv(index=False).encode("utf-8"), "nq_predictions.csv", "text/csv")
    st.download_button("Download metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "nq_metrics.csv", "text/csv")
    st.download_button("Download feature/target dataset CSV", data.to_csv(index=False).encode("utf-8"), "nq_features_targets.csv", "text/csv")
    with st.expander("Features used by this run"):
        st.code(", ".join(results["features"]))
else:
    st.info("Set the options in the sidebar, then run the model. You can start with Yahoo data or upload your own longer, verified intraday CSVs.")

st.divider()
st.caption("Model limitations: small Yahoo sample, proxy futures feed, and no automatic options-data collection. For reliable research, validate data timestamps and test out-of-sample over many months. No prediction guarantees the daily high or low.")
