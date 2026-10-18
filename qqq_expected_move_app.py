from datetime import date, datetime, timedelta
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="QQQ Expected Move & Historical Range Model",
    page_icon="📈",
    layout="wide",
)

NY_TZ = "America/New_York"
PERCENTILES = [50, 70, 80, 90, 95]
QUANTILES = [p / 100 for p in PERCENTILES]
Z_SCORES = {
    "50%": 0.6744897501960817,
    "68%": 1.0,
    "90%": 1.6448536269514722,
    "95%": 1.959963984540054,
    "99%": 2.5758293035489004,
}

st.title("QQQ Expected Move & Historical Range Model")
st.caption(
    "Combines volatility-based estimates with empirical QQQ regular-session range statistics. "
    "All projections are estimates, not guaranteed price boundaries or trading signals."
)


def next_weekday(d: date) -> date:
    """Return d if it is a weekday; otherwise return the next Monday-Friday date."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _column_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        field_names = {"open", "high", "low", "close", "adj close", "volume", "date", "datetime"}
        new_cols = []
        for col in out.columns:
            parts = [str(x) for x in col if str(x) not in ("", "None")]
            field = next((p for p in parts if p.lower() in field_names), parts[0] if parts else "")
            new_cols.append(field)
        out.columns = new_cols
    return out


def _map_columns(df: pd.DataFrame, is_intraday: bool = False) -> pd.DataFrame:
    """Map common CSV/Yahoo Finance labels to canonical column names."""
    out = _flatten_yahoo_columns(df)
    aliases = {
        "date": {"date", "day", "tradingdate"},
        "timestamp": {"datetime", "timestamp", "time", "date", "bar time"},
        "open": {"open", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "c", "last"},
        "volume": {"volume", "vol"},
    }
    key_to_col = {_column_key(col): col for col in out.columns}
    selected = {}
    time_aliases = aliases["timestamp"] if is_intraday else aliases["date"]
    for candidate in time_aliases:
        if _column_key(candidate) in key_to_col:
            selected["timestamp" if is_intraday else "date"] = key_to_col[_column_key(candidate)]
            break
    for canonical in ["open", "high", "low", "close", "volume"]:
        for alias in aliases[canonical]:
            if _column_key(alias) in key_to_col:
                selected[canonical] = key_to_col[_column_key(alias)]
                break
    missing = [c for c in (["timestamp"] if is_intraday else ["date"]) + ["open", "high", "low", "close"] if c not in selected]
    if missing:
        raise ValueError(
            "CSV must contain a date/time column and Open, High, Low, Close. "
            f"Missing: {', '.join(missing)}."
        )
    rename = {original: canonical for canonical, original in selected.items()}
    out = out.rename(columns=rename)
    return out


def normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    out = _map_columns(raw, is_intraday=False)
    dt = pd.to_datetime(out["date"], errors="coerce")
    try:
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    out["date"] = dt.dt.normalize()
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
    out = out[out["high"] >= out["low"]]
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    # Normalize optional enrichment columns if they exist in a user CSV.
    optional_aliases = {
        "overnight_return_pct": {"overnightreturnpct", "overnightreturn", "overnightreturnpercent"},
        "overnight_range_pct": {"overnightrangepct", "overnightrange", "premarketrangepct"},
        "premarket_iv_pct": {"premarketivpct", "premarketiv", "ivbeforeopen"},
        "macro_event": {"macroevent", "macroeconomicevent", "scheduledmacrorelease"},
        "expiration_day": {"expirationday", "isexpirationday", "expiryday"},
        "expiration_week": {"expirationweek", "isexpirationweek", "expiryweek"},
    }
    existing = {_column_key(c): c for c in out.columns}
    for canonical, aliases in optional_aliases.items():
        found = next((existing[a] for a in aliases if a in existing), None)
        if found is not None:
            if found != canonical:
                out = out.rename(columns={found: canonical})
            if canonical in {"overnight_return_pct", "overnight_range_pct", "premarket_iv_pct"}:
                out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
            else:
                out[canonical] = out[canonical].astype(str).str.strip()
    return out


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("date").reset_index(drop=True)
    out["daily_range"] = out["high"] - out["low"]
    out["up_excursion"] = (out["high"] - out["open"]).clip(lower=0)
    out["down_excursion"] = (out["open"] - out["low"]).clip(lower=0)
    out["range_pct"] = out["daily_range"] / out["open"] * 100
    out["up_excursion_pct"] = out["up_excursion"] / out["open"] * 100
    out["down_excursion_pct"] = out["down_excursion"] / out["open"] * 100
    out["previous_close"] = out["close"].shift(1)
    out["gap_pct"] = (out["open"] / out["previous_close"] - 1) * 100
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["previous_close"]).abs(),
            (out["low"] - out["previous_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr20"] = true_range.rolling(20, min_periods=10).mean()
    # Features for session t use only information available before session t.
    out["previous_range"] = out["daily_range"].shift(1)
    out["previous_atr20"] = out["atr20"].shift(1)
    out["previous_range_atr"] = out["previous_range"] / out["previous_atr20"]
    out["previous_5d_return_pct"] = (out["close"].shift(1) / out["close"].shift(6) - 1) * 100
    out["weekday"] = out["date"].dt.day_name()
    out["gap_bucket"] = out["gap_pct"].map(gap_bucket)
    return out


def gap_bucket(value) -> str:
    if pd.isna(value):
        return "Unknown"
    abs_gap = abs(float(value))
    if abs_gap < 0.25:
        return "Small (<0.25%)"
    if abs_gap < 0.75:
        return "Moderate (0.25–0.75%)"
    return "Large (≥0.75%)"


def range_regime(value, low_cut, high_cut) -> str:
    if pd.isna(value):
        return "Unknown"
    if value <= low_cut:
        return "Quiet"
    if value >= high_cut:
        return "Wide"
    return "Normal"


def trend_regime(value) -> str:
    if pd.isna(value):
        return "Unknown"
    if value > 0.5:
        return "Uptrend"
    if value < -0.5:
        return "Downtrend"
    return "Range-bound"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_yahoo() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download("QQQ", period="5y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no daily data.")
    raw = raw.reset_index()
    return normalize_daily(raw)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_intraday_yahoo() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download("QQQ", period="60d", interval="5m", auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no intraday data.")
    raw = raw.reset_index()
    return normalize_intraday(raw)


def normalize_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    out = _map_columns(raw, is_intraday=True)
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    out["timestamp"] = ts
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["session_date"] = out["timestamp"].dt.normalize()
    mins = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    # Regular session, 9:30 a.m. to 4:00 p.m. ET. Bars stamped 16:00 are excluded.
    out = out[(mins >= 9 * 60 + 30) & (mins < 16 * 60)]
    return out.reset_index(drop=True)


with st.sidebar:
    st.header("Market Inputs")
    spot_price = st.number_input("QQQ spot price ($)", min_value=0.01, value=709.80, step=0.10, format="%.2f")
    vxn_volatility = st.number_input("VXN implied volatility (%)", min_value=0.01, value=27.13, step=0.10, format="%.2f")
    trading_days = st.number_input("Trading days per year", min_value=1, max_value=366, value=252, step=1)
    forecast_days = st.number_input("Forecast horizon (trading days)", min_value=1, max_value=252, value=1, step=1)
    st.subheader("Options Expected Move (DTE Method)")
    options_iv = st.number_input(
        "Option implied volatility (%)", min_value=0.01, value=27.13, step=0.10, format="%.2f",
        help="Annualized IV for the option/expiration being analyzed. Enter as a percentage, e.g. 25 for 25%.",
    )
    dte = st.number_input(
        "Days to expiration (DTE)", min_value=0.0, max_value=3650.0, value=1.0, step=1.0,
        help="Calendar days remaining from the chosen data/as-of date to expiration. For a Friday estimate for Monday expiration, use 3 calendar days.",
    )
    selected_levels = st.multiselect(
        "Confidence intervals", options=["50%", "68%", "90%", "95%", "99%"], default=["50%", "68%", "90%", "95%"]
    )
    st.subheader("Gamma Levels")
    gamma_levels_text = st.text_area(
        "Enter gamma price levels ($)", value="", placeholder="Example: 700, 705.5, 710, 715, 720",
        help="Enter levels separated by commas, spaces, or new lines. The app does not fetch gamma data automatically.",
    )

if not selected_levels:
    st.info("Select at least one confidence interval in the sidebar to display results.")
    st.stop()

# Existing volatility-based calculations.
period_volatility = (vxn_volatility / 100) * np.sqrt(forecast_days / trading_days)
one_sigma_move = spot_price * period_volatility
daily_move_pct = period_volatility * 100
rows = []
for label in selected_levels:
    z = Z_SCORES[label]
    move = spot_price * period_volatility * z
    rows.append({
        "Confidence interval": label,
        "Z-score": z,
        "Expected move ($)": move,
        "Expected move (%)": move / spot_price * 100,
        "Lower bound ($)": spot_price - move,
        "Upper bound ($)": spot_price + move,
    })
options_expected_move = spot_price * (options_iv / 100) * np.sqrt(dte / 365)
rows.append({
    "Confidence interval": f"Options EM (DTE {dte:g})",
    "Z-score": 1.0,
    "Expected move ($)": options_expected_move,
    "Expected move (%)": options_expected_move / spot_price * 100,
    "Lower bound ($)": spot_price - options_expected_move,
    "Upper bound ($)": spot_price + options_expected_move,
})
df = pd.DataFrame(rows)

# Parse the manually entered gamma levels.
gamma_levels = []
invalid_gamma_entries = []
for token in gamma_levels_text.replace("\n", ",").replace(";", ",").replace(" ", ",").split(","):
    token = token.strip()
    if not token:
        continue
    try:
        level = float(token.replace("$", ""))
        if level <= 0 or not np.isfinite(level):
            raise ValueError
        gamma_levels.append(level)
    except ValueError:
        invalid_gamma_entries.append(token)
gamma_levels = sorted(set(gamma_levels))
if invalid_gamma_entries:
    st.sidebar.warning("Ignored invalid gamma level(s): " + ", ".join(invalid_gamma_entries))

m1, m2, m3, m4 = st.columns(4)
m1.metric("QQQ Spot Price", f"${spot_price:,.2f}")
m2.metric(f"VXN {forecast_days}-day volatility (approx.)", f"{daily_move_pct:.2f}%")
m3.metric(f"VXN 1σ Move ({forecast_days} day{'s' if forecast_days != 1 else ''})", f"±${one_sigma_move:,.2f}")
m4.metric("Options Expected Move (DTE method)", f"±${options_expected_move:,.2f}")

st.subheader("Volatility-Based Price Levels")
display_df = df.copy()
for col in ["Z-score", "Expected move ($)", "Expected move (%)", "Lower bound ($)", "Upper bound ($)"]:
    if col == "Z-score":
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")
    elif col == "Expected move (%)":
        display_df[col] = display_df[col].map(lambda x: f"±{x:.2f}%")
    elif col == "Expected move ($)":
        display_df[col] = display_df[col].map(lambda x: f"±${x:,.2f}")
    else:
        display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.subheader("Price Range Visualization")
fig, ax = plt.subplots(figsize=(10, max(3.2, 0.8 * len(df))))
for y, row in enumerate(rows):
    ax.plot([row["Lower bound ($)"], row["Upper bound ($)"]], [y, y], linewidth=7, solid_capstyle="round")
    ax.scatter([row["Lower bound ($)"], row["Upper bound ($)"]], [y, y], s=45)
ax.axvline(spot_price, linestyle="--", linewidth=1.5, label=f"Spot ${spot_price:,.2f}")
for i, level in enumerate(gamma_levels):
    ax.axvline(level, linestyle=":", linewidth=1.4, alpha=0.85, label=f"Gamma ${level:,.2f}" if i < 8 else None)
if len(gamma_levels) > 8:
    ax.text(0.99, 0.02, f"{len(gamma_levels) - 8} additional gamma levels not labeled", transform=ax.transAxes, ha="right", fontsize=8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["Confidence interval"] for r in rows])
ax.set_xlabel("QQQ price ($)")
ax.set_ylabel("Volatility estimate")
ax.grid(axis="x", alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if gamma_levels:
    st.subheader("User-Entered Gamma Levels")
    gamma_df = pd.DataFrame({"Gamma level ($)": gamma_levels})
    gamma_df["Distance from spot ($)"] = gamma_df["Gamma level ($)"].map(lambda x: x - spot_price)
    gamma_df["Inside selected volatility range"] = gamma_df["Gamma level ($)"].map(
        lambda x: any(r["Lower bound ($)"] <= x <= r["Upper bound ($)"] for r in rows)
    )
    st.dataframe(gamma_df.style.format({"Gamma level ($)": "${:,.2f}", "Distance from spot ($)": "{:+,.2f}"}), use_container_width=True, hide_index=True)

st.divider()
st.header("Historical Daily Range Distribution")
st.markdown(
    "This section estimates the empirical distribution of QQQ's regular-session high and low relative to its opening price. "
    "It calculates **daily range = high − low**, **upside excursion = high − open**, and **downside excursion = open − low**. "
    "The percentile table is calculated from actual rows in the loaded data; the example percentiles are not hard-coded."
)

with st.expander("Load 3–5 years of daily OHLC data", expanded=True):
    st.write("Choose one source. Yahoo Finance daily data is optional and requires the `yfinance` package plus an internet connection. You can upload your own CSV instead.")
    daily_upload = st.file_uploader("Upload daily OHLC CSV (Date, Open, High, Low, Close)", type=["csv"], key="daily_ohlc_upload")
    fetch_daily = st.checkbox("Fetch about 5 years of daily QQQ data with yfinance", value=False, key="fetch_daily_yahoo")
    st.caption("Daily OHLC data alone supports full-session range distributions. It does not provide historical pre-market IV, true overnight highs/lows, or macro-event labels unless you include those as extra CSV columns.")

daily_history = None
daily_load_error = None
if daily_upload is not None:
    try:
        daily_history = normalize_daily(pd.read_csv(daily_upload))
    except Exception as exc:
        daily_load_error = f"Could not read uploaded daily CSV: {exc}"
elif fetch_daily:
    try:
        with st.spinner("Loading daily QQQ history…"):
            daily_history = fetch_daily_yahoo()
    except Exception as exc:
        daily_load_error = f"Could not fetch Yahoo daily data: {exc}. Upload a CSV or install yfinance."

if daily_load_error:
    st.error(daily_load_error)

if daily_history is None or daily_history.empty:
    st.info("Load daily OHLC history above to calculate real historical percentiles. No historical QQQ distribution is fabricated or assumed.")
else:
    hist = add_daily_features(daily_history)
    if len(hist) < 60:
        st.warning(f"Only {len(hist)} sessions loaded. Prefer at least 3–5 years for more stable estimates.")
    hist_start = hist["date"].min().date()
    hist_end = hist["date"].max().date()
    st.caption(f"Loaded {len(hist):,} sessions, from {hist_start:%b %d, %Y} through {hist_end:%b %d, %Y}.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        forecast_date = st.date_input("Forecast session date", value=next_weekday(date.today()), key="range_forecast_date")
    with col_b:
        projection_open = st.number_input("Projected / actual regular-session open ($)", min_value=0.01, value=float(spot_price), step=0.10, format="%.2f", key="projection_open")
    with col_c:
        sample_start = st.date_input("Start date for historical sample", value=max(hist_start, hist_end - timedelta(days=5 * 365)), min_value=hist_start, max_value=hist_end, key="range_sample_start")

    prev_rows = hist[hist["date"].dt.date < forecast_date]
    if prev_rows.empty:
        previous_close = float(hist.iloc[0]["close"])
        current_prev_range_ratio = np.nan
        current_prev_trend = np.nan
    else:
        last_prev = prev_rows.iloc[-1]
        previous_close = float(last_prev["close"])
        current_prev_range_ratio = float(last_prev["daily_range"] / last_prev["atr20"]) if pd.notna(last_prev["atr20"]) and last_prev["atr20"] != 0 else np.nan
        if len(prev_rows) >= 6:
            current_prev_trend = (float(prev_rows.iloc[-1]["close"]) / float(prev_rows.iloc[-6]["close"]) - 1) * 100
        else:
            current_prev_trend = np.nan
    current_gap_pct = (projection_open / previous_close - 1) * 100 if previous_close > 0 else np.nan
    low_range_cut, high_range_cut = hist["previous_range_atr"].dropna().quantile([1 / 3, 2 / 3]) if hist["previous_range_atr"].notna().sum() >= 10 else (np.nan, np.nan)
    current_range_regime = range_regime(current_prev_range_ratio, low_range_cut, high_range_cut)
    current_trend_regime = trend_regime(current_prev_trend)

    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Assumed opening gap", f"{current_gap_pct:+.2f}%" if pd.notna(current_gap_pct) else "N/A")
    info2.metric("Previous session range / ATR", f"{current_prev_range_ratio:.2f}×" if pd.notna(current_prev_range_ratio) else "N/A")
    info3.metric("Previous 5-session trend", f"{current_prev_trend:+.2f}%" if pd.notna(current_prev_trend) else "N/A")
    info4.metric("Forecast weekday", forecast_date.strftime("%A"))
    st.caption(f"The assumed gap is calculated from your projected opening price (${projection_open:,.2f}) relative to the latest close before {forecast_date:%b %d}. Change the projected open to test gap scenarios.")

    train = hist[(hist["date"].dt.date < forecast_date) & (hist["date"].dt.date >= sample_start)].copy()
    train = train.dropna(subset=["up_excursion_pct", "down_excursion_pct"])

    with st.expander("Condition the distribution on market regime", expanded=True):
        st.caption("These filters use only factors that can be derived from daily OHLC: weekday, opening gap, previous session range relative to ATR, and previous five-session trend. Extra user-CSV fields can add overnight/IV/macro/expiration conditions below.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            weekday_mode = st.selectbox("Day of week", ["All weekdays", "Match forecast weekday"], index=1)
        with c2:
            gap_mode = st.selectbox("Opening-gap regime", ["All gaps", "Match gap size", "Match direction + size", "Up gaps only", "Down gaps only"], index=1)
        with c3:
            range_mode = st.selectbox("Previous range / ATR", ["All regimes", "Match previous session", "Quiet", "Normal", "Wide"], index=1)
        with c4:
            trend_mode = st.selectbox("Previous 5-session trend", ["All trends", "Match previous trend", "Uptrend", "Range-bound", "Downtrend"], index=0)

        numeric_optional = [
            ("overnight_return_pct", "Overnight return (%)", current_gap_pct if pd.notna(current_gap_pct) else 0.0),
            ("overnight_range_pct", "Overnight range (% of prior close)", 0.0),
            ("premarket_iv_pct", "Pre-market IV (%)", float(options_iv)),
        ]
        optional_numeric_settings = {}
        optional_categorical_settings = {}
        available_optional = [item for item in numeric_optional if item[0] in train.columns]
        cat_labels = {"macro_event": "Scheduled macro event", "expiration_day": "Expiration day", "expiration_week": "Expiration week"}
        available_categorical = [(col, label) for col, label in cat_labels.items() if col in train.columns]
        if available_optional or available_categorical:
            st.markdown("**Optional enriched-data filters**")
            st.caption("These appear only if your uploaded CSV contains matching columns. Numeric features are matched by historical tercile; categorical features are matched to the selected label.")
            for col, label, default_value in available_optional:
                enabled = st.checkbox(f"Condition on {label}", value=False, key=f"cond_{col}")
                current_value = st.number_input(f"Current {label}", value=float(default_value), step=0.1, format="%.3f", key=f"value_{col}")
                optional_numeric_settings[col] = (enabled, current_value)
            for col, label in available_categorical:
                values = sorted(train[col].dropna().astype(str).unique().tolist())
                choice = st.selectbox(f"{label} filter", ["Do not filter"] + values, key=f"cat_{col}")
                optional_categorical_settings[col] = choice
        else:
            st.caption("For filters on overnight range, pre-market IV, macro releases, or expiration calendar, upload a CSV with extra columns named OvernightReturnPct, OvernightRangePct, PremarketIVPct, MacroEvent, ExpirationDay, or ExpirationWeek. These fields are optional.")

    conditioned = train.copy()
    if weekday_mode == "Match forecast weekday":
        conditioned = conditioned[conditioned["weekday"] == forecast_date.strftime("%A")]
    if gap_mode != "All gaps":
        target_gap_bucket = gap_bucket(current_gap_pct)
        if gap_mode == "Match gap size":
            conditioned = conditioned[conditioned["gap_pct"].map(gap_bucket) == target_gap_bucket]
        elif gap_mode == "Match direction + size":
            if current_gap_pct > 0:
                conditioned = conditioned[(conditioned["gap_pct"] > 0) & (conditioned["gap_pct"].map(gap_bucket) == target_gap_bucket)]
            elif current_gap_pct < 0:
                conditioned = conditioned[(conditioned["gap_pct"] < 0) & (conditioned["gap_pct"].map(gap_bucket) == target_gap_bucket)]
            else:
                conditioned = conditioned[conditioned["gap_pct"].map(gap_bucket) == target_gap_bucket]
        elif gap_mode == "Up gaps only":
            conditioned = conditioned[conditioned["gap_pct"] > 0]
        elif gap_mode == "Down gaps only":
            conditioned = conditioned[conditioned["gap_pct"] < 0]
    if range_mode != "All regimes" and pd.notna(low_range_cut) and pd.notna(high_range_cut):
        conditioned["range_regime"] = conditioned["previous_range_atr"].map(lambda x: range_regime(x, low_range_cut, high_range_cut))
        if range_mode == "Match previous session":
            target_regime = current_range_regime
        else:
            target_regime = range_mode
        conditioned = conditioned[conditioned["range_regime"] == target_regime]
    if trend_mode != "All trends":
        conditioned["trend_regime"] = conditioned["previous_5d_return_pct"].map(trend_regime)
        target_trend = current_trend_regime if trend_mode == "Match previous trend" else trend_mode
        conditioned = conditioned[conditioned["trend_regime"] == target_trend]

    # Optional numeric enrichment: compare current value with matching historical tercile.
    for col, (enabled, current_value) in optional_numeric_settings.items():
        if enabled:
            valid_vals = train[col].dropna()
            if len(valid_vals) >= 12:
                q1, q2 = valid_vals.quantile([1 / 3, 2 / 3])
                label_current = "Low" if current_value <= q1 else ("High" if current_value >= q2 else "Middle")
                labels = train[col].map(lambda x: "Low" if pd.notna(x) and x <= q1 else ("High" if pd.notna(x) and x >= q2 else ("Middle" if pd.notna(x) else "Unknown")))
                conditioned = conditioned.loc[labels.reindex(conditioned.index).eq(label_current)]
            else:
                st.warning(f"Not enough valid {col} historical observations to condition on this feature; filter skipped.")
    for col, choice in optional_categorical_settings.items():
        if choice != "Do not filter":
            conditioned = conditioned[conditioned[col].astype(str) == choice]

    q_cols = []
    output_rows = []
    for p, q in zip(PERCENTILES, QUANTILES):
        up_pct = float(conditioned["up_excursion_pct"].quantile(q)) if not conditioned.empty else np.nan
        down_pct = float(conditioned["down_excursion_pct"].quantile(q)) if not conditioned.empty else np.nan
        up_dollars = projection_open * up_pct / 100 if pd.notna(up_pct) else np.nan
        down_dollars = projection_open * down_pct / 100 if pd.notna(down_pct) else np.nan
        output_rows.append({
            "Historical percentile": f"{p}th",
            "Upside excursion (%)": up_pct,
            "Upside excursion ($)": up_dollars,
            "Projected high ($)": projection_open + up_dollars if pd.notna(up_dollars) else np.nan,
            "Downside excursion (%)": down_pct,
            "Downside excursion ($)": down_dollars,
            "Projected low ($)": projection_open - down_dollars if pd.notna(down_dollars) else np.nan,
        })
    percentile_df = pd.DataFrame(output_rows)

    n1, n2, n3 = st.columns(3)
    n1.metric("Historical sessions in conditioned sample", f"{len(conditioned):,}")
    n2.metric("Unconditioned sample", f"{len(train):,}")
    n3.metric("Forecast opening price", f"${projection_open:,.2f}")
    if len(conditioned) < 30:
        st.warning("The conditioned sample has fewer than 30 sessions. Treat its percentiles cautiously or relax one or more filters.")
    if conditioned.empty:
        st.error("No sessions match the selected conditions. Relax the filters or expand the historical sample.")
    else:
        pretty = percentile_df.copy()
        for col in ["Upside excursion (%)", "Downside excursion (%)"]:
            pretty[col] = pretty[col].map(lambda x: f"{x:.3f}%" if pd.notna(x) else "N/A")
        for col in ["Upside excursion ($)", "Projected high ($)", "Downside excursion ($)", "Projected low ($)"]:
            pretty[col] = pretty[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        st.dataframe(pretty, use_container_width=True, hide_index=True)
        st.caption("Percentiles show the empirical excursion at or below the specified percentile—not the probability price will reverse at that level. Upside and downside excursions are estimated separately and are not a joint confidence interval.")
        st.download_button(
            "Download historical range percentiles as CSV",
            data=percentile_df.to_csv(index=False).encode("utf-8"),
            file_name="qqq_historical_range_percentiles.csv",
            mime="text/csv",
            key="download_daily_percentiles",
        )

    with st.expander("Historical observations and definitions"):
        sample_view = train[["date", "open", "high", "low", "close", "daily_range", "up_excursion", "down_excursion", "range_pct", "up_excursion_pct", "down_excursion_pct", "gap_pct", "previous_range_atr", "previous_5d_return_pct"]].copy()
        sample_view = sample_view.rename(columns={
            "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "daily_range": "Daily range ($)", "up_excursion": "Upside excursion ($)", "down_excursion": "Downside excursion ($)",
            "range_pct": "Daily range (%)", "up_excursion_pct": "Upside excursion (%)", "down_excursion_pct": "Downside excursion (%)",
            "gap_pct": "Opening gap (%)", "previous_range_atr": "Previous range / ATR", "previous_5d_return_pct": "Previous 5-session return (%)",
        })
        st.dataframe(sample_view.tail(100), use_container_width=True, hide_index=True)

    # Forecast-at-time model based on uploaded or optionally fetched intraday bars.
    st.divider()
    st.subheader("Intraday Remaining-Range Model")
    st.markdown(
        "When intraday bars are available, this model measures how much of the session high/low remained after a forecast checkpoint. "
        "It estimates the distribution of **additional upside beyond the high already observed** and **additional downside below the low already observed**, plus the historical frequency of making a new high/low after that time."
    )
    with st.expander("Load intraday bars (optional)", expanded=False):
        intraday_upload = st.file_uploader("Upload intraday CSV (timestamp, Open, High, Low, Close)", type=["csv"], key="intraday_upload")
        fetch_intraday = st.checkbox("Fetch recent 5-minute bars with yfinance", value=False, key="fetch_intraday_yahoo")
        st.caption("Yahoo Finance generally provides only recent intraday history. Use your own multi-year 1-minute/5-minute exports for deeper time-of-day samples. Uploaded timestamps without a time zone are assumed to be New York time.")

    intraday_history = None
    intraday_error = None
    if intraday_upload is not None:
        try:
            intraday_history = normalize_intraday(pd.read_csv(intraday_upload))
        except Exception as exc:
            intraday_error = f"Could not read intraday CSV: {exc}"
    elif fetch_intraday:
        try:
            with st.spinner("Loading recent QQQ intraday bars…"):
                intraday_history = fetch_intraday_yahoo()
        except Exception as exc:
            intraday_error = f"Could not fetch intraday data: {exc}. Upload a CSV or install yfinance."
    if intraday_error:
        st.warning(intraday_error)

    if intraday_history is None or intraday_history.empty:
        st.info("Load intraday bars to activate the time-conditioned model. Daily OHLC still powers the full-session distribution above.")
    else:
        time_label = st.selectbox("Forecast checkpoint (New York time)", ["Before open (daily baseline)", "9:45 AM", "10:00 AM", "11:00 AM", "12:00 PM"], index=2, key="intraday_checkpoint")
        if time_label == "Before open (daily baseline)":
            st.info("Use the daily percentile table above for a pre-open forecast. A remaining-range calculation requires observed intraday bars.")
        else:
            cutoff_str = {"9:45 AM": "09:45", "10:00 AM": "10:00", "11:00 AM": "11:00", "12:00 PM": "12:00"}[time_label]
            cutoff_minutes = int(cutoff_str[:2]) * 60 + int(cutoff_str[3:])
            i_minutes = intraday_history["timestamp"].dt.hour * 60 + intraday_history["timestamp"].dt.minute
            # Bars labelled with their start time are observed only when timestamp < checkpoint.
            before_cutoff = intraday_history[i_minutes < cutoff_minutes].copy()
            hist_intraday_dates = set(before_cutoff["session_date"].unique())
            target_ts = pd.Timestamp(forecast_date)
            target_bars = before_cutoff[before_cutoff["session_date"] == target_ts]
            if target_bars.empty:
                st.warning(f"No intraday bars for {forecast_date:%b %d, %Y} before {time_label}. The app can show historical time-of-day statistics, but cannot calculate the current session's observed high/low until bars for that date are available.")
            # Join each historical intraday session with daily OHLC targets.
            daily_indexed = hist.set_index("date")
            session_rows = []
            for session_date, bars in before_cutoff.groupby("session_date"):
                if session_date not in daily_indexed.index:
                    continue
                day = daily_indexed.loc[session_date]
                if isinstance(day, pd.DataFrame):
                    day = day.iloc[-1]
                obs_hi = float(bars["high"].max())
                obs_lo = float(bars["low"].min())
                day_hi, day_lo, day_open = float(day["high"]), float(day["low"]), float(day["open"])
                session_rows.append({
                    "date": session_date,
                    "open": day_open,
                    "observed_high": obs_hi,
                    "observed_low": obs_lo,
                    "remaining_up_pct": max(day_hi - obs_hi, 0.0) / day_open * 100,
                    "remaining_down_pct": max(obs_lo - day_lo, 0.0) / day_open * 100,
                    "new_high": day_hi > obs_hi + 0.01,
                    "new_low": day_lo < obs_lo - 0.01,
                })
            time_hist = pd.DataFrame(session_rows)
            time_hist = time_hist[time_hist["date"].dt.date < forecast_date] if not time_hist.empty else time_hist
            if len(time_hist) < 10:
                st.warning(f"Only {len(time_hist)} complete historical sessions are available before {time_label}; use the results as a rough reference.")
            if not time_hist.empty:
                time_rows = []
                for p, q in zip(PERCENTILES, QUANTILES):
                    up_pct = float(time_hist["remaining_up_pct"].quantile(q))
                    down_pct = float(time_hist["remaining_down_pct"].quantile(q))
                    up_dol = projection_open * up_pct / 100
                    down_dol = projection_open * down_pct / 100
                    time_rows.append({
                        "Historical percentile": f"{p}th",
                        "Additional upside beyond observed high (%)": up_pct,
                        "Additional upside ($)": up_dol,
                        "Projected final high ($)": np.nan,
                        "Additional downside beyond observed low (%)": down_pct,
                        "Additional downside ($)": down_dol,
                        "Projected final low ($)": np.nan,
                    })
                time_table = pd.DataFrame(time_rows)
                if not target_bars.empty:
                    current_observed_high = float(target_bars["high"].max())
                    current_observed_low = float(target_bars["low"].min())
                    current_open = float(target_bars.iloc[0]["open"])
                    time_table["Projected final high ($)"] = current_observed_high + time_table["Additional upside ($)"]
                    time_table["Projected final low ($)"] = current_observed_low - time_table["Additional downside ($)"]
                    p_new_high = float(time_hist["new_high"].mean() * 100)
                    p_new_low = float(time_hist["new_low"].mean() * 100)
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Observed session open", f"${current_open:,.2f}")
                    a2.metric("High so far", f"${current_observed_high:,.2f}")
                    a3.metric("Low so far", f"${current_observed_low:,.2f}")
                    a4.metric("Historical chance of a new high", f"{p_new_high:.1f}%")
                    st.metric("Historical chance of a new low after checkpoint", f"{p_new_low:.1f}%")
                else:
                    st.caption("Projected final high/low are shown only when the selected forecast date has intraday bars through the checkpoint.")
                pretty_time = time_table.copy()
                for col in ["Additional upside beyond observed high (%)", "Additional downside beyond observed low (%)"]:
                    pretty_time[col] = pretty_time[col].map(lambda x: f"{x:.3f}%")
                for col in ["Additional upside ($)", "Additional downside ($)", "Projected final high ($)", "Projected final low ($)"]:
                    pretty_time[col] = pretty_time[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
                st.dataframe(pretty_time, use_container_width=True, hide_index=True)
                st.caption("New-high/new-low probabilities are historical frequencies for sessions represented in the intraday dataset, not calibrated probabilities for all market regimes. The remaining-move table conditions on the checkpoint time and the available sample only.")
                st.download_button(
                    "Download intraday remaining-range percentiles as CSV",
                    data=time_table.to_csv(index=False).encode("utf-8"),
                    file_name="qqq_intraday_remaining_range_percentiles.csv",
                    mime="text/csv",
                    key="download_intraday_percentiles",
                )

# Explanations and data requirements.
with st.expander("Method, assumptions, and recommended data fields"):
    st.markdown(
        r"""
        **Volatility estimates**
        - VXN-based move: `S × (VXN / 100) × √(forecast trading days / trading days per year) × z`.
        - Options DTE estimate: `S × (option IV / 100) × √(DTE / 365)`. This is a one-standard-deviation-style estimate and does not apply a Z-score.
        - Bounds equal the selected reference price plus/minus the estimated move. VXN is not identical to the IV of a specific QQQ option.

        **Historical full-session distribution**
        - Daily range = `High − Low`.
        - Upside excursion = `High − regular-session Open`.
        - Downside excursion = `regular-session Open − Low`.
        - The app calculates empirical percentiles (50th, 70th, 80th, 90th, 95th) from loaded history. These are historical quantiles, not guarantees.
        - Baseline conditioning uses weekday, opening gap (daily Open versus prior Close), prior session range relative to 20-session ATR, and prior five-session return.

        **Intraday remaining-range distribution**
        - For each historical session and selected checkpoint, it computes `max(High − observed high by checkpoint, 0)` and `max(observed low by checkpoint − Low, 0)`.
        - It also measures how often the final daily high/low occurred after the checkpoint. Intraday data must include consistent time zones and regular-session bars.

        **Optional enriched daily CSV columns**
        - Core required columns: `Date, Open, High, Low, Close`.
        - Optional numeric: `OvernightReturnPct, OvernightRangePct, PremarketIVPct`.
        - Optional categorical: `MacroEvent, ExpirationDay, ExpirationWeek`.
        - Numeric optional fields are matched by historical tercile; categorical fields can be filtered by label. Daily OHLC by itself cannot reconstruct pre-market ranges or implied volatility and does not identify macro events.

        For meaningful distributions, use 3–5 years of daily OHLC. Use multi-year intraday exports if you want stronger time-of-day estimates; a short intraday sample should be treated as exploratory. Recheck any result against the current tape, options positioning, and market context.
        """
    )
