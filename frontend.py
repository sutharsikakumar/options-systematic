import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model import build_anomaly_watchlist, load_options_data

st.set_page_config(page_title="Options Anomaly Dashboard", page_icon="📈", layout="wide")

st.title("Options Volume Anomaly Dashboard")
st.caption("Explore unusual option volume by symbol, snapshot date, and contract characteristics.")


@st.cache_data(show_spinner=True)
def get_data(path: str) -> pd.DataFrame:
    return load_options_data(path)


def fmt_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date
    if "expiration" in out.columns:
        out["expiration"] = pd.to_datetime(out["expiration"]).dt.date
    return out


parquet_path = st.sidebar.text_input("Parquet path", "spy_options.parquet")

try:
    raw = get_data(parquet_path)
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

all_symbols = sorted(raw["symbol"].dropna().unique().tolist())
all_dates = sorted(raw["date"].dropna().unique().tolist())

if not all_symbols or not all_dates:
    st.error("Dataset has no valid symbols or dates.")
    st.stop()

symbol = st.sidebar.selectbox("Symbol", all_symbols, index=all_symbols.index("SPY") if "SPY" in all_symbols else 0)
snap_date = st.sidebar.selectbox("Snapshot date", all_dates, index=len(all_dates) - 1)

symbol_slice = raw[(raw["symbol"] == symbol) & (raw["date"] == snap_date)].copy()
spot_default = float(symbol_slice["strike"].median()) if not symbol_slice.empty else 470.25

spot = st.sidebar.number_input("Spot price", min_value=1.0, value=round(spot_default, 2), step=0.25)
top_n = st.sidebar.slider("Top N contracts", min_value=5, max_value=50, value=15, step=1)

spikes, duds, scored = build_anomaly_watchlist(
    df=raw,
    symbol=symbol,
    snap_date=snap_date,
    spot=spot,
    top_n=top_n,
    return_scored_data=True,
)

if scored.empty:
    st.warning("No contracts after filters/model prep. Try another symbol/date.")
    st.stop()

scored = scored.copy()
scored["days_to_expiry"] = (pd.to_datetime(scored["expiration"]) - pd.to_datetime(scored["date"])).dt.days
scored["expected_volume"] = np.expm1(scored["logvol_pred"]).clip(lower=0)
scored["actual_minus_expected"] = scored["volume"] - scored["expected_volume"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Contracts scored", f"{len(scored):,}")
m2.metric("Average anomaly score", f"{scored['anomaly_score'].mean():.3f}")
m3.metric("Highest anomaly score", f"{scored['anomaly_score'].max():.3f}")
m4.metric("Lowest anomaly score", f"{scored['anomaly_score'].min():.3f}")

left, right = st.columns(2)
with left:
    st.subheader("Top Unusually High Volume (Spikes)")
    st.dataframe(fmt_frame(spikes), use_container_width=True, hide_index=True)

with right:
    st.subheader("Top Unusually Low Volume (Duds)")
    st.dataframe(fmt_frame(duds), use_container_width=True, hide_index=True)

plot_df = scored.copy()
plot_df["option_label"] = plot_df["type"].str.upper()

fig1 = px.scatter(
    plot_df,
    x="strike",
    y="anomaly_score",
    color="option_label",
    size="volume",
    hover_data=["contract_id", "expiration", "volume", "spread", "resid_z", "implied_volatility"],
    title="Anomaly Score by Strike",
)
fig1.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(
    plot_df,
    x="days_to_expiry",
    y="volume",
    color="anomaly_score",
    color_continuous_scale="RdBu",
    hover_data=["contract_id", "strike", "type", "spread", "logvol_pred"],
    title="Volume vs Days to Expiration",
)
st.plotly_chart(fig2, use_container_width=True)

surface = (
    plot_df.groupby(["days_to_expiry", "type"], as_index=False)["anomaly_score"]
    .mean()
    .rename(columns={"anomaly_score": "avg_anomaly"})
)

fig3 = px.line(
    surface,
    x="days_to_expiry",
    y="avg_anomaly",
    color="type",
    title="Average Anomaly by Tenor",
    markers=True,
)
fig3.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig3, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        """
- Positive `anomaly_score` means observed volume is higher than expected for similar contracts.
- Negative `anomaly_score` means observed volume is lower than expected.
- The model adjusts for moneyness, tenor, strike, option type, and bid/ask spread.
- Wider spreads are penalized, so illiquid contracts score lower than equally unusual liquid contracts.
        """
    )
