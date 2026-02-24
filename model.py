import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor


def build_anomaly_watchlist(df: pd.DataFrame, symbol: str, snap_date, spot: float, top_n: int = 10):
    d = df.copy()

    # --- types ---
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["expiration"] = pd.to_datetime(d["expiration"])

    snap_date = pd.to_datetime(snap_date).normalize()

    # --- filter one snapshot + symbol ---
    d = d[(d["symbol"] == symbol) & (d["date"] == snap_date)].copy()

    # numeric cleanup
    for col in ["bid", "ask", "mark", "implied_volatility", "strike"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["strike", "implied_volatility", "expiration", "type"])

    # --- engineered features ---
    d["t"] = (d["expiration"] - d["date"]).dt.days / 365.0
    d = d[(d["t"] > 0) & (d["t"] < 5)]

    d["spot"] = float(spot)
    d["log_moneyness"] = np.log(d["strike"] / d["spot"])

    d["spread"] = d["ask"] - d["bid"]
    d["mid"] = (d["ask"] + d["bid"]) / 2.0

    d = d[(d["bid"] >= 0) & (d["ask"] >= d["bid"]) & (d["implied_volatility"] > 0)]

    # --- target ---
    y = d["implied_volatility"]

    # --- features ---
    num_cols = ["log_moneyness", "t", "strike", "spot"]
    cat_cols = ["type"]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(d[num_cols + cat_cols], y)

    d["iv_pred"] = pipe.predict(d[num_cols + cat_cols])
    d["iv_resid"] = d["implied_volatility"] - d["iv_pred"]

    # normalize residuals within expiration
    d["resid_z"] = d.groupby("expiration")["iv_resid"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )

    # liquidity penalty
    d["liq_penalty"] = np.log1p(d["spread"].clip(lower=0))
    d["anomaly_score"] = d["resid_z"].abs() - 0.25 * d["liq_penalty"]

    # results
    cheap = d.sort_values(["resid_z", "spread"], ascending=[True, True]).head(top_n)
    rich  = d.sort_values(["resid_z", "spread"], ascending=[False, True]).head(top_n)

    keep = [
        "contract_id", "symbol", "date", "expiration", "strike", "type",
        "bid", "ask", "mark", "spread",
        "implied_volatility", "iv_pred", "iv_resid", "resid_z",
        "delta", "gamma", "theta", "vega", "rho",
        "in_the_money"
    ]

    return cheap[keep], rich[keep]


# =========================
# RUN THE MODEL
# =========================

# Load data
df = pd.read_parquet("options.parquet", engine="fastparquet")

# Normalize date column
df["date"] = pd.to_datetime(df["date"]).dt.normalize()

# Automatically use most recent snapshot
snap_date = df["date"].max()
print("Using snapshot date:", snap_date.date())

# Manually set spot (replace with real NVDA close if you have it)
spot = 140.0  # <-- NVDA price, adjust as needed

cheap, rich = build_anomaly_watchlist(
    df=df,
    symbol="NVDA",
    snap_date=snap_date,
    spot=spot,
    top_n=10
)

print("\n=== CHEAP VOL (IV unusually LOW vs surface) ===")
print(cheap.to_string(index=False))

print("\n=== RICH VOL (IV unusually HIGH vs surface) ===")
print(rich.to_string(index=False))
