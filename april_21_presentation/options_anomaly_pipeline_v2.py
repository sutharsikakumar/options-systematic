"""
Options Anomaly Detection Pipeline  —  v2
==========================================
Stages:
  0  - Load & derive features  (yfinance live OR your own CSV/Parquet)
  1  - Data quality flags       (spread anomalies, inverted quotes, zero mid)
  2  - Rolling z-score engine   (vol/OI, IV, spread coincidence flags)
  3  - IV surface residuals     (OLS per snapshot date)
  4  - Unified rule-based score (0–3)
  5  - Summary dashboard        (6-panel PNG)
  ── NEW IN v2 ──────────────────────────────────────────────────────────
  6  - Put-call parity violations  (true arb signal; requires matched pairs)
  7  - BSM Greeks + greeks flags   (delta bounds, vega spike, gamma ATM)
  8  - Isolation Forest ML score   (unsupervised anomaly; features 1–7)
  9  - Extended dashboard + alert export  (JSON watchlist + enriched CSV)

Usage (zero data needed):
  pip install yfinance pandas numpy matplotlib seaborn scikit-learn scipy pyarrow
  python options_anomaly_pipeline_v2.py

To use your own data:
  Set DATA_SOURCE = "file" and DATA_PATH = "your_file.csv" / ".parquet"
  Fill in COLUMN_MAP with your column names.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
DATA_SOURCE = "yfinance"   # "yfinance" | "file"
TICKER      = "SPY"
DATA_PATH   = "options.csv"

COLUMN_MAP = {
    # "QUOTE_DATE":       "quote_date",
    # "EXPIRE_DATE":      "expiration",
    # "STRIKE":           "strike",
    # "C_BID":            "bid",
    # "C_ASK":            "ask",
    # "C_VOLUME":         "volume",
    # "C_OI":             "open_interest",
    # "C_IV":             "implied_volatility",
    # "UNDERLYING_LAST":  "underlying_price",
}

RISK_FREE_RATE  = 0.052   # annualised (update as needed)
DIV_YIELD       = 0.013   # SPY approximate; set 0 for non-dividend stocks

Z_THRESHOLD     = 2.5
Z_WINDOW        = 20
MIN_CONTRACTS   = 15
PCP_THRESHOLD   = 0.02    # put-call parity deviation as fraction of mid price
IF_CONTAMINATION= 0.05    # expected anomaly fraction for Isolation Forest

STYLE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "accent1": "#58a6ff",
    "accent2": "#f78166",
    "accent3": "#3fb950",
    "accent4": "#d2a8ff",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "grid":    "#21262d",
}


# ═══════════════════════════════════════════════════════════
# STAGE 0 — Load & Derive
# ═══════════════════════════════════════════════════════════

def load_yfinance(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    print(f"[Stage 0] Fetching {ticker} options from Yahoo Finance …")
    stk  = yf.Ticker(ticker)
    exps = stk.options
    print(f"          Found {len(exps)} expiration dates")

    rows  = []
    price = stk.history(period="1d")["Close"].iloc[-1]
    today = pd.Timestamp.today().normalize()

    for exp in exps:
        try:
            chain = stk.option_chain(exp)
            for opt_type, df_ in [("C", chain.calls), ("P", chain.puts)]:
                df_ = df_.copy()
                df_["option_type"]      = opt_type
                df_["expiration"]       = pd.Timestamp(exp)
                df_["quote_date"]       = today
                df_["underlying_price"] = price
                rows.append(df_)
        except Exception as e:
            print(f"          Skip {exp}: {e}")

    raw = pd.concat(rows, ignore_index=True)
    raw = raw.rename(columns={
        "strike":            "strike",
        "bid":               "bid",
        "ask":               "ask",
        "volume":            "volume",
        "openInterest":      "open_interest",
        "impliedVolatility": "implied_volatility",
    })
    print(f"          Loaded {len(raw):,} option contracts")
    return raw


def load_file(path: str, col_map: dict) -> pd.DataFrame:
    print(f"[Stage 0] Loading {path} …")
    df = pd.read_parquet(path) if path.endswith(".parquet") \
         else pd.read_csv(path, low_memory=False)
    if col_map:
        df = df.rename(columns=col_map)
    print(f"          Loaded {len(df):,} rows")
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[Stage 0] Deriving features …")
    for col in ["bid", "ask", "volume", "open_interest",
                "implied_volatility", "underlying_price", "strike"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    df["mid"]          = (df["bid"] + df["ask"]) / 2
    df["spread_pct"]   = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df["vol_oi_ratio"] = df["volume"] / (df["open_interest"].fillna(0) + 1)
    df["moneyness"]    = df["strike"] / df["underlying_price"]
    df["tte"]          = (df["expiration"] - df["quote_date"]).dt.days.clip(lower=1)
    df["tte_years"]    = df["tte"] / 365.0

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper().str[0]

    print(f"          Contracts: {len(df):,}  |  Expiries: {df['expiration'].nunique()}")
    return df


# ═══════════════════════════════════════════════════════════
# STAGE 1 — Data Quality Flags
# ═══════════════════════════════════════════════════════════

def stage1_quality(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 1] Data quality checks …")
    df["flag_inverted"]    = df["bid"] > df["ask"]
    df["flag_wide_spread"] = df["spread_pct"] > 0.5
    df["flag_zero_mid"]    = df["mid"] <= 0.01

    n = len(df)
    for name, col in [("Inverted", "flag_inverted"),
                      ("Wide spread", "flag_wide_spread"),
                      ("Zero mid",   "flag_zero_mid")]:
        cnt = df[col].sum()
        print(f"          {name:12s}: {cnt:>6,}  ({100*cnt/n:.1f}%)")

    df_clean = df[
        ~df["flag_inverted"] & ~df["flag_zero_mid"] &
        df["tte"].gt(0) &
        df["implied_volatility"].notna() &
        df["implied_volatility"].gt(0)
    ].copy()
    print(f"          Clean rows: {len(df_clean):,} / {n:,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=STYLE["bg"])
    fig.suptitle("Stage 1 · Data Quality Overview",
                 color=STYLE["text"], fontsize=14, y=1.02)

    ax = axes[0]
    ax.set_facecolor(STYLE["surface"])
    spread_clean = df_clean["spread_pct"].dropna().clip(0, 2)
    ax.hist(spread_clean, bins=60, color=STYLE["accent1"], alpha=0.8, edgecolor="none")
    ax.axvline(0.5, color=STYLE["accent2"], lw=1.5, linestyle="--",
               label="Wide threshold (0.5)")
    ax.set_xlabel("Spread %", color=STYLE["muted"])
    ax.set_ylabel("Count", color=STYLE["muted"])
    ax.set_title("Bid-Ask Spread Distribution", color=STYLE["text"])
    ax.tick_params(colors=STYLE["muted"])
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    labels = ["Inverted\nQuotes", "Wide\nSpread", "Zero\nMid"]
    counts = [df["flag_inverted"].sum(), df["flag_wide_spread"].sum(),
              df["flag_zero_mid"].sum()]
    colors = [STYLE["accent2"], STYLE["accent1"], STYLE["accent3"]]
    bars = ax2.bar(labels, counts, color=colors, alpha=0.85, edgecolor="none")
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{cnt:,}", ha="center", va="bottom",
                 color=STYLE["text"], fontsize=10)
    ax2.set_title("Flag Counts", color=STYLE["text"])
    ax2.tick_params(colors=STYLE["muted"])
    ax2.set_facecolor(STYLE["surface"])
    for sp in ax2.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    plt.tight_layout()
    plt.savefig("output/stage1_quality.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage1_quality.png")
    return df_clean


# ═══════════════════════════════════════════════════════════
# STAGE 2 — Rolling Z-Score Engine
# ═══════════════════════════════════════════════════════════

def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    m = series.rolling(window, min_periods=3).mean()
    s = series.rolling(window, min_periods=3).std()
    return (series - m) / (s + 1e-9)


def stage2_zscore(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 2] Rolling z-score engine …")
    n_dates = df["quote_date"].nunique()
    if n_dates < 5:
        print(f"          Only {n_dates} date(s) — cross-sectional z-scores")
        for feat in ["vol_oi_ratio", "implied_volatility", "spread_pct"]:
            df[f"z_{feat}"] = df.groupby("quote_date")[feat].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9))
    else:
        grp = df.sort_values("quote_date").groupby(
            ["strike", "expiration", "option_type"])
        for feat in ["vol_oi_ratio", "implied_volatility", "spread_pct"]:
            df[f"z_{feat}"] = grp[feat].transform(
                lambda x: rolling_zscore(x, Z_WINDOW))

    for old, new in [("z_vol_oi_ratio", "z_vol_oi"),
                     ("z_implied_volatility", "z_iv"),
                     ("z_spread_pct", "z_spread")]:
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    df["n_zscore_signals"] = (
        (df["z_vol_oi"].abs() > Z_THRESHOLD).astype(int) +
        (df["z_iv"].abs()     > Z_THRESHOLD).astype(int) +
        (df["z_spread"].abs() > Z_THRESHOLD).astype(int)
    )
    df["flag_zscore_anomaly"] = df["n_zscore_signals"] >= 2

    n_anom = df["flag_zscore_anomaly"].sum()
    print(f"          Z-score anomalies: {n_anom:,}  ({100*n_anom/len(df):.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=STYLE["bg"])
    fig.suptitle("Stage 2 · Z-Score Distributions",
                 color=STYLE["text"], fontsize=14)
    for ax, (feat, label, color) in zip(axes, [
        ("z_vol_oi", "Vol/OI Z-Score",  STYLE["accent1"]),
        ("z_iv",     "IV Z-Score",       STYLE["accent3"]),
        ("z_spread", "Spread Z-Score",   STYLE["accent2"]),
    ]):
        ax.set_facecolor(STYLE["surface"])
        data = df[feat].dropna().clip(-6, 6)
        ax.hist(data, bins=80, color=color, alpha=0.75, edgecolor="none")
        ax.axvline( Z_THRESHOLD, color="white", lw=1, linestyle="--", alpha=0.6)
        ax.axvline(-Z_THRESHOLD, color="white", lw=1, linestyle="--", alpha=0.6)
        ax.set_title(label, color=STYLE["text"])
        ax.set_xlabel("Z-Score", color=STYLE["muted"])
        ax.tick_params(colors=STYLE["muted"])
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["grid"])

    plt.tight_layout()
    plt.savefig("output/stage2_zscore.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage2_zscore.png")
    return df


# ═══════════════════════════════════════════════════════════
# STAGE 3 — IV Surface Residuals (OLS)
# ═══════════════════════════════════════════════════════════

def stage3_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 3] IV surface residual model …")
    results = []

    for date in df["quote_date"].unique():
        grp   = df[df["quote_date"] == date].copy()
        valid = (grp["implied_volatility"].notna() &
                 np.isfinite(grp["implied_volatility"]) &
                 grp["moneyness"].notna() & grp["tte"].notna())
        grp_v = grp[valid]

        if len(grp_v) < MIN_CONTRACTS:
            grp["iv_residual"] = np.nan
            grp["iv_res_std"]  = np.nan
            results.append(grp)
            continue

        X = pd.DataFrame({
            "m":     grp_v["moneyness"],
            "m2":    grp_v["moneyness"] ** 2,
            "lT":    np.log(grp_v["tte"].clip(lower=1)),
            "call":  (grp_v["option_type"] == "C").astype(int),
        })
        y = grp_v["implied_volatility"]

        try:
            model = LinearRegression().fit(X, y)
            res   = y - model.predict(X)
            std   = float(res.std()) + 1e-9
        except Exception:
            grp["iv_residual"] = np.nan
            grp["iv_res_std"]  = np.nan
            results.append(grp)
            continue

        grp.loc[grp_v.index, "iv_residual"] = res.values
        grp["iv_res_std"] = std
        results.append(grp)

    df_out = pd.concat(results, ignore_index=True)
    df_out["flag_surface_anomaly"] = (
        df_out["iv_residual"].abs() > 2 * df_out["iv_res_std"]
    ).fillna(False)

    n_sa = df_out["flag_surface_anomaly"].sum()
    print(f"          Surface anomalies: {n_sa:,}  ({100*n_sa/len(df_out):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=STYLE["bg"])
    fig.suptitle("Stage 3 · IV Surface Residuals", color=STYLE["text"], fontsize=14)

    ax = axes[0]
    ax.set_facecolor(STYLE["surface"])
    plot_df = df_out.dropna(subset=["iv_residual", "moneyness", "implied_volatility"])
    sc = ax.scatter(plot_df["moneyness"], plot_df["implied_volatility"],
                    c=plot_df["iv_residual"], cmap="RdYlGn_r",
                    s=6, alpha=0.6, vmin=-0.3, vmax=0.3)
    anom = plot_df[df_out.loc[plot_df.index, "flag_surface_anomaly"]]
    ax.scatter(anom["moneyness"], anom["implied_volatility"],
               color=STYLE["accent2"], s=20, alpha=0.9,
               label=f"Anomalies ({len(anom):,})", zorder=5)
    plt.colorbar(sc, ax=ax, label="IV Residual").ax.yaxis.label.set_color(STYLE["muted"])
    ax.set_xlabel("Moneyness (K/S)", color=STYLE["muted"])
    ax.set_ylabel("Implied Volatility", color=STYLE["muted"])
    ax.set_title("IV Surface — Residual Colour Map", color=STYLE["text"])
    ax.tick_params(colors=STYLE["muted"])
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"],
              markerscale=2, fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    res_vals = df_out["iv_residual"].dropna().clip(-0.5, 0.5)
    ax2.hist(res_vals, bins=80, color=STYLE["accent1"], alpha=0.75, edgecolor="none")
    std_val = df_out["iv_res_std"].dropna().mean()
    for sign in [-2, 2]:
        ax2.axvline(sign*std_val, color=STYLE["accent2"], lw=1.5, linestyle="--",
                    label=f"±2σ ({abs(sign*std_val):.3f})" if sign > 0 else None)
    ax2.set_xlabel("IV Residual", color=STYLE["muted"])
    ax2.set_ylabel("Count", color=STYLE["muted"])
    ax2.set_title("Residual Distribution", color=STYLE["text"])
    ax2.tick_params(colors=STYLE["muted"])
    ax2.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"])
    for sp in ax2.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    plt.tight_layout()
    plt.savefig("output/stage3_surface.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage3_surface.png")
    return df_out


# ═══════════════════════════════════════════════════════════
# STAGE 4 — Unified Rule-Based Score
# ═══════════════════════════════════════════════════════════

def stage4_score(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 4] Unified rule-based anomaly score …")
    df["anomaly_score"] = (
        df.get("flag_wide_spread",     False).astype(int) +
        df.get("flag_zscore_anomaly",  False).astype(int) +
        df.get("flag_surface_anomaly", False).astype(int)
    )
    df["is_anomaly"] = df["anomaly_score"] >= 1

    print(f"          Total (rule-based): {df['is_anomaly'].sum():,} / {len(df):,}  "
          f"({100*df['is_anomaly'].mean():.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=STYLE["bg"])
    fig.suptitle("Stage 4 · Unified Rule-Based Scores",
                 color=STYLE["text"], fontsize=14)

    ax = axes[0]
    ax.set_facecolor(STYLE["surface"])
    score_counts = df["anomaly_score"].value_counts().sort_index()
    colors_map   = {0: STYLE["muted"], 1: STYLE["accent1"],
                    2: STYLE["accent3"], 3: STYLE["accent2"]}
    ax.bar(score_counts.index.astype(str), score_counts.values,
           color=[colors_map.get(i, STYLE["accent2"]) for i in score_counts.index],
           edgecolor="none", alpha=0.85)
    ax.set_xlabel("Anomaly Score (# flags)", color=STYLE["muted"])
    ax.set_ylabel("Contract Count", color=STYLE["muted"])
    ax.set_title("Score Distribution", color=STYLE["text"])
    ax.tick_params(colors=STYLE["muted"])
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    normal = df[~df["is_anomaly"]].sample(min(500, (~df["is_anomaly"]).sum()),
                                           random_state=42)
    anoms  = df[df["is_anomaly"]]
    ax2.scatter(normal["moneyness"], normal["implied_volatility"],
                color=STYLE["muted"], s=4, alpha=0.3, label="Normal")
    sc2 = ax2.scatter(anoms["moneyness"], anoms["implied_volatility"],
                      c=anoms["anomaly_score"], cmap="YlOrRd",
                      s=anoms["anomaly_score"]*30+10, alpha=0.85,
                      zorder=5, label="Anomaly")
    plt.colorbar(sc2, ax=ax2, label="Score").ax.yaxis.label.set_color(STYLE["muted"])
    ax2.set_xlabel("Moneyness", color=STYLE["muted"])
    ax2.set_ylabel("IV", color=STYLE["muted"])
    ax2.set_title("Anomalies on IV Surface", color=STYLE["text"])
    ax2.tick_params(colors=STYLE["muted"])
    ax2.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"],
               markerscale=2, fontsize=9)
    for sp in ax2.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    plt.tight_layout()
    plt.savefig("output/stage4_scores.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage4_scores.png")
    return df


# ═══════════════════════════════════════════════════════════
# STAGE 5 — Summary Dashboard (unchanged from v1)
# ═══════════════════════════════════════════════════════════

def stage5_dashboard(df: pd.DataFrame) -> None:
    print("\n[Stage 5] Rendering rule-based summary dashboard …")
    fig = plt.figure(figsize=(16, 10), facecolor=STYLE["bg"])
    fig.suptitle(
        f"Options Anomaly Dashboard (Stages 1–4)  ·  {TICKER}  ·  "
        f"{df['quote_date'].max().date()}",
        color=STYLE["text"], fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A – IV smile
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(STYLE["surface"])
    for ot, col, lbl in [("C", STYLE["accent3"], "Calls"),
                          ("P", STYLE["accent1"], "Puts")]:
        sub = df[df["option_type"] == ot]
        ax_a.scatter(sub["moneyness"], sub["implied_volatility"],
                     s=4, alpha=0.4, color=col, label=lbl)
    ax_a.set_title("IV Smile", color=STYLE["text"], fontsize=11)
    ax_a.set_xlabel("Moneyness", color=STYLE["muted"], fontsize=9)
    ax_a.set_ylabel("IV", color=STYLE["muted"], fontsize=9)
    ax_a.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"],
                fontsize=8, markerscale=3)
    ax_a.tick_params(colors=STYLE["muted"], labelsize=8)
    for sp in ax_a.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    # Panel B – IV vs DTE
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(STYLE["surface"])
    sc = ax_b.scatter(df["tte"], df["implied_volatility"],
                      c=df["moneyness"], cmap="plasma",
                      s=4, alpha=0.45, vmin=0.7, vmax=1.3)
    plt.colorbar(sc, ax=ax_b, label="Moneyness").ax.yaxis.label.set_color(STYLE["muted"])
    ax_b.set_title("IV vs DTE", color=STYLE["text"], fontsize=11)
    ax_b.set_xlabel("Days to Expiry", color=STYLE["muted"], fontsize=9)
    ax_b.set_ylabel("IV", color=STYLE["muted"], fontsize=9)
    ax_b.tick_params(colors=STYLE["muted"], labelsize=8)
    for sp in ax_b.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    # Panel C – Vol/OI
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor(STYLE["surface"])
    ax_c.hist(df["vol_oi_ratio"].clip(0, 5).dropna(), bins=60,
              color=STYLE["accent2"], alpha=0.75, edgecolor="none")
    ax_c.axvline(1.0, color="white", lw=1.2, linestyle="--", label="Vol = OI")
    ax_c.set_title("Volume / OI Ratio", color=STYLE["text"], fontsize=11)
    ax_c.set_xlabel("Vol/OI", color=STYLE["muted"], fontsize=9)
    ax_c.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=8)
    ax_c.tick_params(colors=STYLE["muted"], labelsize=8)
    for sp in ax_c.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    # Panel D – Flag breakdown
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.set_facecolor(STYLE["surface"])
    flag_cols = [f for f in ["flag_wide_spread", "flag_zscore_anomaly",
                              "flag_surface_anomaly"] if f in df.columns]
    counts = [df[f].sum() for f in flag_cols]
    labels = ["Wide\nSpread", "Z-Score\nAnomaly", "IV Surface\nAnomaly"]
    clrs   = [STYLE["accent1"], STYLE["accent3"], STYLE["accent2"]]
    bars   = ax_d.bar(labels[:len(counts)], counts,
                      color=clrs[:len(counts)], edgecolor="none", alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  f"{cnt:,}", ha="center", va="bottom",
                  color=STYLE["text"], fontsize=9)
    ax_d.set_title("Flag Counts by Type", color=STYLE["text"], fontsize=11)
    ax_d.tick_params(colors=STYLE["muted"], labelsize=8)
    for sp in ax_d.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    # Panel E – Z-score heatmap of top anomalies
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_facecolor(STYLE["surface"])
    sample = (df[df["is_anomaly"]]
              .sort_values("anomaly_score", ascending=False)
              .head(30))
    if len(sample) > 3 and all(c in sample.columns for c in ["z_vol_oi","z_iv","z_spread"]):
        z_data = sample[["z_vol_oi", "z_iv", "z_spread"]].fillna(0)
        z_data.columns = ["Vol/OI", "IV", "Spread"]
        im = ax_e.imshow(z_data.T.values, aspect="auto",
                         cmap="RdYlGn_r", vmin=-5, vmax=5)
        plt.colorbar(im, ax=ax_e, label="Z-Score").ax.yaxis.label.set_color(STYLE["muted"])
        ax_e.set_yticks([0, 1, 2])
        ax_e.set_yticklabels(["Vol/OI", "IV", "Spread"],
                              color=STYLE["muted"], fontsize=8)
        ax_e.set_xticks([])
        ax_e.set_title(f"Z-Scores — Top {len(sample)} Anomalies",
                       color=STYLE["text"], fontsize=11)
    else:
        ax_e.text(0.5, 0.5, "Not enough\nanomaly data",
                  ha="center", va="center", color=STYLE["muted"], fontsize=11,
                  transform=ax_e.transAxes)
        ax_e.set_title("Z-Score Heatmap", color=STYLE["text"], fontsize=11)

    # Panel F – Stats table
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor(STYLE["surface"])
    ax_f.axis("off")
    total  = len(df)
    n_anom = df["is_anomaly"].sum()
    rows = [
        ["Ticker",          TICKER],
        ["Quote Date",      str(df["quote_date"].max().date())],
        ["Total Contracts", f"{total:,}"],
        ["Expiries",        str(df["expiration"].nunique())],
        ["Anomalies",       f"{n_anom:,}  ({100*n_anom/total:.1f}%)"],
        ["Avg IV (calls)",  f"{df[df['option_type']=='C']['implied_volatility'].mean():.3f}"],
        ["Avg IV (puts)",   f"{df[df['option_type']=='P']['implied_volatility'].mean():.3f}"],
        ["Med Spread%",     f"{df['spread_pct'].median():.2%}"],
    ]
    tbl = ax_f.table(cellText=rows, colLabels=["Metric", "Value"],
                     cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(STYLE["surface"] if r % 2 == 0 else STYLE["bg"])
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["text"] if r > 0 else STYLE["accent1"])
    ax_f.set_title("Summary Statistics", color=STYLE["text"], fontsize=11)

    plt.savefig("output/stage5_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage5_dashboard.png")


# ═══════════════════════════════════════════════════════════
# STAGE 6 — Put-Call Parity Violations  ← NEW
# ═══════════════════════════════════════════════════════════

def bsm_discount(r: float, T: float, q: float = 0.0) -> tuple:
    """Return (discount_factor, forward_price_factor)."""
    return np.exp(-r * T), np.exp((r - q) * T)


def stage6_pcp_violations(df: pd.DataFrame,
                           r: float = RISK_FREE_RATE,
                           q: float = DIV_YIELD) -> pd.DataFrame:
    """
    For each (quote_date, expiration, strike) pair detect put-call parity
    violations.

    Put-Call Parity (European):
        C - P = S·e^{-(q)T} - K·e^{-rT}

    We compute:
        pcp_lhs  = call_mid - put_mid
        pcp_rhs  = S·e^{-qT} - K·e^{-rT}
        pcp_dev  = pcp_lhs - pcp_rhs          (should be ≈ 0)
        pcp_dev_pct = pcp_dev / mean_mid       (normalised deviation)

    A violation is flagged when |pcp_dev_pct| > PCP_THRESHOLD and
    |pcp_dev| > combined bid-ask half-spread (i.e., not explained by spread).
    """
    print("\n[Stage 6] Put-call parity violation detector …")

    df["flag_pcp_violation"] = False
    df["pcp_deviation"]      = np.nan
    df["pcp_dev_pct"]        = np.nan

    # Pivot to wide: one row per (date, expiration, strike)
    calls = df[df["option_type"] == "C"].copy()
    puts  = df[df["option_type"] == "P"].copy()

    # Merge on the triple key
    merge_cols = ["quote_date", "expiration", "strike",
                  "underlying_price", "tte_years"]
    pair = calls[merge_cols + ["mid", "bid", "ask"]].merge(
        puts [merge_cols + ["mid", "bid", "ask"]],
        on=merge_cols, suffixes=("_c", "_p")
    )

    if len(pair) == 0:
        print("          No matched pairs found — skipping PCP stage.")
        return df

    T  = pair["tte_years"].values
    S  = pair["underlying_price"].values
    K  = pair["strike"].values

    pcp_rhs  = S * np.exp(-q * T) - K * np.exp(-r * T)
    pcp_lhs  = pair["mid_c"].values - pair["mid_p"].values
    pcp_dev  = pcp_lhs - pcp_rhs
    mean_mid = (pair["mid_c"].values + pair["mid_p"].values) / 2 + 1e-6

    # Combined half-spread: min profit after paying both spreads
    half_spread_c = (pair["ask_c"].values - pair["bid_c"].values) / 2
    half_spread_p = (pair["ask_p"].values - pair["bid_p"].values) / 2
    min_arb_profit = np.abs(pcp_dev) - half_spread_c - half_spread_p

    pair["pcp_dev"]     = pcp_dev
    pair["pcp_dev_pct"] = pcp_dev / mean_mid
    pair["flag_pcp"]    = (
        (np.abs(pair["pcp_dev_pct"]) > PCP_THRESHOLD) &
        (min_arb_profit > 0)
    )

    n_viol = pair["flag_pcp"].sum()
    print(f"          Matched pairs:      {len(pair):,}")
    print(f"          PCP violations:     {n_viol:,}  "
          f"({100*n_viol/max(len(pair),1):.1f}%)")

    # Write violation flags back to df rows (both call and put legs)
    viol_keys = pair[pair["flag_pcp"]][["quote_date", "expiration", "strike"]]
    viol_set  = set(
        zip(viol_keys["quote_date"], viol_keys["expiration"], viol_keys["strike"])
    )
    df_key = list(zip(df["quote_date"], df["expiration"], df["strike"]))
    df["flag_pcp_violation"] = [k in viol_set for k in df_key]

    # Scalar deviation  (assign call's deviation; put gets same magnitude)
    dev_map = dict(zip(
        zip(pair["quote_date"], pair["expiration"], pair["strike"]),
        zip(pair["pcp_dev"], pair["pcp_dev_pct"])
    ))
    df["pcp_deviation"] = [dev_map.get(k, (np.nan, np.nan))[0] for k in df_key]
    df["pcp_dev_pct"]   = [dev_map.get(k, (np.nan, np.nan))[1] for k in df_key]

    # ── PCP visualisation ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=STYLE["bg"])
    fig.suptitle("Stage 6 · Put-Call Parity Violations",
                 color=STYLE["text"], fontsize=14)

    ax = axes[0]
    ax.set_facecolor(STYLE["surface"])
    dev_vals = pair["pcp_dev_pct"].clip(-0.3, 0.3)
    ax.hist(dev_vals, bins=80, color=STYLE["accent4"], alpha=0.75, edgecolor="none")
    ax.axvline( PCP_THRESHOLD, color=STYLE["accent2"], lw=1.5,
               linestyle="--", label=f"+{PCP_THRESHOLD:.0%}")
    ax.axvline(-PCP_THRESHOLD, color=STYLE["accent2"], lw=1.5,
               linestyle="--", label=f"-{PCP_THRESHOLD:.0%}")
    ax.set_xlabel("PCP Deviation (% of mid)", color=STYLE["muted"])
    ax.set_ylabel("Pair Count", color=STYLE["muted"])
    ax.set_title("PCP Deviation Distribution", color=STYLE["text"])
    ax.tick_params(colors=STYLE["muted"])
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    ax2 = axes[1]
    ax2.set_facecolor(STYLE["surface"])
    norm_pairs = pair[~pair["flag_pcp"]]
    viol_pairs = pair[ pair["flag_pcp"]]
    ax2.scatter(norm_pairs["strike"], norm_pairs["pcp_dev"],
                color=STYLE["muted"], s=5, alpha=0.3, label="No violation")
    ax2.scatter(viol_pairs["strike"], viol_pairs["pcp_dev"],
                color=STYLE["accent2"], s=25, alpha=0.85, zorder=5,
                label=f"Violation ({len(viol_pairs):,})")
    ax2.axhline(0, color="white", lw=0.8, linestyle="--")
    ax2.set_xlabel("Strike", color=STYLE["muted"])
    ax2.set_ylabel("PCP Deviation ($)", color=STYLE["muted"])
    ax2.set_title("Violations by Strike", color=STYLE["text"])
    ax2.tick_params(colors=STYLE["muted"])
    ax2.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"],
               markerscale=2, fontsize=9)
    for sp in ax2.spines.values():
        sp.set_edgecolor(STYLE["grid"])

    plt.tight_layout()
    plt.savefig("output/stage6_pcp.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage6_pcp.png")
    return df


# ═══════════════════════════════════════════════════════════
# STAGE 7 — BSM Greeks + Greeks Anomaly Flags  ← NEW
# ═══════════════════════════════════════════════════════════

def _bsm_row(S, K, T, r, q, sigma, opt_type):
    """
    Return (price, delta, gamma, vega, theta, rho) or NaN tuple on failure.
    vega  = per 1% move in vol  (i.e. raw_vega / 100)
    theta = per calendar day
    """
    nan6 = (np.nan,) * 6
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return nan6
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        nd1, nd2   = norm.cdf(d1),  norm.cdf(d2)
        nd1_, nd2_ = norm.cdf(-d1), norm.cdf(-d2)
        pdf_d1     = norm.pdf(d1)

        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q * T)

        if opt_type == "C":
            price = S * disc_q * nd1 - K * disc_r * nd2
            delta = disc_q * nd1
            rho   = K * T * disc_r * nd2
        else:
            price = K * disc_r * nd2_ - S * disc_q * nd1_
            delta = disc_q * (nd1 - 1)
            rho   = -K * T * disc_r * nd2_

        gamma = pdf_d1 * disc_q / (S * sigma * sqrt_T)
        vega  = S * disc_q * pdf_d1 * sqrt_T / 100          # per 1% IV
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            + (-1 if opt_type == "C" else 1) * r * K * disc_r * (nd2 if opt_type == "C" else nd2_)
            - q * S * disc_q * (nd1 if opt_type == "C" else nd1_)
        ) / 365

        return price, delta, gamma, vega, theta, rho
    except Exception:
        return nan6


def stage7_greeks(df: pd.DataFrame,
                  r: float = RISK_FREE_RATE,
                  q: float = DIV_YIELD) -> pd.DataFrame:
    """
    Compute BSM greeks for every clean row.  Flag contracts where:
      - delta_bound_flag : call delta outside (0,1) or put delta outside (-1,0)
      - vega_spike_flag  : vega > 2.5 std above median for that expiry bucket
      - gamma_atm_flag   : gamma spike — top 2% near-ATM (0.97–1.03)
    """
    print("\n[Stage 7] Computing BSM Greeks …")

    S     = df["underlying_price"].values
    K     = df["strike"].values
    T     = df["tte_years"].values
    sigma = df["implied_volatility"].values
    otype = df["option_type"].values

    prices, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []
    for i in range(len(df)):
        p, d, g, v, t, rh = _bsm_row(S[i], K[i], T[i], r, q, sigma[i], otype[i])
        prices.append(p); deltas.append(d); gammas.append(g)
        vegas.append(v); thetas.append(t); rhos.append(rh)

    df["bsm_price"] = prices
    df["delta"]     = deltas
    df["gamma"]     = gammas
    df["vega"]      = vegas
    df["theta"]     = thetas
    df["rho"]       = rhos

    # ── Flag: delta out of theoretical bounds ──
    call_mask = df["option_type"] == "C"
    put_mask  = df["option_type"] == "P"
    df["flag_delta_bound"] = (
        (call_mask & ((df["delta"] < 0) | (df["delta"] > 1))) |
        (put_mask  & ((df["delta"] > 0) | (df["delta"] < -1)))
    ).fillna(False)

    # ── Flag: vega spike (within same DTE bucket) ──
    df["dte_bucket"] = pd.cut(df["tte"], bins=[0, 7, 30, 90, 180, 365, 9999],
                              labels=["1w", "1m", "3m", "6m", "1y", "LEAP"])
    df["vega_z"] = df.groupby("dte_bucket")["vega"].transform(
        lambda x: (x - x.median()) / (x.std() + 1e-9))
    df["flag_vega_spike"] = df["vega_z"].abs() > Z_THRESHOLD

    # ── Flag: gamma spike near ATM ──
    atm_mask = df["moneyness"].between(0.97, 1.03)
    if atm_mask.sum() > 10:
        gamma_atm = df.loc[atm_mask, "gamma"]
        gamma_thresh = gamma_atm.quantile(0.98)
        df["flag_gamma_spike"] = atm_mask & (df["gamma"] > gamma_thresh)
    else:
        df["flag_gamma_spike"] = False

    n_db = df["flag_delta_bound"].sum()
    n_vs = df["flag_vega_spike"].sum()
    n_gs = df["flag_gamma_spike"].sum()
    print(f"          Delta bound violations: {n_db:,}")
    print(f"          Vega spike flags:        {n_vs:,}")
    print(f"          Gamma ATM spikes:        {n_gs:,}")

    df["flag_greeks_anomaly"] = (
        df["flag_delta_bound"] | df["flag_vega_spike"] | df["flag_gamma_spike"]
    )
    print(f"          Greeks anomaly (any):    {df['flag_greeks_anomaly'].sum():,}")

    # ── Greeks visualisation ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=STYLE["bg"])
    fig.suptitle("Stage 7 · BSM Greeks Analysis",
                 color=STYLE["text"], fontsize=14, y=1.01)

    def _style(ax, title, xlabel, ylabel="Count"):
        ax.set_facecolor(STYLE["surface"])
        ax.set_title(title, color=STYLE["text"], fontsize=10)
        ax.set_xlabel(xlabel, color=STYLE["muted"], fontsize=8)
        ax.set_ylabel(ylabel, color=STYLE["muted"], fontsize=8)
        ax.tick_params(colors=STYLE["muted"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["grid"])

    # A – Delta distribution split by type
    ax = axes[0, 0]
    for ot, col in [("C", STYLE["accent3"]), ("P", STYLE["accent1"])]:
        sub = df[df["option_type"] == ot]["delta"].dropna().clip(-1.05, 1.05)
        ax.hist(sub, bins=60, color=col, alpha=0.6, edgecolor="none",
                label=f"{'Calls' if ot=='C' else 'Puts'}")
    ax.axvline(0, color="white", lw=0.8, linestyle="--")
    ax.axvline(1, color=STYLE["accent2"], lw=0.8, linestyle="--")
    ax.axvline(-1, color=STYLE["accent2"], lw=0.8, linestyle="--")
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=7)
    _style(ax, "Delta Distribution", "Delta")

    # B – Gamma vs Moneyness (scatter)
    ax = axes[0, 1]
    ax.set_facecolor(STYLE["surface"])
    valid_g = df.dropna(subset=["gamma", "moneyness"])
    ax.scatter(valid_g["moneyness"], valid_g["gamma"].clip(0, 0.1),
               s=3, alpha=0.3, color=STYLE["accent1"])
    spikes = valid_g[valid_g["flag_gamma_spike"]]
    ax.scatter(spikes["moneyness"], spikes["gamma"].clip(0, 0.1),
               s=20, alpha=0.9, color=STYLE["accent2"], zorder=5,
               label=f"Gamma spike ({len(spikes):,})")
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=7)
    _style(ax, "Gamma vs Moneyness", "Moneyness", "Gamma")

    # C – Vega by DTE bucket
    ax = axes[0, 2]
    ax.set_facecolor(STYLE["surface"])
    valid_v = df.dropna(subset=["vega", "dte_bucket"])
    bucket_order = ["1w", "1m", "3m", "6m", "1y", "LEAP"]
    bucket_order = [b for b in bucket_order if b in valid_v["dte_bucket"].cat.categories]
    data_by_bucket = [valid_v[valid_v["dte_bucket"] == b]["vega"].clip(0, 50).dropna().values
                      for b in bucket_order]
    bp = ax.boxplot(data_by_bucket, patch_artist=True, notch=False,
                    boxprops=dict(facecolor=STYLE["accent1"], color=STYLE["grid"]),
                    whiskerprops=dict(color=STYLE["muted"]),
                    capprops=dict(color=STYLE["muted"]),
                    medianprops=dict(color=STYLE["accent2"], lw=2),
                    flierprops=dict(marker=".", color=STYLE["accent3"],
                                   alpha=0.4, markersize=3))
    ax.set_xticklabels(bucket_order, rotation=30)
    _style(ax, "Vega by DTE Bucket", "DTE Bucket", "Vega (per 1% IV)")

    # D – Theta vs DTE (time decay)
    ax = axes[1, 0]
    ax.set_facecolor(STYLE["surface"])
    valid_t = df.dropna(subset=["theta", "tte"])
    ax.scatter(valid_t["tte"], valid_t["theta"].clip(-5, 0),
               s=3, alpha=0.3, color=STYLE["accent4"])
    _style(ax, "Theta vs DTE (time decay)", "Days to Expiry", "Theta ($/day)")

    # E – BSM price vs Market mid
    ax = axes[1, 1]
    ax.set_facecolor(STYLE["surface"])
    valid_p = df.dropna(subset=["bsm_price", "mid"]).query("bsm_price > 0 and mid > 0")
    sample_p = valid_p.sample(min(2000, len(valid_p)), random_state=42)
    ax.scatter(sample_p["mid"], sample_p["bsm_price"],
               s=3, alpha=0.3, color=STYLE["accent3"])
    lim = max(sample_p["mid"].max(), sample_p["bsm_price"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="white", lw=1, linestyle="--", label="Parity")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=7)
    _style(ax, "BSM Price vs Market Mid", "Market Mid ($)", "BSM Price ($)")

    # F – Greeks anomaly flag heatmap by bucket
    ax = axes[1, 2]
    ax.set_facecolor(STYLE["surface"])
    greek_flags  = ["flag_delta_bound", "flag_vega_spike", "flag_gamma_spike"]
    greek_labels = ["Delta\nBound", "Vega\nSpike", "Gamma\nATM"]
    gcounts = [df[f].sum() for f in greek_flags]
    gcolors = [STYLE["accent2"], STYLE["accent4"], STYLE["accent3"]]
    gbars   = ax.bar(greek_labels, gcounts, color=gcolors, alpha=0.85, edgecolor="none")
    for bar, cnt in zip(gbars, gcounts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{cnt:,}", ha="center", va="bottom",
                color=STYLE["text"], fontsize=9)
    _style(ax, "Greeks Flag Counts", "Flag Type")

    plt.tight_layout()
    plt.savefig("output/stage7_greeks.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage7_greeks.png")
    return df


# ═══════════════════════════════════════════════════════════
# STAGE 8 — Isolation Forest ML Anomaly Score  ← NEW
# ═══════════════════════════════════════════════════════════

_IF_FEATURES = [
    "moneyness", "tte_years", "spread_pct", "vol_oi_ratio",
    "implied_volatility", "iv_residual",
    "z_vol_oi", "z_iv", "z_spread",
    "delta", "gamma", "vega",
]


def stage8_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train an Isolation Forest on clean numeric features, producing:
      if_score     — raw decision function score (lower = more anomalous)
      if_anomaly   — boolean flag
      if_score_pct — percentile rank (0–1, higher = more anomalous)

    The model is fit per-date when multiple dates exist, else globally.
    """
    print("\n[Stage 8] Isolation Forest anomaly detection …")

    feats = [f for f in _IF_FEATURES if f in df.columns]
    print(f"          Features used: {feats}")

    df["if_score"]     = np.nan
    df["if_anomaly"]   = False
    df["if_score_pct"] = np.nan

    n_dates = df["quote_date"].nunique()
    groups  = df["quote_date"].unique() if n_dates > 1 else [None]

    for grp_date in groups:
        if grp_date is None:
            mask = pd.Series(True, index=df.index)
        else:
            mask = df["quote_date"] == grp_date

        sub = df.loc[mask, feats].copy()
        valid_rows = sub.dropna()

        if len(valid_rows) < 30:
            print(f"          Skip {grp_date}: too few rows ({len(valid_rows)})")
            continue

        scaler = RobustScaler()
        X      = scaler.fit_transform(valid_rows.values)

        clf = IsolationForest(
            n_estimators=200,
            contamination=IF_CONTAMINATION,
            max_features=min(len(feats), 8),
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X)

        scores   = clf.decision_function(X)           # negative = outlier
        labels   = clf.predict(X) == -1               # True = anomaly

        # Scale scores to 0–1 percentile (1 = most anomalous)
        score_pct = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        df.loc[valid_rows.index, "if_score"]     = scores
        df.loc[valid_rows.index, "if_anomaly"]   = labels
        df.loc[valid_rows.index, "if_score_pct"] = score_pct

    n_if = df["if_anomaly"].sum()
    print(f"          IF anomalies flagged: {n_if:,}  ({100*n_if/len(df):.1f}%)")

    # ── Isolation Forest visualisation ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=STYLE["bg"])
    fig.suptitle("Stage 8 · Isolation Forest ML Anomaly Detection",
                 color=STYLE["text"], fontsize=14)

    def _style8(ax, title, xlabel, ylabel="Count"):
        ax.set_facecolor(STYLE["surface"])
        ax.set_title(title, color=STYLE["text"], fontsize=10)
        ax.set_xlabel(xlabel, color=STYLE["muted"], fontsize=9)
        ax.set_ylabel(ylabel, color=STYLE["muted"], fontsize=9)
        ax.tick_params(colors=STYLE["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["grid"])

    # A – IF score distribution
    ax = axes[0]
    valid_scores = df["if_score"].dropna()
    ax.hist(valid_scores, bins=80, color=STYLE["accent4"], alpha=0.75, edgecolor="none")
    threshold = np.percentile(valid_scores, 100 * IF_CONTAMINATION)
    ax.axvline(threshold, color=STYLE["accent2"], lw=1.5, linestyle="--",
               label=f"Threshold ({IF_CONTAMINATION:.0%})")
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"])
    _style8(ax, "IF Decision Score Distribution", "Decision Score")

    # B – IV surface with IF overlay
    ax = axes[1]
    ax.set_facecolor(STYLE["surface"])
    normal_if = df[~df["if_anomaly"]].sample(min(600, (~df["if_anomaly"]).sum()),
                                              random_state=42)
    anom_if   = df[df["if_anomaly"]].dropna(subset=["moneyness", "implied_volatility"])
    ax.scatter(normal_if["moneyness"], normal_if["implied_volatility"],
               s=4, alpha=0.25, color=STYLE["muted"], label="Normal")
    sc = ax.scatter(anom_if["moneyness"], anom_if["implied_volatility"],
                    c=anom_if["if_score_pct"], cmap="plasma",
                    s=15, alpha=0.8, zorder=5, label=f"IF Anomaly ({len(anom_if):,})")
    plt.colorbar(sc, ax=ax, label="Anomaly Strength").ax.yaxis.label.set_color(STYLE["muted"])
    ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"],
              markerscale=2, fontsize=8)
    _style8(ax, "IF Anomalies on IV Surface", "Moneyness", "Implied Volatility")

    # C – Overlap: rule-based vs IF
    ax = axes[2]
    ax.set_facecolor(STYLE["surface"])
    rule_flag = df["is_anomaly"].fillna(False)
    if_flag   = df["if_anomaly"].fillna(False)

    rule_only   = (rule_flag & ~if_flag).sum()
    if_only     = (~rule_flag & if_flag).sum()
    both        = (rule_flag & if_flag).sum()
    neither     = (~rule_flag & ~if_flag).sum()

    labels  = ["Rule\nOnly", "IF\nOnly", "Both", "Neither"]
    vals    = [rule_only, if_only, both, neither]
    clrs    = [STYLE["accent1"], STYLE["accent4"], STYLE["accent2"], STYLE["muted"]]
    bars    = ax.bar(labels, vals, color=clrs, alpha=0.85, edgecolor="none")
    for bar, cnt in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{cnt:,}", ha="center", va="bottom",
                color=STYLE["text"], fontsize=9)
    _style8(ax, "Rule-Based vs IF Overlap", "Detector Agreement")

    plt.tight_layout()
    plt.savefig("output/stage8_isolation_forest.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage8_isolation_forest.png")
    return df


# ═══════════════════════════════════════════════════════════
# STAGE 9 — Extended Score, CSV Export, Alert JSON  ← NEW
# ═══════════════════════════════════════════════════════════

def stage9_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all flags into a composite score (0–7) and export:
      output/top_anomalies_v2.csv  — top 100 contracts, all signal columns
      output/alerts.json           — machine-readable watchlist for daily monitoring
      output/stage9_extended.png   — extended composite dashboard
    """
    print("\n[Stage 9] Composite scoring & alert export …")

    flag_cols = [
        "flag_wide_spread",      # stage 1
        "flag_zscore_anomaly",   # stage 2
        "flag_surface_anomaly",  # stage 3
        "flag_pcp_violation",    # stage 6
        "flag_greeks_anomaly",   # stage 7
        "if_anomaly",            # stage 8
    ]
    flag_cols = [f for f in flag_cols if f in df.columns]

    df["composite_score"] = sum(df[f].astype(int) for f in flag_cols)
    df["is_high_confidence"] = df["composite_score"] >= 3

    top = (df.sort_values("composite_score", ascending=False).head(100))
    export_cols = [
        "quote_date", "expiration", "strike", "option_type",
        "underlying_price", "bid", "ask", "mid", "volume", "open_interest",
        "implied_volatility", "moneyness", "tte",
        "delta", "gamma", "vega", "theta",
        "z_vol_oi", "z_iv", "z_spread",
        "iv_residual", "pcp_deviation", "pcp_dev_pct",
        "if_score_pct",
        "composite_score",
    ] + flag_cols
    present = [c for c in export_cols if c in top.columns]
    top[present].to_csv("output/top_anomalies_v2.csv", index=False)
    print(f"          → output/top_anomalies_v2.csv  (top 100, {len(flag_cols)} signals)")

    # JSON alert watchlist
    high_conf = df[df["is_high_confidence"]].sort_values(
        "composite_score", ascending=False).head(25)
    alerts = []
    for _, row in high_conf.iterrows():
        fired = [f.replace("flag_", "").replace("_anomaly", "")
                 for f in flag_cols if f in row.index and row[f]]
        alert = {
            "ticker":       TICKER,
            "quote_date":   str(row["quote_date"].date()),
            "expiration":   str(row["expiration"].date()),
            "strike":       float(row["strike"]),
            "option_type":  str(row["option_type"]),
            "moneyness":    round(float(row["moneyness"]), 4),
            "tte_days":     int(row["tte"]),
            "iv":           round(float(row["implied_volatility"]), 4),
            "volume":       int(row.get("volume", 0) or 0),
            "open_interest":int(row.get("open_interest", 0) or 0),
            "composite_score": int(row["composite_score"]),
            "signals_fired":   fired,
            "delta":        round(float(row["delta"]), 4) if not pd.isna(row.get("delta")) else None,
            "if_score_pct": round(float(row["if_score_pct"]), 4) if not pd.isna(row.get("if_score_pct")) else None,
        }
        alerts.append(alert)

    alert_doc = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ticker":        TICKER,
        "pipeline_version": "v2",
        "total_contracts":  len(df),
        "total_anomalies":  int(df["is_high_confidence"].sum()),
        "alerts": alerts,
    }
    with open("output/alerts.json", "w") as f:
        json.dump(alert_doc, f, indent=2)
    print(f"          → output/alerts.json  ({len(alerts)} high-confidence alerts)")

    # ── Extended composite dashboard ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=STYLE["bg"])
    fig.suptitle(
        f"Options Anomaly Pipeline v2  ·  {TICKER}  ·  {df['quote_date'].max().date()}",
        color=STYLE["text"], fontsize=15, fontweight="bold", y=1.0)

    def _style9(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(STYLE["surface"])
        ax.set_title(title, color=STYLE["text"], fontsize=10)
        if xlabel: ax.set_xlabel(xlabel, color=STYLE["muted"], fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color=STYLE["muted"], fontsize=8)
        ax.tick_params(colors=STYLE["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["grid"])

    # A – Composite score histogram
    ax = axes[0, 0]
    sc_counts = df["composite_score"].value_counts().sort_index()
    color_ramp = [STYLE["muted"], STYLE["accent1"], STYLE["accent3"],
                  STYLE["accent2"], STYLE["accent2"], STYLE["accent2"],
                  STYLE["accent2"]]
    ax.bar(sc_counts.index.astype(str), sc_counts.values,
           color=[color_ramp[min(i, len(color_ramp)-1)] for i in sc_counts.index],
           edgecolor="none", alpha=0.85)
    _style9(ax, "Composite Score Distribution", "Score (0–6)", "Count")

    # B – Signal co-occurrence matrix
    ax = axes[0, 1]
    present_flags = [f for f in flag_cols if f in df.columns]
    short_names   = [f.replace("flag_", "").replace("_anomaly", "")
                     for f in present_flags]
    comat = np.zeros((len(present_flags), len(present_flags)), dtype=int)
    bool_arr = df[present_flags].fillna(False).values.astype(bool)
    for i in range(len(present_flags)):
        for j in range(len(present_flags)):
            comat[i, j] = int((bool_arr[:, i] & bool_arr[:, j]).sum())
    np.fill_diagonal(comat, 0)
    im = ax.imshow(comat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=30, ha="right",
                       fontsize=7, color=STYLE["muted"])
    ax.set_yticklabels(short_names, fontsize=7, color=STYLE["muted"])
    for i in range(len(present_flags)):
        for j in range(len(present_flags)):
            ax.text(j, i, str(comat[i, j]), ha="center", va="center",
                    fontsize=6, color="white" if comat[i, j] > comat.max()*0.5 else STYLE["text"])
    plt.colorbar(im, ax=ax, label="Co-occurrences").ax.yaxis.label.set_color(STYLE["muted"])
    _style9(ax, "Signal Co-Occurrence Matrix")

    # C – High-confidence anomalies on IV surface
    ax = axes[0, 2]
    ax.set_facecolor(STYLE["surface"])
    norm_mask = ~df["is_high_confidence"]
    hc_mask   =  df["is_high_confidence"]
    norm_s    = df[norm_mask].sample(min(600, norm_mask.sum()), random_state=42)
    hc_s      = df[hc_mask]
    ax.scatter(norm_s["moneyness"], norm_s["implied_volatility"],
               s=4, alpha=0.2, color=STYLE["muted"])
    sc9 = ax.scatter(hc_s["moneyness"], hc_s["implied_volatility"],
                     c=hc_s["composite_score"], cmap="hot_r",
                     s=hc_s["composite_score"]*20+15, alpha=0.85, zorder=5)
    plt.colorbar(sc9, ax=ax, label="Score").ax.yaxis.label.set_color(STYLE["muted"])
    _style9(ax, f"High-Confidence Anomalies ({hc_mask.sum():,})",
            "Moneyness", "IV")

    # D – Signal heatmap: top 20 anomalies × flags
    ax = axes[1, 0]
    top20 = df.sort_values("composite_score", ascending=False).head(20)
    if len(top20) > 0 and len(present_flags) > 0:
        hm_data = top20[present_flags].fillna(False).astype(int).T
        hm_data.index = short_names
        im2 = ax.imshow(hm_data.values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=7, color=STYLE["muted"])
        ax.set_xticks(range(len(top20)))
        ax.set_xticklabels(range(1, len(top20)+1), fontsize=6, color=STYLE["muted"])
        ax.set_xlabel("Anomaly rank", color=STYLE["muted"], fontsize=8)
    _style9(ax, "Signal Flags — Top 20 Anomalies")

    # E – Delta distribution of anomalies vs normal
    ax = axes[1, 1]
    ax.set_facecolor(STYLE["surface"])
    if "delta" in df.columns:
        norm_d = df[~df["is_high_confidence"]]["delta"].dropna().clip(-1.1, 1.1)
        hc_d   = df[ df["is_high_confidence"]]["delta"].dropna().clip(-1.1, 1.1)
        ax.hist(norm_d, bins=60, color=STYLE["accent1"], alpha=0.5,
                edgecolor="none", label="Normal", density=True)
        ax.hist(hc_d,   bins=60, color=STYLE["accent2"], alpha=0.75,
                edgecolor="none", label="High-Conf Anom", density=True)
        ax.legend(facecolor=STYLE["surface"], labelcolor=STYLE["text"], fontsize=7)
    _style9(ax, "Delta: Normal vs High-Conf Anomaly", "Delta", "Density")

    # F – Summary table v2
    ax = axes[1, 2]
    ax.set_facecolor(STYLE["surface"])
    ax.axis("off")
    total   = len(df)
    n_hc    = df["is_high_confidence"].sum()
    n_pcp   = df.get("flag_pcp_violation", pd.Series(False, index=df.index)).sum()
    n_greek = df.get("flag_greeks_anomaly", pd.Series(False, index=df.index)).sum()
    n_if    = df.get("if_anomaly", pd.Series(False, index=df.index)).sum()
    tbl_rows = [
        ["Pipeline",          "v2 (Stages 0–9)"],
        ["Ticker",            TICKER],
        ["Quote Date",        str(df["quote_date"].max().date())],
        ["Total Contracts",   f"{total:,}"],
        ["Expiries",          str(df["expiration"].nunique())],
        ["PCP Violations",    f"{n_pcp:,}"],
        ["Greeks Anomalies",  f"{n_greek:,}"],
        ["IF Anomalies",      f"{n_if:,}"],
        ["High-Conf (≥3)",    f"{n_hc:,}  ({100*n_hc/total:.1f}%)"],
        ["Top Alert Score",   str(df["composite_score"].max())],
    ]
    tbl = ax.table(cellText=tbl_rows, colLabels=["Metric", "Value"],
                   cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(STYLE["surface"] if r % 2 == 0 else STYLE["bg"])
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["text"] if r > 0 else STYLE["accent1"])
    _style9(ax, "Pipeline v2 Summary")

    plt.tight_layout()
    plt.savefig("output/stage9_extended.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print("          → output/stage9_extended.png")

    return df


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Options Anomaly Detection Pipeline  v2")
    print("=" * 62)

    # Stage 0
    if DATA_SOURCE == "yfinance":
        raw = load_yfinance(TICKER)
    else:
        raw = load_file(DATA_PATH, COLUMN_MAP)
    df = derive_features(raw)

    # Stages 1–5  (original pipeline)
    df = stage1_quality(df)
    df = stage2_zscore(df)
    df = stage3_iv_surface(df)
    df = stage4_score(df)
    stage5_dashboard(df)

    # Stages 6–9  (new in v2)
    df = stage6_pcp_violations(df)
    df = stage7_greeks(df)
    df = stage8_isolation_forest(df)
    df = stage9_export(df)

    print("\n" + "=" * 62)
    print("  DONE — all outputs in ./output/")
    print("=" * 62)
    outputs = [
        ("stage1_quality.png",        "Data quality overview"),
        ("stage2_zscore.png",         "Z-score distributions"),
        ("stage3_surface.png",        "IV surface residuals"),
        ("stage4_scores.png",         "Rule-based scores"),
        ("stage5_dashboard.png",      "Stage 1-4 dashboard"),
        ("stage6_pcp.png",            "Put-call parity violations  [NEW]"),
        ("stage7_greeks.png",         "BSM Greeks analysis         [NEW]"),
        ("stage8_isolation_forest.png","Isolation Forest detection  [NEW]"),
        ("stage9_extended.png",       "Composite dashboard         [NEW]"),
        ("top_anomalies_v2.csv",      "Top 100 anomalies, all signals [NEW]"),
        ("alerts.json",               "Machine-readable watchlist  [NEW]"),
    ]
    for fname, desc in outputs:
        print(f"  {fname:<38}  {desc}")
    print("=" * 62)

    return df


if __name__ == "__main__":
    df_result = main()
