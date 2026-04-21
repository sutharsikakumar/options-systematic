"""
options_alert_system.py
=======================
Standalone daily alert runner built on top of the v2 pipeline.

Purpose
-------
Run this script once a day (cron, Task Scheduler, GitHub Action) to:
  1. Fetch fresh option data via yfinance
  2. Run the full v2 pipeline
  3. Diff today's alerts against yesterday's  (new / resolved / persisting)
  4. Output  output/alert_report_YYYYMMDD.json
             output/alert_summary_YYYYMMDD.txt   (human-readable)

Usage
-----
  python options_alert_system.py
  python options_alert_system.py --tickers SPY QQQ AAPL NVDA
  python options_alert_system.py --tickers SPY --min-score 4

Requirements
------------
  pip install yfinance pandas numpy scipy scikit-learn matplotlib seaborn pyarrow

Scheduling  (examples)
-----------
  # Linux cron  — run at 4:35 PM ET every weekday
  35 16 * * 1-5  cd /your/project && python options_alert_system.py >> logs/alerts.log 2>&1

  # macOS launchd  — see Apple docs for plist templates

  # GitHub Actions  — see .github/workflows/daily_scan.yml example in README
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────
# Import the v2 pipeline stages
# ──────────────────────────────────────────────────────────
try:
    from options_anomaly_pipeline_v2 import (
        load_yfinance, derive_features,
        stage1_quality, stage2_zscore, stage3_iv_surface,
        stage4_score, stage6_pcp_violations,
        stage7_greeks, stage8_isolation_forest,
        RISK_FREE_RATE, DIV_YIELD, STYLE,
    )
except ImportError:
    sys.exit("ERROR: options_anomaly_pipeline_v2.py must be in the same directory.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
DEFAULT_TICKERS  = ["SPY"]
MIN_SCORE        = 3           # composite score threshold for alerts
HISTORY_DIR      = Path("output/history")
TODAY_STR        = date.today().strftime("%Y%m%d")

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs("output", exist_ok=True)


# ──────────────────────────────────────────────────────────
# Core pipeline runner (one ticker)
# ──────────────────────────────────────────────────────────

def run_pipeline_for_ticker(ticker: str) -> pd.DataFrame:
    """Run full v2 pipeline for a single ticker; return enriched DataFrame."""
    import importlib, options_anomaly_pipeline_v2 as pipe
    pipe.TICKER = ticker          # patch global (simple enough for a script)

    raw = load_yfinance(ticker)
    df  = derive_features(raw)
    df  = stage1_quality(df)
    df  = stage2_zscore(df)
    df  = stage3_iv_surface(df)
    df  = stage4_score(df)
    df  = stage6_pcp_violations(df)
    df  = stage7_greeks(df)
    df  = stage8_isolation_forest(df)

    # Composite score
    flag_cols = [
        "flag_wide_spread", "flag_zscore_anomaly", "flag_surface_anomaly",
        "flag_pcp_violation", "flag_greeks_anomaly", "if_anomaly",
    ]
    flag_cols = [f for f in flag_cols if f in df.columns]
    df["composite_score"]    = sum(df[f].astype(int) for f in flag_cols)
    df["is_high_confidence"] = df["composite_score"] >= MIN_SCORE
    df["ticker"]             = ticker

    return df


# ──────────────────────────────────────────────────────────
# Contract identity key
# ──────────────────────────────────────────────────────────

def contract_key(row) -> str:
    exp = pd.Timestamp(row["expiration"]).strftime("%Y%m%d")
    return f"{row.get('ticker','')}-{exp}-{row['strike']:.2f}{row['option_type']}"


# ──────────────────────────────────────────────────────────
# Delta detection: new vs resolved vs persisting
# ──────────────────────────────────────────────────────────

def diff_alerts(today_alerts: list, yesterday_path: Path) -> dict:
    """
    Compare today's alert keys with yesterday's.
    Returns dict with lists: new, resolved, persisting.
    """
    today_keys = {a["contract_key"] for a in today_alerts}

    if not yesterday_path.exists():
        return {"new": today_keys, "resolved": set(), "persisting": set()}

    with open(yesterday_path) as f:
        prev = json.load(f)
    prev_keys = {a["contract_key"] for a in prev.get("alerts", [])}

    return {
        "new":        today_keys - prev_keys,
        "resolved":   prev_keys  - today_keys,
        "persisting": today_keys & prev_keys,
    }


# ──────────────────────────────────────────────────────────
# Visualisation: multi-ticker summary
# ──────────────────────────────────────────────────────────

def plot_summary(all_df: pd.DataFrame, tickers: list, report_path: str) -> None:
    n_tickers = len(tickers)
    fig = plt.figure(figsize=(6 * min(n_tickers, 3), 9), facecolor=STYLE["bg"])
    fig.suptitle(f"Daily Alert Summary  ·  {TODAY_STR}",
                 color=STYLE["text"], fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, min(n_tickers, 3), figure=fig,
                           hspace=0.45, wspace=0.35)

    for i, ticker in enumerate(tickers[:3]):
        sub = all_df[all_df["ticker"] == ticker]
        col = i % 3

        # Top panel: IV smile coloured by composite score
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.set_facecolor(STYLE["surface"])
        norm = sub[~sub["is_high_confidence"]].sample(
            min(300, (~sub["is_high_confidence"]).sum()), random_state=42)
        hc   = sub[sub["is_high_confidence"]]
        ax_top.scatter(norm["moneyness"], norm["implied_volatility"],
                       s=4, alpha=0.2, color=STYLE["muted"])
        if len(hc) > 0:
            sc = ax_top.scatter(hc["moneyness"], hc["implied_volatility"],
                                c=hc["composite_score"], cmap="hot_r",
                                s=hc["composite_score"]*25+10,
                                alpha=0.85, zorder=5,
                                vmin=MIN_SCORE, vmax=6)
            plt.colorbar(sc, ax=ax_top).ax.yaxis.label.set_color(STYLE["muted"])
        ax_top.set_title(f"{ticker} — IV Smile  ({len(hc):,} alerts)",
                         color=STYLE["text"], fontsize=10)
        ax_top.set_xlabel("Moneyness", color=STYLE["muted"], fontsize=8)
        ax_top.set_ylabel("IV", color=STYLE["muted"], fontsize=8)
        ax_top.tick_params(colors=STYLE["muted"], labelsize=7)
        for sp in ax_top.spines.values():
            sp.set_edgecolor(STYLE["grid"])

        # Bottom panel: composite score bar
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.set_facecolor(STYLE["surface"])
        sc_cnt = sub["composite_score"].value_counts().sort_index()
        color_ramp = {0: STYLE["muted"], 1: STYLE["accent1"],
                      2: STYLE["accent3"], 3: STYLE["accent2"],
                      4: STYLE["accent2"], 5: STYLE["accent2"],
                      6: STYLE["accent2"]}
        ax_bot.bar(sc_cnt.index.astype(str), sc_cnt.values,
                   color=[color_ramp.get(s, STYLE["accent2"]) for s in sc_cnt.index],
                   edgecolor="none", alpha=0.85)
        ax_bot.set_title(f"{ticker} — Score Distribution",
                         color=STYLE["text"], fontsize=10)
        ax_bot.set_xlabel("Composite Score", color=STYLE["muted"], fontsize=8)
        ax_bot.set_ylabel("Count", color=STYLE["muted"], fontsize=8)
        ax_bot.tick_params(colors=STYLE["muted"], labelsize=7)
        for sp in ax_bot.spines.values():
            sp.set_edgecolor(STYLE["grid"])

    plt.tight_layout()
    plt.savefig(report_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  → {report_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main(tickers: list, min_score: int):
    print("=" * 62)
    print(f"  Options Alert System  —  {TODAY_STR}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Min composite score for alert: {min_score}")
    print("=" * 62)

    all_dfs  = []
    all_alerts = []

    for ticker in tickers:
        print(f"\n{'─'*30}  {ticker}")
        try:
            df = run_pipeline_for_ticker(ticker)
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            continue

        all_dfs.append(df)

        # Build alert records
        hc = df[df["composite_score"] >= min_score].sort_values(
            "composite_score", ascending=False)

        flag_cols = [f for f in [
            "flag_wide_spread", "flag_zscore_anomaly", "flag_surface_anomaly",
            "flag_pcp_violation", "flag_greeks_anomaly", "if_anomaly",
        ] if f in df.columns]

        for _, row in hc.iterrows():
            fired = [f.replace("flag_", "").replace("_anomaly", "")
                     for f in flag_cols if row.get(f, False)]
            key = contract_key({**row.to_dict(), "ticker": ticker})
            alert = {
                "contract_key":  key,
                "ticker":        ticker,
                "quote_date":    str(pd.Timestamp(row["quote_date"]).date()),
                "expiration":    str(pd.Timestamp(row["expiration"]).date()),
                "strike":        float(row["strike"]),
                "option_type":   str(row["option_type"]),
                "moneyness":     round(float(row["moneyness"]), 4),
                "tte_days":      int(row["tte"]),
                "iv":            round(float(row["implied_volatility"]), 4),
                "volume":        int(row.get("volume") or 0),
                "open_interest": int(row.get("open_interest") or 0),
                "composite_score": int(row["composite_score"]),
                "signals_fired": fired,
                "delta":         round(float(row["delta"]), 4)
                                 if "delta" in row and not pd.isna(row["delta"]) else None,
                "if_score_pct":  round(float(row["if_score_pct"]), 4)
                                 if "if_score_pct" in row and not pd.isna(row["if_score_pct"]) else None,
            }
            all_alerts.append(alert)

    if not all_dfs:
        print("\nNo data processed. Exiting.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Delta diff vs yesterday
    yesterday_path = HISTORY_DIR / f"alerts_{(date.today()-timedelta(1)).strftime('%Y%m%d')}.json"
    delta = diff_alerts(all_alerts, yesterday_path)

    # Save today's alert archive
    today_archive = HISTORY_DIR / f"alerts_{TODAY_STR}.json"
    with open(today_archive, "w") as f:
        json.dump({"alerts": all_alerts}, f, indent=2)

    # Full report JSON
    report = {
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "date":            TODAY_STR,
        "tickers":         tickers,
        "min_score":       min_score,
        "total_alerts":    len(all_alerts),
        "new_alerts":      len(delta["new"]),
        "resolved_alerts": len(delta["resolved"]),
        "persisting_alerts": len(delta["persisting"]),
        "alerts": all_alerts,
        "delta": {
            "new":       list(delta["new"]),
            "resolved":  list(delta["resolved"]),
            "persisting":list(delta["persisting"]),
        }
    }
    report_json = f"output/alert_report_{TODAY_STR}.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  → {report_json}")

    # Human-readable summary text
    summary_lines = [
        f"Options Alert Summary  —  {TODAY_STR}",
        f"Tickers:  {', '.join(tickers)}",
        f"Total alerts (score ≥ {min_score}): {len(all_alerts)}",
        f"  NEW      : {len(delta['new'])}",
        f"  RESOLVED : {len(delta['resolved'])}",
        f"  PERSISTING: {len(delta['persisting'])}",
        "",
        "Top 10 Highest-Score Contracts:",
        "-" * 62,
    ]
    sorted_alerts = sorted(all_alerts, key=lambda x: -x["composite_score"])
    for a in sorted_alerts[:10]:
        line = (f"  {a['ticker']:<5} {a['option_type']}  "
                f"K={a['strike']:<8.2f} exp={a['expiration']}  "
                f"score={a['composite_score']}  "
                f"signals=[{', '.join(a['signals_fired'])}]  "
                f"IV={a['iv']:.3f}")
        summary_lines.append(line)

    summary_txt = "\n".join(summary_lines)
    txt_path    = f"output/alert_summary_{TODAY_STR}.txt"
    with open(txt_path, "w") as f:
        f.write(summary_txt)

    print("\n" + summary_txt)
    print(f"\n  → {txt_path}")

    # Visualisation
    plot_summary(combined_df, tickers,
                 f"output/alert_chart_{TODAY_STR}.png")

    print("\n" + "=" * 62)
    print("  Alert run complete.")
    print("=" * 62)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily options anomaly alert runner")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Space-separated ticker list  (default: SPY)")
    parser.add_argument(
        "--min-score", type=int, default=MIN_SCORE,
        help=f"Min composite score to raise alert  (default: {MIN_SCORE})")
    args = parser.parse_args()

    main(args.tickers, args.min_score)
