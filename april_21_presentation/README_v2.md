# Options Anomaly Detection Pipeline  —  v2

A full 9-stage options anomaly detection system with rule-based, arbitrage,
Greeks-based, and machine-learning detectors.

## Quick Start (zero data needed)

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn scipy pyarrow

# Full 9-stage pipeline
python options_anomaly_pipeline_v2.py

# Daily multi-ticker alert runner
python options_alert_system.py --tickers SPY QQQ AAPL --min-score 3
```

---

## What's New in v2

| Stage | Detector | Signal Type |
|-------|----------|-------------|
| **6** | **Put-Call Parity Violations** | True arbitrage — deviation exceeds bid-ask spread |
| **7** | **BSM Greeks + Anomaly Flags** | Delta bounds, vega spikes, gamma ATM outliers |
| **8** | **Isolation Forest (ML)** | Unsupervised; 12 features; per-date or global |
| **9** | **Composite Score (0–6) + Alert Export** | JSON watchlist, enriched CSV, extended dashboard |

Plus a new **`options_alert_system.py`** daily runner with delta-diff (new / resolved /
persisting alerts) and multi-ticker support.

---

## All Outputs

All files land in `./output/`:

| File | Stage | Description |
|------|-------|-------------|
| `stage1_quality.png`          | 1 | Spread distribution, flag counts |
| `stage2_zscore.png`           | 2 | Z-score distributions (Vol/OI, IV, Spread) |
| `stage3_surface.png`          | 3 | IV surface residual scatter + histogram |
| `stage4_scores.png`           | 4 | Rule-based score distribution |
| `stage5_dashboard.png`        | 5 | 6-panel overview of Stages 1–4 |
| `stage6_pcp.png`              | 6 | Put-call parity deviation histogram + scatter |
| `stage7_greeks.png`           | 7 | 6-panel Greeks analysis (delta, gamma, vega, theta) |
| `stage8_isolation_forest.png` | 8 | IF score distribution, IV surface overlay, detector overlap |
| `stage9_extended.png`         | 9 | Composite dashboard: co-occurrence matrix, score heatmap |
| `top_anomalies_v2.csv`        | 9 | Top 100 contracts with all signal columns |
| `alerts.json`                 | 9 | Machine-readable watchlist (top 25 high-confidence) |

Daily runner outputs (`options_alert_system.py`):

| File | Description |
|------|-------------|
| `alert_report_YYYYMMDD.json` | Full report with delta diff (new/resolved/persisting) |
| `alert_summary_YYYYMMDD.txt` | Human-readable top-10 summary |
| `alert_chart_YYYYMMDD.png`   | Multi-ticker IV smile + score distribution |

---

## Pipeline Architecture

```
Raw Data (yfinance / CSV / Parquet)
    │
    ▼
[Stage 0] Load & Derive
          mid, spread_pct, vol_oi_ratio, moneyness, tte, tte_years
    │
    ▼
[Stage 1] Data Quality Flags
          inverted bids, wide spreads, zero mid
          → stage1_quality.png
    │
    ▼
[Stage 2] Rolling Z-Score Engine
          vol/OI, IV, spread; coincidence flag (≥2 of 3)
          → stage2_zscore.png
    │
    ▼
[Stage 3] IV Surface Residuals
          OLS per date: IV ~ moneyness + moneyness² + log(TTE) + is_call
          flag: |residual| > 2σ
          → stage3_surface.png
    │
    ▼
[Stage 4] Rule-Based Unified Score (0–3)
          → stage4_scores.png
    │
    ▼
[Stage 5] Summary Dashboard  (Stages 1–4)
          → stage5_dashboard.png
    │
    ▼─── NEW IN v2 ─────────────────────────────────────────────────
    │
    ▼
[Stage 6] Put-Call Parity Violations
          C - P = S·e^{-qT} - K·e^{-rT}
          flag: |deviation| > PCP_THRESHOLD and deviation > combined half-spread
          → stage6_pcp.png
    │
    ▼
[Stage 7] BSM Greeks + Flags
          delta (bounds), gamma (ATM spike), vega (DTE-bucket spike), theta
          BSM price vs market mid scatter
          → stage7_greeks.png
    │
    ▼
[Stage 8] Isolation Forest ML Score
          12 features · 200 trees · per-date or global
          → stage8_isolation_forest.png
    │
    ▼
[Stage 9] Composite Score (0–6) + Export
          if_score_pct, co-occurrence matrix, delta distribution
          → stage9_extended.png, top_anomalies_v2.csv, alerts.json
```

---

## Anomaly Score Interpretation

### Rule-Based (Stages 1–4, 6–7)

| Score | Meaning |
|-------|---------|
| **3+** | All rule detectors fired — high-confidence anomaly |
| **2**  | Two rule detectors agree — investigate |
| **1**  | Soft flag — noise possible, worth logging |

### Composite Score (Stage 9)

Sums 6 binary flags: wide spread, z-score, IV surface, PCP, Greeks, IF

| Score | Interpretation |
|-------|---------------|
| **5–6** | Extreme outlier — all detectors agree |
| **3–4** | High-confidence — act on this |
| **2**   | Moderate — monitor, verify with Greeks |
| **1**   | Soft flag — log only |

### Isolation Forest Score (Stage 8)

`if_score_pct` is 0–1, with 1 = most anomalous.  
The model is unsupervised — it will find clusters you haven't defined rules for.

---

## Stage 6: Put-Call Parity — Theory

For European options (no early exercise):

```
C - P  =  S · e^{-qT}  -  K · e^{-rT}
```

Where:
- `C`, `P` = call and put mid prices at same strike and expiry
- `S` = underlying spot price
- `K` = strike price
- `T` = time to expiry in years
- `r` = risk-free rate (annualised)
- `q` = continuous dividend yield

A violation is flagged only when:
1. `|deviation %| > PCP_THRESHOLD` (default 2%)  **and**
2. `|deviation| > combined_half_spread` (i.e., the profit survives transaction costs)

American-style options (like SPY) have early-exercise optionality — the parity
holds as a lower bound, not an equality.  Use PCp violations as a *signal
strength* indicator rather than a guaranteed arbitrage.

---

## Stage 7: BSM Greeks

Computed using Black-Scholes-Merton with continuous dividend yield:

| Greek | Formula | Anomaly Condition |
|-------|---------|-------------------|
| **Delta** | `∂V/∂S` | Outside [0,1] for calls or [-1,0] for puts |
| **Gamma** | `∂²V/∂S²` | Top 2% near-ATM (0.97–1.03 moneyness) |
| **Vega**  | `∂V/∂σ · 1/100` | >2.5σ above median within DTE bucket |
| **Theta** | `∂V/∂t / 365` | Visualised only |
| **Rho**   | `∂V/∂r` | Computed, available in DataFrame |

---

## Stage 8: Isolation Forest

**Features** (12 dimensions):
```
moneyness, tte_years, spread_pct, vol_oi_ratio,
implied_volatility, iv_residual,
z_vol_oi, z_iv, z_spread,
delta, gamma, vega
```

- Uses `RobustScaler` (median/IQR) — robust to outliers in the feature space
- `contamination = 0.05` (5% expected anomaly rate — tune to your data)
- Fit per quote_date when multiple dates exist; global otherwise
- `if_score_pct` ∈ [0,1]; 1.0 = most anomalous in the batch

---

## Using Your Own Data

Edit the top of `options_anomaly_pipeline_v2.py`:

```python
DATA_SOURCE = "file"
DATA_PATH   = "your_file.csv"   # or .parquet

COLUMN_MAP = {
    "YOUR_DATE_COL":       "quote_date",
    "YOUR_EXPIRY_COL":     "expiration",
    "YOUR_STRIKE_COL":     "strike",
    "YOUR_BID_COL":        "bid",
    "YOUR_ASK_COL":        "ask",
    "YOUR_VOLUME_COL":     "volume",
    "YOUR_OI_COL":         "open_interest",
    "YOUR_IV_COL":         "implied_volatility",
    "YOUR_UNDERLYING_COL": "underlying_price",
    "YOUR_TYPE_COL":       "option_type",   # 'C' or 'P'
}

# Adjust for your underlying
RISK_FREE_RATE = 0.052   # current annualised risk-free rate
DIV_YIELD      = 0.0     # set to 0 for non-dividend stocks
```

---

## Free Data Sources

| Source | What's Free | Notes |
|--------|-------------|-------|
| **yfinance** | Live option chains, any ticker | Built into pipeline |
| **OptionsDX** | EOD samples (SPY, SPX, AAPL, NVDA...) | Free checkout at optionsdx.com — no card |
| **CBOE** | Volume and put/call ratio data | cboe.com/us/options/market_statistics |
| **DoltHub** | 2019–present options DB (~6GB CSV) | see medium.com/@codythedatainvestor |

---

## Scheduling the Alert Runner

```bash
# Run for multiple tickers with a higher confidence bar
python options_alert_system.py --tickers SPY QQQ AAPL TSLA NVDA --min-score 4
```

### Linux cron (4:35 PM ET, weekdays)
```cron
35 20 * * 1-5  cd /path/to/project && python options_alert_system.py >> logs/alerts.log 2>&1
```

### GitHub Actions (`.github/workflows/daily_scan.yml`)
```yaml
name: Daily Options Scan
on:
  schedule:
    - cron: '35 20 * * 1-5'   # 20:35 UTC = 4:35 PM ET
  workflow_dispatch:

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install yfinance pandas numpy scipy scikit-learn matplotlib seaborn pyarrow
      - run: python options_alert_system.py --tickers SPY QQQ AAPL
      - uses: actions/upload-artifact@v4
        with:
          name: alerts-${{ github.run_id }}
          path: output/
```

---

## Fallback for Limited Data

Single-date snapshot (yfinance default):
- Rolling z-scores → cross-sectional z-scores (within that date)
- Put-call parity still runs (only needs matching call/put pairs)
- BSM Greeks run per-row (no time series needed)
- Isolation Forest runs globally (not per-date)

Everything still produces valid output.
