# options-systematic

This project scores option contracts by how unusual their volume is relative to peers on the same snapshot date.

## What `model.py` does

`model.py`:
1. Loads `spy_options.parquet`
2. Uses the latest snapshot date in the file
3. Fits a gradient-boosting model to predict `log(1 + volume)` using:
   - moneyness (`log(strike / spot)`)
   - time to expiration
   - strike
   - spot
   - bid/ask spread
   - option type (`call`/`put`)
4. Computes residuals (`actual - predicted` in log-volume space)
5. Standardizes residuals inside each expiration bucket (`resid_z`)
6. Applies a liquidity penalty for wide spreads
7. Prints two ranked terminal tables:
   - top positive anomalies (`UNUSUALLY HIGH VOLUME VS PEERS`)
   - top negative anomalies (`UNUSUALLY LOW VOLUME VS PEERS`)

Run it with:

```bash
python3 model.py
```

## Frontend dashboard

A Streamlit dashboard is included in `frontend.py` so you can interact with symbol/date/spot and visualize results.

Run it with:

```bash
streamlit run frontend.py
```

You can then:
- Filter symbol, snapshot date, spot, and watchlist size
- Inspect high/low anomaly contract tables
- Explore scatter/tenor charts of anomaly behavior

## Data

Download options parquet data (replace `<ticker>`):

https://static.philippdubach.com/data/options/<ticker>/options.parquet

By default this repo uses `spy_options.parquet`.
