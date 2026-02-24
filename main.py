import pandas as pd

# Load options data
df = pd.read_parquet("options.parquet", engine="fastparquet")

# Filter to calls expiring in 30 days
from datetime import datetime, timedelta
target_exp = datetime.now() + timedelta(days=30)
calls = df[(df['type'] == 'call') & (df['expiration'] <= target_exp)]

calls_recent10 = calls.sort_values("expiration", ascending=False).head(10)

print(calls_recent10)