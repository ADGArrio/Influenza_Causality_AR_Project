import pandas as pd
import numpy as np

# Load your CSV file
input_file = "flunet_noaa_features.csv"  # Replace with your actual file path
df = pd.read_csv(input_file, parse_dates=["iso_weekstartdate"])

# Sort by country and date for proper lagging
df = df.sort_values(by=["country_code", "iso_weekstartdate"])

# Features to create lags and rolling averages for
lag_features = ["inf_all", "avg_temp", "humidity"]

# Add lag features (1–3 weeks)
for feature in lag_features:
    for lag in [1, 2, 3]:
        df[f"{feature}_lag{lag}"] = df.groupby("country_code")[feature].shift(lag)

# Add rolling averages (3-week window)
for feature in lag_features:
    df[f"{feature}_roll3"] = (
        df.groupby("country_code")[feature]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Drop rows with NaN in lag features (optional for Granger causality)
df_clean = df.dropna(subset=[f"{feature}_lag3" for feature in lag_features])

# Save prepared dataset
output_file = "flunet_weather_features_lagged.csv"
df_clean.to_csv(output_file, index=False)

print(f"Lag features and rolling averages added. Saved as {output_file}")
print(df_clean.head())
