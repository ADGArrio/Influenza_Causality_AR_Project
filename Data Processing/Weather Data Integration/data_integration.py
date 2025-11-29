import requests
import pandas as pd
import numpy as np
import datetime
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
WHO_FLUNET_URL = "https://www.who.int/docs/default-source/influenza/flunet-data.csv"  # WHO FluNet CSV URL
NOAA_API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/"
NOAA_TOKEN = "bjjrctwQTjWUvgLIyzelyAMxyDKyTAvr"  # Replace with your NOAA token
OUTPUT_FILE = "flunet_noaa_features.csv"

# -----------------------------
# STEP 1: Download WHO FluNet data
# -----------------------------
# print("Downloading WHO FluNet data...")
# response = requests.get(WHO_FLUNET_URL)
# if response.status_code != 200:
#     raise Exception(f"Failed to download WHO FluNet data: {response.status_code}")

# with open("flunet_raw.csv", "wb") as f:
#     f.write(response.content)

flunet_df = pd.read_csv("VIW_FNT_Cleaned.csv", parse_dates=["iso_weekstartdate"])
flunet_df.columns = [c.lower().replace(" ", "_") for c in flunet_df.columns]

# Keep relevant columns
flunet_df = flunet_df[["country_code", "iso_weekstartdate", "isoyw", "inf_all", "inf_a", "inf_b"]]

# -----------------------------
# STEP 2: Fetch NOAA weather data
# -----------------------------
def fetch_noaa_weather(country_code, start_date, end_date):
    """Fetch NOAA weather data for a given country and date range."""
    headers = {"token": NOAA_TOKEN}
    params = {
        "datasetid": "GSOM",  # Monthly summaries (can aggregate to weekly)
        "locationid": f"FIPS:{country_code}",  # Adjust for country codes
        "startdate": start_date,
        "enddate": end_date,
        "units": "metric",
        "limit": 1000
    }
    r = requests.get(NOAA_API_BASE + "data", headers=headers, params=params)
    if r.status_code != 200:
        print(f"NOAA API error for {country_code}: {r.status_code}")
        return pd.DataFrame()
    data = r.json().get("results", [])
    return pd.DataFrame(data)

# For demo: simulate NOAA data (replace with real API calls)
print("Fetching NOAA data (simulated)...")
weather_data = []
for _, row in flunet_df.iterrows():
    weather_data.append({
        "country_code": row["country_code"],
        "iso_weekstartdate": row["iso_weekstartdate"],
        "avg_temp": np.random.uniform(-5, 30),
        "humidity": np.random.uniform(40, 90)
    })
weather_df = pd.DataFrame(weather_data)

# -----------------------------
# STEP 3: Merge and Feature Engineering
# -----------------------------
merged = pd.merge(flunet_df, weather_df, on=["country_code", "iso_weekstartdate"], how="left")

# Derived indicators
merged["pct_a"] = merged["inf_a"] / merged["inf_all"].replace(0, np.nan)
merged["pct_b"] = merged["inf_b"] / merged["inf_all"].replace(0, np.nan)

# Seasonal features
merged["week"] = merged["iso_weekstartdate"].dt.isocalendar().week
merged["season_sin"] = np.sin(2 * np.pi * merged["week"] / 52)
merged["season_cos"] = np.cos(2 * np.pi * merged["week"] / 52)

# Save combined dataset
merged.to_csv(OUTPUT_FILE, index=False)
print(f"Automation complete. Combined dataset saved as: {OUTPUT_FILE}")
print(merged.head())
