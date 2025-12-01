import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Aruba GHCNh data
# -----------------------------
# Update this filename to your actual file path
file_path = "./GHCNh_AAI0000TNCA_por.psv"

df = pd.read_csv(file_path, sep="|")

# -----------------------------
# 2. Build a proper datetime
# -----------------------------
# Some files have "Year"/"Month"/"Day"/"Hour"/"Minute" as ints; we coerce if needed
for col in ["Year", "Month", "Day", "Hour", "Minute"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["datetime"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=df["Day"],
        hour=df["Hour"],
        minute=df["Minute"]
    ),
    errors="coerce",
)

df = df.dropna(subset=["datetime"]).sort_values("datetime")

# -----------------------------
# 3. Select only useful columns
# -----------------------------

core_and_optional = [
    # Time/meta
    "Station_ID", "Station_name", "Year", "Month", "Day", "Hour", "Minute",
    "Latitude", "Longitude", "Elevation",
    # Core weather
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "sea_level_pressure",
    "station_level_pressure",
    # Optional weather
    "visibility",
    "wind_gust",
    "wet_bulb_temperature",
    "altimeter",
    "pressure_3hr_change",
    # Optional precip windows
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_9_hour",
    "precipitation_12_hour",
    "precipitation_15_hour",
    "precipitation_18_hour",
    "precipitation_21_hour",
    "precipitation_24_hour",
    # Optional cloud cover (state, not base heights)
    "sky_cover_1",
    "sky_cover_2",
    "sky_cover_3",
]

# Keep only columns that actually exist in this station file
keep_cols = [c for c in core_and_optional if c in df.columns]
keep_cols = ["datetime"] + keep_cols  # make sure datetime is included

df = df[keep_cols].copy()

# -----------------------------
# 4. Convert units to interpretable ones
# -----------------------------
# According to GHCNh documentation:
# - temperature, dew_point_temperature, wet_bulb_temperature: tenths of °C
# - precipitation & precipitation_X_hour: mm
# - relative_humidity: percent
# - wind_speed, wind_gust: m/s
# - pressures: hPa
# - visibility: km

for col in ["temperature", "dew_point_temperature", "wet_bulb_temperature"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 10.0

for col in [
    "precipitation",
    "precipitation_3_hour", "precipitation_6_hour",
    "precipitation_9_hour", "precipitation_12_hour",
    "precipitation_15_hour", "precipitation_18_hour",
    "precipitation_21_hour", "precipitation_24_hour",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in [
    "relative_humidity",
    "wind_speed", "wind_gust",
    "sea_level_pressure", "station_level_pressure",
    "visibility",
    "altimeter",
    "pressure_3hr_change",
    "wind_direction",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 5. Aggregate hourly → weekly
# -----------------------------
# We’ll use weeks starting Monday to match ISO-like weeks (W-MON).
# First, set datetime as index.
df = df.set_index("datetime")

# Define aggregation rules:
agg_dict = {}

# Mean-type variables
for col in [
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",          # note: linear mean, not circular; OK as a rough summary
    "sea_level_pressure",
    "station_level_pressure",
    "visibility",
    "wind_gust",
    "wet_bulb_temperature",
    "altimeter",
    "pressure_3hr_change",
]:
    if col in df.columns:
        agg_dict[col] = "mean"

# Sum-type variables (precipitation)
for col in [
    "precipitation",
    "precipitation_3_hour", "precipitation_6_hour",
    "precipitation_9_hour", "precipitation_12_hour",
    "precipitation_15_hour", "precipitation_18_hour",
    "precipitation_21_hour", "precipitation_24_hour",
]:
    if col in df.columns:
        agg_dict[col] = "sum"

# For station metadata, we can take the first record of the week (it won't change)
for col in ["Station_ID", "Station_name", "Latitude", "Longitude", "Elevation"]:
    if col in df.columns:
        agg_dict[col] = "first"

weekly = df.resample("W-MON").agg(agg_dict).reset_index()
weekly = weekly.rename(columns={"datetime": "iso_weekstartdate"})

# -----------------------------
# 6. Create ISO year-week code (to match FluNet)
# -----------------------------
iso = weekly["iso_weekstartdate"].dt.isocalendar()
weekly["iso_year"] = iso["year"]
weekly["iso_week"] = iso["week"]
weekly["isoyw"] = weekly["iso_year"] * 100 + weekly["iso_week"]

# Final column ordering (you can adjust)
cols_order = (
    ["iso_weekstartdate", "iso_year", "iso_week", "isoyw"] +
    [c for c in weekly.columns if c not in ["iso_weekstartdate", "iso_year", "iso_week", "isoyw"]]
)
weekly = weekly[cols_order]

print(weekly.head())

# Optionally, save to CSV for merging with FluNet
weekly.to_csv("aruba_weather_weekly.csv", index=False)
