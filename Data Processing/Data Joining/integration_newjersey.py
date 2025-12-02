# %%
import pandas as pd
import numpy as np

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------

weather_path = "/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Raw Data/newjersey/newjersey_weather_weekly_last25yrs.csv"
flu_path = "/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Raw Data/Final_FluNet.csv"

weather = pd.read_csv(weather_path)
flu = pd.read_csv(flu_path)

# Filter FluNet data to USA only (handle both uppercase and lowercase column names)
# Note: New Jersey is part of USA, FluNet has country-level data
country_col = "COUNTRY_CODE" if "COUNTRY_CODE" in flu.columns else "country_code"
if country_col in flu.columns:
    flu = flu[flu[country_col].astype(str).str.upper() == "USA"].copy()
    print("Flu rows after USA filter:", flu.shape)
else:
    print("[WARN] Country code column not found in flu data; no country filter applied.")

print("Weather cols:", weather.columns.tolist())
print("Flu cols:", flu.columns.tolist())

# %%
# -----------------------------------------------------
# 2. DROP UNNECESSARY WEATHER COLUMNS
# -----------------------------------------------------

# Columns to drop - metadata and redundant/high-NaN columns
# New Jersey has higher NaN for: wind_gust (50%), visibility (44%), altimeter (40%)
DROP_COLS = [
    "wind_gust",              # 50% NaN
    "visibility",             # 44% NaN
    "altimeter",              # 40% NaN
    "pressure_3hr_change",    # Keep - 0% NaN for NJ
    "station_id",
    "Station_ID",
    "Station_name",
    "wind_dir",
    "wind_dir_deg",
    "wind_direction",         # Circular variable, harder to use in VAR
    "Latitude",
    "Longitude",
    "Elevation",
    "station_level_pressure", # Redundant with sea_level_pressure
    "isoyw",
    "iso_weekstartdate",
    # Drop hourly precipitation breakdowns, keep daily total
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_9_hour",
    "precipitation_12_hour",
    "precipitation_15_hour",
    "precipitation_18_hour",
    "precipitation_21_hour",
]

weather = weather.drop(columns=[c for c in DROP_COLS if c in weather.columns])

# Relevant weather features for influenza VAR analysis
likely_relevant = [
    "iso_year",
    "iso_week",
    # Core meteorological variables
    "temperature",
    "dew_point_temperature",  # Affects virus survival in aerosols
    "relative_humidity",
    "wind_speed",
    # Atmospheric conditions
    "sea_level_pressure",     # Barometric pressure affects respiratory health
    "wet_bulb_temperature",   # Heat stress metric
    # Precipitation
    "precipitation",          # Instant precipitation
    "precipitation_24_hour",  # Daily cumulative rainfall
]

weather = weather[[c for c in likely_relevant if c in weather.columns]]

# Sort by ISO year and week to ensure proper temporal order
if "iso_year" in weather.columns and "iso_week" in weather.columns:
    weather = weather.sort_values(["iso_year", "iso_week"])

# Interpolate weather variables to fill small gaps
INTERPOLATE_COLS = [
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    "sea_level_pressure",
    "wet_bulb_temperature",
    "precipitation",
    "precipitation_24_hour",
]
for col in INTERPOLATE_COLS:
    if col in weather.columns:
        weather[col] = weather[col].interpolate(method="linear", limit_direction="both")

print("Weather after cleaning:", weather.columns.tolist())

# %%
# -----------------------------------------------------
# 3. ALIGN MERGING KEYS USING ISO WEEK
# -----------------------------------------------------

# Handle uppercase column names in flu data
iso_year_col = "ISO_YEAR" if "ISO_YEAR" in flu.columns else "iso_year"
iso_week_col = "ISO_WEEK" if "ISO_WEEK" in flu.columns else "iso_week"

flu["iso_year"] = pd.to_numeric(flu[iso_year_col], errors="coerce")
flu["iso_week"] = pd.to_numeric(flu[iso_week_col], errors="coerce")
flu = flu[flu["iso_year"].notna() & flu["iso_week"].notna()].copy()
flu["iso_year"] = flu["iso_year"].astype(int)
flu["iso_week"] = flu["iso_week"].astype(int)

# Sort flu data by ISO year and week and interpolate counts
flu = flu.sort_values(["iso_year", "iso_week"])

# Handle uppercase INF columns
inf_a_col = "INF_A" if "INF_A" in flu.columns else "inf_a"
inf_b_col = "INF_B" if "INF_B" in flu.columns else "inf_b"

for col in [inf_a_col, inf_b_col]:
    if col in flu.columns:
        flu[col] = flu[col].interpolate(method="linear", limit_direction="both")

weather["iso_year"] = pd.to_numeric(weather["iso_year"], errors="coerce")
weather["iso_week"] = pd.to_numeric(weather["iso_week"], errors="coerce")
weather = weather[weather["iso_year"].notna() & weather["iso_week"].notna()].copy()
weather["iso_year"] = weather["iso_year"].astype(int)
weather["iso_week"] = weather["iso_week"].astype(int)

# Ensure exactly one weather row per (iso_year, iso_week)
value_cols = [c for c in weather.columns if c not in ["iso_year", "iso_week"]]
weather_weekly = (
    weather
    .groupby(["iso_year", "iso_week"], as_index=False)[value_cols]
    .mean()
)

# Merge on iso_year + iso_week
merged = pd.merge(
    flu,
    weather_weekly,
    on=["iso_year", "iso_week"],
    how="inner",
    validate="many_to_one"
)

print("Merged shape:", merged.shape)

# -----------------------------------------------------
# REMOVE DATE/TIME COLUMNS (NOT USED FOR VAR)
# -----------------------------------------------------
DATE_COLS = [
    "iso_weekstartdate",
    "ISO_WEEKSTARTDATE",
    "mmwr_weekstartdate",
    "mmwr_year",
    "mmwr_week",
    "isoyw",
    "mmwryw",
    "iso_year",
    "iso_week",
    "ISO_YEAR",
    "ISO_WEEK",
    "COUNTRY_CODE",
    "country_code",
    "COUNTRY_AREA_TERRITORY",
]

merged = merged.drop(columns=[c for c in DATE_COLS if c in merged.columns])
print("Removed date/time columns. Remaining cols:", merged.columns.tolist())

# %%
# -----------------------------------------------------
# 4. HANDLE LOG TRANSFORMATIONS
# -----------------------------------------------------

# Check if LOG columns already exist, otherwise create them
log_inf_a_col = "LOG_INF_A" if "LOG_INF_A" in merged.columns else None
log_inf_b_col = "LOG_INF_B" if "LOG_INF_B" in merged.columns else None

if log_inf_a_col is None:
    # Create LOG_INF_A if it doesn't exist
    if inf_a_col in merged.columns:
        merged["LOG_INF_A"] = np.log1p(merged[inf_a_col])
        print("[INFO] Created LOG_INF_A from", inf_a_col)
    else:
        print("[WARN] Cannot create LOG_INF_A - source column not found")
else:
    print("[INFO] LOG_INF_A already exists in data")

if log_inf_b_col is None:
    # Create LOG_INF_B if it doesn't exist
    if inf_b_col in merged.columns:
        merged["LOG_INF_B"] = np.log1p(merged[inf_b_col])
        print("[INFO] Created LOG_INF_B from", inf_b_col)
    else:
        print("[WARN] Cannot create LOG_INF_B - source column not found")
else:
    print("[INFO] LOG_INF_B already exists in data")

print("Final dataset shape:", merged.shape)
print("Final columns:", merged.columns.tolist())

# %%
# -----------------------------------------------------
# 5. CREATE TWO SEPARATE DATASETS (INF_A and INF_B)
# -----------------------------------------------------

# Common weather columns for both datasets
# Note: No visibility for NJ due to high NaN
WEATHER_COLS = [
    # Core meteorological
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    # Atmospheric conditions
    "sea_level_pressure",
    "wet_bulb_temperature",
    # Precipitation
    "precipitation",
    "precipitation_24_hour",
]

# Dataset with LOG_INF_A
KEEP_COLS_A = ["LOG_INF_A"] + WEATHER_COLS
merged_a = merged[[c for c in KEEP_COLS_A if c in merged.columns]].copy()
print("\n--- INF_A Dataset ---")
print("Shape:", merged_a.shape)
print("Columns:", merged_a.columns.tolist())

# Dataset with LOG_INF_B
KEEP_COLS_B = ["LOG_INF_B"] + WEATHER_COLS
merged_b = merged[[c for c in KEEP_COLS_B if c in merged.columns]].copy()
print("\n--- INF_B Dataset ---")
print("Shape:", merged_b.shape)
print("Columns:", merged_b.columns.tolist())

# %%
# -----------------------------------------------------
# 6. SAVE OUTPUTS TO CSV
# -----------------------------------------------------

output_dir = "/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/NewJersey"

# Create output directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

output_path_a = f"{output_dir}/NJ_Training_Data_INF_A.csv"
merged_a.to_csv(output_path_a, index=False)
print(f"\nSaved INF_A dataset to: {output_path_a}")

output_path_b = f"{output_dir}/NJ_Training_Data_INF_B.csv"
merged_b.to_csv(output_path_b, index=False)
print(f"Saved INF_B dataset to: {output_path_b}")

# %%

