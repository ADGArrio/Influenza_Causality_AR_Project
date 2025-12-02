# %%
import pandas as pd
import numpy as np

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------

weather_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Data Processing/WeatherDataReal/Singapore/singapore_weather_weekly_last25yrs.csv"
flu_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/Raw Data/VIW_FNT_Cleaned.csv"

weather = pd.read_csv(weather_path)
flu = pd.read_csv(flu_path)

# Filter FluNet data to Singapore only
if "country_code" in flu.columns:
    flu = flu[flu["country_code"].astype(str).str.upper() == "SGP"].copy()
    print("Flu rows after Singapore filter:", flu.shape)
else:
    print("[WARN] 'country_code' column not found in flu data; no country filter applied.")

print("Weather cols:", weather.columns.tolist())
print("Flu cols:", flu.columns.tolist())

# %%
# -----------------------------------------------------
# 2. DROP UNNECESSARY WEATHER COLUMNS
# -----------------------------------------------------

DROP_COLS = [
    "wind_gust",
    "pressure_3hr_change",
    "station_id",
    "wind_dir",
    "wind_dir_deg",
    "dew_point",
    "precipitation",
]

weather = weather.drop(columns=[c for c in DROP_COLS if c in weather.columns])


likely_relevant = [
    "iso_week",
    "relative_humidity",
    "wind_speed",
    "visibility",
    "dew_point_temperature",
    "temperature",
    "iso_year"
]

weather = weather[[c for c in likely_relevant if c in weather.columns]]

# Sort by ISO year and week to ensure proper temporal order
if "iso_year" in weather.columns and "iso_week" in weather.columns:
    weather = weather.sort_values(["iso_year", "iso_week"])

# Interpolate weather variables to fill small gaps
for col in ["temperature", "relative_humidity", "wind_speed", "dew_point_temperature", "visibility"]:
    if col in weather.columns:
        weather[col] = weather[col].interpolate(method="linear", limit_direction="both")

print("Weather after cleaning:", weather.columns.tolist())

# %%
# -----------------------------------------------------
# 3. ALIGN MERGING KEYS USING ISO WEEK
# -----------------------------------------------------


flu["iso_year"] = pd.to_numeric(flu["iso_year"], errors="coerce")
flu["iso_week"] = pd.to_numeric(flu["iso_week"], errors="coerce")
flu = flu[flu["iso_year"].notna() & flu["iso_week"].notna()].copy()
flu["iso_year"] = flu["iso_year"].astype(int)
flu["iso_week"] = flu["iso_week"].astype(int)

# Sort Singapore flu data by ISO year and week and interpolate counts
flu = flu.sort_values(["iso_year", "iso_week"])
for col in ["inf_all", "inf_a", "inf_b"]:
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

# %%
# -----------------------------------------------------
# 4. APPLY LOG TRANSFORMATIONS
# -----------------------------------------------------

FLU_COUNT_COLS = ["inf_all", "inf_a", "inf_b"]
LOG_TRANSFORM_COLS = FLU_COUNT_COLS

def add_log_transforms(df: pd.DataFrame, cols_to_log) -> pd.DataFrame:
    df = df.copy()
    for col in cols_to_log:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found. Skipping log transform.")
            continue

        new_col = f"{col}_log"
        df[new_col] = np.log1p(df[col])
        print(f"[INFO] Added log transform: {new_col}")

    return df

merged = add_log_transforms(merged, LOG_TRANSFORM_COLS)

# %%
# -----------------------------------------------------
# 5. OPTIONAL — Add first differences for VAR
# -----------------------------------------------------

for col in FLU_COUNT_COLS:
    merged[f"{col}_diff"] = merged[col].diff()

KEEP_COLS = [
    "iso_year", "iso_week",

    # Flu (use only log-transformed total flu)
    "inf_all_log",

    # Weather variables
    "temperature",
    "relative_humidity",
    "wind_speed",
    "dew_point_temperature",
    "visibility"
]

merged = merged[KEEP_COLS]
print("Final dataset shape:", merged.shape)
print("Final columns:", merged.columns.tolist())

# %%
# -----------------------------------------------------
# 6. SAVE OUTPUT TO CSV
# -----------------------------------------------------
output_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/Final_Training_Data/Singapore/SG_Training_Data_With_ExtraCols.csv"
merged.to_csv(output_path, index=False)
print(f"Saved merged dataset to: {output_path}")

# %%