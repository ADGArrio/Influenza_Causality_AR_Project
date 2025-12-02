import pandas as pd
import numpy as np

# --------------------------------------------------
# 0. Config
# --------------------------------------------------

STATION_FILES = {
    "Jakarta":  "./GHCNh_IDU096749-1_por.psv",
    "Surabaya": "./GHCNh_IDU096937-1_por.psv",
    "Bandung":  "./GHCNh_IDA00967830_por.psv",
}

DATE_MIN = "2000-01-01"
DATE_MAX = "2025-12-31"

USECOLS = [
    "Station_ID", "Station_name", "Year", "Month", "Day", "Hour", "Minute",
    "Latitude", "Longitude", "Elevation",
    "temperature", "dew_point_temperature", "relative_humidity",
    "precipitation", "wind_speed", "wind_direction",
    "sea_level_pressure", "station_level_pressure",
    "visibility", "wind_gust", "wet_bulb_temperature",
    "altimeter", "pressure_3hr_change",
    "precipitation_3_hour", "precipitation_6_hour", "precipitation_9_hour",
    "precipitation_12_hour", "precipitation_15_hour", "precipitation_18_hour",
    "precipitation_21_hour", "precipitation_24_hour",
    "sky_cover_1", "sky_cover_2", "sky_cover_3",
]


# --------------------------------------------------
# 1. Load and clean hourly PSV files
# --------------------------------------------------

def load_and_clean_hourly(file_path):
    """
    Loads a PSV file (GHCNh) and:
    - Filters years 2000–2025
    - Converts units
    - Reindexes to continuous hourly timeline
    """
    chunks = []
    chunksize = 200_000

    for chunk in pd.read_csv(
        file_path,
        sep="|",
        usecols=lambda c: c in USECOLS or c in ["Year", "Month", "Day", "Hour", "Minute"],
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
        chunk = chunk[(chunk["Year"] >= 2000) & (chunk["Year"] <= 2025)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No data between 2000–2025 for {file_path}")

    df = pd.concat(chunks, ignore_index=True)

    # Build datetime
    for col in ["Month", "Day", "Hour", "Minute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=df["Day"],
            hour=df["Hour"],
            minute=df["Minute"],
        ),
        errors="coerce",
    )

    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    df = df[(df["datetime"] >= DATE_MIN) & (df["datetime"] <= DATE_MAX)]

    # Full hourly index (fills missing rows with NAs)
    full_index = pd.date_range(start=DATE_MIN, end=DATE_MAX, freq="h")
    df = df.set_index("datetime").reindex(full_index)
    df.index.name = "datetime"

    # Reduce to known columns
    keep_cols = [c for c in USECOLS if c in df.columns]
    df = df[keep_cols]

    # Unit conversions
    for col in ["temperature", "dew_point_temperature", "wet_bulb_temperature"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 10.0

    for col in [
        "precipitation", "precipitation_3_hour", "precipitation_6_hour",
        "precipitation_9_hour", "precipitation_12_hour", "precipitation_15_hour",
        "precipitation_18_hour", "precipitation_21_hour", "precipitation_24_hour",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "relative_humidity", "wind_speed", "wind_gust",
        "sea_level_pressure", "station_level_pressure", "visibility",
        "altimeter", "pressure_3hr_change", "wind_direction",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# 2. Hourly → Weekly
# --------------------------------------------------

def hourly_to_weekly(df_hourly):
    agg = {}

    # Means
    for col in [
        "temperature", "dew_point_temperature", "relative_humidity",
        "wind_speed", "wind_direction", "sea_level_pressure",
        "station_level_pressure", "visibility", "wind_gust",
        "wet_bulb_temperature", "altimeter", "pressure_3hr_change",
    ]:
        if col in df_hourly.columns:
            agg[col] = "mean"

    # Sums
    for col in [
        "precipitation", "precipitation_3_hour", "precipitation_6_hour",
        "precipitation_9_hour", "precipitation_12_hour", "precipitation_15_hour",
        "precipitation_18_hour", "precipitation_21_hour", "precipitation_24_hour",
    ]:
        if col in df_hourly.columns:
            agg[col] = "sum"

    # Metadata
    for col in ["Station_ID", "Station_name", "Latitude", "Longitude", "Elevation"]:
        if col in df_hourly.columns:
            agg[col] = "first"

    weekly = df_hourly.resample("W-MON").agg(agg).reset_index()
    weekly = weekly.rename(columns={"datetime": "iso_weekstartdate"})

    iso = weekly["iso_weekstartdate"].dt.isocalendar()
    weekly["iso_year"], weekly["iso_week"] = iso["year"], iso["week"]
    weekly["isoyw"] = weekly["iso_year"] * 100 + weekly["iso_week"]

    cols = ["iso_weekstartdate", "iso_year", "iso_week", "isoyw"]
    cols += [c for c in weekly.columns if c not in cols]
    return weekly[cols]


# --------------------------------------------------
# 3. Main
# --------------------------------------------------

def main():
    city_weekly = []

    for city, path in STATION_FILES.items():
        print(f"\n=== Processing {city} ===")

        df_hourly = load_and_clean_hourly(path)

        # ------------------- NaN Report (printed only) -------------------
        print(f"\nNaN summary for hourly data ({city}):")
        print(df_hourly.isna().sum().sort_values(ascending=False).head(20))

        weekly = hourly_to_weekly(df_hourly)
        weekly["city"] = city

        # ------------------- Save ONLY weekly -------------------
        out_path = f"indonesia_{city.lower()}_weekly_2000_2025.csv"
        weekly.to_csv(out_path, index=False)
        print(f"Saved weekly file: {out_path}")

        print(f"\nNaN summary for WEEKLY data ({city}):")
        print(weekly.isna().sum().sort_values(ascending=False).head(20))

        city_weekly.append(weekly)

    # ========= Combine Across Cities =========

    combined = pd.concat(city_weekly, ignore_index=True)

    ignore = [
        "iso_weekstartdate", "iso_year", "iso_week", "isoyw",
        "city", "Station_ID", "Station_name", "Latitude", "Longitude", "Elevation",
    ]
    numeric_cols = [c for c in combined.columns if c not in ignore]

    grouped = combined.groupby(["iso_year", "iso_week", "isoyw"], as_index=False)
    agg_spec = {"iso_weekstartdate": "first"}

    for col in numeric_cols:
        agg_spec[col] = "mean"

    # average metadata positions
    for col in ["Latitude", "Longitude", "Elevation"]:
        if col in combined.columns:
            agg_spec[col] = "mean"

    weekly_mean = grouped.agg(agg_spec)

    cols = ["iso_weekstartdate", "iso_year", "iso_week", "isoyw"]
    cols += [c for c in weekly_mean.columns if c not in cols]
    weekly_mean = weekly_mean[cols]

    # Save final averaged file
    weekly_mean.to_csv("indonesia_3cities_weekly_mean_2000_2025.csv", index=False)
    print("\nSaved averaged weekly file: indonesia_3cities_weekly_mean_2000_2025.csv")

    print("\nNaN summary for 3-city AVERAGED weekly:")
    print(weekly_mean.isna().sum().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()

