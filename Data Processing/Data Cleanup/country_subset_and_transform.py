# %%
import os
import numpy as np
import pandas as pd

# %%
# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

csv_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/FluNet_Training_Data.csv"

# Column name that identifies country
COUNTRY_COL = "country_code"

# Output folder for per-country CSVs 
OUTPUT_DIR = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/country_subsets"

# Flu count columns to log-transform 
FLU_COUNT_COLS = ["inf_all", "inf_a", "inf_b"]

# Columns to log-transform 
LOG_TRANSFORM_COLS = FLU_COUNT_COLS

# %%
# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame and check for the country column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if COUNTRY_COL not in df.columns:
        raise KeyError(
            f"Country column '{COUNTRY_COL}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def filter_country(df: pd.DataFrame, country_col: str, target_names) -> pd.DataFrame:
    """
    Filter the DataFrame for rows where `country_col` matches any of the target_names.
    """
    col_str = df[country_col].astype(str).str.upper()
    targets_upper = {name.upper() for name in target_names}
    mask = col_str.isin(targets_upper)
    return df[mask].copy()


def add_log_transforms(df: pd.DataFrame, cols_to_log) -> pd.DataFrame:
    """
    For each column in cols_to_log that exists in df, add a log1p-transformed version.
    """
    df = df.copy()
    for col in cols_to_log:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found in DataFrame. Skipping log transform.")
            continue

        new_col = f"{col}_log"
        # Use log1p to safely handle zeros: log1p(x) = log(1 + x)
        df[new_col] = np.log1p(df[col])

        print(f"[INFO] Added log transform: {new_col} = log1p({col})")

    return df


def save_country_subset(df_country: pd.DataFrame, country_label: str, base_name: str):
    """
    Save a country subset to CSV.
    """
    if df_country.empty:
        print(f"[WARN] No rows found for {country_label}. Not writing a file.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{country_label}_log.csv")
    df_country.to_csv(out_path, index=False)
    print(f"[OK] Wrote {country_label} subset with log transforms to: {out_path}")

# %%
# --------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------

df = load_data(csv_path)
print("[INFO] Loaded data.")
print("[INFO] Data shape:", df.shape)
print("[INFO] Columns:", list(df.columns))

base_name = os.path.splitext(os.path.basename(csv_path))[0]

# %%
# --------------------------------------------------------------------
# Subset for USA, China, India
# --------------------------------------------------------------------

# US, India, China Splits
usa_targets = ["USA", "US", "UNITED STATES", "UNITED STATES OF AMERICA"]
china_targets = ["CHN", "CN", "CHINA"]
india_targets = ["IND", "IN", "INDIA"]

df_usa = filter_country(df, COUNTRY_COL, usa_targets)
df_china = filter_country(df, COUNTRY_COL, china_targets)
df_india = filter_country(df, COUNTRY_COL, india_targets)

print(f"[INFO] USA rows   : {df_usa.shape[0]}")
print(f"[INFO] China rows : {df_china.shape[0]}")
print(f"[INFO] India rows : {df_india.shape[0]}")

if df_usa.empty:
    print("No USA rows found.")
if df_china.empty:
    print("No China rows found.")
if df_india.empty:
    print("No India rows found.")

# %% Log Transforms for Variance Stabilization
df_usa_log = add_log_transforms(df_usa, LOG_TRANSFORM_COLS)
df_china_log = add_log_transforms(df_china, LOG_TRANSFORM_COLS)
df_india_log = add_log_transforms(df_india, LOG_TRANSFORM_COLS)

# %% Saving Outputs
save_country_subset(df_usa_log, "USA", base_name)
save_country_subset(df_china_log, "China", base_name)
save_country_subset(df_india_log, "India", base_name)

print("[DONE] Country splits + log transforms complete.")
# %%
