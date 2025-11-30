# %%
import os
import numpy as np
import pandas as pd

# %%
# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

BASE_DIR = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files"

# Input Paths
INDIA_IN  = os.path.join(BASE_DIR, "Country Subsets with Transforms/India_Data.csv")
CHINA_IN  = os.path.join(BASE_DIR, "Country Subsets with Transforms/China_Data.csv")
USA_IN    = os.path.join(BASE_DIR, "Country Subsets with Transforms/USA_Data.csv")

# Output Paths
INDIA_OUT = os.path.join(BASE_DIR, "Final_Training_Data/India_Training_Data.csv")
CHINA_OUT = os.path.join(BASE_DIR, "Final_Training_Data/China_Training_Data.csv")
USA_OUT   = os.path.join(BASE_DIR, "Final_Training_Data/USA_Training_Data.csv")

# Column that encodes time ordering
TIME_COL = "isoyw"  # ISO year-week format

# Flu count columns and their log names
FLU_COLS = ["inf_all", "inf_a", "inf_b"]
FLU_LOG_COLS = ["inf_all_log", "inf_a_log", "inf_b_log"]

# %%
# --------------------------------------------------------------------
# Helper: load and sanity check
# --------------------------------------------------------------------

def load_country_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {path} with shape {df.shape}")
    return df

# %%
# --------------------------------------------------------------------
# Helper: ensure log columns exist
# --------------------------------------------------------------------

def ensure_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for raw, log in zip(FLU_COLS, FLU_LOG_COLS):
        if log not in df.columns:
            if raw not in df.columns:
                print(f"[WARN] Neither {raw} nor {log} found. Skipping.")
                continue
            df[log] = np.log1p(df[raw])
            print(f"[INFO] Created {log} = log1p({raw})")
    return df

# %%
# --------------------------------------------------------------------
# Helper: build VAR-ready dataset
# --------------------------------------------------------------------

def build_var_ready(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # If we have a time column, use it to sort and keep it for reference
    if TIME_COL in df.columns:
        df = df.sort_values(by=TIME_COL)
    else:
        print(f"[WARN] TIME_COL '{TIME_COL}' not found. Using existing order.")

    # Make sure log columns exist
    df = ensure_log_columns(df)

    # Create differenced log flu variables (improve stationarity)
    for log_col in ["inf_a_log", "inf_b_log"]:
        if log_col in df.columns:
            diff_col = f"{log_col}_diff"
            df[diff_col] = df[log_col].diff()
            print(f"[INFO] Created {diff_col} = diff({log_col})")
        else:
            print(f"[WARN] {log_col} not found. Cannot create diff.")

    # Drop the first row with NaNs introduced by differencing
    df = df.dropna().reset_index(drop=True)

    keep_cols = []

    # Time column for reference
    # if TIME_COL in df.columns:
    #     keep_cols.append(TIME_COL)

    # differenced logs
    for c in ["inf_a_log_diff", "inf_b_log_diff"]:
        if c in df.columns:
            keep_cols.append(c)
        else:
            print(f"[WARN] {c} not found in dataframe.")

    # weather & season
    for c in ["avg_temp", "humidity", "season_sin", "season_cos"]:
        if c in df.columns:
            keep_cols.append(c)
        else:
            print(f"[WARN] {c} not found in dataframe.")

    var_df = df[keep_cols].copy()
    print("[INFO] VAR-ready columns:", keep_cols)
    print("[INFO] VAR-ready shape:", var_df.shape)
    return var_df

# %%
# --------------------------------------------------------------------
# Helper: process and save one country
# --------------------------------------------------------------------

def process_country(in_path: str, out_path: str, label: str):
    print("\n" + "="*70)
    print(f"[INFO] Processing country: {label}")
    print("="*70)

    df = load_country_df(in_path)
    var_df = build_var_ready(df)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    var_df.to_csv(out_path, index=False)
    print(f"[OK] Saved VAR-ready dataset for {label} to: {out_path}")

# %%
# --------------------------------------------------------------------
# Run for all three countries
# --------------------------------------------------------------------

process_country(INDIA_IN, INDIA_OUT, "India")
process_country(CHINA_IN, CHINA_OUT, "China")
process_country(USA_IN,   USA_OUT,   "USA")

print("\n[DONE] All VAR-ready datasets created.")

# %%