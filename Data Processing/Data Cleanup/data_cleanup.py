"""
WHO FluNet (VIW_FNT) - Basic Cleanup
------------------------------------
- Standardize column names (snake_case)
- Parse dates (ISO/MMWR)
- Normalize identifiers (trim/uppercase)
- Coerce numeric columns
- Create simple derived features:
    * pct_pos_influenza = 100 * inf_all / spec_processed_nb
    * share_a_among_pos, share_b_among_pos
    * subtype shares among A (e.g., share_ah3_among_a)
- Build consistent weekly key isoyw (YYYYWW)
- Drop easy duplicates on [country_code, isoyw, origin_source]
- Add a couple QC flags
- Save cleaned outputs

Requirements: pandas, numpy. (Optional for Parquet: fastparquet or pyarrow)
"""

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- Config ----------
INPUT_CSV  = "VIW_FNT.csv"          # change if needed
OUTPUT_CSV = "VIW_FNT_Cleaned.csv"
OUTPUT_PARQUET = "VIW_FNT_Cleaned.parquet"  # will try to write if engine available

# %%
# ---------- Helpers ----------
def clean_cols(cols):
    """Standardize column names to snake_case."""
    def fix(c):
        c = str(c).strip()
        c = c.replace("/", "_per_")
        c = re.sub(r"[^\w]+", "_", c)     # non-word -> underscore
        c = re.sub(r"_+", "_", c).strip("_")
        return c.lower()
    return [fix(c) for c in cols]

def safe_div(num, den):
    """Vectorized safe division returning NaN when denominator <= 0 or missing."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
    return pd.Series(out, index=num.index)

# Columns that should stay as text even if they sometimes look numeric
TEXT_COLS = {
    "whoregion","fluseason","hemisphere","itz",
    "country_code","country_area_territory","iso2",
    "origin_source","aother_subtype_details","other_respvirus_details",
    "lab_result_comment","wcr_comment"
}
DATE_COLS = {"iso_weekstartdate","mmwr_weekstartdate"}

# %%
# ---------- Load ----------
df = pd.read_csv(INPUT_CSV, dtype=str)   # start as strings; we’ll coerce later
df.columns = clean_cols(df.columns)

# ---------- Parse dates ----------
for dcol in ["iso_weekstartdate", "mmwr_weekstartdate"]:
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce", utc=True)

# ---------- Normalize identifiers ----------
for c in ["country_code","iso2","whoregion","hemisphere","fluseason","itz","origin_source"]:
    if c in df.columns:
        df[c] = (
            df[c].astype(str)
                 .str.strip()
                 .str.upper()
                 .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )

# %%
# ---------- Coerce numerics ----------
candidate_num = [c for c in df.columns if c not in TEXT_COLS and c not in DATE_COLS]
# first attempt
for c in candidate_num:
    df[c] = pd.to_numeric(df[c], errors="ignore")
# second pass for object columns with commas/spaces
for c in candidate_num:
    if df[c].dtype == "object":
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Build isoyw if missing ----------
if "isoyw" not in df.columns or df["isoyw"].isna().all():
    if "iso_weekstartdate" in df.columns:
        iso = df["iso_weekstartdate"].dt.isocalendar()
        df["isoyw"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).astype("Int64")
else:
    df["isoyw"] = pd.to_numeric(df["isoyw"], errors="coerce").astype("Int64")

# ---------- Light de-duplication ----------
subset_keys = [k for k in ["country_code", "isoyw", "origin_source"] if k in df.columns]
before = len(df)
if subset_keys:
    df = df.drop_duplicates(subset=subset_keys, keep="last")
removed = before - len(df)

# %%
# ---------- Derived features ----------
# Percent positive among processed
if set(["inf_all","spec_processed_nb"]).issubset(df.columns):
    df["pct_pos_influenza"] = 100 * safe_div(df["inf_all"], df["spec_processed_nb"])

# A vs B shares among positives
if set(["inf_a","inf_b","inf_all"]).issubset(df.columns):
    df["share_a_among_pos"] = safe_div(df["inf_a"], df["inf_all"])
    df["share_b_among_pos"] = safe_div(df["inf_b"], df["inf_all"])

# Subtype shares among influenza A
for sub in ["ah1n12009","ah1","ah3","ah5","ah7n9","aother_subtype","anotsubtyped","anotsubtypable"]:
    if "inf_a" in df.columns and sub in df.columns:
        df[f"share_{sub}_among_a"] = safe_div(df[sub], df["inf_a"])

# ---------- Simple QC flags ----------
# Negative > processed (shouldn’t happen)
if set(["inf_negative","spec_processed_nb"]).issubset(df.columns):
    neg = pd.to_numeric(df["inf_negative"], errors="coerce")
    proc = pd.to_numeric(df["spec_processed_nb"], errors="coerce")
    df["qc_neg_gt_processed"] = (neg > proc)

# INF_A + INF_B > INF_ALL (type sum exceeds total positives)
if set(["inf_a","inf_b","inf_all"]).issubset(df.columns):
    a = pd.to_numeric(df["inf_a"], errors="coerce").fillna(0)
    b = pd.to_numeric(df["inf_b"], errors="coerce").fillna(0)
    allp = pd.to_numeric(df["inf_all"], errors="coerce")
    df["qc_type_sum_gt_all"] = (a + b > allp)

# %%
# ---------- Save ----------
df.to_csv(OUTPUT_CSV, index=False)
saved_parquet = False
parquet_msg = ""
try:
    # tries pyarrow first, then fastparquet if available
    df.to_parquet(OUTPUT_PARQUET, index=False)
    saved_parquet = True
except Exception as e:
    parquet_msg = f"(Parquet not saved: {e})"

# ---------- Report ----------
print(f"Rows after cleanup: {len(df):,}  | Duplicates removed: {removed:,}")
print(f"Saved CSV -> {Path(OUTPUT_CSV).resolve()}")
if saved_parquet:
    print(f"Saved Parquet -> {Path(OUTPUT_PARQUET).resolve()}")
else:
    print(parquet_msg or "Parquet not saved.")
# %%
