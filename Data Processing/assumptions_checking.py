# %%

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
# --------------------------------------------------------------------
# Stationarity
# --------------------------------------------------------------------

def check_stationarity(df, alpha=0.05, issues=None):
    """
    ADF unit root test for each series.

    If p-value > alpha -> non-stationary (consider differencing/detrending).
    """
    print("\n=== Stationarity (ADF tests) ===")
    for col in df.columns:
        series = df[col].dropna()
        res = adfuller(series, autolag='AIC')
        test_stat, p_value, used_lag, n_obs, crit, _ = res

        print(f"\nSeries: {col}")
        print(f"  ADF statistic: {test_stat:.3f}")
        print(f"  p-value      : {p_value:.4f}")
        print(f"  Used lags    : {used_lag}")
        print(f"  # of obs     : {n_obs}")
        print("  Critical values:")
        for k, v in crit.items():
            print(f"    {k}: {v:.3f}")

        if p_value < alpha:
            print(f"  -> Likely STATIONARY at {alpha} level.")
        else:
            print(f"  -> Likely NON-STATIONARY at {alpha} level.")
            print("     Remedy: detrend or difference this series.")
            if issues is not None:
                issues.append(f"[Stationarity] {col} appears NON-stationary (ADF p={p_value:.4f})")

# %%
# --------------------------------------------------------------------
# Homoscedasticity & residual autocorrelation
# --------------------------------------------------------------------

def check_variance_stability(df, alpha=0.05, issues=None):
    print("\n=== Variance Stability / Homoscedasticity ===")
    for col in df.columns:
        y = df[col].dropna()
        arch_stat, arch_p, _, _ = het_arch(y)
        print(f"{col}: ARCH p={arch_p:.4f}")
        if arch_p < alpha:
            print("  -> Variance changes over time.")
            if issues is not None:
                issues.append(f"[Variance] {col} shows time-varying variance (ARCH p={arch_p:.4f})")



# --------------------------------------------------------------------
# Multicollinearity (VIF)
# --------------------------------------------------------------------

def check_multicollinearity(df, vif_thresh=10.0, issues=None):
    print("\n=== Multicollinearity (VIF on raw data) ===")
    clean = df.dropna()
    X = sm.add_constant(clean)
    for i, col in enumerate(["const"] + list(clean.columns)):
        if col == "const":
            continue
        v = variance_inflation_factor(X.values, i)
        print(f"{col}: VIF={v:.2f}")
        if v > vif_thresh or np.isinf(v):
            if issues is not None:
                issues.append(f"[Multicollinearity] {col} has high VIF={v:.2f}")



# --------------------------------------------------------------------
# Linearity (RESET-style check)
# --------------------------------------------------------------------

def check_linearity(df, lags=3, alpha=0.05, issues=None):
    print("\n=== Linearity (RESET-like on raw series) ===")
    for col in df.columns:
        y = df[col].dropna()
        # Build lag matrix
        lagged = pd.concat([y.shift(i) for i in range(1, lags + 1)], axis=1)
        lagged.columns = [f"{col}_lag{i}" for i in range(1, lags + 1)]

        data = pd.concat([y, lagged], axis=1).dropna()
        if data.shape[0] <= lags + 2:
            print(f"{col}: not enough observations for RESET test, skipping.")
            continue

        y_clean = data[col]
        X = sm.add_constant(data.drop(columns=[col]))

        model = sm.OLS(y_clean, X).fit()
        reset_res = linear_reset(model, power=2, use_f=True)

        # ContrastResults attributes
        f_stat = reset_res.fvalue
        p_value = reset_res.pvalue
        df_num = reset_res.df_num
        df_denom = reset_res.df_denom

        print(f"{col}: RESET F={f_stat:.3f}, p={p_value:.4f} "
              f"(df_num={df_num}, df_denom={df_denom})")
        if p_value < alpha:
            print("  -> Nonlinearity detected.")
            if issues is not None:
                issues.append(f"[Linearity] {col} shows nonlinearity (RESET p={p_value:.4f})")
   
# %%
# --------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------

def run_all_checks(csv_path, columns=None, alpha=0.05):
    df = pd.read_csv(csv_path)

    if columns:
        df = df[columns]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].dropna()

    print("Using numeric columns for diagnostics:", list(df.columns))
    print("Data shape after filtering numeric cols and dropping NA:", df.shape)

    # Data Irregularity Issues Summary
    issues = []

    check_stationarity(df, alpha, issues=issues)
    check_linearity(df, alpha=alpha, issues=issues)
    check_variance_stability(df, alpha, issues=issues)
    check_multicollinearity(df, issues=issues)

    print("\n=== SUMMARY OF IRREGULARITIES ===")
    if not issues:
        print("No major assumption violations detected at the chosen thresholds.")
    else:
        for msg in issues:
            print("-", msg)

    report_path = csv_path.replace(".csv", "_Assumption_Report.txt")

    with open(report_path, "w") as f:
        f.write("Assumption Check Report\n")
        f.write("Source file: " + csv_path + "\n\n")

        if not issues:
            f.write("No major assumption violations detected.\n")
        else:
            for msg in issues:
                f.write(msg + "\n")

    print(f"\n[OK] Assumption report written to: {report_path}\n")


# %% For Running All Checks
def running_all_checks(country_name, country_path):
    print("\n" + "="*50)
    print("Running Checks for:", country_name)
    run_all_checks(
        csv_path=country_path,
        columns=None,
        alpha=0.05,
    )
    print("\n" + "="*50 + "\n")

# %% CSV Locations

india_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/Final_Training_Data/India_Training_Data.csv"
china_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/Final_Training_Data/China_Training_Data.csv"
usa_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/Final_Training_Data/USA_Training_Data.csv"
sg_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Data Processing/WeatherDataReal/Singapore/SG_Training_Dataset.csv"

# %% Running All Checks
running_all_checks("India", india_path)
print("India checks done.")

running_all_checks("China", china_path)
print("China checks done.")

running_all_checks("USA", usa_path)
print("USA checks done.")

# %%
running_all_checks("Singapore", sg_path)
print("Singapore checks done.")

# %% 





# %% For Running Distinct Checks Individually

# %% For Running Linearity Check Only
csv_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/FluNet_Training_Data.csv"

columns = None
df = pd.read_csv(csv_path)
if columns:
    df = df[columns]
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols].dropna()
print("Using numeric columns for diagnostics:", list(df.columns))
print("Data shape after filtering numeric cols and dropping NA:", df.shape)
check_linearity(df)

# %% For Running Variance Stability Check Only
csv_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/FluNet_Training_Data.csv"

columns = None
df = pd.read_csv(csv_path)
if columns:
    df = df[columns]
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols].dropna()
print("Using numeric columns for diagnostics:", list(df.columns))
print("Data shape after filtering numeric cols and dropping NA:", df.shape)
check_variance_stability(df, alpha=0.05)

# %% For Running Multicollinearity Check Only
csv_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Files/FluNet_Training_Data.csv"

columns = None
df = pd.read_csv(csv_path)
if columns:
    df = df[columns]
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols].dropna()
print("Using numeric columns for diagnostics:", list(df.columns))
print("Data shape after filtering numeric cols and dropping NA:", df.shape)
check_multicollinearity(df)

# %%