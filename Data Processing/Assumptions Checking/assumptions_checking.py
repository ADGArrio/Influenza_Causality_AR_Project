# %%
"""

assumptions_checking.py

Diagnostics for a VAR model used in the Influenza Causality AR Project.
Checks:
- Linearity
- Stationarity
- Stability
- Homoscedasticity / residual autocorrelation
- Multicollinearity
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
# %%
# --------------------------------------------------------------------
# Stationarity
# --------------------------------------------------------------------

def check_stationarity(df, alpha=0.05):
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


# --------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------

def check_stability(var_results):
    """
    Check stability of VAR using roots of the characteristic polynomial.

    Stability requires all |roots| < 1.
    """
    print("\n=== Stability (roots of VAR) ===")
    print("stability applies to fitted models; skipping.")


# --------------------------------------------------------------------
# Homoscedasticity & residual autocorrelation
# --------------------------------------------------------------------

def check_variance_stability(df, alpha=0.05):
    print("\n=== Variance Stability / Homoscedasticity ===")
    for col in df.columns:
        y = df[col].dropna()
        arch_stat, arch_p, _, _ = het_arch(y)
        print(f"{col}: ARCH p={arch_p:.4f}")
        if arch_p < alpha:
            print("  -> Variance changes over time.")


# --------------------------------------------------------------------
# Multicollinearity (VIF)
# --------------------------------------------------------------------

def check_multicollinearity(df, vif_thresh=10.0):
    print("\n=== Multicollinearity (VIF on raw data) ===")
    clean = df.dropna()
    X = sm.add_constant(clean)
    for i, col in enumerate(["const"] + list(clean.columns)):
        if col == "const": continue
        v = variance_inflation_factor(X.values, i)
        print(f"{col}: VIF={v:.2f}")


# --------------------------------------------------------------------
# Linearity (RESET-style check)
# --------------------------------------------------------------------

def check_linearity(df, lags=3, alpha=0.05):
    print("\n=== Linearity (RESET-like on raw series) ===")
    for col in df.columns:
        y = df[col].dropna()
        # Build lag matrix
        lagged = pd.concat([y.shift(i) for i in range(1, lags+1)], axis=1)
        lagged.columns = [f"{col}_lag{i}" for i in range(1, lags+1)]
        data = pd.concat([y, lagged], axis=1).dropna()
        y_clean = data[col]
        X = sm.add_constant(data.drop(columns=[col]))
        model = sm.OLS(y_clean, X).fit()
        reset = linear_reset(model, power=2, use_f=True)
        f_stat, p_value, df_denom, df_num = reset
        print(f"{col}: RESET p={p_value:.4f}")
        if p_value < alpha:
            print("  -> Nonlinearity detected.")

# %%
# --------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------

def run_all_checks(csv_path, columns=None, alpha=0.05):
    df = pd.read_csv(csv_path)

    # Optional: keep only selected columns first
    if columns:
        df = df[columns]

    # Keep only numeric columns for the tests
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].dropna()

    print("Using numeric columns for diagnostics:", list(df.columns))
    print("Data shape after filtering numeric cols and dropping NA:", df.shape)

    check_stationarity(df, alpha)
    check_linearity(df)
    check_variance_stability(df, alpha)
    check_multicollinearity(df)

# %% For Running Linearity Check Only
csv_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Data Processing/Weather Data Integration/flunet_noaa_features.csv"

columns = None
df = pd.read_csv(csv_path)
if columns:
    df = df[columns]
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols].dropna()
print("Using numeric columns for diagnostics:", list(df.columns))
print("Data shape after filtering numeric cols and dropping NA:", df.shape)
check_linearity(df)

# %% For Running All Checks
if __name__ == "__main__":

    csv_path = "/Users/adgarrio/go/src/Semester Project/Influenza_Causality_AR_Project/Data Processing/Weather Data Integration/flunet_noaa_features.csv"

    columns = None

    run_all_checks(
        csv_path=csv_path,
        columns=columns,
        alpha=0.05,
    )
    
# %%