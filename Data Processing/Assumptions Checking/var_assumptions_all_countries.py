# %%
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define datasets to process
DATASETS = {
    'Singapore': {
        'inf_a': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/Singapore/SG_Training_Data_INF_A.csv',
        'inf_b': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/Singapore/SG_Training_Data_INF_B.csv',
        'output_dir': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/Singapore',
    },
    'Qatar': {
        'inf_a': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/Qatar/Qatar_Training_Data_INF_A.csv',
        'inf_b': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/Qatar/Qatar_Training_Data_INF_B.csv',
        'output_dir': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/Qatar',
    },
    'NewJersey': {
        'inf_a': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/NewJersey/NJ_Training_Data_INF_A.csv',
        'inf_b': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/NewJersey/NJ_Training_Data_INF_B.csv',
        'output_dir': '/Users/shreyan/Downloads/VAR/Influenza_Causality_AR_Project/Files/Final_Training_Data/NewJersey',
    },
}

# Columns that should NOT be log-transformed (already log or can be negative)
NO_LOG_COLS = ['LOG_INF_A', 'LOG_INF_B', 'temperature', 'dew_point_temperature', 'wet_bulb_temperature']

# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def test_stationarity(series, col_name):
    """Test stationarity using ADF and KPSS tests"""
    results = {'column': col_name}
    
    series = series.dropna()
    if len(series) < 20:
        return {'column': col_name, 'adf_pvalue': np.nan, 'kpss_pvalue': np.nan, 
                'adf_stationary': False, 'kpss_stationary': False, 'error': 'Insufficient data'}
    
    # ADF Test
    try:
        adf_result = adfuller(series, autolag='AIC')
        results['adf_stat'] = adf_result[0]
        results['adf_pvalue'] = adf_result[1]
        results['adf_stationary'] = adf_result[1] < 0.05
    except Exception as e:
        results['adf_pvalue'] = np.nan
        results['adf_stationary'] = False
        results['adf_error'] = str(e)
    
    # KPSS Test
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        results['kpss_stat'] = kpss_result[0]
        results['kpss_pvalue'] = kpss_result[1]
        results['kpss_stationary'] = kpss_result[1] >= 0.05
    except Exception as e:
        results['kpss_pvalue'] = np.nan
        results['kpss_stationary'] = False
        results['kpss_error'] = str(e)
    
    return results


def test_linearity(series, col_name):
    """Test linearity using RESET test"""
    results = {'column': col_name}
    
    series = series.dropna().values
    n = len(series)
    
    if n < 20:
        return {'column': col_name, 'reset_pvalue': np.nan, 'is_linear': False, 'error': 'Insufficient data'}
    
    try:
        X = np.arange(n).reshape(-1, 1)
        y = series
        
        X_const = add_constant(X)
        model_linear = OLS(y, X_const).fit()
        
        y_hat = model_linear.fittedvalues
        X_squared = np.column_stack([X_const, y_hat**2])
        model_extended = OLS(y, X_squared).fit()
        
        rss_linear = model_linear.ssr
        rss_extended = model_extended.ssr
        
        df_num = 1
        df_denom = n - model_extended.df_model - 1
        
        if rss_extended > 0 and df_denom > 0:
            f_stat = ((rss_linear - rss_extended) / df_num) / (rss_extended / df_denom)
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)
        else:
            f_stat = np.nan
            p_value = np.nan
        
        results['reset_f_stat'] = f_stat
        results['reset_pvalue'] = p_value
        results['is_linear'] = p_value >= 0.05 if not np.isnan(p_value) else False
        
    except Exception as e:
        results['reset_pvalue'] = np.nan
        results['is_linear'] = False
        results['error'] = str(e)
    
    return results


def calculate_vif(df):
    """Calculate VIF for all columns"""
    vif_data = []
    df_clean = df.dropna()
    
    if len(df_clean) < 10:
        return pd.DataFrame({'column': df.columns, 'VIF': [np.nan]*len(df.columns)})
    
    for i, col in enumerate(df_clean.columns):
        try:
            vif = variance_inflation_factor(df_clean.values, i)
            vif_data.append({'column': col, 'VIF': vif})
        except Exception as e:
            vif_data.append({'column': col, 'VIF': np.nan, 'error': str(e)})
    
    return pd.DataFrame(vif_data)


def apply_log_transform(df, columns_to_transform):
    """Apply log(x + 1) transformation to specified columns"""
    df_transformed = df.copy()
    transformed_cols = []
    
    for col in columns_to_transform:
        if col in df.columns and col not in NO_LOG_COLS:
            # Check if column has all positive values
            min_val = df[col].min()
            if min_val >= 0:
                new_col = f"LOG_{col.upper().replace(' ', '_')}"
                df_transformed[new_col] = np.log1p(df[col])
                transformed_cols.append((col, new_col))
                print(f"    ✓ Created {new_col} from {col}")
            else:
                print(f"    ⚠ Skipping {col} (has negative values, min={min_val:.2f})")
    
    return df_transformed, transformed_cols


# %%
# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_dataset(country_name, data_path, flu_type, output_dir):
    """Process a single dataset - test assumptions and apply transformations"""
    
    print("\n" + "="*80)
    print(f"PROCESSING: {country_name} - {flu_type.upper()}")
    print(f"File: {data_path}")
    print("="*80)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\n1. Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    # Get all columns
    all_cols = df.columns.tolist()
    
    # =========================================================================
    # TEST 1: STATIONARITY
    # =========================================================================
    print("\n" + "-"*60)
    print("TEST 1: STATIONARITY (ADF & KPSS)")
    print("-"*60)
    
    stationarity_results = []
    non_stationary_cols = []
    
    for col in all_cols:
        result = test_stationarity(df[col], col)
        stationarity_results.append(result)
        
        adf_ok = result.get('adf_stationary', False)
        kpss_ok = result.get('kpss_stationary', False)
        
        status = ""
        if adf_ok and kpss_ok:
            status = "✓ Stationary (both tests)"
        elif adf_ok:
            status = "~ Stationary (ADF only)"
        elif kpss_ok:
            status = "~ Stationary (KPSS only)"
        else:
            status = "✗ Non-stationary"
            non_stationary_cols.append(col)
        
        adf_p = result.get('adf_pvalue', np.nan)
        kpss_p = result.get('kpss_pvalue', np.nan)
        print(f"   {col:25s} ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f} → {status}")
    
    # =========================================================================
    # TEST 2: LINEARITY
    # =========================================================================
    print("\n" + "-"*60)
    print("TEST 2: LINEARITY (RESET Test)")
    print("-"*60)
    
    linearity_results = []
    non_linear_cols = []
    
    for col in all_cols:
        result = test_linearity(df[col], col)
        linearity_results.append(result)
        
        is_linear = result.get('is_linear', False)
        p_val = result.get('reset_pvalue', np.nan)
        
        if is_linear:
            status = "✓ Linear"
        else:
            status = "✗ Non-linear"
            non_linear_cols.append(col)
        
        print(f"   {col:25s} p={p_val:.4f} → {status}")
    
    # =========================================================================
    # TEST 3: MULTICOLLINEARITY (VIF)
    # =========================================================================
    print("\n" + "-"*60)
    print("TEST 3: MULTICOLLINEARITY (VIF)")
    print("-"*60)
    
    vif_df = calculate_vif(df)
    high_vif_cols = []
    
    for _, row in vif_df.iterrows():
        col = row['column']
        vif = row['VIF']
        
        if np.isnan(vif):
            status = "⚠ Could not calculate"
        elif vif < 5:
            status = "✓ No multicollinearity"
        elif vif < 10:
            status = "~ Moderate"
        else:
            status = "✗ High multicollinearity"
            high_vif_cols.append(col)
        
        print(f"   {col:25s} VIF={vif:10.2f} → {status}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    
    print(f"   Non-stationary columns: {non_stationary_cols if non_stationary_cols else 'None'}")
    print(f"   Non-linear columns: {non_linear_cols if non_linear_cols else 'None'}")
    print(f"   High VIF columns: {high_vif_cols if high_vif_cols else 'None'}")
    
    # =========================================================================
    # APPLY LOG TRANSFORMATIONS IF NEEDED
    # =========================================================================
    # Identify columns that need transformation (non-stationary OR non-linear)
    cols_needing_transform = list(set(non_stationary_cols + non_linear_cols))
    # Remove columns that shouldn't be transformed
    cols_to_transform = [c for c in cols_needing_transform if c not in NO_LOG_COLS]
    
    if cols_to_transform:
        print("\n" + "-"*60)
        print("APPLYING LOG TRANSFORMATIONS")
        print("-"*60)
        print(f"   Columns to transform: {cols_to_transform}")
        
        df_transformed, transformed_cols = apply_log_transform(df, cols_to_transform)
        
        # Re-test transformed columns
        print("\n   Re-testing transformed columns...")
        for orig_col, new_col in transformed_cols:
            stat_result = test_stationarity(df_transformed[new_col], new_col)
            lin_result = test_linearity(df_transformed[new_col], new_col)
            
            adf_ok = "✓" if stat_result.get('adf_stationary', False) else "✗"
            kpss_ok = "✓" if stat_result.get('kpss_stationary', False) else "✗"
            lin_ok = "✓" if lin_result.get('is_linear', False) else "✗"
            
            print(f"   {new_col:25s} ADF:{adf_ok} KPSS:{kpss_ok} Linear:{lin_ok}")
        
        # Save transformed data
        output_prefix = os.path.basename(data_path).replace('.csv', '')
        output_path = os.path.join(output_dir, f"{output_prefix}_transformed.csv")
        df_transformed.to_csv(output_path, index=False)
        print(f"\n   ✓ Saved transformed data to: {output_path}")
        
        return df_transformed, True
    else:
        print("\n   ✓ No transformations needed - data looks good!")
        return df, False


# %%
# =============================================================================
# RUN ALL DATASETS
# =============================================================================

print("="*80)
print("VAR ASSUMPTIONS TESTING FOR ALL COUNTRIES")
print("Testing: Stationarity, Linearity, Multicollinearity")
print("="*80)

all_results = {}

for country, paths in DATASETS.items():
    print(f"\n\n{'#'*80}")
    print(f"# {country.upper()}")
    print(f"{'#'*80}")
    
    # Process INF_A
    df_a, transformed_a = process_dataset(
        country, 
        paths['inf_a'], 
        'INF_A',
        paths['output_dir']
    )
    
    # Process INF_B
    df_b, transformed_b = process_dataset(
        country, 
        paths['inf_b'], 
        'INF_B',
        paths['output_dir']
    )
    
    all_results[country] = {
        'inf_a_transformed': transformed_a,
        'inf_b_transformed': transformed_b
    }

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n\n" + "="*80)
print("FINAL SUMMARY - ALL COUNTRIES")
print("="*80)

for country, result in all_results.items():
    a_status = "Transformed" if result['inf_a_transformed'] else "Original OK"
    b_status = "Transformed" if result['inf_b_transformed'] else "Original OK"
    print(f"   {country:12s}: INF_A={a_status:15s}, INF_B={b_status}")

print("\n" + "="*80)
print("ASSUMPTIONS TESTING COMPLETE!")
print("="*80)

# %%

