import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load the lagged dataset
input_file = "flunet_weather_features_lagged.csv"  # Replace with your actual file path
df = pd.read_csv(input_file, parse_dates=["iso_weekstartdate"])

# Select relevant columns for stationarity tests
variables = ["inf_all", "avg_temp", "humidity"]

# Function to perform ADF test
def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF Test for {name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values: {result[4]}")
    if result[1] <= 0.05:
        print("  => Stationary (reject null hypothesis)")
    else:
        print("  => Non-stationary (fail to reject null hypothesis)")
    print("\n")

# Run ADF test for each variable
for var in variables:
    adf_test(df[var], var)

print("Stationarity tests complete. If variables are non-stationary, apply differencing before Granger causality.")
