package main

type TimeSeriesPoint struct {
	// Need to change based on data source
	A_H1N1_Count int     // Specific subtype count
	FluA_Percent float64 // Percentage positive flu A tests
	ILI_Activity float64 // ILI Syndromic Indicator

	// Need to change based on data source
	Avg_Temperature float64
	Avg_Humidity    float64
	Sin_Seasonality float64

	// Lagged Features (Engineered)
}

type VARModel struct {
	LagP         int         // Optimal lag order used (p)
	Coefficients [][]float64 // The fitted A_1...A_p matrices (the core model parameters)
	Residuals    []float64   // Model residuals (for checking assumptions)
	Variables    []string    // List of variables included in the model (e.g., ["A_H1N1_Count", "Avg_Temperature"])

	// Granger Causality Results
	GrangerPValues map[string]map[string]float64 // Map[CauseVar][EffectVar] = PValue
}
