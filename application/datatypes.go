package main

import (
	"gonum.org/v1/gonum/mat"
)

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

// Simple struct for time series data
type TimeSeries struct {
	// Matrix for data
	Y *mat.Dense
	// Tracks number of time points, basically rows
	Time []float64
	// List of variable Names
	VarNames []string
}

type Deterministic int

// Deterministic Constants for VAR
const (
	DetNone Deterministic = iota
	DetConst
	DetTrend
	DetConstTrend
)

// What kind of model to fit
type ModelSpec struct {
	// How many lags?
	Lags int
	// What kind of constant to include
	Deterministic Deterministic
	// Does it have extra variables?
	HasExogenous bool
}

type ReducedFormVAR struct {
	Model ModelSpec

	// Coefficient matrices for each lag A_1, A_2, etc (each KxK matrix)
	// Stored as a slice of matrices
	A []*mat.Dense

	// Deterministic Terms: e.g. constant (Kx1) and trend (Kx1) if included
	C *mat.Dense

	// Covariance of residuals (KxK)
	SigmaU *mat.SymDense
}

type ReducedForm interface {
	// Returns the model specification
	Spec() ModelSpec
	// Returns the coefficient matrices
	Phi() []*mat.Dense
	// Returns the error covariance
	CovU() *mat.SymDense

	// compute the forcasts for a given initial state
	Forecast(y0 *mat.Dense, steps int) (*mat.Dense, error)
	// Simulates effect of one-time shock in 1 variable on all variables over time
	IRF(horizon int, shockIndex int) (*mat.Dense, error)
}

// EstimationOptions contains options like regularization strngth, priors, etc.
type EstimationOptions struct {
	// For standard VAr
	UseGeneralizedLeastSquares bool

	// EX: if BVAR is implemented
	//Prior Prior
}

type Estimator interface {
	// Turns the data we have into a reduced form VAR
	Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error)
}

// --- Plain OLS VAR estimator ---

type OLSEstimator struct{}

type VARModel struct {
	LagP         int         // Optimal lag order used (p)
	Coefficients [][]float64 // The fitted A_1...A_p matrices (the core model parameters)
	Residuals    []float64   // Model residuals (for checking assumptions)
	Variables    []string    // List of variables included in the model (e.g., ["A_H1N1_Count", "Avg_Temperature"])

	// Granger Causality Results
	GrangerPValues map[string]map[string]float64 // Map[CauseVar][EffectVar] = PValue
}


// --- GRANGER CAUSALITY TEST ---

// GrangerCausalityResult holds the result of a Granger causality test
type GrangerCausalityResult struct {
	CauseVar    string  // Variable being tested as the cause
	EffectVar   string  // Variable being tested as the effect
	FStatistic  float64 // F-statistic value
	PValue      float64 // P-value
	Lags        int     // Number of lags used
	Significant bool    // True if p-value < 0.05
}