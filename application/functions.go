package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// --- FUNCTIONS FOR BASE ---
// Returns current Model Spec
func (rf *ReducedFormVAR) Spec() ModelSpec { return rf.Model }

// Returns coefficient matrices
func (rf *ReducedFormVAR) Phi() []*mat.Dense { return rf.A }

// Returns error covariance matrix
func (rf *ReducedFormVAR) CovU() *mat.SymDense { return rf.SigmaU }

// Forecasat produces multi-step ahead forecases given the historical data of yHist.
// yHist: T x K (rows: time, cols: variables). Only last p rows are used as lags.
// steps: number of steps ahead to forecast
// Returns: Steps xK matrix of forecasts
// HOW TO USE:
// rf, _ := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
// fcst, err := rf.Forecast(ts.Y, 10) //10-step ahead forecast
func (rf *ReducedFormVAR) Forecast(yHist *mat.Dense, steps int) (*mat.Dense, error) {
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}
	if steps <= 0 {
		return nil, fmt.Errorf("steps must be > 0")
	}

	p := rf.Model.Lags

	if p <= 0 {
		return nil, fmt.Errorf("lags must be > 0 to forecast")
	}

	// dimensions of yHist, T rows, K cols
	T, K := yHist.Dims()
	if T < p {
		return nil, fmt.Errorf("need at least %d rows in yHist, got %d", p, T)
	}

	totalRows := p + steps

	data := make([]float64, totalRows*K)

	// Gonum.mat stores matrices as a 1d slice at first, so we need to multiply by K to fill out
	// Can maybe parallelize later?
	for i := 0; i < p; i++ {
		for k := 0; k < K; k++ {
			data[i*K+k] = yHist.At(T-p+i, k)
		}
	}

	out := mat.NewDense(totalRows, K, data)

	// Deterministic structure
	hasConst := rf.Model.Deterministic == DetConst || rf.Model.Deterministic == DetConstTrend
	hasTrend := rf.Model.Deterministic == DetTrend || rf.Model.Deterministic == DetConstTrend

	detConstIdx := 0
	detTrendIdx := 0
	detCols := 0

	if hasConst {
		detCols += 1
	}
	if hasTrend {
		detTrendIdx = detCols
		detCols += 1
	}

	// for each equation in the VAR (for each variable in the time series)
	for step := 0; step < steps; step++ {
		row := p + step
		// time index from last row of yHist
		tIdx := float64(T + step + 1)

		for eq := 0; eq < K; eq++ {
			// val is where we store the equation for the current step for A SINGLE variable
			val := 0.0

			// If model has deterministic trends, include in forecast value by adding it
			if rf.C != nil && detCols > 0 {
				if hasConst {
					val += rf.C.At(eq, detConstIdx)
				}

				if hasTrend {
					val += rf.C.At(eq, detTrendIdx) * tIdx
				}

				// lagged part: sum_j A_j * y_{t-j}
				for lag := 1; lag <= p; lag++ {
					A := rf.A[lag-1]
					prevRow := row - lag
					for j := 0; j < K; j++ {
						val += A.At(eq, j) * out.At(prevRow, j)
					}
				}
			}
			// Sets each row of the forecast with the current value at each column
			out.Set(row, eq, val)
		}
	}
	// Returns only the forecasted rows
	forecast := mat.DenseCopyOf(out.Slice(p, totalRows, 0, K))
	return forecast, nil
}

// --- OLS IMPLEMENTATION ---
func (e *OLSEstimator) Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error) {
	// 1. Build lagged design matrix X and Y
	// 2. Run (possibly multiple) linear regressions
	// 3. Assemble Phi, C, SigmaU
	// 4. Return &ReducedFormVAR{...}
	return nil, nil
}
