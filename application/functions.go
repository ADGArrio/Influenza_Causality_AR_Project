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

// IRF computes impulse responses to a one-time structural shock in a variable shockIndex
// Horizon: number of periods to compute (h=0, ..., horizon-1)
// shockIndex: index of variable to shock, (0-based)
// Returns: horizon x K matrix. where row h is response of all K vars at horizon h
// Usage:
// rf, _ := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
// irfMat, err := rf.IRF(20, 0) // 20-period IRF to shock in variable 0
func (rf *ReducedFormVAR) IRF(horizon int, shockIndex int) (*mat.Dense, error) {
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be > 0")
	}

	p := rf.Model.Lags
	if p <= 0 {
		return nil, fmt.Errorf("lags must be > 0 to IRF")
	}

	K, _ := rf.A[0].Dims()
	if shockIndex < 0 || shockIndex >= K {
		return nil, fmt.Errorf("shockIndex must be between 0 and %d", K-1)
	}

	// Makes the shock matrix
	shock := make([]float64, K)
	if rf.SigmaU != nil {
		var chol mat.Cholesky
		// Cholesky decomposition applied to SigmaU, calculates LL'
		if chol.Factorize(rf.SigmaU) {
			L := mat.NewTriDense(K, mat.Lower, nil)
			chol.LTo(L) // SigmaU = L * L^T
			// get the shock vector
			for i := 0; i < K; i++ {
				shock[i] = L.At(i, shockIndex)
			}
		} else {
			// fallback if SigmaU is not positive definite
			shock[shockIndex] = 1.0
		}
	} else {
		// fall back if SigmaU is not provided
		shock[shockIndex] = 1.0
	}

	// Moving-average coeff matrix Psi_h
	Psi := make([]*mat.Dense, horizon)

	// Psi_0 = I_K, makes matrix using mat
	Idata := make([]float64, K*K)

	for i := 0; i < K; i++ {
		Idata[i*K+i] = 1.0
	}
	// makes a new identity matrix
	Psi[0] = mat.NewDense(K, K, Idata)

	// Recursively computes Psi_h
	for h := 1; h < horizon; h++ {
		M := mat.NewDense(K, K, nil)
		maxLag := p
		if h < p {
			maxLag = h
		}
		for j := 1; j <= maxLag; j++ {
			var tmp mat.Dense
			tmp.Mul(rf.A[j-1], Psi[h-j]) // A_j * Psi_{h-j}
			M.Add(M, &tmp)
		}
		Psi[h] = M
	}

	// IRF[h] = Psi_h * shock

	irf := mat.NewDense(horizon, K, nil)
	shockVec := mat.NewVecDense(K, shock)

	for h := 0; h < horizon; h++ {
		var resp mat.VecDense
		resp.MulVec(Psi[h], shockVec)
		for i := 0; i < K; i++ {
			irf.Set(h, i, resp.AtVec(i))
		}
	}

	return irf, nil
}

// --- OLS IMPLEMENTATION ---
func (e *OLSEstimator) Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error) {
	// 1. Build lagged design matrix X and Y
	// 2. Run (possibly multiple) linear regressions
	// 3. Assemble Phi, C, SigmaU
	// 4. Return &ReducedFormVAR{...}
	return nil, nil
}
