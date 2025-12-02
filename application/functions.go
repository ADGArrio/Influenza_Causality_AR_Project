package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
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
			}

			// lagged part: sum_j A_j * y_{t-j}
			for lag := 1; lag <= p; lag++ {
				A := rf.A[lag-1]
				prevRow := row - lag
				for j := 0; j < K; j++ {
					val += A.At(eq, j) * out.At(prevRow, j)
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

// Run IRF for all variables to look for changes in varible var, then compile results
// of how much each varible changed var during its shock into a map
// varIndex: index of variable to analyze, 0-based
// horizon: number of periods to compute (h=0, ..., horizon-1)
// Returns: map[shockIndex] =  impact on varIndex
func (rf *ReducedFormVAR) RunIRFAnalysis(varIndex int, horizon int) (map[int][]float64, error) {
	// Check if the model is estimated and varIndex is valid
	if rf == nil || len(rf.A) == 0 {
		return nil, fmt.Errorf("VAR model not estimated")
	}

	K, _ := rf.A[0].Dims()
	if varIndex < 0 || varIndex >= K {
		return nil, fmt.Errorf("varIndex must be between 0 and %d", K-1)
	}

	results := make(map[int][]float64)
	for shockIdx := 0; shockIdx < K; shockIdx++ {
		irfMat, err := rf.IRF(horizon, shockIdx)
		if err != nil {
			return nil, fmt.Errorf("IRF failed for shockIdx %d: %v", shockIdx, err)
		}

		series := make([]float64, horizon)
		for h := 0; h < horizon; h++ {
			series[h] = irfMat.At(h, varIndex)
		}

		results[shockIdx] = series
	}

	return results, nil
}

func (rf *ReducedFormVAR) OutputIRFAnalysisToCSV(path string, analysis map[int][]float64, varNames []string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}

	defer file.Close()

	// Initialize a new CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush() // Ensure all buffered data is written

	// Write header
	header := []string{"Horizon"}
	for shockIdx := range analysis {
		var varName string
		if len(varNames) == len(analysis) {
			varName = varNames[shockIdx]
		} else {
			varName = fmt.Sprintf("Var%d", shockIdx+1)
		}
		header = append(header, "Shock_in_"+varName)
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Determine horizon from one of the analysis entries
	var horizon int
	for _, series := range analysis {
		horizon = len(series)
		break
	}

	// Write data rows
	for h := 0; h < horizon; h++ {
		record := []string{fmt.Sprintf("%d", h)}
		for shockIdx := range analysis {
			record = append(record, fmt.Sprintf("%f", analysis[shockIdx][h]))
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	return nil
}

func (rf *ReducedFormVAR) OutputForecastsToCSV(path string, fc *mat.Dense, varNames []string) error {

	rows, cols := fc.Dims()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Initialize a new CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush() // Ensure all buffered data is written

	// Write header
	header := make([]string, cols)
	for j := 0; j < cols; j++ {
		if len(varNames) == cols {
			header[j] = varNames[j]
		} else {
			header[j] = fmt.Sprintf("Var%d", j+1)
		}
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Write data rows
	for i := 0; i < rows; i++ {
		record := make([]string, cols)
		for j := 0; j < cols; j++ {
			record[j] = fmt.Sprintf("%f", fc.At(i, j))
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	return nil
}

// --- OLS IMPLEMENTATION ---
func (e *OLSEstimator) Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}

	T, K := ts.Y.Dims()
	p := spec.Lags

	if p <= 0 {
		return nil, fmt.Errorf("lags must be > 0")
	}

	if T <= p {
		return nil, fmt.Errorf("need at least p+1 observations: p = %d, T = %d", p, T)
	}
	if spec.HasExogenous {
		return nil, fmt.Errorf("exogenous variables not supported yet")
	}

	// Builds the response matrix for later use
	Treg := T - p // Usable rows

	// Response matrix Yreg: rows are y_p, y_{p+1}, ..., y_{T-1}
	Yreg := mat.NewDense(Treg, K, nil)
	for t := 0; t < Treg; t++ {
		for k := 0; k < K; k++ {
			Yreg.Set(t, k, ts.Y.At(t+p, k))
		}
	}

	// Deterministic structure
	hasConst := spec.Deterministic == DetConst || spec.Deterministic == DetConstTrend
	hasTrend := spec.Deterministic == DetTrend || spec.Deterministic == DetConstTrend

	detCols := 0
	if hasConst {
		detCols++
	}
	if hasTrend {
		detCols++
	}

	lagCols := p * K
	m := detCols + lagCols // total regressors

	X := mat.NewDense(Treg, m, nil)

	// Fill X row-by-row

	for t := 0; t < Treg; t++ {
		col := 0
		// time index
		timeIndex := float64(t + p + 1)

		if hasConst {
			X.Set(t, col, 1.0)
			col++
		}
		if hasTrend {
			X.Set(t, col, timeIndex)
			col++
		}

		// Lagged Y's: [ y_{t+p-1}, y_{t+p-2}, ..., y_{t+p-p}]
		for j := 1; j <= p; j++ {
			srcRow := t + p - j
			for k := 0; k < K; k++ {
				X.Set(t, col, ts.Y.At(srcRow, k))
				col++
			}
		}
	}

	// B = (X'X)^(-1) X'Y
	// Calculates closed form
	var B mat.Dense

	// First try: normal equations B = (X'X)^(-1) X'Y
	var xtx mat.Dense
	xtx.Mul(X.T(), X)

	var xtxInv mat.Dense

	xtxError := xtxInv.Inverse(&xtx)

	if xtxError == nil {
		// X'X is invertible: standard OLS
		var xty mat.Dense
		xty.Mul(X.T(), Yreg)
		B.Mul(&xtxInv, &xty)
	} else {
		// Fallback: X'X is singular or badly conditioned.
		// Use SVD-based least squares: minimize ||Yreg - X B||_F with minimum-norm B.

		var svd mat.SVD
		ok := svd.Factorize(X, mat.SVDFullU|mat.SVDFullV)
		if !ok {
			return nil, fmt.Errorf("OLS failed: X'X singular and SVD factorization failed: %v", xtxError)
		}

		// Choose an effective numerical rank (tolerance can be tuned)
		rank := svd.Rank(1e-12)

		// Solve X * B ≈ Yreg in least-squares sense; B will be (m × K)
		// This gives us the Moore_penrose pseudoinverse commonly used in regression too
		// If rank == 0, the matrix X is (numerically) all-zero.
		// The minimum-norm least-squares solution to X B ≈ Y is just B = 0.
		if rank == 0 {
			B = *mat.NewDense(m, K, nil) // all zeros
		} else {
			// Solve X * B ≈ Yreg in least-squares sense; B will be (m × K)
			svd.SolveTo(&B, Yreg, rank)
		}
	}

	// Split B into C (deterministic) and A_j's
	var C *mat.Dense
	if detCols > 0 {
		C = mat.NewDense(K, detCols, nil)
		for k := 0; k < K; k++ {
			for d := 0; d < detCols; d++ {
				C.Set(k, d, B.At(d, k))
			}
		}
	}

	A := make([]*mat.Dense, p)
	for j := 0; j < p; j++ {
		Aj := mat.NewDense(K, K, nil)
		rowOffset := detCols + j*K // start row of this lag block in B

		for eq := 0; eq < K; eq++ {
			for colVar := 0; colVar < K; colVar++ {
				Aj.Set(eq, colVar, B.At(rowOffset+colVar, eq))
			}
		}
		A[j] = Aj
	}

	// Residual covariance SigmaU
	var Yhat mat.Dense
	Yhat.Mul(X, &B)

	var U mat.Dense
	U.Sub(Yreg, &Yhat) // Treg x K

	var utu mat.Dense
	utu.Mul(U.T(), &U) // K x K

	df := float64(Treg - m)
	if df <= 0 {
		df = float64(Treg) // fallback
	}

	sigmaData := make([]float64, K*K)
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			sigmaData[i*K+j] = utu.At(i, j) / df
		}
	}
	SigmaU := mat.NewSymDense(K, sigmaData)

	rf := &ReducedFormVAR{
		Model:  spec,
		A:      A,
		C:      C,
		SigmaU: SigmaU,
	}

	return rf, nil
}

// GrangerCausality tests whether causeIdx Granger-causes effectIdx
// Returns the F-statistic and p-value
func (rf *ReducedFormVAR) GrangerCausality(ts *TimeSeries, causeIdx, effectIdx int) (*GrangerCausalityResult, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}

	T, K := ts.Y.Dims()
	p := rf.Model.Lags

	if causeIdx < 0 || causeIdx >= K {
		return nil, fmt.Errorf("causeIdx out of range: %d", causeIdx)
	}
	if effectIdx < 0 || effectIdx >= K {
		return nil, fmt.Errorf("effectIdx out of range: %d", effectIdx)
	}
	if causeIdx == effectIdx {
		return nil, fmt.Errorf("causeIdx and effectIdx cannot be the same")
	}

	// Build response vector for the effect variable
	Treg := T - p
	yEffect := mat.NewVecDense(Treg, nil)
	for t := 0; t < Treg; t++ {
		yEffect.SetVec(t, ts.Y.At(t+p, effectIdx))
	}

	// Deterministic structure
	hasConst := rf.Model.Deterministic == DetConst || rf.Model.Deterministic == DetConstTrend
	hasTrend := rf.Model.Deterministic == DetTrend || rf.Model.Deterministic == DetConstTrend

	detCols := 0
	if hasConst {
		detCols++
	}
	if hasTrend {
		detCols++
	}

	// Build UNRESTRICTED model: includes all lagged variables
	lagCols := p * K
	mUnrestricted := detCols + lagCols
	XUnrestricted := mat.NewDense(Treg, mUnrestricted, nil)

	for t := 0; t < Treg; t++ {
		col := 0
		timeIndex := float64(t + p + 1)

		if hasConst {
			XUnrestricted.Set(t, col, 1.0)
			col++
		}
		if hasTrend {
			XUnrestricted.Set(t, col, timeIndex)
			col++
		}

		for j := 1; j <= p; j++ {
			srcRow := t + p - j
			for k := 0; k < K; k++ {
				XUnrestricted.Set(t, col, ts.Y.At(srcRow, k))
				col++
			}
		}
	}

	// Fit unrestricted model
	var betaUnrestricted mat.VecDense
	err := betaUnrestricted.SolveVec(XUnrestricted, yEffect)
	if err != nil {
		return nil, fmt.Errorf("failed to solve unrestricted model: %v", err)
	}

	// Calculate RSS for unrestricted model
	var yHatUnrestricted mat.VecDense
	yHatUnrestricted.MulVec(XUnrestricted, &betaUnrestricted)

	var residUnrestricted mat.VecDense
	residUnrestricted.SubVec(yEffect, &yHatUnrestricted)

	rssUnrestricted := mat.Dot(&residUnrestricted, &residUnrestricted)

	// Build RESTRICTED model: excludes lags of the cause variable
	mRestricted := detCols + p*(K-1) // exclude p lags of cause variable
	XRestricted := mat.NewDense(Treg, mRestricted, nil)

	for t := 0; t < Treg; t++ {
		col := 0
		timeIndex := float64(t + p + 1)

		if hasConst {
			XRestricted.Set(t, col, 1.0)
			col++
		}
		if hasTrend {
			XRestricted.Set(t, col, timeIndex)
			col++
		}

		// Add lags but skip the cause variable
		for j := 1; j <= p; j++ {
			srcRow := t + p - j
			for k := 0; k < K; k++ {
				if k != causeIdx { // Skip the cause variable
					XRestricted.Set(t, col, ts.Y.At(srcRow, k))
					col++
				}
			}
		}
	}

	// Fit restricted model
	var betaRestricted mat.VecDense
	err = betaRestricted.SolveVec(XRestricted, yEffect)
	if err != nil {
		return nil, fmt.Errorf("failed to solve restricted model: %v", err)
	}

	// Calculate RSS for restricted model
	var yHatRestricted mat.VecDense
	yHatRestricted.MulVec(XRestricted, &betaRestricted)

	var residRestricted mat.VecDense
	residRestricted.SubVec(yEffect, &yHatRestricted)

	rssRestricted := mat.Dot(&residRestricted, &residRestricted)

	// Calculate F-statistic
	// F = [(RSS_r - RSS_ur) / q] / [RSS_ur / (T - k)]
	// where q = number of restrictions = p (p lags of cause variable)
	// k = number of parameters in unrestricted model
	q := float64(p)
	k := float64(mUnrestricted)
	dof := float64(Treg) - k

	if dof <= 0 {
		return nil, fmt.Errorf("insufficient degrees of freedom: %f", dof)
	}

	fStatistic := ((rssRestricted - rssUnrestricted) / q) / (rssUnrestricted / dof)

	// Calculate p-value using F-distribution
	fDist := distuv.F{
		D1: q,
		D2: dof,
	}
	pValue := 1.0 - fDist.CDF(fStatistic)

	// Handle edge cases
	if math.IsNaN(fStatistic) || math.IsInf(fStatistic, 0) {
		fStatistic = 0
		pValue = 1.0
	}
	if pValue < 0 {
		pValue = 0
	}
	if pValue > 1 {
		pValue = 1.0
	}

	result := &GrangerCausalityResult{
		CauseVar:    ts.VarNames[causeIdx],
		EffectVar:   ts.VarNames[effectIdx],
		FStatistic:  fStatistic,
		PValue:      pValue,
		Lags:        p,
		Significant: pValue < 0.05,
	}

	return result, nil
}

// GrangerCausalityMatrix performs pairwise Granger causality tests for all variables
func (rf *ReducedFormVAR) GrangerCausalityMatrix(ts *TimeSeries) ([][]*GrangerCausalityResult, error) {
	if ts == nil || ts.Y == nil {
		return nil, fmt.Errorf("time series data not provided")
	}

	_, K := ts.Y.Dims()

	// Create matrix to store results
	results := make([][]*GrangerCausalityResult, K)
	for i := range results {
		results[i] = make([]*GrangerCausalityResult, K)
	}

	// Perform pairwise tests
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				// No self-causality test
				results[i][j] = nil
				continue
			}

			result, err := rf.GrangerCausality(ts, i, j)
			if err != nil {
				return nil, fmt.Errorf("error testing %s -> %s: %v", ts.VarNames[i], ts.VarNames[j], err)
			}
			results[i][j] = result
		}
	}

	return results, nil
}

// This function takes in the created Granger Matrix and outputs it to a CSV file with
// the columns: CauseVar, EffectVar, FStatistic, PValue, Lags, Significant
func (rf *ReducedFormVAR) OutputGrangerMatrixToCSV(path string, gcMatrix [][]*GrangerCausalityResult, varNames []string) error {
	file, err := os.Create(path)

	if err != nil {
		return err
	}

	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{"CauseVar", "EffectVar", "FStatistic", "PValue", "Lags", "Significant"}
	if err := writer.Write(header); err != nil {
		return err
	}

	K := len(varNames)

	// Write data rows
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				continue // Skip self-causality
			}
			result := gcMatrix[i][j]
			if result == nil {
				continue
			}
			record := []string{
				result.CauseVar,
				result.EffectVar,
				fmt.Sprintf("%f", result.FStatistic),
				fmt.Sprintf("%f", result.PValue),
				fmt.Sprintf("%d", result.Lags),
				fmt.Sprintf("%t", result.Significant),
			}
			if err := writer.Write(record); err != nil {
				return err
			}
		}
	}

	return nil
}
