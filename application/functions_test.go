// Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
// Date: Dec 12th 2025
// Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
// Class: 02-613 at Caregie Mellon University

package main

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// helper: compare floats with tolerance
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// Forecast tests

// VAR(1) scalar without deterministics: y_t = 0.5 y_{t-1}
// If last observed value is y_T = 1/16, then forecasts should be:
// y_{T+1} = 0.5 * 1/16 = 1/32, etc.
func TestForecast_SimpleVAR1_NoDeterministic(t *testing.T) {
	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetNone,
		HasExogenous:  false,
	}

	// A_1 = [0.5]
	A1 := mat.NewDense(1, 1, []float64{0.5})
	rf := &ReducedFormVAR{
		Model: spec,
		A:     []*mat.Dense{A1},
		// no C, no SigmaU needed for forecasting
	}

	// History: y = [1, 1/2, 1/4, 1/8, 1/16]
	histData := []float64{
		1.0,
		0.5,
		0.25,
		0.125,
		0.0625,
	}
	yHist := mat.NewDense(len(histData), 1, histData)

	steps := 3
	fcst, err := rf.Forecast(yHist, steps)
	if err != nil {
		t.Fatalf("Forecast returned error: %v", err)
	}

	if r, c := fcst.Dims(); r != steps || c != 1 {
		t.Fatalf("Forecast dims = %dx%d, want %dx1", r, c, steps)
	}

	expected := []float64{
		0.03125,   // 1/32
		0.015625,  // 1/64
		0.0078125, // 1/128
	}

	for i := 0; i < steps; i++ {
		got := fcst.At(i, 0)
		if !almostEqual(got, expected[i], 1e-6) {
			t.Errorf("Forecast[%d] = %v, want %v", i, got, expected[i])
		}
	}
}

// VAR(1) scalar with constant only: y_t = c, c = 1.0
// A_1 = 0, C = 1, so all forecasts should be 1.
func TestForecast_Var1_ConstantOnly(t *testing.T) {
	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetConst,
		HasExogenous:  false,
	}

	A1 := mat.NewDense(1, 1, []float64{0.0})
	C := mat.NewDense(1, 1, []float64{1.0})
	rf := &ReducedFormVAR{
		Model: spec,
		A:     []*mat.Dense{A1},
		C:     C,
	}

	// History can be anything; we use zeros for convenience.
	histData := []float64{0, 0, 0}
	yHist := mat.NewDense(len(histData), 1, histData)

	steps := 4
	fcst, err := rf.Forecast(yHist, steps)
	if err != nil {
		t.Fatalf("Forecast returned error: %v", err)
	}

	if r, c := fcst.Dims(); r != steps || c != 1 {
		t.Fatalf("Forecast dims = %dx%d, want %dx1", r, c, steps)
	}

	for i := 0; i < steps; i++ {
		got := fcst.At(i, 0)
		if !almostEqual(got, 1.0, 1e-6) {
			t.Errorf("Forecast[%d] = %v, want 1.0", i, got)
		}
	}
}

// IRF tests

// Scalar VAR(1): y_t = a y_{t-1} + u_t, Var(u_t) = 1
// With Cholesky, shock = 1, and Psi_h = a^h, so IRF(h) = a^h.
func TestIRF_ScalarVAR1(t *testing.T) {
	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetNone,
		HasExogenous:  false,
	}

	a := 0.5
	A1 := mat.NewDense(1, 1, []float64{a})

	// SigmaU = [1]
	sigmaData := []float64{1.0}
	SigmaU := mat.NewSymDense(1, sigmaData)

	rf := &ReducedFormVAR{
		Model:  spec,
		A:      []*mat.Dense{A1},
		SigmaU: SigmaU,
	}

	horizon := 5
	irf, err := rf.IRF(horizon, 0)
	if err != nil {
		t.Fatalf("IRF returned error: %v", err)
	}

	if r, c := irf.Dims(); r != horizon || c != 1 {
		t.Fatalf("IRF dims = %dx%d, want %dx1", r, c, horizon)
	}

	// expected: [1, a, a^2, ..., a^(horizon-1)]
	val := 1.0
	for h := 0; h < horizon; h++ {
		got := irf.At(h, 0)
		if !almostEqual(got, val, 1e-6) {
			t.Errorf("IRF[%d] = %v, want %v", h, got, val)
		}
		val *= a
	}
}

// Estimate tests

// Check that Estimate recovers roughly the correct coefficient
// for y_t = 0.5 y_{t-1} with no deterministic terms.
func TestEstimate_SimpleVAR1_NoDeterministic(t *testing.T) {
	// Generate data exactly following y_t = 0.5 y_{t-1}
	data := []float64{
		1.0,      // y_0
		0.5,      // y_1
		0.25,     // y_2
		0.125,    // y_3
		0.0625,   // y_4
		0.03125,  // y_5
		0.015625, // y_6
	}
	T := len(data)
	Y := mat.NewDense(T, 1, data)

	ts := &TimeSeries{
		Y:        Y,
		Time:     nil,
		VarNames: []string{"y"},
	}

	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetNone,
		HasExogenous:  false,
	}

	opts := EstimationOptions{}

	est := &OLSEstimator{}
	rf, err := est.Estimate(ts, spec, opts)
	if err != nil {
		t.Fatalf("Estimate returned error: %v", err)
	}

	if len(rf.A) != 1 {
		t.Fatalf("len(rf.A) = %d, want 1", len(rf.A))
	}

	phiHat := rf.A[0].At(0, 0)
	if !almostEqual(phiHat, 0.5, 1e-2) {
		t.Errorf("Estimated phi = %v, want approx 0.5", phiHat)
	}

	if rf.C != nil {
		t.Errorf("Expected no deterministic coefficients (C == nil), got C != nil")
	}
}

// Force X'X to be singular to test the SVD / pseudoinverse path.
// We do this by using all-zero regressors: y_t = 0 for all t, so lagged y are all zero.
func TestEstimate_PseudoinverseFallback(t *testing.T) {
	// All zeros
	data := []float64{0, 0, 0, 0}
	T := len(data)
	Y := mat.NewDense(T, 1, data)

	ts := &TimeSeries{
		Y:        Y,
		Time:     nil,
		VarNames: []string{"y"},
	}

	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetNone,
		HasExogenous:  false,
	}

	opts := EstimationOptions{}
	est := &OLSEstimator{}
	rf, err := est.Estimate(ts, spec, opts)
	if err != nil {
		t.Fatalf("Estimate returned error (pseudoinverse path): %v", err)
	}

	if len(rf.A) != 1 {
		t.Fatalf("len(rf.A) = %d, want 1", len(rf.A))
	}

	phiHat := rf.A[0].At(0, 0)
	// With all-zero regressors and responses, the least-squares solution should be 0.
	if !almostEqual(phiHat, 0.0, 1e-6) {
		t.Errorf("Estimated phi (pseudoinverse) = %v, want 0.0", phiHat)
	}
}

// Getter tests

func TestSpec(t *testing.T) {
	spec := ModelSpec{
		Lags:          3,
		Deterministic: DetConst,
		HasExogenous:  false,
	}
	rf := &ReducedFormVAR{Model: spec}

	got := rf.Spec()
	if got.Lags != 3 {
		t.Errorf("Spec().Lags = %d, want 3", got.Lags)
	}
	if got.Deterministic != DetConst {
		t.Errorf("Spec().Deterministic = %v, want DetConst", got.Deterministic)
	}
}

func TestPhi(t *testing.T) {
	A1 := mat.NewDense(2, 2, []float64{0.5, 0.1, 0.2, 0.3})
	A2 := mat.NewDense(2, 2, []float64{0.1, 0.0, 0.0, 0.1})
	rf := &ReducedFormVAR{A: []*mat.Dense{A1, A2}}

	phi := rf.Phi()
	if len(phi) != 2 {
		t.Errorf("len(Phi()) = %d, want 2", len(phi))
	}
	if phi[0].At(0, 0) != 0.5 {
		t.Errorf("Phi()[0].At(0,0) = %v, want 0.5", phi[0].At(0, 0))
	}
}

func TestCovU(t *testing.T) {
	sigmaData := []float64{1.0, 0.5, 0.5, 2.0}
	sigma := mat.NewSymDense(2, sigmaData)
	rf := &ReducedFormVAR{SigmaU: sigma}

	got := rf.CovU()
	if got.At(0, 0) != 1.0 {
		t.Errorf("CovU().At(0,0) = %v, want 1.0", got.At(0, 0))
	}
	if got.At(0, 1) != 0.5 {
		t.Errorf("CovU().At(0,1) = %v, want 0.5", got.At(0, 1))
	}
}

// RunIRFAnalysis tests

func TestRunIRFAnalysis(t *testing.T) {
	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetNone,
		HasExogenous:  false,
	}

	// 2x2 VAR(1) with diagonal coefficients
	A1 := mat.NewDense(2, 2, []float64{0.5, 0.0, 0.0, 0.3})
	sigmaData := []float64{1.0, 0.0, 0.0, 1.0}
	SigmaU := mat.NewSymDense(2, sigmaData)

	rf := &ReducedFormVAR{
		Model:  spec,
		A:      []*mat.Dense{A1},
		SigmaU: SigmaU,
	}

	results, err := rf.RunIRFAnalysis(0, 5)
	if err != nil {
		t.Fatalf("RunIRFAnalysis returned error: %v", err)
	}

	// Should have results for 2 shocks (one for each variable)
	if len(results) != 2 {
		t.Errorf("len(results) = %d, want 2", len(results))
	}

	// Check that each shock has 5 horizons
	for shockIdx, series := range results {
		if len(series) != 5 {
			t.Errorf("shock %d: len(series) = %d, want 5", shockIdx, len(series))
		}
	}
}

// GrangerCausality tests

func TestGrangerCausality_Basic(t *testing.T) {
	// Generate simple VAR(1) data where var1 causes var2
	// y1_t = 0.5 * y1_{t-1} + e1
	// y2_t = 0.3 * y1_{t-1} + 0.2 * y2_{t-1} + e2
	T := 100
	data := make([]float64, T*2)

	// Initialize
	data[0] = 1.0 // y1_0
	data[1] = 0.0 // y2_0

	for t := 1; t < T; t++ {
		y1_prev := data[(t-1)*2]
		y2_prev := data[(t-1)*2+1]
		data[t*2] = 0.5*y1_prev + 0.01*float64(t%5)    // y1_t
		data[t*2+1] = 0.3*y1_prev + 0.2*y2_prev + 0.01 // y2_t (y1 causes y2)
	}

	Y := mat.NewDense(T, 2, data)
	ts := &TimeSeries{
		Y:        Y,
		VarNames: []string{"y1", "y2"},
	}

	spec := ModelSpec{
		Lags:          2,
		Deterministic: DetConst,
		HasExogenous:  false,
	}

	rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// Test y1 -> y2 (should show some causality)
	result, err := rf.GrangerCausality(ts, 0, 1)
	if err != nil {
		t.Fatalf("GrangerCausality returned error: %v", err)
	}

	if result.CauseVar != "y1" {
		t.Errorf("CauseVar = %s, want y1", result.CauseVar)
	}
	if result.EffectVar != "y2" {
		t.Errorf("EffectVar = %s, want y2", result.EffectVar)
	}
	if result.Lags != 2 {
		t.Errorf("Lags = %d, want 2", result.Lags)
	}

	// F-statistic and p-value should be valid numbers
	if math.IsNaN(result.FStatistic) || math.IsInf(result.FStatistic, 0) {
		t.Errorf("FStatistic is NaN or Inf: %v", result.FStatistic)
	}
	if result.PValue < 0 || result.PValue > 1 {
		t.Errorf("PValue out of range: %v", result.PValue)
	}
}

func TestGrangerCausality_SameIndex(t *testing.T) {
	Y := mat.NewDense(10, 2, nil)
	ts := &TimeSeries{Y: Y, VarNames: []string{"a", "b"}}
	spec := ModelSpec{Lags: 1, Deterministic: DetNone}
	rf := &ReducedFormVAR{
		Model: spec,
		A:     []*mat.Dense{mat.NewDense(2, 2, nil)},
	}

	_, err := rf.GrangerCausality(ts, 0, 0)
	if err == nil {
		t.Error("Expected error for causeIdx == effectIdx, got nil")
	}
}

// GrangerCausalityMatrix tests

func TestGrangerCausalityMatrix(t *testing.T) {
	T := 50
	K := 3
	data := make([]float64, T*K)
	for i := range data {
		data[i] = float64(i%10) * 0.1
	}

	Y := mat.NewDense(T, K, data)
	ts := &TimeSeries{
		Y:        Y,
		VarNames: []string{"x", "y", "z"},
	}

	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetConst,
	}

	rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	matrix, err := rf.GrangerCausalityMatrix(ts)
	if err != nil {
		t.Fatalf("GrangerCausalityMatrix returned error: %v", err)
	}

	if len(matrix) != K {
		t.Errorf("len(matrix) = %d, want %d", len(matrix), K)
	}

	// Diagonal should be nil (no self-causality)
	for i := 0; i < K; i++ {
		if matrix[i][i] != nil {
			t.Errorf("matrix[%d][%d] should be nil", i, i)
		}
	}

	// Off-diagonal should have results
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i != j && matrix[i][j] == nil {
				t.Errorf("matrix[%d][%d] should not be nil", i, j)
			}
		}
	}
}

// CSV Output tests

func TestOutputForecastsToCSV(t *testing.T) {
	tmpFile := "test_forecasts.csv"
	defer os.Remove(tmpFile)

	fc := mat.NewDense(3, 2, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	varNames := []string{"var1", "var2"}

	rf := &ReducedFormVAR{}
	err := rf.OutputForecastsToCSV(tmpFile, fc, varNames)
	if err != nil {
		t.Fatalf("OutputForecastsToCSV returned error: %v", err)
	}

	// Check file exists
	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Output file was not created")
	}
}

func TestOutputIRFAnalysisToCSV(t *testing.T) {
	tmpFile := "test_irf_analysis.csv"
	defer os.Remove(tmpFile)

	analysis := map[int][]float64{
		0: {1.0, 0.5, 0.25},
		1: {0.0, 0.1, 0.2},
	}
	varNames := []string{"x", "y"}

	rf := &ReducedFormVAR{}
	err := rf.OutputIRFAnalysisToCSV(tmpFile, analysis, varNames)
	if err != nil {
		t.Fatalf("OutputIRFAnalysisToCSV returned error: %v", err)
	}

	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Output file was not created")
	}
}

func TestOutputGrangerMatrixToCSV(t *testing.T) {
	tmpFile := "test_granger_matrix.csv"
	defer os.Remove(tmpFile)

	varNames := []string{"a", "b"}
	gcMatrix := [][]*GrangerCausalityResult{
		{nil, {CauseVar: "a", EffectVar: "b", FStatistic: 2.5, PValue: 0.1, Lags: 1, Significant: false}},
		{{CauseVar: "b", EffectVar: "a", FStatistic: 5.0, PValue: 0.02, Lags: 1, Significant: true}, nil},
	}

	rf := &ReducedFormVAR{}
	err := rf.OutputGrangerMatrixToCSV(tmpFile, gcMatrix, varNames)
	if err != nil {
		t.Fatalf("OutputGrangerMatrixToCSV returned error: %v", err)
	}

	if _, err := os.Stat(tmpFile); os.IsNotExist(err) {
		t.Error("Output file was not created")
	}
}

// computeResiduals tests

func TestComputeResiduals(t *testing.T) {
	// Generate data from known VAR(1): y_t = 0.5 * y_{t-1}
	data := []float64{1.0, 0.5, 0.25, 0.125, 0.0625}
	T := len(data)
	Y := mat.NewDense(T, 1, data)

	ts := &TimeSeries{
		Y:        Y,
		VarNames: []string{"y"},
	}

	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetNone,
	}

	A1 := mat.NewDense(1, 1, []float64{0.5})
	rf := &ReducedFormVAR{
		Model: spec,
		A:     []*mat.Dense{A1},
	}

	resU, err := rf.computeResiduals(ts)
	if err != nil {
		t.Fatalf("computeResiduals returned error: %v", err)
	}

	rows, cols := resU.Dims()
	expectedRows := T - 1 // T - p
	if rows != expectedRows {
		t.Errorf("residuals rows = %d, want %d", rows, expectedRows)
	}
	if cols != 1 {
		t.Errorf("residuals cols = %d, want 1", cols)
	}

	// Residuals should be very close to 0 for perfect data
	for i := 0; i < rows; i++ {
		if !almostEqual(resU.At(i, 0), 0.0, 1e-6) {
			t.Errorf("residual[%d] = %v, want ~0", i, resU.At(i, 0))
		}
	}
}

// bootstrapQuantile tests

func TestBootstrapQuantile(t *testing.T) {
	samples := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	// Test median (q=0.5)
	median := bootstrapQuantile(samples, 0.5)
	if !almostEqual(median, 3.0, 1e-6) {
		t.Errorf("median = %v, want 3.0", median)
	}

	// Test min (q=0)
	min := bootstrapQuantile(samples, 0.0)
	if !almostEqual(min, 1.0, 1e-6) {
		t.Errorf("min = %v, want 1.0", min)
	}

	// Test max (q=1)
	max := bootstrapQuantile(samples, 1.0)
	if !almostEqual(max, 5.0, 1e-6) {
		t.Errorf("max = %v, want 5.0", max)
	}

	// Test 25th percentile
	q25 := bootstrapQuantile(samples, 0.25)
	if q25 < 1.0 || q25 > 3.0 {
		t.Errorf("q25 = %v, want between 1.0 and 3.0", q25)
	}

	// Test empty slice
	empty := bootstrapQuantile([]float64{}, 0.5)
	if !math.IsNaN(empty) {
		t.Errorf("quantile of empty slice = %v, want NaN", empty)
	}
}

// simulateBootstrapSeries tests

func TestSimulateBootstrapSeries(t *testing.T) {
	T := 20
	K := 2
	p := 2

	// Create simple data
	data := make([]float64, T*K)
	for i := range data {
		data[i] = float64(i) * 0.1
	}
	Y := mat.NewDense(T, K, data)

	ts := &TimeSeries{
		Y:        Y,
		Time:     nil,
		VarNames: []string{"x", "y"},
	}

	spec := ModelSpec{
		Lags:          p,
		Deterministic: DetConst,
	}

	// Simple coefficient matrices
	A1 := mat.NewDense(K, K, []float64{0.3, 0.1, 0.1, 0.3})
	A2 := mat.NewDense(K, K, []float64{0.1, 0.0, 0.0, 0.1})
	C := mat.NewDense(K, 1, []float64{0.1, 0.1})

	rf := &ReducedFormVAR{
		Model: spec,
		A:     []*mat.Dense{A1, A2},
		C:     C,
	}

	// Create residuals matrix (T-p x K)
	resU := mat.NewDense(T-p, K, nil)
	for i := 0; i < T-p; i++ {
		for j := 0; j < K; j++ {
			resU.Set(i, j, 0.01*float64(i+j))
		}
	}

	// Create RNG
	rng := rand.New(rand.NewSource(42))

	tsStar, err := rf.simulateBootstrapSeries(ts, resU, rng)
	if err != nil {
		t.Fatalf("simulateBootstrapSeries returned error: %v", err)
	}

	// Check dimensions
	rows, cols := tsStar.Y.Dims()
	if rows != T {
		t.Errorf("bootstrap Y rows = %d, want %d", rows, T)
	}
	if cols != K {
		t.Errorf("bootstrap Y cols = %d, want %d", cols, K)
	}

	// First p rows should match original
	for i := 0; i < p; i++ {
		for j := 0; j < K; j++ {
			if tsStar.Y.At(i, j) != ts.Y.At(i, j) {
				t.Errorf("bootstrap Y[%d][%d] = %v, want %v (first p rows should match)",
					i, j, tsStar.Y.At(i, j), ts.Y.At(i, j))
			}
		}
	}
}

// BootstrapIRF tests

func TestBootstrapIRF_SmallSample(t *testing.T) {
	// This is a smoke test - just verify it runs without error
	T := 30
	K := 2

	data := make([]float64, T*K)
	for i := range data {
		data[i] = float64(i%10) * 0.1
	}
	Y := mat.NewDense(T, K, data)

	ts := &TimeSeries{
		Y:        Y,
		VarNames: []string{"x", "y"},
	}

	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetConst,
	}

	rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	opts := BootstrapOptions{
		NReplications: 10, // Small for testing
		Horizon:       5,
		Alpha:         0.05,
		Seed:          42,
	}

	results, err := rf.BootstrapIRF(ts, opts)
	if err != nil {
		t.Fatalf("BootstrapIRF returned error: %v", err)
	}

	// Should have results for each shock variable
	if len(results) != K {
		t.Errorf("len(results) = %d, want %d", len(results), K)
	}

	// Check structure of results
	for shockIdx, res := range results {
		if res.ShockIndex != shockIdx {
			t.Errorf("ShockIndex = %d, want %d", res.ShockIndex, shockIdx)
		}
		if res.Horizon != opts.Horizon {
			t.Errorf("Horizon = %d, want %d", res.Horizon, opts.Horizon)
		}
		if res.Point == nil {
			t.Errorf("Point IRF is nil for shock %d", shockIdx)
		}
		if res.Lower == nil {
			t.Errorf("Lower CI is nil for shock %d", shockIdx)
		}
		if res.Upper == nil {
			t.Errorf("Upper CI is nil for shock %d", shockIdx)
		}
	}
}

// BootstrapGrangerMatrix tests

func TestBootstrapGrangerMatrix_SmallSample(t *testing.T) {
	// Smoke test - verify it runs
	T := 30
	K := 2

	data := make([]float64, T*K)
	for i := range data {
		data[i] = float64(i%10)*0.1 + float64(i/10)*0.01
	}
	Y := mat.NewDense(T, K, data)

	ts := &TimeSeries{
		Y:        Y,
		VarNames: []string{"x", "y"},
	}

	spec := ModelSpec{
		Lags:          1,
		Deterministic: DetConst,
	}

	rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	opts := GrangerBootstrapOptions{
		NReplications: 10, // Small for testing
		Alpha:         0.05,
		Seed:          42,
	}

	results, err := rf.BootstrapGrangerMatrix(ts, opts)
	if err != nil {
		t.Fatalf("BootstrapGrangerMatrix returned error: %v", err)
	}

	// Should be KxK matrix
	if len(results) != K {
		t.Errorf("len(results) = %d, want %d", len(results), K)
	}

	// Diagonal should be nil
	for i := 0; i < K; i++ {
		if results[i][i] != nil {
			t.Errorf("results[%d][%d] should be nil (diagonal)", i, i)
		}
	}

	// Off-diagonal should have results
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i != j {
				if results[i][j] == nil {
					t.Errorf("results[%d][%d] should not be nil", i, j)
				} else {
					// Check bootstrap p-value is valid
					if results[i][j].BootPValue < 0 || results[i][j].BootPValue > 1 {
						t.Errorf("BootPValue out of range: %v", results[i][j].BootPValue)
					}
				}
			}
		}
	}
}
