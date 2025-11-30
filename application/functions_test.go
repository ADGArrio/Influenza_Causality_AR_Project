package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// helper: compare floats with tolerance
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// --- Forecast tests ---

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

// --- IRF tests ---

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

// --- Estimate tests ---

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
