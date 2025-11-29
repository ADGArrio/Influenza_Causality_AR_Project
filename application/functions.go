package main

import (
	"gonum.org/v1/gonum/mat"
)

// --- FUNCTIONS FOR BASE ---
// Returns current Model Spec
func (rf *ReducedFormVAR) Spec() ModelSpec { return rf.Model }

// Returns coefficient matrices
func (rf *ReducedFormVAR) Phi() []*mat.Dense { return rf.A }

// Returns error covariance matrix
func (rf *ReducedFormVAR) CovU() *mat.SymDense { return rf.SigmaU }

// --- OLS IMPLEMENTATION ---
func (e *OLSEstimator) Estimate(ts *TimeSeries, spec ModelSpec, opts EstimationOptions) (*ReducedFormVAR, error) {
	// 1. Build lagged design matrix X and Y
	// 2. Run (possibly multiple) linear regressions
	// 3. Assemble Phi, C, SigmaU
	// 4. Return &ReducedFormVAR{...}
	return nil, nil
}
