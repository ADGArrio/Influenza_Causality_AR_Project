package main

import (
	"fmt"
	"os"
)

func main() {
	// expect 1 argument: country name
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <country_name>")
		return
	}
	country := os.Args[1]
	fmt.Println("Running VAR analysis for country:", country)
	// Determine filename based on country
	var filename string
	switch country {
	case "India":
		filename = "India_Training_Data.csv"
	case "USA":
		filename = "USA_Training_Data.csv"
	case "China":
		filename = "China_Training_Data.csv"
	default:
		panic("Unsupported country: " + country + ". Options: India, USA, China")
	}

	// 1. Load CSV into TimeSeries
	ts, err := LoadCSVToTimeSeries("../Files/Final_Training_Data/" + filename)
	if err != nil {
		panic(err)
	}

	fmt.Println("Loaded series with", ts.Y.RawMatrix().Rows, "rows and",
		ts.Y.RawMatrix().Cols, "variables:", ts.VarNames)

	// 2. Set up VAR spec
	spec := ModelSpec{
		Lags:          2,
		Deterministic: DetConst, // or DetConstTrend, etc.
		HasExogenous:  false,
	}

	// 3. Estimate VAR
	rf, err := (&OLSEstimator{}).Estimate(ts, spec, EstimationOptions{})
	if err != nil {
		panic(err)
	}

	rf.PrintCoefficients()

	// 4. Forecast 10 steps ahead
	fcst, err := rf.Forecast(ts.Y, 10)
	if err != nil {
		panic(err)
	}

	PrintForecast(fcst)

	// 5. IRF to shock sample variable 2
	irfMat, err := rf.IRF(12, 2)
	if err != nil {
		panic(err)
	}
	PrintIRF(irfMat, ts.VarNames, 2)

	// 6. Prints Summary
	rf.Summary(ts)
}
