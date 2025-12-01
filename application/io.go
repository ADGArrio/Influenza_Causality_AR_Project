package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// LoadCSVToTimeSeries reads CSV file:
//
//   - The first row is a header with variable names
//   - All remaining rows are numeric values
//   - There is no explicit time column; time is taken as 0,1,2,...
//
// Returns TimeSeries with:
//   - Y: T x K matrix (rows: time points, cols: variables)
//   - Time: []float64 of length T
//   - VarNames: []string of length K
func LoadCSVToTimeSeries(path string) (*TimeSeries, error) {
	// 1. Open file
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	// 2. Make CSV reader
	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	// 3. Read header row
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if len(header) == 0 {
		return nil, fmt.Errorf("empty header in %s", path)
	}
	K := len(header) // number of variables

	var (
		data  []float64 // flat data for mat.Dense
		times []float64 // time index
		row   int       // row counter
	)

	// 4. Read each data row
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row %d: %w", row+2, err) // +2 for header + 1-based
		}

		// Skip completely empty lines (optional, but nice to have)
		if len(record) == 1 && record[0] == "" {
			continue
		}

		if len(record) != K {
			return nil, fmt.Errorf(
				"row %d: expected %d columns, got %d",
				row+2, K, len(record),
			)
		}

		for j, s := range record {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, fmt.Errorf(
					"parse float at row %d col %d (%q): %w",
					row+2, j+1, s, err,
				)
			}
			data = append(data, v)
		}

		// Here we just use a simple time index: 0,1,2,...
		times = append(times, float64(row))
		row++
	}

	if row == 0 {
		return nil, fmt.Errorf("no data rows in %s", path)
	}

	T := row

	// 5. Build mat.Dense
	Y := mat.NewDense(T, K, data)

	// 6. Build TimeSeries
	ts := &TimeSeries{
		Y:        Y,
		Time:     times,
		VarNames: header,
	}

	return ts, nil
}

// Helper function to print coefficient matrices
func (rf *ReducedFormVAR) PrintCoefficients() {
	for i, Ai := range rf.A {
		fmt.Printf("\n=== A_%d ===\n", i+1)
		fmt.Printf("%v\n", mat.Formatted(Ai, mat.Prefix(" ")))
	}

	fmt.Println("\n=== Covariance Matrix Σ_u ===")
	fmt.Printf("%v\n", mat.Formatted(rf.SigmaU, mat.Prefix(" ")))
}

// Helper function to print forecasts
func PrintForecast(fc *mat.Dense) {
	fmt.Println("\n=== Forecast Matrix ===")
	fmt.Printf("%v\n", mat.Formatted(fc, mat.Prefix(" ")))
}
