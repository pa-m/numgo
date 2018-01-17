# numgo
##is a small numpy-like module for my own use. use gonum instead

[![Build Status](https://travis-ci.org/pa-m/numgo.svg?branch=master)](https://travis-ci.org/pa-m/numgo)
[![Code Coverage](https://codecov.io/gh/pa-m/numgo/branch/master/graph/badge.svg)](https://codecov.io/gh/pa-m/numgo)
[![Go Report Card](https://goreportcard.com/badge/github.com/pa-m/numgo)](https://goreportcard.com/report/github.com/pa-m/numgo)
[![GoDoc](https://godoc.org/github.com/pa-m/numgo?status.svg)](https://godoc.org/github.com/pa-m/numgo)

###usage:
```go
import (
  "fmt"
  "pa-m/numgo"
)

var np = numgo.NomGo{}

func fl(d ...float64) []float64 {
	return d
}

func Example() {
	fmt.Println("zeros:", np.Zeros([]int{3}))
	fmt.Println("ones:", np.Ones([]int{3}))
	fmt.Println("sum:", np.Sum(fl(1., 2., 3.)))
	fmt.Println("min:", np.Min(fl(1., 2., 3.)))
	fmt.Println("max:", np.Max(fl(1., 2., 3.)))
	fmt.Println("mean:", np.Mean(fl(1., 2., 3.)))
	fmt.Println("minimum:", np.Minimum(fl(1., 2., 3.), fl(2., 2., 2.)))
	fmt.Println("maximum:", np.Maximum(fl(1., 2., 3.), fl(2., 2., 2.)))
	fmt.Println("linspace:", np.Linspace(-5., 5., 11, true), np.Linspace(-5, 5, 10, false))
	fmt.Println("logspace:", np.Logspace(-5., 5., 11, true, 10.), np.Logspace(-5, 5, 10, false, 10.))
	fmt.Println("argsort:", np.Argsort(fl(1., 3., 2.)))
	fmt.Println("argmin:", np.Argmin(fl(1., 3., 2.)))
	fmt.Println("argmax:", np.Argmax(fl(1., 3., 2.)))
	fmt.Println("add:", np.Add(fl(1., 2., 3.), fl(1., 2., 3.)))
	fmt.Println("sub:", np.Sub(fl(1., 2., 3.), fl(1., 2., 3.)))
	fmt.Println("multiply:", np.Multiply(fl(1., 2., 3.), fl(1., 2., 3.)))
	fmt.Println("divide:", np.Divide(fl(1., 2., 3.), fl(1., 2., 3.)))
	fmt.Println("square:", np.Square(fl(1., 2., 3.)))
	// Output:
	// zeros: [0 0 0]
	// ones: [1 1 1]
	// sum: 6
	// min: 1
	// max: 3
	// mean: 2
	// minimum: [1 2 2]
	// maximum: [2 2 3]
	// linspace: [-5 -4 -3 -2 -1 0 1 2 3 4 5] [-5 -4 -3 -2 -1 0 1 2 3 4]
	// logspace: [1e-05 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000] [1e-05 0.0001 0.001 0.01 0.1 1 10 100 1000 10000]
	// argsort: [0 2 1]
	// argmin: 0
	// argmax: 1
	// add: [2 4 6]
	// sub: [0 0 0]
	// multiply: [1 4 9]
	// divide: [1 1 1]
	// square: [1 4 9]
}

```
