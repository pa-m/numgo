// a simple numpy-like subset for 1d float64 slices
package numgo

import (
	"fmt"
	"math"
	"sort"
)

type NumGo struct{}

func ewize1(a []float64, f func(x float64) float64) []float64 {
	r := make([]float64, len(a), len(a))
	for i, xi := range a {
		r[i] = f(xi)
	}
	return r
}
func ewize2(a, b []float64, f func(x, y float64) float64) []float64 {
	if len(a) != len(b) {
		panic("maximum:len mismatch")
	}
	r := make([]float64, len(a), len(a))
	for i := range a {
		r[i] = f(a[i], b[i])
	}
	return r
}

func (NumGo) Sum(a []float64) float64 {
	s := 0.
	for _, v := range a {
		s += v
	}
	return s
}
func (NumGo) Min(a []float64) float64 {
	r := a[0]
	for _, v := range a[1:] {
		if r > v {
			r = v
		}
	}
	return r
}
func (NumGo) Max(a []float64) float64 {
	r := a[0]
	for _, v := range a[1:] {
		if r < v {
			r = v
		}
	}
	return r
}
func (NumGo) Mean(a []float64) float64 {
	return (NumGo{}).Sum(a) / float64(len(a))
}
func (NumGo) Full(ishape interface{}, fill_value float64) []float64 {
	n, ok := ishape.(int)
	if !ok {
		shape := ishape.([]int)
		n = shape[0]
	}
	r := make([]float64, n, n)
	for i := 0; i < n; i++ {
		r[i] = fill_value
	}
	return r
}
func (NumGo) Zeros(shape interface{}) []float64 {
	return (NumGo{}).Full(shape, 0.)
}
func (NumGo) Ones(shape interface{}) []float64 {
	return (NumGo{}).Full(shape, 1.)
}
func (NumGo) Minimum(a, b []float64) []float64 {
	return ewize2(a, b, math.Min)
}
func (NumGo) Maximum(a, b []float64) []float64 {
	return ewize2(a, b, math.Max)
}

//NumGo.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]¶
func (NumGo) Linspace(start, stop float64, num int, endPoint bool) []float64 {
	step := 0.
	if endPoint {
		if num == 1 {
			return []float64{start}
		}
		step = (stop - start) / float64(num-1)
	} else {
		if num == 0 {
			return []float64{}
		}
		step = (stop - start) / float64(num)
	}
	r := make([]float64, num, num)
	for i := 0; i < num; i++ {
		r[i] = start + float64(i)*step
	}
	return r
}

func (NumGo) Logspace(start, stop float64, num int, endPoint bool, base float64) []float64 {
	return ewize1((NumGo{}).Linspace(start, stop, num, endPoint), func(x float64) float64 { return math.Pow(base, x) })
}

func (NumGo) Argsort(a []float64) []int {
	type T struct {
		I int
		V float64
	}
	aa := make([]T, len(a), len(a))
	for i, v := range a {
		aa[i] = T{i, v}
	}
	sort.Slice(aa, func(a, b int) bool { return aa[a].V < aa[b].V })
	r := make([]int, len(a), len(a))
	for i, aai := range aa {
		r[i] = aai.I
	}
	return r
}
func (NumGo) Argmin(a []float64) int {
	r := 0
	for i, v := range a {
		if a[r] > v {
			r = i
		}
	}
	return r
}
func (NumGo) Argmax(a []float64) int {
	r := 0
	for i, v := range a {
		if a[r] < v {
			r = i
		}
	}
	return r
}

func (NumGo) Add(a, b []float64) []float64 {
	return ewize2(a, b, func(ai, bi float64) float64 { return ai + bi })
}
func (NumGo) Sub(a, b []float64) []float64 {
	return ewize2(a, b, func(ai, bi float64) float64 { return ai - bi })
}
func (NumGo) Multiply(a, b []float64) []float64 {
	return ewize2(a, b, func(ai, bi float64) float64 { return ai * bi })
}
func (NumGo) Divide(a, b []float64) []float64 {
	return ewize2(a, b, func(ai, bi float64) float64 { return ai / bi })
}
func (NumGo) Square(a []float64) []float64 {
	return (NumGo{}).Multiply(a, a)
}

func (NumGo) Expit(x []float64) []float64 {
	return ewize1(x, func(xi float64) float64 { return 1. / (1 + math.Exp(-xi)) })
}

func (NumGo) Logit(x []float64) []float64 {
	return ewize1(x, func(xi float64) float64 { return math.Log(xi / (1. - xi)) })
}
func (NumGo) Absolute(x []float64) []float64 {
	return ewize1(x, math.Abs)
}

func (NumGo) Allclose(a, b []float64, opts ...float64) bool {
	if len(a) != len(b) {
		panic(fmt.Sprintf("allclose:len mismatch %d %d", len(a), len(b)))
	}
	rtol := 1e-5
	atol := 1e-8
	if len(opts) > 0 {
		rtol = opts[0]
	}
	if len(opts) > 1 {
		atol = opts[1]
	}
	// absolute(a - b) <= (atol + rtol * absolute(b))
	for i := range a {
		if math.Abs(a[i]-b[i]) > (atol + rtol*math.Abs(b[i])) {
			//fmt.Printf("At i=%d a=%g b=%g abs(a-b)=%g atol + rtol*math.Abs(b)=%g", i, a[i], b[i], math.Abs(a[i]-b[i]), (atol + rtol*math.Abs(b[i])))
			return false
		}

	}
	//fmt.Println("allclose:true")
	return true
}

func (NumGo) Reciprocal(x []float64) []float64 {
	return ewize1(x, func(x float64) float64 { return 1. / x })
}
