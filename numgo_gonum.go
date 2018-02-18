// +build gonum

package numgo

import (
	"fmt"
	"github.com/gonum/floats"
	"math"
)

func (NumGo) Sum(ai interface{}) float64 {
	return floats.Sum(np.Array(ai))
}
func (NumGo) Allclose(ai, bi interface{}, opts ...float64) bool {
	a, b := np.Array(ai), np.Array(bi)
	if a.Len() != b.Len() {
		panic(fmt.Sprintf("allclose:len mismatch %d %d", a.Len(), b.Len()))
	}
	rtol := 1e-5
	atol := 1e-8
	if len(opts) > 0 {
		rtol = opts[0]
	}
	if len(opts) > 1 {
		atol = opts[1]
	}
	for i := range a {
		if !floats.EqualWithinAbsOrRel(a.Index(i), b.Index(i), atol, rtol) {
			return false
		}
	}
	return true
}

func broadcast(r *Float1, bi interface{}) Float1 {
	b := np.Array(bi)
	if (*r).Len() == 1 && b.Len() > 1 {
		*r = np.Full(b.Len(), (*r).Index(0))
	} else if (*r).Len() > 1 && b.Len() == 1 {
		b = np.Full((*r).Len(), b.Index(0))
	}
	return b
}

func (NumGo) Add(ais ...interface{}) Float1 {
	r := np.Copy(ais[0])
	for _, bi := range ais[1:] {
		b := broadcast(&r, bi)
		floats.Add(r, b)
	}
	return r
}

func (NumGo) Sub(ais ...interface{}) Float1 {
	r := np.Copy(ais[0])
	for _, bi := range ais[1:] {
		b := broadcast(&r, bi)
		floats.Sub(r, b)
	}
	return r
}
func (NumGo) Multiply(ais ...interface{}) Float1 {
	r := np.Copy(ais[0])
	for _, bi := range ais[1:] {
		b := broadcast(&r, bi)
		floats.Mul(r, b)
	}
	return r
}
func (NumGo) Divide(ais ...interface{}) Float1 {
	r := np.Copy(ais[0])
	for _, bi := range ais[1:] {
		b := broadcast(&r, bi)
		floats.Div(r, b)
	}
	return r
}

func (NumGo) Power(ais ...interface{}) Float1 {
	f := func(a, b float64) float64 { return math.Pow(a, b) }
	a := np.Array(ais[0])
	r := make(Float1, a.Len(), a.Len())
	for i := 0; i < a.Len(); i++ {
		r[i] = a.Index(i)
	}
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
	}
	return r
}

func (NumGo) Square(a interface{}) Float1 {
	return (NumGo{}).Multiply(a, a)
}

func (NumGo) Min(ai interface{}) float64 {
	a := np.Array(ai)
	return floats.Min(a)
}
func (NumGo) Max(ai interface{}) float64 {
	a := np.Array(ai)
	return floats.Max(a)
}
func (NumGo) Mean(ai interface{}) float64 {
	a := np.Array(ai)
	return floats.Sum(a) / float64(a.Len())
}
func (NumGo) Median(ai interface{}) float64 {
	a := np.Array(ai)
	l := a.Len()
	ids := make([]int, l, l)
	floats.Argsort(a, ids)
	switch {
	case l == 0:
		return math.NaN()
	case l&1 == 1:
		return a[ids[(l-1)/2]]
	default:
		return (a[ids[l/2-1]] + a[ids[l/2]]) / 2.
	}
}

func (NumGo) Argsort(ai interface{}) []int {
	a := np.Array(ai)
	l := a.Len()
	i := make([]int, l, l)
	floats.Argsort(a, i)
	return i
}
func (NumGo) Argmin(ai interface{}) int {
	a := np.Array(ai)
	return floats.MinIdx(a)
}
func (NumGo) Argmax(ai interface{}) int {
	a := np.Array(ai)
	return floats.MaxIdx(a)
}

//NumGo.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]¶
func (NumGo) Linspace(start, stop float64, num int, endPoint bool) Float1 {
	if num == 0 {
		return []float64{}
	} else if num == 1 {
		return []float64{start}
	}
	num1 := num
	if !endPoint {
		num1 += 1
	}
	a := make([]float64, num1, num1)
	floats.Span(a, start, stop)
	return a[0:num]
}

func (NumGo) Logspace(start, stop float64, num int, endPoint bool, base float64) Float1 {
	return ewize1((NumGo{}).Linspace(start, stop, num, endPoint), func(x float64) float64 { return math.Pow(base, x) })
}

/*
func (NumGo) Logspace(start, stop float64, num int, endPoint bool, base float64) Float1 {
	if num == 0 {
		return []float64{}
	} else if num == 1 {
		return []float64{start}
	}
	num1 := num
	if !endPoint {
		num1 += 1
	}
	a := make([]float64, num1, num1)
	floats.LogSpan(a, math.Pow(10., start), math.Pow(10, stop))
	return a[0:num]
}*/
