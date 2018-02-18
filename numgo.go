// +build !gonum

// a simple numpy-like subset for 1d float64 slices. For numpy addict. You may use gonum instead
package numgo

import (
	"fmt"
	"math"
	"sort"
)

func (NumGo) Sum(a interface{}) float64 {
	return reduce(a, func(carry, item float64) float64 { return carry + item }, 0.)
}
func (NumGo) Min(ai interface{}) float64 {
	a := np.Array(ai)
	return reduce(a, func(carry, item float64) float64 { return math.Min(carry, item) }, a.Index(0))
}
func (NumGo) Max(ai interface{}) float64 {
	a := np.Array(ai)
	return reduce(a, func(carry, item float64) float64 { return math.Max(carry, item) }, a.Index(0))
}
func (NumGo) Mean(ai interface{}) float64 {
	a := np.Array(ai)
	return (NumGo{}).Sum(a) / float64(a.Len())
}
func (NumGo) Median(ai interface{}) float64 {
	a := np.Array(ai)
	a1 := np.Copy(a)
	sort.Sort(a1)
	l := len(a)
	switch {
	case l == 0:
		return math.NaN()
	case l&1 == 1:
		return a1[(l-1)/2]
	default:
		return (a1[l/2-1] + a1[l/2]) / 2.
	}
}

func (NumGo) Argsort(ai interface{}) []int {
	a := np.Array(ai)
	l := a.Len()
	type T struct {
		I int
		V float64
	}
	aa := make([]T, l, l)
	for i, v := range a {
		aa[i] = T{i, v}
	}
	sort.Slice(aa, func(a, b int) bool { return aa[a].V < aa[b].V })
	r := make([]int, l, l)
	for i, aai := range aa {
		r[i] = aai.I
	}
	return r
}
func (NumGo) Argmin(ai interface{}) int {
	a := np.Array(ai)
	r := 0
	for i := 0; i < a.Len(); i++ {
		if a.Index(r) > a.Index(i) {
			r = i
		}
	}
	return r
}
func (NumGo) Argmax(ai interface{}) int {
	a := np.Array(ai)
	r := 0
	for i := 0; i < a.Len(); i++ {
		if a.Index(r) < a.Index(i) {
			r = i
		}
	}
	return r
}

func (NumGo) Add(ais ...interface{}) Float1 {
	f := func(a, b float64) float64 { return a + b }
	r := np.Copy(ais[0])
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
	}
	return r
}

func (NumGo) Sub(ais ...interface{}) Float1 {
	f := func(a, b float64) float64 { return a - b }
	r := np.Copy(ais[0])
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
	}
	return r
}
func (NumGo) Multiply(ais ...interface{}) Float1 {
	f := func(a, b float64) float64 { return a * b }
	r := np.Copy(ais[0])
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
	}
	return r
}
func (NumGo) Divide(ais ...interface{}) Float1 {
	f := func(a, b float64) float64 { return a / b }
	r := np.Copy(ais[0])
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
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
	// absolute(a - b) <= (atol + rtol * absolute(b))
	for i := 0; i < a.Len(); i++ {
		if math.Abs(a.Index(i)-b.Index(i)) > (atol + rtol*math.Abs(b.Index(i))) {
			//fmt.Printf("At i=%d a=%g b=%g abs(a-b)=%g atol + rtol*math.Abs(b)=%g", i, a[i], b[i], math.Abs(a[i]-b[i]), (atol + rtol*math.Abs(b[i])))
			return false
		}

	}
	//fmt.Println("allclose:true")
	return true
}

//NumGo.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)[source]¶
func (NumGo) Linspace(start, stop float64, num int, endPoint bool) Float1 {
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

func (NumGo) Logspace(start, stop float64, num int, endPoint bool, base float64) Float1 {
	return ewize1((NumGo{}).Linspace(start, stop, num, endPoint), func(x float64) float64 { return math.Pow(base, x) })
}
