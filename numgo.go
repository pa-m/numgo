// a simple numpy-like subset for 1d float64 slices. For numpy addict. You may use gonum instead
package numgo

import (
	"fmt"
	"math"
	"reflect"
	"sort"
)

type ifloatslice interface {
	Len() int
	Index(int) float64
}

type NumGo struct{}

var np = NumGo{}

type Float0 float64

func (Float0) Len() int { return 1 }
func (o Float0) Index(i int) float64 {
	if i != 0 {
		panic("invalid index")
	}
	return float64(o)
}

type Float1 []float64

func (o Float1) Len() int            { return len(o) }
func (o Float1) Index(i int) float64 { return o[i] }
func (o Float1) Swap(i, j int)       { a := o[i]; o[i] = o[j]; o[j] = a }
func (o Float1) Less(i, j int) bool  { return o[i] < o[j] }
func (NumGo) Array(ai interface{}) Float1 {
	var a Float1
	switch atyped := ai.(type) {

	case int:
		a = Float1([]float64{float64(atyped)})
	case float64:
		a = Float1([]float64{(atyped)})
	case Float0:
		a = Float1([]float64{float64(atyped)})
	case []float64:
		a = Float1(atyped)
	case Float1:
		a = atyped
	default:
		panic(fmt.Sprintf("unhandled type %T", atyped))
	}
	for i, ai := range a {
		if math.IsNaN(ai) {
			panic(fmt.Sprintf("nan at index %d", i))
		}
	}
	return a
}

func ewize1(ai interface{}, f func(x float64) float64) Float1 {
	a := np.Array(ai)
	r := make([]float64, a.Len(), a.Len())
	for i := 0; i < a.Len(); i++ {
		r[i] = f(a.Index(i))
	}
	return Float1(r)
}

func ewize2(ai, bi interface{}, f func(x, y float64) float64) Float1 {
	a, b := np.Array(ai), np.Array(bi)
	if a.Len() != b.Len() {
		if a.Len() == 1 && b.Len() > 1 {
			a = (NumGo{}).Full(b.Len(), a.Index(0))
		} else if a.Len() > 1 && b.Len() == 1 {
			b = (NumGo{}).Full(a.Len(), b.Index(0))

		} else {
			panic("ewize2:len mismatch")
		}
	}
	r := make([]float64, a.Len(), a.Len())
	for i := 0; i < a.Len(); i++ {
		r[i] = f(a.Index(i), b.Index(i))
	}
	return Float1(r)
}

func reduce_util(rv reflect.Value, f func(float64)) {
	if rv.Kind() == reflect.Float64 {
		f(rv.Float())
	} else if rv.Kind() == reflect.Slice {
		for i := 0; i < rv.Len(); i++ {
			reduce_util(rv.Index(i), f)
		}
	} else {
		panic(fmt.Sprintf("invalid Kind %s", rv.Kind()))
	}
}

func reduce(ai interface{}, f func(carry, item float64) float64, init float64) float64 {
	carry := init
	reduce_util(reflect.ValueOf(ai), func(item float64) { carry = f(carry, item) })
	return carry
}

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

func (NumGo) Full(ishape interface{}, fill_value float64) Float1 {
	n, ok := ishape.(int)
	if !ok {
		shape := ishape.([]int)
		n = shape[0]
	}
	r := make([]float64, n, n)
	for i := 0; i < n; i++ {
		r[i] = fill_value
	}
	return Float1(r)
}
func (NumGo) Zeros(shape interface{}) Float1 {
	return (NumGo{}).Full(shape, 0.)
}
func (NumGo) Ones(shape interface{}) Float1 {
	return (NumGo{}).Full(shape, 1.)
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

func (NumGo) Minimum(ais ...interface{}) Float1 {
	f := math.Min
	r := np.Copy(ais[0])
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
	}
	return r
}
func (NumGo) Maximum(ais ...interface{}) Float1 {
	f := math.Max
	r := np.Copy(ais[0])
	for _, b := range ais[1:] {
		r = ewize2(r, b, f)
	}
	return r
}

func (NumGo) Copy(ai interface{}) Float1 {
	a := np.Array(ai)
	r := make(Float1, a.Len(), a.Len())
	copy(r, a)
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

func (NumGo) Expit(x interface{}) Float1 {
	return ewize1(x, func(xi float64) float64 { return 1. / (1 + math.Exp(-xi)) })
}

func (NumGo) Logit(x interface{}) Float1 {
	return ewize1(x, func(xi float64) float64 { return math.Log(xi / (1. - xi)) })
}
func (NumGo) Absolute(x interface{}) Float1 {
	return ewize1(x, math.Abs)
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

func (NumGo) Reciprocal(x interface{}) Float1 {
	return ewize1(x, func(x float64) float64 { return 1. / x })
}
