// a simple numpy-like subset for 1d float64 slices. For numpy addict. You may use gonum instead
package numgo

import (
	"fmt"
	"math"
	"reflect"
)

type ifloatslice interface {
	Len() int
	Index(int) float64
}

type NumGo struct{}

func New() NumGo { return NumGo{} }

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

func (NumGo) Copy(ai interface{}) Float1 {
	a := np.Array(ai)
	r := make(Float1, a.Len(), a.Len())
	copy(r, a)
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

func (NumGo) Logit(x interface{}) Float1 {
	return ewize1(x, func(xi float64) float64 { return math.Log(xi / (1. - xi)) })
}

func (NumGo) Expit(x interface{}) Float1 {
	return ewize1(x, func(xi float64) float64 { return 1. / (1 + math.Exp(-xi)) })
}

func (NumGo) Absolute(x interface{}) Float1 {
	return ewize1(x, math.Abs)
}

func (NumGo) Reciprocal(x interface{}) Float1 {
	return ewize1(x, func(x float64) float64 { return 1. / x })
}
