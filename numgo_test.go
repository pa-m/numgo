// a simple numpy-like subset for 1d float64 slices
package numgo

import (
	"fmt"
	"testing"
  "math"
)

var np = NumGo{}

func fl(d ...float64) []float64 {
	return d
}

func TestAddSub(t *testing.T) {
	r := np.Sub(np.Add(fl(1, 2), fl(4, 3)), fl(4, 4))
	if r[0] != 1. || r[1] != 1. {
		t.Fail()
	}
	fmt.Println("TestAddSub ok")
}

func TestMulDiv(t *testing.T) {
	r := np.Divide(np.Multiply(fl(1, 2), fl(4, 3)), fl(2, 2))
	if r[0] != 2. || r[1] != 3. {
		fmt.Println(r)
		t.Fail()
	}
	fmt.Println("TestMulDiv ok")
}

func TestZeros(t *testing.T) {
	if !np.Allclose(fl(0, 0, 0), np.Zeros([]int{3})) {
		t.Fail()
	}
  if !np.Allclose(fl(0, 0, 0), np.Zeros(3)) {
		t.Fail()
	}
}
func TestOnes(t *testing.T) {
	if !np.Allclose(fl(1, 1, 1), np.Ones([]int{3})) {
		t.Fail()
	}
  if !np.Allclose(fl(1, 1, 1), np.Ones(3)) {
		t.Fail()
	}
}
func TestSum(t *testing.T) {
	if 6. != np.Sum(fl(1, 2, 3)) {
		t.Fail()
	}
}
func TestMin(t *testing.T) {
	e := 1.
	a := np.Min(fl(1, 2, 3))
	if e != a {
		fmt.Println(a)
		t.Fail()
	}
}
func TestMax(t *testing.T) {
	e := 3.
	a := np.Max(fl(1, 2, 3))
	if e != a {
		fmt.Println(a)
		t.Fail()
	}
}
func TestMean(t *testing.T) {
	if 2. != np.Mean(fl(1, 2, 3)) {
		t.Fail()
	}
}
func TestMinimum(t *testing.T) {
	if !np.Allclose(fl(1, 2, 2), np.Minimum(fl(1, 2, 3), fl(2, 2, 2))) {
		t.Fail()
	}
}
func TestMaximum(t *testing.T) {
	e := fl(2, 2, 3)
	a := np.Maximum(fl(1, 2, 3), fl(2, 2, 2))
	if !np.Allclose(e, a) {
		fmt.Println(a)
		t.Fail()
	}
}
func TestLinspace(t *testing.T) {
	e := fl(-3, -2, -1, 0, 1, 2, 3)
	a := np.Linspace(-3, 3, 7, true)
	if !np.Allclose(e, a) {
		fmt.Println(a)
		t.Fail()
	}
}
func TestLinspace2(t *testing.T) {
	e := fl(-3, -2, -1, 0, 1, 2)
	a := np.Linspace(-3, 3, 6, false)
	if !np.Allclose(e, a) {
		fmt.Println(a)
		t.Fail()
	}
}
func TestLinspace3(t *testing.T) {
  e:=fl(0)
  a:=np.Linspace(0,0,1,true)
  if !np.Allclose(e, a) || math.IsNaN(a[0]) {
		fmt.Println(a)
		t.Fail()
	}
}
func TestLogspace(t *testing.T) {
	e := fl(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3)
	a := np.Logspace(-3, 3, 7, true, 10)
	if !np.Allclose(e, a) {
		fmt.Println(a)
		t.Fail()
	}
}
func TestLogspace2(t *testing.T) {
	e := fl(1e-3, 1e-2, 1e-1, 1, 1e1, 1e2)
	a := np.Logspace(-3, 3, 6, false, 10.)
	if !np.Allclose(e, a) {
		fmt.Println(a)
		t.Fail()
	}
}
func TestArgsort(t *testing.T) {
	e := []int{0, 2, 1}
	a := np.Argsort(fl(1, 3, 2))
	if e[0] != a[0] || e[1] != a[1] || e[2] != a[2] {
		t.Fail()
	}
}
func TestArgmin(t *testing.T) {
	e := 0
	a := np.Argmin(fl(1, 3, 2))
	if e != a {
		t.Fail()
	}
}
func TestArgmax(t *testing.T) {
	e := 1
	a := np.Argmax(fl(1, 3, 2))
	if e != a {
		fmt.Println(a)
		t.Fail()
	}
}
func TestSquare(t *testing.T) {
	e := fl(1, 4, 9)
	a := np.Square(fl(1, -2, 3))
	if !np.Allclose(e, a) {
		t.Fail()
	}
}

func TestExpitLogit(t *testing.T) {
	e := fl(1, -2, 3)
	a := np.Logit(np.Expit(fl(1, -2, 3)))
	if !np.Allclose(e, a) {
		t.Fail()
	}
}
func TestAbsolute(t *testing.T) {
	e := fl(1, 2, 3)
	a := np.Absolute(fl(1, -2, 3))
	if !np.Allclose(e, a) {
		t.Fail()
	}
}

func TestReciprocal(t *testing.T) {
  e:= fl(1.,-.5,1./3.)
  a := np.Reciprocal(fl(1,-2,3))
  if !np.Allclose(e, a) {
    t.Fail()
  }
}
//square,expit,logit,absolute,allclose
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
