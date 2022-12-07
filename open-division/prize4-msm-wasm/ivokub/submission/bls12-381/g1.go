// Copyright 2020 ConsenSys Software Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bls12381

import (
	"github.com/gbotrel/msmwasm/bls12-381/fp"
)

type pool[T fp.Element | g1JacExtended | schedItem] struct {
	data [1 << 18]T
	ptr  int
}

func (p *pool[T]) Get() *T {
	p.ptr = (p.ptr + 1) % len(p.data)
	return &(p.data[p.ptr])
}

func (p *pool[T]) Reset() {
	for i := range p.data {
		var fp T
		p.data[i] = fp
	}
	p.ptr = 0
}

var fpPool = &pool[fp.Element]{}

// G1Affine point in affine coordinates
type G1Affine struct {
	X, Y fp.Element
}

// G1Jac is a point with fp.Element coordinates
type G1Jac struct {
	X, Y, Z fp.Element
}

// g1JacExtended parameterized Jacobian coordinates (x=X/ZZ, y=Y/ZZZ, ZZ³=ZZZ²)
type g1JacExtended struct {
	X, Y, ZZ, ZZZ fp.Element
}

// -------------------------------------------------------------------------------------------------
// Affine

// Set sets p to the provided point
func (p *G1Affine) Set(a *G1Affine) *G1Affine {
	p.X, p.Y = a.X, a.Y
	return p
}

// setInfinity sets p to O
func (p *G1Affine) setInfinity() *G1Affine {
	p.X.SetZero()
	p.Y.SetZero()
	return p
}

// Equal tests if two points (in Affine coordinates) are equal
func (p *G1Affine) Equal(a *G1Affine) bool {
	return p.X.Equal(&a.X) && p.Y.Equal(&a.Y)
}

// Neg computes -G
func (p *G1Affine) Neg(a *G1Affine) *G1Affine {
	p.X = a.X
	p.Y.Neg(&a.Y)
	return p
}

// IsInfinity checks if the point is infinity
// in affine, it's encoded as (0,0)
// (0,0) is never on the curve for j=0 curves
func (p *G1Affine) IsInfinity() bool {
	return p.X.IsZero() && p.Y.IsZero()
}

func (p *G1Affine) FromJacobian(p1 *G1Jac) *G1Affine {

	// var a, b fp.Element
	a := fpPool.Get()
	b := fpPool.Get()

	if p1.Z.IsZero() {
		p.X.SetZero()
		p.Y.SetZero()
		return p
	}

	a.Inverse(&p1.Z)
	b.Square(a)
	p.X.Mul(&p1.X, b)
	p.Y.Mul(&p1.Y, b).Mul(&p.Y, a)

	return p
}

// -------------------------------------------------------------------------------------------------
// Jacobian

// Set sets p to the provided point
func (p *G1Jac) Set(a *G1Jac) *G1Jac {
	p.X, p.Y, p.Z = a.X, a.Y, a.Z
	return p
}

// Equal tests if two points (in Jacobian coordinates) are equal
func (p *G1Jac) Equal(a *G1Jac) bool {

	if p.Z.IsZero() && a.Z.IsZero() {
		return true
	}
	_p := G1Affine{}
	_p.FromJacobian(p)

	_a := G1Affine{}
	_a.FromJacobian(a)

	return _p.X.Equal(&_a.X) && _p.Y.Equal(&_a.Y)
}

// Neg computes -G
func (p *G1Jac) Neg(a *G1Jac) *G1Jac {
	*p = *a
	p.Y.Neg(&a.Y)
	return p
}

// SubAssign subtracts two points on the curve
func (p *G1Jac) SubAssign(a *G1Jac) *G1Jac {
	var tmp G1Jac
	tmp.Set(a)
	tmp.Y.Neg(&tmp.Y)
	p.AddAssign(&tmp)
	return p
}

// AddAssign point addition in montgomery form
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#addition-add-2007-bl
func (p *G1Jac) AddAssign(a *G1Jac) *G1Jac {

	// p is infinity, return a
	if p.Z.IsZero() {
		p.Set(a)
		return p
	}

	// a is infinity, return p
	if a.Z.IsZero() {
		return p
	}

	var Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r, V fp.Element
	Z1Z1.Square(&a.Z)
	Z2Z2.Square(&p.Z)
	U1.Mul(&a.X, &Z2Z2)
	U2.Mul(&p.X, &Z1Z1)
	S1.Mul(&a.Y, &p.Z).
		Mul(&S1, &Z2Z2)
	S2.Mul(&p.Y, &a.Z).
		Mul(&S2, &Z1Z1)

	// if p == a, we double instead
	if U1.Equal(&U2) && S1.Equal(&S2) {
		return p.DoubleAssign()
	}

	H.Sub(&U2, &U1)
	I.Double(&H).
		Square(&I)
	J.Mul(&H, &I)
	r.Sub(&S2, &S1).Double(&r)
	V.Mul(&U1, &I)
	p.X.Square(&r).
		Sub(&p.X, &J).
		Sub(&p.X, &V).
		Sub(&p.X, &V)
	p.Y.Sub(&V, &p.X).
		Mul(&p.Y, &r)
	S1.Mul(&S1, &J).Double(&S1)
	p.Y.Sub(&p.Y, &S1)
	p.Z.Add(&p.Z, &a.Z)
	p.Z.Square(&p.Z).
		Sub(&p.Z, &Z1Z1).
		Sub(&p.Z, &Z2Z2).
		Mul(&p.Z, &H)

	return p
}

// AddMixed point addition
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
func (p *G1Jac) AddMixed(a *G1Affine) *G1Jac {

	//if a is infinity return p
	if a.IsInfinity() {
		return p
	}
	// p is infinity, return a
	if p.Z.IsZero() {
		p.X = a.X
		p.Y = a.Y
		p.Z.SetOne()
		return p
	}

	var Z1Z1, U2, S2, H, HH, I, J, r, V fp.Element
	Z1Z1.Square(&p.Z)
	U2.Mul(&a.X, &Z1Z1)
	S2.Mul(&a.Y, &p.Z).
		Mul(&S2, &Z1Z1)

	// if p == a, we double instead
	if U2.Equal(&p.X) && S2.Equal(&p.Y) {
		return p.DoubleAssign()
	}

	H.Sub(&U2, &p.X)
	HH.Square(&H)
	I.Double(&HH).Double(&I)
	J.Mul(&H, &I)
	r.Sub(&S2, &p.Y).Double(&r)
	V.Mul(&p.X, &I)
	p.X.Square(&r).
		Sub(&p.X, &J).
		Sub(&p.X, &V).
		Sub(&p.X, &V)
	J.Mul(&J, &p.Y).Double(&J)
	p.Y.Sub(&V, &p.X).
		Mul(&p.Y, &r)
	p.Y.Sub(&p.Y, &J)
	p.Z.Add(&p.Z, &H)
	p.Z.Square(&p.Z).
		Sub(&p.Z, &Z1Z1).
		Sub(&p.Z, &HH)

	return p
}

// Double doubles a point in Jacobian coordinates
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2007-bl
func (p *G1Jac) Double(q *G1Jac) *G1Jac {
	p.Set(q)
	p.DoubleAssign()
	return p
}

// DoubleAssign doubles a point in Jacobian coordinates
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-3.html#doubling-dbl-2007-bl
func (p *G1Jac) DoubleAssign() *G1Jac {

	var XX, YY, YYYY, ZZ, S, M, T fp.Element

	XX.Square(&p.X)
	YY.Square(&p.Y)
	YYYY.Square(&YY)
	ZZ.Square(&p.Z)
	S.Add(&p.X, &YY)
	S.Square(&S).
		Sub(&S, &XX).
		Sub(&S, &YYYY).
		Double(&S)
	M.Double(&XX).Add(&M, &XX)
	p.Z.Add(&p.Z, &p.Y).
		Square(&p.Z).
		Sub(&p.Z, &YY).
		Sub(&p.Z, &ZZ)
	T.Square(&M)
	p.X = T
	T.Double(&S)
	p.X.Sub(&p.X, &T)
	p.Y.Sub(&S, &p.X).
		Mul(&p.Y, &M)
	YYYY.Double(&YYYY).Double(&YYYY).Double(&YYYY)
	p.Y.Sub(&p.Y, &YYYY)

	return p
}

// FromAffine sets p = Q, p in Jacboian, Q in affine
func (p *G1Jac) FromAffine(Q *G1Affine) *G1Jac {
	if Q.IsInfinity() {
		p.Z.SetZero()
		p.X.SetOne()
		p.Y.SetOne()
		return p
	}
	p.Z.SetOne()
	p.X.Set(&Q.X)
	p.Y.Set(&Q.Y)
	return p
}

// -------------------------------------------------------------------------------------------------
// Jacobian extended

// Set sets p to the provided point
func (p *g1JacExtended) Set(a *g1JacExtended) *g1JacExtended {
	p.X, p.Y, p.ZZ, p.ZZZ = a.X, a.Y, a.ZZ, a.ZZZ
	return p
}

// setInfinity sets p to O
func (p *g1JacExtended) setInfinity() *g1JacExtended {
	p.X.SetOne()
	p.Y.SetOne()
	p.ZZ = fp.Element{}
	p.ZZZ = fp.Element{}
	return p
}

// unsafeFromJacExtended sets p in Jacobian coordinates, but don't check for infinity
func (p *G1Jac) unsafeFromJacExtended(Q *g1JacExtended) *G1Jac {
	p.X.Square(&Q.ZZ).Mul(&p.X, &Q.X)
	p.Y.Square(&Q.ZZZ).Mul(&p.Y, &Q.Y)
	p.Z = Q.ZZZ
	return p
}

// add point in Jacobian extended coordinates
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
func (p *g1JacExtended) add(q *g1JacExtended) *g1JacExtended {
	//if q is infinity return p
	if q.ZZ.IsZero() {
		return p
	}
	// p is infinity, return q
	if p.ZZ.IsZero() {
		p.Set(q)
		return p
	}

	A := fpPool.Get()
	B := fpPool.Get()
	U1 := fpPool.Get()
	U2 := fpPool.Get()
	S1 := fpPool.Get()
	S2 := fpPool.Get()
	// var A, B, U1, U2, S1, S2 fp.Element

	// p2: q, p1: p
	U2.Mul(&q.X, &p.ZZ)
	U1.Mul(&p.X, &q.ZZ)
	A.Sub(U2, U1)
	S2.Mul(&q.Y, &p.ZZZ)
	S1.Mul(&p.Y, &q.ZZZ)
	B.Sub(S2, S1)

	if A.IsZero() {
		if B.IsZero() {
			return p.double(q)

		}
		p.ZZ = fp.Element{}
		p.ZZZ = fp.Element{}
		return p
	}

	P := fpPool.Get()
	R := fpPool.Get()
	PP := fpPool.Get()
	PPP := fpPool.Get()
	Q := fpPool.Get()
	V := fpPool.Get()
	// var P, R, PP, PPP, Q, V fp.Element
	P.Sub(U2, U1)
	R.Sub(S2, S1)
	PP.Square(P)
	PPP.Mul(P, PP)
	Q.Mul(U1, PP)
	V.Mul(S1, PPP)

	p.X.Square(R).
		Sub(&p.X, PPP).
		Sub(&p.X, Q).
		Sub(&p.X, Q)
	p.Y.Sub(Q, &p.X).
		Mul(&p.Y, R).
		Sub(&p.Y, V)
	p.ZZ.Mul(&p.ZZ, &q.ZZ).
		Mul(&p.ZZ, PP)
	p.ZZZ.Mul(&p.ZZZ, &q.ZZZ).
		Mul(&p.ZZZ, PPP)

	return p
}

// double point in Jacobian extended coordinates
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
func (p *g1JacExtended) double(q *g1JacExtended) *g1JacExtended {
	// var U, V, W, S, XX, M fp.Element
	U := fpPool.Get()
	V := fpPool.Get()
	W := fpPool.Get()
	S := fpPool.Get()
	XX := fpPool.Get()
	M := fpPool.Get()

	U.Double(&q.Y)
	V.Square(U)
	W.Mul(U, V)
	S.Mul(&q.X, V)
	XX.Square(&q.X)
	M.Double(XX).
		Add(M, XX) // -> + a, but a=0 here
	U.Mul(W, &q.Y)

	p.X.Square(M).
		Sub(&p.X, S).
		Sub(&p.X, S)
	p.Y.Sub(S, &p.X).
		Mul(&p.Y, M).
		Sub(&p.Y, U)
	p.ZZ.Mul(V, &q.ZZ)
	p.ZZZ.Mul(W, &q.ZZZ)

	return p
}

// subMixed same as addMixed, but will negate a.Y
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
func (p *g1JacExtended) subMixed(a *G1Affine) *g1JacExtended {

	//if a is infinity return p
	if a.IsInfinity() {
		return p
	}
	// p is infinity, return a
	if p.ZZ.IsZero() {
		p.X = a.X
		p.Y.Neg(&a.Y)
		p.ZZ.SetOne()
		p.ZZZ.SetOne()
		return p
	}

	// var P, R fp.Element
	P := fpPool.Get()
	R := fpPool.Get()

	// p2: a, p1: p
	P.Mul(&a.X, &p.ZZ)
	P.Sub(P, &p.X)

	R.Mul(&a.Y, &p.ZZZ)
	R.Neg(R)
	R.Sub(R, &p.Y)

	if P.IsZero() {
		if R.IsZero() {
			return p.doubleNegMixed(a)

		}
		p.ZZ = fp.Element{}
		p.ZZZ = fp.Element{}
		return p
	}

	// var PP, PPP, Q, Q2, RR, X3, Y3 fp.Element
	PP := fpPool.Get()
	PPP := fpPool.Get()
	Q := fpPool.Get()
	Q2 := fpPool.Get()
	RR := fpPool.Get()
	X3 := fpPool.Get()
	Y3 := fpPool.Get()

	PP.Square(P)
	PPP.Mul(P, PP)
	Q.Mul(&p.X, PP)
	RR.Square(R)
	X3.Sub(RR, PPP)
	Q2.Double(Q)
	p.X.Sub(X3, Q2)
	Y3.Sub(Q, &p.X).Mul(Y3, R)
	R.Mul(&p.Y, PPP)
	p.Y.Sub(Y3, R)
	p.ZZ.Mul(&p.ZZ, PP)
	p.ZZZ.Mul(&p.ZZZ, PPP)

	return p

}

var PP1, PPP1, Q1, Q21, RR1, X31, Y31 fp.Element

// addMixed
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
func (p *g1JacExtended) addMixed(a *G1Affine) *g1JacExtended {

	//if a is infinity return p
	if a.IsInfinity() {
		return p
	}
	// p is infinity, return a
	if p.ZZ.IsZero() {
		p.X = a.X
		p.Y = a.Y
		p.ZZ.SetOne()
		p.ZZZ.SetOne()
		return p
	}

	// var P, R fp.Element
	P := fpPool.Get()
	R := fpPool.Get()

	// p2: a, p1: p
	P.Mul(&a.X, &p.ZZ)
	P.Sub(P, &p.X)

	R.Mul(&a.Y, &p.ZZZ)
	R.Sub(R, &p.Y)

	if P.IsZero() {
		if R.IsZero() {
			return p.doubleMixed(a)

		}
		p.ZZ = fp.Element{}
		p.ZZZ = fp.Element{}
		return p
	}

	PP1.Square(P)
	PPP1.Mul(P, &PP1)
	Q1.Mul(&p.X, &PP1)
	RR1.Square(R)
	X31.Sub(&RR1, &PPP1)
	Q21.Double(&Q1)
	p.X.Sub(&X31, &Q21)
	Y31.Sub(&Q1, &p.X).Mul(&Y31, R)
	R.Mul(&p.Y, &PPP1)
	p.Y.Sub(&Y31, R)
	p.ZZ.Mul(&p.ZZ, &PP1)
	p.ZZZ.Mul(&p.ZZZ, &PPP1)

	return p

}

// doubleNegMixed same as double, but will negate q.Y
func (p *g1JacExtended) doubleNegMixed(q *G1Affine) *g1JacExtended {
	U := fpPool.Get()
	V := fpPool.Get()
	W := fpPool.Get()
	S := fpPool.Get()
	XX := fpPool.Get()
	M := fpPool.Get()
	S2 := fpPool.Get()
	L := fpPool.Get()

	// var U, V, W, S, XX, M, S2, L fp.Element

	U.Double(&q.Y)
	U.Neg(U)
	V.Square(U)
	W.Mul(U, V)
	S.Mul(&q.X, V)
	XX.Square(&q.X)
	M.Double(XX).
		Add(M, XX) // -> + a, but a=0 here
	S2.Double(S)
	L.Mul(W, &q.Y)

	p.X.Square(M).
		Sub(&p.X, S2)
	p.Y.Sub(S, &p.X).
		Mul(&p.Y, M).
		Add(&p.Y, L)
	p.ZZ.Set(V)
	p.ZZZ.Set(W)

	return p
}

// doubleMixed point in Jacobian extended coordinates
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
func (p *g1JacExtended) doubleMixed(q *G1Affine) *g1JacExtended {

	U := fpPool.Get()
	V := fpPool.Get()
	W := fpPool.Get()
	S := fpPool.Get()
	XX := fpPool.Get()
	M := fpPool.Get()
	S2 := fpPool.Get()
	L := fpPool.Get()
	// var V, W, S, XX, M, S2, L fp.Element

	U.Double(&q.Y)
	V.Square(U)
	W.Mul(U, V)
	S.Mul(&q.X, V)
	XX.Square(&q.X)
	M.Double(XX).
		Add(M, XX) // -> + a, but a=0 here
	S2.Double(S)
	L.Mul(W, &q.Y)

	p.X.Square(M).
		Sub(&p.X, S2)
	p.Y.Sub(S, &p.X).
		Mul(&p.Y, M).
		Sub(&p.Y, L)
	p.ZZ.Set(V)
	p.ZZZ.Set(W)

	return p
}

// // ScalarMultiplication computes and returns p = a ⋅ s
// func (p *G1Affine) ScalarMultiplication(a *G1Affine, s *big.Int) *G1Affine {
// 	var _p G1Jac
// 	_p.FromAffine(a)
// 	_p.ScalarMultiplication(&_p, s)
// 	p.FromJacobian(&_p)
// 	return p
// }

// // ScalarMultiplication computes and returns p = a ⋅ s
// // windowed-Mul
// func (p *G1Jac) ScalarMultiplication(a *G1Jac, s *big.Int) *G1Jac {
// 	var res G1Jac
// 	var ops [3]G1Jac

// 	res.Set(&g1Infinity)
// 	ops[0].Set(a)
// 	ops[1].Double(&ops[0])
// 	ops[2].Set(&ops[0]).AddAssign(&ops[1])

// 	b := s.Bytes()
// 	for i := range b {
// 		w := b[i]
// 		mask := byte(0xc0)
// 		for j := 0; j < 4; j++ {
// 			res.DoubleAssign().DoubleAssign()
// 			c := (w & mask) >> (6 - 2*j)
// 			if c != 0 {
// 				res.AddAssign(&ops[c-1])
// 			}
// 			mask = mask >> 2
// 		}
// 	}
// 	p.Set(&res)

// 	return p
// }
