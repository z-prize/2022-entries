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

package fp

import (
	"encoding/binary"
	"math/bits"
)

// Element represents a field element stored on 6 words (uint64)
//
// Element are assumed to be in Montgomery form in all methods.
//
// Modulus q =
//
//	q[base10] = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
//	q[base16] = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
type Element [6]uint64

const (
	Limbs = 6         // number of 64 bits words needed to represent a Element
	Bits  = 381       // number of bits needed to represent a Element
	Bytes = Limbs * 8 // number of bytes needed to represent a Element
)

// Field modulus q
const (
	q0 uint64 = 13402431016077863595
	q1 uint64 = 2210141511517208575
	q2 uint64 = 7435674573564081700
	q3 uint64 = 7239337960414712511
	q4 uint64 = 5412103778470702295
	q5 uint64 = 1873798617647539866
)

var qElement = Element{
	q0,
	q1,
	q2,
	q3,
	q4,
	q5,
}

// q + r'.r = 1, i.e., qInvNeg = - q⁻¹ mod r
// used for Montgomery reduction
const qInvNeg uint64 = 9940570264628428797

// SetUint64 sets z to v and returns z
func (z *Element) SetUint64(v uint64) *Element {
	//  sets z LSB to v (non-Montgomery form) and convert z to Montgomery form
	*z = Element{v}
	return z.Mul(z, &rSquare) // z.ToMont()
}

// SetInt64 sets z to v and returns z
func (z *Element) SetInt64(v int64) *Element {

	// absolute value of v
	m := v >> 63
	z.SetUint64(uint64((v ^ m) - m))

	if m != 0 {
		// v is negative
		z.Neg(z)
	}

	return z
}

// Set z = x and returns z
func (z *Element) Set(x *Element) *Element {
	z[0] = x[0]
	z[1] = x[1]
	z[2] = x[2]
	z[3] = x[3]
	z[4] = x[4]
	z[5] = x[5]
	return z
}

// SetZero z = 0
func (z *Element) SetZero() *Element {
	z[0] = 0
	z[1] = 0
	z[2] = 0
	z[3] = 0
	z[4] = 0
	z[5] = 0
	return z
}

// SetOne z = 1 (in Montgomery form)
func (z *Element) SetOne() *Element {
	z[0] = 8505329371266088957
	z[1] = 17002214543764226050
	z[2] = 6865905132761471162
	z[3] = 8632934651105793861
	z[4] = 6631298214892334189
	z[5] = 1582556514881692819
	return z
}

// One returns 1
func One() Element {
	var one Element
	one.SetOne()
	return one
}

// Equal returns z == x; constant-time
func (z *Element) Equal(x *Element) bool {
	return z.NotEqual(x) == 0
}

// NotEqual returns 0 if and only if z == x; constant-time
func (z *Element) NotEqual(x *Element) uint64 {
	return (z[5] ^ x[5]) | (z[4] ^ x[4]) | (z[3] ^ x[3]) | (z[2] ^ x[2]) | (z[1] ^ x[1]) | (z[0] ^ x[0])
}

// IsZero returns z == 0
func (z *Element) IsZero() bool {
	return (z[5] | z[4] | z[3] | z[2] | z[1] | z[0]) == 0
}

// IsOne returns z == 1
func (z *Element) IsOne() bool {
	return (z[5] ^ 1582556514881692819 | z[4] ^ 6631298214892334189 | z[3] ^ 8632934651105793861 | z[2] ^ 6865905132761471162 | z[1] ^ 17002214543764226050 | z[0] ^ 8505329371266088957) == 0
}

// smallerThanModulus returns true if z < q
// This is not constant time
func (z *Element) smallerThanModulus() bool {
	return (z[5] < q5 || (z[5] == q5 && (z[4] < q4 || (z[4] == q4 && (z[3] < q3 || (z[3] == q3 && (z[2] < q2 || (z[2] == q2 && (z[1] < q1 || (z[1] == q1 && (z[0] < q0)))))))))))
}

// Mul z = x * y (mod q)
//
// x and y must be strictly inferior to q
func (z *Element) Mul(x, y *Element) *Element {
	// Implements CIOS multiplication -- section 2.3.2 of Tolga Acar's thesis
	// https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
	//
	// The algorithm:
	//
	// for i=0 to N-1
	// 		C := 0
	// 		for j=0 to N-1
	// 			(C,t[j]) := t[j] + x[j]*y[i] + C
	// 		(t[N+1],t[N]) := t[N] + C
	//
	// 		C := 0
	// 		m := t[0]*q'[0] mod D
	// 		(C,_) := t[0] + m*q[0]
	// 		for j=1 to N-1
	// 			(C,t[j-1]) := t[j] + m*q[j] + C
	//
	// 		(C,t[N-1]) := t[N] + C
	// 		t[N] := t[N+1] + C
	//
	// → N is the number of machine words needed to store the modulus q
	// → D is the word size. For example, on a 64-bit architecture D is 2	64
	// → x[i], y[i], q[i] is the ith word of the numbers x,y,q
	// → q'[0] is the lowest word of the number -q⁻¹ mod r. This quantity is pre-computed, as it does not depend on the inputs.
	// → t is a temporary array of size N+2
	// → C, S are machine words. A pair (C,S) refers to (hi-bits, lo-bits) of a two-word number
	//
	// As described here https://hackmd.io/@gnark/modular_multiplication we can get rid of one carry chain and simplify:
	//
	// for i=0 to N-1
	// 		(A,t[0]) := t[0] + x[0]*y[i]
	// 		m := t[0]*q'[0] mod W
	// 		C,_ := t[0] + m*q[0]
	// 		for j=1 to N-1
	// 			(A,t[j])  := t[j] + x[j]*y[i] + A
	// 			(C,t[j-1]) := t[j] + m*q[j] + C
	//
	// 		t[N-1] = C + A
	//
	// This optimization saves 5N + 2 additions in the algorithm, and can be used whenever the highest bit
	// of the modulus is zero (and not all of the remaining bits are set).
	_mulGeneric(z, x, y)
	return z
}

// Square z = x * x (mod q)
//
// x must be strictly inferior to q
func (z *Element) Square(x *Element) *Element {
	// see Mul for algorithm documentation
	_mulGeneric(z, x, x)
	return z
}

// FromMont converts z in place (i.e. mutates) from Montgomery to regular representation
// sets and returns z = z * 1
func (z *Element) FromMont() *Element {
	_fromMontGeneric(z)
	return z
}

// Add z = x + y (mod q)
func (z *Element) Add(x, y *Element) *Element {

	var carry uint64
	z[0], carry = bits.Add64(x[0], y[0], 0)
	z[1], carry = bits.Add64(x[1], y[1], carry)
	z[2], carry = bits.Add64(x[2], y[2], carry)
	z[3], carry = bits.Add64(x[3], y[3], carry)
	z[4], carry = bits.Add64(x[4], y[4], carry)
	z[5], _ = bits.Add64(x[5], y[5], carry)

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], b = bits.Sub64(z[3], q3, b)
		z[4], b = bits.Sub64(z[4], q4, b)
		z[5], _ = bits.Sub64(z[5], q5, b)
	}
	return z
}

// Double z = x + x (mod q), aka Lsh 1
func (z *Element) Double(x *Element) *Element {

	var carry uint64
	z[0], carry = bits.Add64(x[0], x[0], 0)
	z[1], carry = bits.Add64(x[1], x[1], carry)
	z[2], carry = bits.Add64(x[2], x[2], carry)
	z[3], carry = bits.Add64(x[3], x[3], carry)
	z[4], carry = bits.Add64(x[4], x[4], carry)
	z[5], _ = bits.Add64(x[5], x[5], carry)

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], b = bits.Sub64(z[3], q3, b)
		z[4], b = bits.Sub64(z[4], q4, b)
		z[5], _ = bits.Sub64(z[5], q5, b)
	}
	return z
}

// Sub z = x - y (mod q)
func (z *Element) Sub(x, y *Element) *Element {
	var b uint64
	z[0], b = bits.Sub64(x[0], y[0], 0)
	z[1], b = bits.Sub64(x[1], y[1], b)
	z[2], b = bits.Sub64(x[2], y[2], b)
	z[3], b = bits.Sub64(x[3], y[3], b)
	z[4], b = bits.Sub64(x[4], y[4], b)
	z[5], b = bits.Sub64(x[5], y[5], b)
	if b != 0 {
		var c uint64
		z[0], c = bits.Add64(z[0], q0, 0)
		z[1], c = bits.Add64(z[1], q1, c)
		z[2], c = bits.Add64(z[2], q2, c)
		z[3], c = bits.Add64(z[3], q3, c)
		z[4], c = bits.Add64(z[4], q4, c)
		z[5], _ = bits.Add64(z[5], q5, c)
	}
	return z
}

// Neg z = q - x
func (z *Element) Neg(x *Element) *Element {
	if x.IsZero() {
		z.SetZero()
		return z
	}
	var borrow uint64
	z[0], borrow = bits.Sub64(q0, x[0], 0)
	z[1], borrow = bits.Sub64(q1, x[1], borrow)
	z[2], borrow = bits.Sub64(q2, x[2], borrow)
	z[3], borrow = bits.Sub64(q3, x[3], borrow)
	z[4], borrow = bits.Sub64(q4, x[4], borrow)
	z[5], _ = bits.Sub64(q5, x[5], borrow)
	return z
}

func _mulGeneric(z, x, y *Element) {
	// see Mul for algorithm documentation

	var t [6]uint64
	var c [3]uint64
	{
		// round 0
		v := x[0]
		c[1], c[0] = bits.Mul64(v, y[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd1(v, y[1], c[1])
		c[2], t[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd1(v, y[2], c[1])
		c[2], t[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd1(v, y[3], c[1])
		c[2], t[2] = madd2(m, q3, c[2], c[0])
		c[1], c[0] = madd1(v, y[4], c[1])
		c[2], t[3] = madd2(m, q4, c[2], c[0])
		c[1], c[0] = madd1(v, y[5], c[1])
		t[5], t[4] = madd3(m, q5, c[0], c[2], c[1])
	}
	{
		// round 1
		v := x[1]
		c[1], c[0] = madd1(v, y[0], t[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd2(v, y[1], c[1], t[1])
		c[2], t[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd2(v, y[2], c[1], t[2])
		c[2], t[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd2(v, y[3], c[1], t[3])
		c[2], t[2] = madd2(m, q3, c[2], c[0])
		c[1], c[0] = madd2(v, y[4], c[1], t[4])
		c[2], t[3] = madd2(m, q4, c[2], c[0])
		c[1], c[0] = madd2(v, y[5], c[1], t[5])
		t[5], t[4] = madd3(m, q5, c[0], c[2], c[1])
	}
	{
		// round 2
		v := x[2]
		c[1], c[0] = madd1(v, y[0], t[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd2(v, y[1], c[1], t[1])
		c[2], t[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd2(v, y[2], c[1], t[2])
		c[2], t[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd2(v, y[3], c[1], t[3])
		c[2], t[2] = madd2(m, q3, c[2], c[0])
		c[1], c[0] = madd2(v, y[4], c[1], t[4])
		c[2], t[3] = madd2(m, q4, c[2], c[0])
		c[1], c[0] = madd2(v, y[5], c[1], t[5])
		t[5], t[4] = madd3(m, q5, c[0], c[2], c[1])
	}
	{
		// round 3
		v := x[3]
		c[1], c[0] = madd1(v, y[0], t[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd2(v, y[1], c[1], t[1])
		c[2], t[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd2(v, y[2], c[1], t[2])
		c[2], t[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd2(v, y[3], c[1], t[3])
		c[2], t[2] = madd2(m, q3, c[2], c[0])
		c[1], c[0] = madd2(v, y[4], c[1], t[4])
		c[2], t[3] = madd2(m, q4, c[2], c[0])
		c[1], c[0] = madd2(v, y[5], c[1], t[5])
		t[5], t[4] = madd3(m, q5, c[0], c[2], c[1])
	}
	{
		// round 4
		v := x[4]
		c[1], c[0] = madd1(v, y[0], t[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd2(v, y[1], c[1], t[1])
		c[2], t[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd2(v, y[2], c[1], t[2])
		c[2], t[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd2(v, y[3], c[1], t[3])
		c[2], t[2] = madd2(m, q3, c[2], c[0])
		c[1], c[0] = madd2(v, y[4], c[1], t[4])
		c[2], t[3] = madd2(m, q4, c[2], c[0])
		c[1], c[0] = madd2(v, y[5], c[1], t[5])
		t[5], t[4] = madd3(m, q5, c[0], c[2], c[1])
	}
	{
		// round 5
		v := x[5]
		c[1], c[0] = madd1(v, y[0], t[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd2(v, y[1], c[1], t[1])
		c[2], z[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd2(v, y[2], c[1], t[2])
		c[2], z[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd2(v, y[3], c[1], t[3])
		c[2], z[2] = madd2(m, q3, c[2], c[0])
		c[1], c[0] = madd2(v, y[4], c[1], t[4])
		c[2], z[3] = madd2(m, q4, c[2], c[0])
		c[1], c[0] = madd2(v, y[5], c[1], t[5])
		z[5], z[4] = madd3(m, q5, c[0], c[2], c[1])
	}

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], b = bits.Sub64(z[3], q3, b)
		z[4], b = bits.Sub64(z[4], q4, b)
		z[5], _ = bits.Sub64(z[5], q5, b)
	}

}

func _fromMontGeneric(z *Element) {
	// the following lines implement z = z * 1
	// with a modified CIOS montgomery multiplication
	// see Mul for algorithm documentation
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		C, z[3] = madd2(m, q4, z[4], C)
		C, z[4] = madd2(m, q5, z[5], C)
		z[5] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		C, z[3] = madd2(m, q4, z[4], C)
		C, z[4] = madd2(m, q5, z[5], C)
		z[5] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		C, z[3] = madd2(m, q4, z[4], C)
		C, z[4] = madd2(m, q5, z[5], C)
		z[5] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		C, z[3] = madd2(m, q4, z[4], C)
		C, z[4] = madd2(m, q5, z[5], C)
		z[5] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		C, z[3] = madd2(m, q4, z[4], C)
		C, z[4] = madd2(m, q5, z[5], C)
		z[5] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		C, z[3] = madd2(m, q4, z[4], C)
		C, z[4] = madd2(m, q5, z[5], C)
		z[5] = C
	}

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], b = bits.Sub64(z[3], q3, b)
		z[4], b = bits.Sub64(z[4], q4, b)
		z[5], _ = bits.Sub64(z[5], q5, b)
	}
}

// BitLen returns the minimum number of bits needed to represent z
// returns 0 if z == 0
func (z *Element) BitLen() int {
	if z[5] != 0 {
		return 320 + bits.Len64(z[5])
	}
	if z[4] != 0 {
		return 256 + bits.Len64(z[4])
	}
	if z[3] != 0 {
		return 192 + bits.Len64(z[3])
	}
	if z[2] != 0 {
		return 128 + bits.Len64(z[2])
	}
	if z[1] != 0 {
		return 64 + bits.Len64(z[1])
	}
	return bits.Len64(z[0])
}

// rSquare where r is the Montgommery constant
// see section 2.3.2 of Tolga Acar's thesis
// https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
var rSquare = Element{
	17644856173732828998,
	754043588434789617,
	10224657059481499349,
	7488229067341005760,
	11130996698012816685,
	1267921511277847466,
}

// ToMont converts z to Montgomery form
// sets and returns z = z * r²
func (z *Element) ToMont() *Element {
	return z.Mul(z, &rSquare)
}

// // ToRegular returns z in regular form (doesn't mutate z)
// func (z Element) ToRegular() Element {
// 	return *z.FromMont()
// }

// Bytes returns the value of z as a big-endian byte array
func (z *Element) Bytes() (res [Limbs * 8]byte) {
	// _z := z.ToRegular()
	binary.BigEndian.PutUint64(res[40:48], z[0])
	binary.BigEndian.PutUint64(res[32:40], z[1])
	binary.BigEndian.PutUint64(res[24:32], z[2])
	binary.BigEndian.PutUint64(res[16:24], z[3])
	binary.BigEndian.PutUint64(res[8:16], z[4])
	binary.BigEndian.PutUint64(res[0:8], z[5])

	return
}

const (
	k               = 32 // word size / 2
	signBitSelector = uint64(1) << 63
	approxLowBitsN  = k - 1
	approxHighBitsN = k + 1
)

const (
	inversionCorrectionFactorWord0 = 8737414717120368535
	inversionCorrectionFactorWord1 = 10094300570241649429
	inversionCorrectionFactorWord2 = 6339946188669102923
	inversionCorrectionFactorWord3 = 10492640117780001228
	inversionCorrectionFactorWord4 = 12201317704601795701
	inversionCorrectionFactorWord5 = 1158882751927031822
	invIterationsN                 = 26
)

// Inverse z = x⁻¹ (mod q)
//
// if x == 0, sets and returns z = x
func (z *Element) Inverse(x *Element) *Element {
	// Implements "Optimized Binary GCD for Modular Inversion"
	// https://github.com/pornin/bingcd/blob/main/doc/bingcd.pdf

	a := *x
	b := Element{
		q0,
		q1,
		q2,
		q3,
		q4,
		q5,
	} // b := q

	u := Element{1}

	// Update factors: we get [u; v] ← [f₀ g₀; f₁ g₁] [u; v]
	// cᵢ = fᵢ + 2³¹ - 1 + 2³² * (gᵢ + 2³¹ - 1)
	var c0, c1 int64

	// Saved update factors to reduce the number of field multiplications
	var pf0, pf1, pg0, pg1 int64

	var i uint

	var v, s Element

	// Since u,v are updated every other iteration, we must make sure we terminate after evenly many iterations
	// This also lets us get away with half as many updates to u,v
	// To make this constant-time-ish, replace the condition with i < invIterationsN
	for i = 0; i&1 == 1 || !a.IsZero(); i++ {
		n := max(a.BitLen(), b.BitLen())
		aApprox, bApprox := approximate(&a, n), approximate(&b, n)

		// f₀, g₀, f₁, g₁ = 1, 0, 0, 1
		c0, c1 = updateFactorIdentityMatrixRow0, updateFactorIdentityMatrixRow1

		for j := 0; j < approxLowBitsN; j++ {

			// -2ʲ < f₀, f₁ ≤ 2ʲ
			// |f₀| + |f₁| < 2ʲ⁺¹

			if aApprox&1 == 0 {
				aApprox /= 2
			} else {
				s, borrow := bits.Sub64(aApprox, bApprox, 0)
				if borrow == 1 {
					s = bApprox - aApprox
					bApprox = aApprox
					c0, c1 = c1, c0
					// invariants unchanged
				}

				aApprox = s / 2
				c0 = c0 - c1

				// Now |f₀| < 2ʲ⁺¹ ≤ 2ʲ⁺¹ (only the weaker inequality is needed, strictly speaking)
				// Started with f₀ > -2ʲ and f₁ ≤ 2ʲ, so f₀ - f₁ > -2ʲ⁺¹
				// Invariants unchanged for f₁
			}

			c1 *= 2
			// -2ʲ⁺¹ < f₁ ≤ 2ʲ⁺¹
			// So now |f₀| + |f₁| < 2ʲ⁺²
		}

		s = a

		var g0 int64
		// from this point on c0 aliases for f0
		c0, g0 = updateFactorsDecompose(c0)
		aHi := a.linearCombNonModular(&s, c0, &b, g0)
		if aHi&signBitSelector != 0 {
			// if aHi < 0
			c0, g0 = -c0, -g0
			aHi = negL(&a, aHi)
		}
		// right-shift a by k-1 bits
		a[0] = (a[0] >> approxLowBitsN) | ((a[1]) << approxHighBitsN)
		a[1] = (a[1] >> approxLowBitsN) | ((a[2]) << approxHighBitsN)
		a[2] = (a[2] >> approxLowBitsN) | ((a[3]) << approxHighBitsN)
		a[3] = (a[3] >> approxLowBitsN) | ((a[4]) << approxHighBitsN)
		a[4] = (a[4] >> approxLowBitsN) | ((a[5]) << approxHighBitsN)
		a[5] = (a[5] >> approxLowBitsN) | (aHi << approxHighBitsN)

		var f1 int64
		// from this point on c1 aliases for g0
		f1, c1 = updateFactorsDecompose(c1)
		bHi := b.linearCombNonModular(&s, f1, &b, c1)
		if bHi&signBitSelector != 0 {
			// if bHi < 0
			f1, c1 = -f1, -c1
			bHi = negL(&b, bHi)
		}
		// right-shift b by k-1 bits
		b[0] = (b[0] >> approxLowBitsN) | ((b[1]) << approxHighBitsN)
		b[1] = (b[1] >> approxLowBitsN) | ((b[2]) << approxHighBitsN)
		b[2] = (b[2] >> approxLowBitsN) | ((b[3]) << approxHighBitsN)
		b[3] = (b[3] >> approxLowBitsN) | ((b[4]) << approxHighBitsN)
		b[4] = (b[4] >> approxLowBitsN) | ((b[5]) << approxHighBitsN)
		b[5] = (b[5] >> approxLowBitsN) | (bHi << approxHighBitsN)

		if i&1 == 1 {
			// Combine current update factors with previously stored ones
			// [F₀, G₀; F₁, G₁] ← [f₀, g₀; f₁, g₁] [pf₀, pg₀; pf₁, pg₁], with capital letters denoting new combined values
			// We get |F₀| = | f₀pf₀ + g₀pf₁ | ≤ |f₀pf₀| + |g₀pf₁| = |f₀| |pf₀| + |g₀| |pf₁| ≤ 2ᵏ⁻¹|pf₀| + 2ᵏ⁻¹|pf₁|
			// = 2ᵏ⁻¹ (|pf₀| + |pf₁|) < 2ᵏ⁻¹ 2ᵏ = 2²ᵏ⁻¹
			// So |F₀| < 2²ᵏ⁻¹ meaning it fits in a 2k-bit signed register

			// c₀ aliases f₀, c₁ aliases g₁
			c0, g0, f1, c1 = c0*pf0+g0*pf1,
				c0*pg0+g0*pg1,
				f1*pf0+c1*pf1,
				f1*pg0+c1*pg1

			s = u

			// 0 ≤ u, v < 2²⁵⁵
			// |F₀|, |G₀| < 2⁶³
			u.linearComb(&u, c0, &v, g0)
			// |F₁|, |G₁| < 2⁶³
			v.linearComb(&s, f1, &v, c1)

		} else {
			// Save update factors
			pf0, pg0, pf1, pg1 = c0, g0, f1, c1
		}
	}

	// For every iteration that we miss, v is not being multiplied by 2ᵏ⁻²
	const pSq uint64 = 1 << (2 * (k - 1))
	a = Element{pSq}
	// If the function is constant-time ish, this loop will not run (no need to take it out explicitly)
	for ; i < invIterationsN; i += 2 {
		// could optimize further with mul by word routine or by pre-computing a table since with k=26,
		// we would multiply by pSq up to 13times;
		// on x86, the assembly routine outperforms generic code for mul by word
		// on arm64, we may loose up to ~5% for 6 limbs
		_mulGeneric(&v, &v, &a)
	}

	// u.Set(x) // for correctness check

	z.Mul(&v, &Element{
		inversionCorrectionFactorWord0,
		inversionCorrectionFactorWord1,
		inversionCorrectionFactorWord2,
		inversionCorrectionFactorWord3,
		inversionCorrectionFactorWord4,
		inversionCorrectionFactorWord5,
	})

	// correctness check
	// v.Mul(&u, z)
	// if !v.IsOne() && !u.IsZero() {
	// 	return z.inverseExp(&u)
	// }

	return z
}

// approximate a big number x into a single 64 bit word using its uppermost and lowermost bits
// if x fits in a word as is, no approximation necessary
func approximate(x *Element, nBits int) uint64 {

	if nBits <= 64 {
		return x[0]
	}

	const mask = (uint64(1) << (k - 1)) - 1 // k-1 ones
	lo := mask & x[0]

	hiWordIndex := (nBits - 1) / 64

	hiWordBitsAvailable := nBits - hiWordIndex*64
	hiWordBitsUsed := min(hiWordBitsAvailable, approxHighBitsN)

	mask_ := uint64(^((1 << (hiWordBitsAvailable - hiWordBitsUsed)) - 1))
	hi := (x[hiWordIndex] & mask_) << (64 - hiWordBitsAvailable)

	mask_ = ^(1<<(approxLowBitsN+hiWordBitsUsed) - 1)
	mid := (mask_ & x[hiWordIndex-1]) >> hiWordBitsUsed

	return lo | mid | hi
}

// linearComb z = xC * x + yC * y;
// 0 ≤ x, y < 2³⁸¹
// |xC|, |yC| < 2⁶³
func (z *Element) linearComb(x *Element, xC int64, y *Element, yC int64) {
	// | (hi, z) | < 2 * 2⁶³ * 2³⁸¹ = 2⁴⁴⁵
	// therefore | hi | < 2⁶¹ ≤ 2⁶³
	hi := z.linearCombNonModular(x, xC, y, yC)
	z.montReduceSigned(z, hi)
}

// montReduceSigned z = (xHi * r + x) * r⁻¹ using the SOS algorithm
// Requires |xHi| < 2⁶³. Most significant bit of xHi is the sign bit.
func (z *Element) montReduceSigned(x *Element, xHi uint64) {
	const signBitRemover = ^signBitSelector
	mustNeg := xHi&signBitSelector != 0
	// the SOS implementation requires that most significant bit is 0
	// Let X be xHi*r + x
	// If X is negative we would have initially stored it as 2⁶⁴ r + X (à la 2's complement)
	xHi &= signBitRemover
	// with this a negative X is now represented as 2⁶³ r + X

	var t [2*Limbs - 1]uint64
	var C uint64

	m := x[0] * qInvNeg

	C = madd0(m, q0, x[0])
	C, t[1] = madd2(m, q1, x[1], C)
	C, t[2] = madd2(m, q2, x[2], C)
	C, t[3] = madd2(m, q3, x[3], C)
	C, t[4] = madd2(m, q4, x[4], C)
	C, t[5] = madd2(m, q5, x[5], C)

	// m * qElement[5] ≤ (2⁶⁴ - 1) * (2⁶³ - 1) = 2¹²⁷ - 2⁶⁴ - 2⁶³ + 1
	// x[5] + C ≤ 2*(2⁶⁴ - 1) = 2⁶⁵ - 2
	// On LHS, (C, t[5]) ≤ 2¹²⁷ - 2⁶⁴ - 2⁶³ + 1 + 2⁶⁵ - 2 = 2¹²⁷ + 2⁶³ - 1
	// So on LHS, C ≤ 2⁶³
	t[6] = xHi + C
	// xHi + C < 2⁶³ + 2⁶³ = 2⁶⁴

	// <standard SOS>
	{
		const i = 1
		m = t[i] * qInvNeg

		C = madd0(m, q0, t[i+0])
		C, t[i+1] = madd2(m, q1, t[i+1], C)
		C, t[i+2] = madd2(m, q2, t[i+2], C)
		C, t[i+3] = madd2(m, q3, t[i+3], C)
		C, t[i+4] = madd2(m, q4, t[i+4], C)
		C, t[i+5] = madd2(m, q5, t[i+5], C)

		t[i+Limbs] += C
	}
	{
		const i = 2
		m = t[i] * qInvNeg

		C = madd0(m, q0, t[i+0])
		C, t[i+1] = madd2(m, q1, t[i+1], C)
		C, t[i+2] = madd2(m, q2, t[i+2], C)
		C, t[i+3] = madd2(m, q3, t[i+3], C)
		C, t[i+4] = madd2(m, q4, t[i+4], C)
		C, t[i+5] = madd2(m, q5, t[i+5], C)

		t[i+Limbs] += C
	}
	{
		const i = 3
		m = t[i] * qInvNeg

		C = madd0(m, q0, t[i+0])
		C, t[i+1] = madd2(m, q1, t[i+1], C)
		C, t[i+2] = madd2(m, q2, t[i+2], C)
		C, t[i+3] = madd2(m, q3, t[i+3], C)
		C, t[i+4] = madd2(m, q4, t[i+4], C)
		C, t[i+5] = madd2(m, q5, t[i+5], C)

		t[i+Limbs] += C
	}
	{
		const i = 4
		m = t[i] * qInvNeg

		C = madd0(m, q0, t[i+0])
		C, t[i+1] = madd2(m, q1, t[i+1], C)
		C, t[i+2] = madd2(m, q2, t[i+2], C)
		C, t[i+3] = madd2(m, q3, t[i+3], C)
		C, t[i+4] = madd2(m, q4, t[i+4], C)
		C, t[i+5] = madd2(m, q5, t[i+5], C)

		t[i+Limbs] += C
	}
	{
		const i = 5
		m := t[i] * qInvNeg

		C = madd0(m, q0, t[i+0])
		C, z[0] = madd2(m, q1, t[i+1], C)
		C, z[1] = madd2(m, q2, t[i+2], C)
		C, z[2] = madd2(m, q3, t[i+3], C)
		C, z[3] = madd2(m, q4, t[i+4], C)
		z[5], z[4] = madd2(m, q5, t[i+5], C)
	}

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], b = bits.Sub64(z[3], q3, b)
		z[4], b = bits.Sub64(z[4], q4, b)
		z[5], _ = bits.Sub64(z[5], q5, b)
	}
	// </standard SOS>

	if mustNeg {
		// We have computed ( 2⁶³ r + X ) r⁻¹ = 2⁶³ + X r⁻¹ instead
		var b uint64
		z[0], b = bits.Sub64(z[0], signBitSelector, 0)
		z[1], b = bits.Sub64(z[1], 0, b)
		z[2], b = bits.Sub64(z[2], 0, b)
		z[3], b = bits.Sub64(z[3], 0, b)
		z[4], b = bits.Sub64(z[4], 0, b)
		z[5], b = bits.Sub64(z[5], 0, b)

		// Occurs iff x == 0 && xHi < 0, i.e. X = rX' for -2⁶³ ≤ X' < 0

		if b != 0 {
			// z[5] = -1
			// negative: add q
			const neg1 = 0xFFFFFFFFFFFFFFFF

			var carry uint64

			z[0], carry = bits.Add64(z[0], q0, 0)
			z[1], carry = bits.Add64(z[1], q1, carry)
			z[2], carry = bits.Add64(z[2], q2, carry)
			z[3], carry = bits.Add64(z[3], q3, carry)
			z[4], carry = bits.Add64(z[4], q4, carry)
			z[5], _ = bits.Add64(neg1, q5, carry)
		}
	}
}

const (
	updateFactorsConversionBias    int64 = 0x7fffffff7fffffff // (2³¹ - 1)(2³² + 1)
	updateFactorIdentityMatrixRow0       = 1
	updateFactorIdentityMatrixRow1       = 1 << 32
)

func updateFactorsDecompose(c int64) (int64, int64) {
	c += updateFactorsConversionBias
	const low32BitsFilter int64 = 0xFFFFFFFF
	f := c&low32BitsFilter - 0x7FFFFFFF
	g := c>>32&low32BitsFilter - 0x7FFFFFFF
	return f, g
}

// negL negates in place [x | xHi] and return the new most significant word xHi
func negL(x *Element, xHi uint64) uint64 {
	var b uint64

	x[0], b = bits.Sub64(0, x[0], 0)
	x[1], b = bits.Sub64(0, x[1], b)
	x[2], b = bits.Sub64(0, x[2], b)
	x[3], b = bits.Sub64(0, x[3], b)
	x[4], b = bits.Sub64(0, x[4], b)
	x[5], b = bits.Sub64(0, x[5], b)
	xHi, _ = bits.Sub64(0, xHi, b)

	return xHi
}

// mulWNonModular multiplies by one word in non-montgomery, without reducing
func (z *Element) mulWNonModular(x *Element, y int64) uint64 {

	// w := abs(y)
	m := y >> 63
	w := uint64((y ^ m) - m)

	var c uint64
	c, z[0] = bits.Mul64(x[0], w)
	c, z[1] = madd1(x[1], w, c)
	c, z[2] = madd1(x[2], w, c)
	c, z[3] = madd1(x[3], w, c)
	c, z[4] = madd1(x[4], w, c)
	c, z[5] = madd1(x[5], w, c)

	if y < 0 {
		c = negL(z, c)
	}

	return c
}

// linearCombNonModular computes a linear combination without modular reduction
func (z *Element) linearCombNonModular(x *Element, xC int64, y *Element, yC int64) uint64 {
	var yTimes Element

	yHi := yTimes.mulWNonModular(y, yC)
	xHi := z.mulWNonModular(x, xC)

	var carry uint64
	z[0], carry = bits.Add64(z[0], yTimes[0], 0)
	z[1], carry = bits.Add64(z[1], yTimes[1], carry)
	z[2], carry = bits.Add64(z[2], yTimes[2], carry)
	z[3], carry = bits.Add64(z[3], yTimes[3], carry)
	z[4], carry = bits.Add64(z[4], yTimes[4], carry)
	z[5], carry = bits.Add64(z[5], yTimes[5], carry)

	yHi, _ = bits.Add64(xHi, yHi, carry)

	return yHi
}
