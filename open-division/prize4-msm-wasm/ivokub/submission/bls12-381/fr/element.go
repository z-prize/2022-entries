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

package fr

import (
	"math/bits"
)

// Element represents a field element stored on 4 words (uint64)
//
// Element are assumed to be in Montgomery form in all methods.
//
// Modulus q =
//
//	q[base10] = 52435875175126190479447740508185965837690552500527637822603658699938581184513
//	q[base16] = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
//
// # Warning
//
// This code has not been audited and is provided as-is. In particular, there is no security guarantees such as constant time implementation or side-channel attack resistance.
type Element [4]uint64

const (
	Limbs = 4         // number of 64 bits words needed to represent a Element
	Bits  = 255       // number of bits needed to represent a Element
	Bytes = Limbs * 8 // number of bytes needed to represent a Element
)

// Field modulus q
const (
	q0 uint64 = 18446744069414584321
	q1 uint64 = 6034159408538082302
	q2 uint64 = 3691218898639771653
	q3 uint64 = 8353516859464449352
)

var qElement = Element{
	q0,
	q1,
	q2,
	q3,
}

// q + r'.r = 1, i.e., qInvNeg = - q⁻¹ mod r
// used for Montgomery reduction
const qInvNeg uint64 = 18446744069414584319

// NewElement returns a new Element from a uint64 value
//
// it is equivalent to
//
//	var v Element
//	v.SetUint64(...)
func NewElement(v uint64) Element {
	z := Element{v}
	z.Mul(&z, &rSquare)
	return z
}

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

// FitsOnOneWord reports whether z words (except the least significant word) are 0
//
// It is the responsibility of the caller to convert from Montgomery to Regular form if needed.
func (z *Element) FitsOnOneWord() bool {
	return (z[3] | z[2] | z[1]) == 0
}

// smallerThanModulus returns true if z < q
// This is not constant time
func (z *Element) smallerThanModulus() bool {
	return (z[3] < q3 || (z[3] == q3 && (z[2] < q2 || (z[2] == q2 && (z[1] < q1 || (z[1] == q1 && (z[0] < q0)))))))
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

// SetZero z = 0
func (z *Element) SetZero() *Element {
	z[0] = 0
	z[1] = 0
	z[2] = 0
	z[3] = 0
	return z
}

// IsZero returns z == 0
func (z *Element) IsZero() bool {
	return (z[3] | z[2] | z[1] | z[0]) == 0
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
	z[3], _ = bits.Sub64(q3, x[3], borrow)
	return z
}

// Add z = x + y (mod q)
func (z *Element) Add(x, y *Element) *Element {

	var carry uint64
	z[0], carry = bits.Add64(x[0], y[0], 0)
	z[1], carry = bits.Add64(x[1], y[1], carry)
	z[2], carry = bits.Add64(x[2], y[2], carry)
	z[3], _ = bits.Add64(x[3], y[3], carry)

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], _ = bits.Sub64(z[3], q3, b)
	}
	return z
}

// Double z = x + x (mod q), aka Lsh 1
func (z *Element) Double(x *Element) *Element {

	var carry uint64
	z[0], carry = bits.Add64(x[0], x[0], 0)
	z[1], carry = bits.Add64(x[1], x[1], carry)
	z[2], carry = bits.Add64(x[2], x[2], carry)
	z[3], _ = bits.Add64(x[3], x[3], carry)

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], _ = bits.Sub64(z[3], q3, b)
	}
	return z
}

func _mulGeneric(z, x, y *Element) {
	// see Mul for algorithm documentation

	var t [4]uint64
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
		t[3], t[2] = madd3(m, q3, c[0], c[2], c[1])
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
		t[3], t[2] = madd3(m, q3, c[0], c[2], c[1])
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
		t[3], t[2] = madd3(m, q3, c[0], c[2], c[1])
	}
	{
		// round 3
		v := x[3]
		c[1], c[0] = madd1(v, y[0], t[0])
		m := c[0] * qInvNeg
		c[2] = madd0(m, q0, c[0])
		c[1], c[0] = madd2(v, y[1], c[1], t[1])
		c[2], z[0] = madd2(m, q1, c[2], c[0])
		c[1], c[0] = madd2(v, y[2], c[1], t[2])
		c[2], z[1] = madd2(m, q2, c[2], c[0])
		c[1], c[0] = madd2(v, y[3], c[1], t[3])
		z[3], z[2] = madd3(m, q3, c[0], c[2], c[1])
	}

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], _ = bits.Sub64(z[3], q3, b)
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
		z[3] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		z[3] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		z[3] = C
	}
	{
		// m = z[0]n'[0] mod W
		m := z[0] * qInvNeg
		C := madd0(m, q0, z[0])
		C, z[0] = madd2(m, q1, z[1], C)
		C, z[1] = madd2(m, q2, z[2], C)
		C, z[2] = madd2(m, q3, z[3], C)
		z[3] = C
	}

	// if z >= q → z -= q
	if !z.smallerThanModulus() {
		var b uint64
		z[0], b = bits.Sub64(z[0], q0, 0)
		z[1], b = bits.Sub64(z[1], q1, b)
		z[2], b = bits.Sub64(z[2], q2, b)
		z[3], _ = bits.Sub64(z[3], q3, b)
	}
}

// rSquare where r is the Montgommery constant
// see section 2.3.2 of Tolga Acar's thesis
// https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
var rSquare = Element{
	14526898881837571181,
	3129137299524312099,
	419701826671360399,
	524908885293268753,
}

// // ToBigInt returns z as a big.Int in Montgomery form
// func (z *Element) ToBigInt(res *big.Int) *big.Int {
// 	var b [Limbs * 8]byte
// 	binary.BigEndian.PutUint64(b[24:32], z[0])
// 	binary.BigEndian.PutUint64(b[16:24], z[1])
// 	binary.BigEndian.PutUint64(b[8:16], z[2])
// 	binary.BigEndian.PutUint64(b[0:8], z[3])

// 	return res.SetBytes(b[:])
// }

// // ToBigIntRegular returns z as a big.Int in regular form
// func (z Element) ToBigIntRegular(res *big.Int) *big.Int {
// 	z.FromMont()
// 	return z.ToBigInt(res)
// }

// // SetRandom sets z to a uniform random value in [0, q).
// //
// // This might error only if reading from crypto/rand.Reader errors,
// // in which case, value of z is undefined.
// func (z *Element) SetRandom() (*Element, error) {
// 	// this code is generated for all modulus
// 	// and derived from go/src/crypto/rand/util.go

// 	// l is number of limbs * 8; the number of bytes needed to reconstruct 4 uint64
// 	const l = 32

// 	// bitLen is the maximum bit length needed to encode a value < q.
// 	const bitLen = 255

// 	// k is the maximum byte length needed to encode a value < q.
// 	const k = (bitLen + 7) / 8

// 	// b is the number of bits in the most significant byte of q-1.
// 	b := uint(bitLen % 8)
// 	if b == 0 {
// 		b = 8
// 	}

// 	var bytes [l]byte

// 	for {
// 		// note that bytes[k:l] is always 0
// 		if _, err := io.ReadFull(rand.Reader, bytes[:k]); err != nil {
// 			return nil, err
// 		}

// 		// Clear unused bits in in the most signicant byte to increase probability
// 		// that the candidate is < q.
// 		bytes[k-1] &= uint8(int(1<<b) - 1)
// 		z[0] = binary.LittleEndian.Uint64(bytes[0:8])
// 		z[1] = binary.LittleEndian.Uint64(bytes[8:16])
// 		z[2] = binary.LittleEndian.Uint64(bytes[16:24])
// 		z[3] = binary.LittleEndian.Uint64(bytes[24:32])

// 		if !z.smallerThanModulus() {
// 			continue // ignore the candidate and re-sample
// 		}

// 		return z, nil
// 	}
// }
