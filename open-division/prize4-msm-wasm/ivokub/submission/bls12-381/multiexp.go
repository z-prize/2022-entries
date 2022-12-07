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
	"github.com/gbotrel/msmwasm/bls12-381/fr"
)

type PointIdType uint32
type schedItem struct {
	pointId      PointIdType
	packedScalar uint32
}

const BATCH_SIZE = 250

var g1jePool = &pool[g1JacExtended]{}
var schedItemPool = &pool[schedItem]{}

// selector stores the index, mask and shifts needed to select bits from a scalar
// it is used during the multiExp algorithm or the batch scalar multiplication
type selector struct {
	index uint64 // index in the multi-word scalar to select bits from
	mask  uint64 // mask (c-bit wide)
	shift uint64 // shift needed to get our bits on low positions

	multiWordSelect bool   // set to true if we need to select bits from 2 words (case where c doesn't divide 64)
	maskHigh        uint64 // same than mask, for index+1
	shiftHigh       uint64 // same than shift, for index+1
}

var (
	dataToReturn [1 << 18]fr.Element
	dataNbChunks [fr.Limbs * 64 / 4]selector
)

// partitionScalars  compute, for each scalars over c-bit wide windows, nbChunk digits
// if the digit is larger than 2^{c-1}, then, we borrow 2^c from the next window and substract
// 2^{c} to the current digit, making it negative.
// negative digits can be processed in a later step as adding -G into the bucket instead of G
// (computing -G is cheap, and this saves us half of the buckets in the MultiExp or BatchScalarMul)
// scalarsMont indicates wheter the provided scalars are in montgomery form
// returns smallValues, which represent the number of scalars which meets the following condition
// 0 < scalar < 2^c (in other words, scalars where only the c-least significant bits are non zero)
func partitionScalars(scalars []fr.Element, c uint64) []fr.Element {
	const scalarsMont = false
	toReturn := dataToReturn[:len(scalars)]
	// toReturn := make([]fr.Element, len(scalars))

	// number of c-bit radixes in a scalar
	nbChunks := fr.Limbs * 64 / c
	if (fr.Limbs*64)%c != 0 {
		nbChunks++
	}

	mask := uint64((1 << c) - 1)      // low c bits are 1
	msbWindow := uint64(1 << (c - 1)) // msb of the c-bit window
	max := int(1 << (c - 1))          // max value we want for our digits
	cDivides64 := (64 % c) == 0       // if c doesn't divide 64, we may need to select over multiple words

	// compute offset and word selector / shift to select the right bits of our windows
	selectors := dataNbChunks[:nbChunks]
	// selectors := make([]selector, nbChunks)
	for chunk := uint64(0); chunk < nbChunks; chunk++ {
		jc := uint64(chunk * c)
		d := selector{}
		d.index = jc / 64
		d.shift = jc - (d.index * 64)
		d.mask = mask << d.shift
		d.multiWordSelect = !cDivides64 && d.shift > (64-c) && d.index < (fr.Limbs-1)
		if d.multiWordSelect {
			nbBitsHigh := d.shift - uint64(64-c)
			d.maskHigh = (1 << nbBitsHigh) - 1
			d.shiftHigh = (c - nbBitsHigh)
		}
		selectors[chunk] = d
	}

	// for each chunk, we could track the number of non-zeros points we will need to process
	// this way, if a chunk has more work to do than others, we can spawn off more go routines
	// (at the cost of more buckets allocated)
	// a simplified approach is to track the small values where only the first word is set
	// if this number represent a significant number of points, then we will split first chunk
	// processing in the msm in 2, to ensure all go routines finish at ~same time
	// /!\ nbTasks is enough as parallel.Execute is not going to spawn more than nbTasks go routine
	// if it does, though, this will deadlocK.
	for i := 0; i < len(scalars); i++ {
		var carry int

		scalar := scalars[i]
		if scalarsMont {
			scalar.FromMont()
		}
		if scalar.FitsOnOneWord() {
			// everything is 0, no need to process this scalar
			if scalar[0] == 0 {
				continue
			}
		}

		// for each chunk in the scalar, compute the current digit, and an eventual carry
		for chunk := uint64(0); chunk < nbChunks; chunk++ {
			s := selectors[chunk]

			// init with carry if any
			digit := carry
			carry = 0

			// digit = value of the c-bit window
			digit += int((scalar[s.index] & s.mask) >> s.shift)

			if s.multiWordSelect {
				// we are selecting bits over 2 words
				digit += int(scalar[s.index+1]&s.maskHigh) << s.shiftHigh
			}

			// if digit is zero, no impact on result
			if digit == 0 {
				continue
			}

			// if the digit is larger than 2^{c-1}, then, we borrow 2^c from the next window and substract
			// 2^{c} to the current digit, making it negative.
			if digit >= max {
				digit -= (1 << c)
				carry = 1
			}

			var bits uint64
			if digit >= 0 {
				bits = uint64(digit)
			} else {
				bits = uint64(-digit-1) | msbWindow
			}

			toReturn[i][s.index] |= (bits << s.shift)
			if s.multiWordSelect {
				toReturn[i][s.index+1] |= (bits >> s.shiftHigh)
			}

		}
	}

	return toReturn
}

// MultiExp implements section 4 of https://eprint.iacr.org/2012/549.pdf
func (p *G1Jac) MultiExp(points []G1Affine, scalars []fr.Element) *G1Jac {
	initMem()

	// ensure len(points) == len(scalars)
	nbPoints := len(points)
	if nbPoints != len(scalars) {
		panic("len(points) != len(scalars)")
	}

	const maxFloat = 0x1p1023 * (1 + (1 - 0x1p-52))
	// here, we compute the best C for nbPoints
	// we split recursively until nbChunks(c) >= nbTasks,
	bestC := func(nbPoints int) uint64 {
		// implemented msmC methods (the c we use must be in this slice)
		implementedCs := []uint64{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21}
		var C uint64
		// approximate cost (in group operations)
		// cost = bits/c * (nbPoints + 2^{c})
		// this needs to be verified empirically.
		// for example, on a MBP 2016, for G2 MultiExp > 8M points, hand picking c gives better results
		min := maxFloat
		for _, c := range implementedCs {
			cc := fr.Limbs * 64 * (nbPoints + (1 << (c)))
			cost := float64(cc) / float64(c)
			if cost < min {
				min = cost
				C = c
			}
		}
		// empirical, needs to be tuned.
		// if C > 16 && nbPoints < 1 << 23 {
		// 	C = 16
		// }
		return C
	}

	C := bestC(nbPoints)

	// partition the scalars
	// note: we do that before the actual chunk processing, as for each c-bit window (starting from LSW)
	// if it's larger than 2^{c-1}, we have a carry we need to propagate up to the higher window
	scalars = partitionScalars(scalars, C)

	p.innerMSM(points, scalars, C)
	return p
}

var (
	dataBucketsJacExt [1 << 20]g1JacExtended
	dataBucketsAffine [1 << 20]G1Affine
)

func (p *G1Jac) innerMSM(points []G1Affine, scalars []fr.Element, C uint64) *G1Jac {
	nbChunks := int(fr.Limbs * 64 / C) // number of c-bit radixes in a scalar

	_p := g1jePool.Get()
	tj := g1jePool.Get()

	if (fr.Limbs*64)%C != 0 {
		lastC := (fr.Limbs * 64) - (C * (fr.Limbs * 64 / C))
		// lastC := C
		msmProcessChunkG1JacExtended(uint64(nbChunks), dataBucketsJacExt[:1<<(lastC-1)], C, points, scalars, tj)
		_p.Set(tj)
	}

	buckets = dataBucketsAffine[:1<<(C-1)]

	for j := int(nbChunks - 1); j >= 0; j-- {
		msmProcessChunkG1BatchAffine(uint64(j), buckets, C, points, scalars, tj)
		for l := uint64(0); l < C; l++ {
			_p.double(_p)
		}
		_p.add(tj)
	}

	p.unsafeFromJacExtended(_p)
	return p

}

func msmProcessChunkG1BatchAffine(chunk uint64,
	buckets []G1Affine,
	c uint64,
	points []G1Affine,
	scalars []fr.Element,
	total *g1JacExtended) {

	mask := uint64((1 << c) - 1) // low c bits are 1

	for i := 0; i < len(buckets); i++ {
		buckets[i].setInfinity()
	}

	jc := uint64(chunk * c)
	s := selector{}
	s.index = jc / 64
	s.shift = jc - (s.index * 64)
	s.mask = mask << s.shift
	s.multiWordSelect = (64%c) != 0 && s.shift > (64-c) && s.index < (fr.Limbs-1)
	if s.multiWordSelect {
		nbBitsHigh := s.shift - uint64(64-c)
		s.maskHigh = (1 << nbBitsHigh) - 1
		s.shiftHigh = (c - nbBitsHigh)
	}

	sched := newBatchSchedulerG1Affine(points, scalars, s, c, chunk)
	for batch := sched(points, scalars, s, c); batch != nil; batch = sched(points, scalars, s, c) {
		BatchAddG1Affine(batch.buckets, batch.points, batch.len)
	}

	// reduce buckets into total
	// total =  bucket[0] + 2*bucket[1] + 3*bucket[2] ... + n*bucket[n-1]

	// var runningSum g1JacExtended
	runningSum := g1jePool.Get()
	runningSum.setInfinity()
	total.setInfinity()
	for k := len(buckets) - 1; k >= 0; k-- {
		if !buckets[k].IsInfinity() {
			runningSum.addMixed(&buckets[k])
		}
		total.add(runningSum)
	}
}

func msmProcessChunkG1JacExtended(chunk uint64,
	buckets []g1JacExtended,
	c uint64,
	points []G1Affine,
	scalars []fr.Element,
	total *g1JacExtended) {

	mask := uint64((1 << c) - 1) // low c bits are 1
	msbWindow := uint64(1 << (c - 1))

	for i := 0; i < len(buckets); i++ {
		buckets[i].setInfinity()
	}

	jc := uint64(chunk * c)
	s := selector{}
	s.index = jc / 64
	s.shift = jc - (s.index * 64)
	s.mask = mask << s.shift
	s.multiWordSelect = (64%c) != 0 && s.shift > (64-c) && s.index < (fr.Limbs-1)
	if s.multiWordSelect {
		nbBitsHigh := s.shift - uint64(64-c)
		s.maskHigh = (1 << nbBitsHigh) - 1
		s.shiftHigh = (c - nbBitsHigh)
	}

	// for each scalars, get the digit corresponding to the chunk we're processing.
	for i := 0; i < len(scalars); i++ {
		bits := (scalars[i][s.index] & s.mask) >> s.shift
		if s.multiWordSelect {
			bits += (scalars[i][s.index+1] & s.maskHigh) << s.shiftHigh
		}

		if bits == 0 {
			continue
		}

		// if msbWindow bit is set, we need to substract
		if bits&msbWindow == 0 {
			// add
			buckets[bits-1].addMixed(&points[i])
		} else {
			// sub
			buckets[bits & ^msbWindow].subMixed(&points[i])
		}
	}

	// reduce buckets into total
	// total =  bucket[0] + 2*bucket[1] + 3*bucket[2] ... + n*bucket[n-1]

	// var runningSum g1JacExtended
	runningSum := g1jePool.Get()
	runningSum.setInfinity()
	total.setInfinity()
	for k := len(buckets) - 1; k >= 0; k-- {
		if !buckets[k].ZZ.IsZero() {
			runningSum.add(&buckets[k])
		}
		total.add(runningSum)
	}
}

type batchSchedItemG1Affine struct {
	buckets []*G1Affine
	points  []*G1Affine
	len     int
}

var batchBuckets, batchPoints [BATCH_SIZE]*G1Affine
var negPoints [BATCH_SIZE]G1Affine
var bucketIds [BATCH_SIZE]uint32
var preBatch [BATCH_SIZE]*schedItem
var lambda [BATCH_SIZE]fp.Element
var lambdain [BATCH_SIZE]fp.Element
var isDbl [BATCH_SIZE]bool
var dataQueue [1 << 18]*schedItem
var item *schedItem
var curBatch = 0
var i int
var q int
var fetchFromInput bool
var msbWindow uint64
var buckets []G1Affine
var queue []*schedItem
var retBatchSchedItemsG1Affine batchSchedItemG1Affine
var rr G1Affine

func initMem() {
	for ii := 0; i < BATCH_SIZE; i++ {
		batchBuckets[ii] = nil
		batchPoints[ii] = nil
		negPoints[ii] = G1Affine{}
		bucketIds[ii] = 0
		preBatch[i] = nil
		lambda[ii] = fp.Element{}
		lambdain[ii] = fp.Element{}
		isDbl[ii] = false
	}
	for ii := range dataQueue {
		dataQueue[ii] = nil
	}
	item = nil
	curBatch = 0
	// i = -1
	// q = -1
	// fetchFromInput = true
	// queue = dataQueue[:0]
	rr = G1Affine{}
	retBatchSchedItemsG1Affine = batchSchedItemG1Affine{}
	g1jePool.Reset()
	schedItemPool.Reset()
	fpPool.Reset()
	for ii := range dataToReturn {
		dataToReturn[ii] = fr.Element{}
	}
	for ii := range dataNbChunks {
		dataNbChunks[ii] = selector{}
	}
}

func newBatchSchedulerG1Affine(
	// buckets []G1Affine,
	points []G1Affine,
	scalars []fr.Element,
	s selector,
	c uint64,
	chunk uint64) (sched func(
	// buckets []G1Affine,
	points []G1Affine,
	scalars []fr.Element,
	s selector,
	c uint64,
	// queue []*schedItem,
) *batchSchedItemG1Affine) {

	queue = dataQueue[:0] // queue for conflicting points
	msbWindow = uint64(1 << (c - 1))
	i = -1                // idx to iterate over points
	q = -1                // idx to iterate over queue, after points
	fetchFromInput = true // whether to get iterate from points or queue

	// scheduler
	// curBatch := 0

	return func(
		// buckets []G1Affine,
		points []G1Affine,
		scalars []fr.Element,
		s selector,
		c uint64,
		// queue []*schedItem,
	) *batchSchedItemG1Affine {
		// iterator on points/scalars
		var nextPoint = func(points []G1Affine, scalars []fr.Element, s selector) *schedItem {
			for i++; i < len(scalars); i++ {
				// extract the bits
				bits := (scalars[i][s.index] & s.mask) >> s.shift
				if s.multiWordSelect {
					bits += (scalars[i][s.index+1] & s.maskHigh) << s.shiftHigh
				}

				if bits == 0 {
					continue
				}

				if bits&msbWindow == 0 {
					// add
					bucketId := uint32(bits - 1)
					if buckets[bucketId].IsInfinity() {
						buckets[bucketId].Set(&points[i])
						continue
					}
					ret := schedItemPool.Get()
					*ret = schedItem{
						pointId:      PointIdType(i),
						packedScalar: (bucketId << 1) + 1,
					}
					return ret
				} else {
					// sub
					bucketId := uint32(bits & ^msbWindow)
					if buckets[bucketId].IsInfinity() {
						buckets[bucketId].Neg(&points[i])
						continue
					}
					ret := schedItemPool.Get()
					*ret = schedItem{
						pointId:      PointIdType(i),
						packedScalar: bucketId << 1,
					}
					return ret
				}
			}
			return nil
		}

		// batch handler
		// it deals with trivial cases

		var addToCurrentBatch = func(pos int, item *schedItem, points []G1Affine, scalars []fr.Element) (bool, bool) {
			bucketId := item.packedScalar >> 1
			bucket := &buckets[bucketId]
			point := &points[item.pointId]
			negY := fpPool.Get()
			if point.IsInfinity() {
				return false, false
			}

			if point.IsInfinity() {
				return false, false
			}

			// collision detection
			for i := 0; i < pos; i++ {
				if bucketId == bucketIds[i] {
					return false, true
				}
			}

			if (item.packedScalar % 2) == 1 {
				// bucket = 0 => bucket := P
				if bucket.IsInfinity() {
					bucket.Set(point)
					return false, false
				}
				// bucket = -P => bucket := 0
				if bucket.X.Equal(&point.X) && bucket.Y.Equal(negY.Neg(&point.Y)) {
					bucket.X.SetZero()
					bucket.Y.SetZero()
					return false, false
				}
				// else, add P to the batch
				batchPoints[pos] = point
			} else {
				// bucket = 0 => bucket := -P
				if bucket.IsInfinity() {
					bucket.Neg(point)
					return false, false
				}
				// bucket = P => bucket := 0
				if bucket.Equal(point) {
					bucket.X.SetZero()
					bucket.Y.SetZero()
					return false, false
				}
				// else, add -P to the batch
				negPoints[pos].Neg(point)
				batchPoints[pos] = &negPoints[pos]
			}
			batchBuckets[pos] = bucket
			bucketIds[pos] = uint32(bucketId)
			return true, false
		}

		// queue manager
		var nextQueue = func() *schedItem {
			for q++; q < len(queue); q++ {
				return queue[q]
			}
			q--
			return nil
		}
		BatchSize := len(buckets) / 8
		if BatchSize > BATCH_SIZE {
			BatchSize = BATCH_SIZE
		}
		curBatch++
		// fetch BatchSize points, possibly with conflicts
		batchIdx := 0
		for batchIdx = 0; batchIdx < BatchSize; batchIdx++ {
			if fetchFromInput {
				// fetch from input points/scalars
				item = nextPoint(points, scalars, s)
				if item == nil {
					// switch from input to queue
					fetchFromInput = false
					if item = nextQueue(); item == nil {
						// no more points in input nor queue, we're done
						break
					}
				}
			} else {
				// fetch from queue
				item = nextQueue()
				if item == nil {
					// no more points in input nor queue, we're done
					break
				}
			}
			// // when there's no more points, close the batch
			// // note: below we may enqueue more points
			// if item == nil {
			// 	// switch from input to queue
			// 	fetchFromInput = false
			// 	break
			// }
			preBatch[batchIdx] = item
		}

		// if the batch is empty, we're done
		if batchIdx == 0 {
			return nil
		}

		// prepare final batch, removing conflicts and trivial cases
		count := 0
		for j := 0; j < batchIdx; j++ {
			isAdded, isConflict := addToCurrentBatch(count, preBatch[j], points, scalars)
			if isConflict {
				// enqueue
				queue = append(queue, preBatch[j])
			}
			if isAdded {
				// count the actual number of points in the batch
				count++
			}
		}

		// return the batch
		retBatchSchedItemsG1Affine = batchSchedItemG1Affine{
			buckets: batchBuckets[:],
			points:  batchPoints[:],
			len:     count,
		}
		return &retBatchSchedItemsG1Affine
	}
}

// batch add/dbl in affine coordinates
// using batch inversion
// cost add: 5*batchSize M + 1I, dbl: +1M
func BatchAddG1Affine(R []*G1Affine, P []*G1Affine, batchSize int) {
	if batchSize == 0 {
		return
	}
	d := fpPool.Get()
	// var d fp.Element
	// var rr G1Affine

	for j := 0; j < batchSize; j++ {
		// detect dbl vs add
		if P[j].X.Equal(&R[j].X) {
			if P[j].Y.Equal(&R[j].Y) {
				isDbl[j] = true
			}
		}
		// compute denominator
		if isDbl[j] {
			lambdain[j].Double(&P[j].Y)
		} else {
			lambdain[j].Sub(&P[j].X, &R[j].X)
		}
	}

	// invert denominator
	BatchInvertG1Affine(&lambda, &lambdain, batchSize)

	for j := 0; j < batchSize; j++ {
		// computa lambda, distinguishing dbl / add
		if isDbl[j] {
			d.Square(&P[j].X)
			lambda[j].Mul(&lambda[j], d)
			d.Double(&lambda[j])
			lambda[j].Add(&lambda[j], d)
		} else {
			d.Sub(&P[j].Y, &R[j].Y)
			lambda[j].Mul(&lambda[j], d)
		}

		// compute X, Y
		rr.X.Square(&lambda[j])
		rr.X.Sub(&rr.X, &R[j].X)
		rr.X.Sub(&rr.X, &P[j].X)
		d.Sub(&R[j].X, &rr.X)
		rr.Y.Mul(&lambda[j], d)
		rr.Y.Sub(&rr.Y, &R[j].Y)
		R[j].Set(&rr)
	}
}

// batch inversion
// similar to fp.BatchInvert, ignores edge cases
func BatchInvertG1Affine(res *[BATCH_SIZE]fp.Element, a *[BATCH_SIZE]fp.Element, n int) {
	accumulator := fpPool.Get()
	accumulator.SetOne()
	// accumulator := fp.One()

	for i := 0; i < n; i++ {
		res[i] = *accumulator
		accumulator.Mul(accumulator, &a[i])
	}

	accumulator.Inverse(accumulator)

	for i := n - 1; i >= 0; i-- {
		res[i].Mul(&res[i], accumulator)
		accumulator.Mul(accumulator, &a[i])
	}
}
