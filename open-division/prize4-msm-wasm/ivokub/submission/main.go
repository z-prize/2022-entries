package main

import (
	"reflect"
	"unsafe"

	bls12381 "github.com/gbotrel/msmwasm/bls12-381"
	"github.com/gbotrel/msmwasm/bls12-381/fp"
	"github.com/gbotrel/msmwasm/bls12-381/fr"
)

var (
	dataScalars [1 << 18]fr.Element
	dataPoints  [1 << 18]bls12381.G1Affine
	dataMsmRes  [2 * 8 * fp.Limbs]byte
)

func main() {}

//export getScalarsMem
func getScalarsMem() uint {
	return uint(uintptr(unsafe.Pointer(&dataScalars)))
}

//export getPointsMem
func getPointsMem() uint {
	return uint(uintptr(unsafe.Pointer(&dataPoints)))
}

//export toMontgomery
func toMontgomery(ptr uint, count uint) {
	hdr := reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  uintptr(count),
		Cap:  uintptr(count),
	}
	pts := *(*[]fp.Element)(unsafe.Pointer(&hdr))
	for i := 0; i < int(count); i++ {
		pts[i].ToMont()
	}
}

var (
	pt  = new(bls12381.G1Jac)
	pta = new(bls12381.G1Affine)
)

//export compute_msm
func computeMSM(ptrPts, ptrScs, size uint) uint {
	hdrPts := reflect.SliceHeader{
		Data: uintptr(ptrPts),
		Len:  uintptr(size),
		Cap:  uintptr(size),
	}
	hdrScs := reflect.SliceHeader{
		Data: uintptr(ptrScs),
		Len:  uintptr(size),
		Cap:  uintptr(size),
	}
	pts := *(*[]bls12381.G1Affine)(unsafe.Pointer(&hdrPts))
	scs := *(*[]fr.Element)(unsafe.Pointer(&hdrScs))
	pt.MultiExp(pts, scs)
	pta.FromJacobian(pt)
	xb := pta.X.FromMont().Bytes()
	yb := pta.Y.FromMont().Bytes()
	copy(dataMsmRes[0:8*fp.Limbs], xb[:])
	copy(dataMsmRes[8*fp.Limbs:], yb[:])
	return uint(uintptr(unsafe.Pointer(&dataMsmRes)))
}

//export fill
func fillPoints(ptrPts, ptrScs, size uint) uint {
	hdrPts := reflect.SliceHeader{
		Data: uintptr(ptrPts),
		Len:  uintptr(size),
		Cap:  uintptr(size),
	}
	hdrScs := reflect.SliceHeader{
		Data: uintptr(ptrScs),
		Len:  uintptr(size),
		Cap:  uintptr(size),
	}
	pts := *(*[]bls12381.G1Affine)(unsafe.Pointer(&hdrPts))
	scs := *(*[]fr.Element)(unsafe.Pointer(&hdrScs))
	_, _ = pts, scs
	fillBenchBasesG1(pts)
	fillBenchScalars(scs)
	return 0
}

var one fp.Element

func fillBenchBasesG1(samplePoints []bls12381.G1Affine) {
	one.SetOne()
	samplePoints[0].X = fp.Element{0xfce2c12afc6a3412, 0xdc43eeb49b814b6a, 0x3aa1a2864a336cca, 0xce59a3aaa9ed1d85, 0x974b5a207c066000, 0x75b44cabdf94a9c}
	samplePoints[0].Y = fp.Element{0x44d5cd6e2b899bd8, 0x3d7c397a067bcbf0, 0x6efb751548de89b9, 0xb91d84fe65051c63, 0xa47a5ce550ff52e3, 0xc01b8e1aa2b253}

	for i := 1; i < len(samplePoints); i++ {
		samplePoints[i].X.Add(&samplePoints[i-1].X, &one)
		samplePoints[i].Y.Sub(&samplePoints[i-1].Y, &one)
	}
}

func fillBenchScalars(sampleScalars []fr.Element) {
	mixer := fr.Element{0x74809303d1e6b5dd, 0x7f098fe9b1d5b09c, 0x595341574e4c03ff, 0x6ea5ed3db3dc499d}
	for i := 1; i <= len(sampleScalars); i++ {
		sampleScalars[i-1].SetUint64(uint64(i)).
			Mul(&sampleScalars[i-1], &mixer).
			FromMont()
	}
}
