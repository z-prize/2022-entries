package bls12381

import (
	"github.com/gbotrel/msmwasm/bls12-381/fp"
)

// BLS12-381: A Barreto--Lynn--Scott curve of embedding degree k=12 with seed x₀=-15132376222941642752
// 𝔽r: r=52435875175126190479447740508185965837690552500527637822603658699938581184513 (x₀⁴-x₀²+1)
// 𝔽p: p=4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787 ((x₀-1)² ⋅ r(x₀)/3+x₀)
// (E/𝔽p): Y²=X³+4
// r ∣ #E(Fp)

// bCurveCoeff b coeff of the curve Y²=X³+b
var bCurveCoeff fp.Element

// generators of the r-torsion group, resp. in ker(pi-id), ker(Tr)
var g1Gen G1Jac

// point at infinity
var g1Infinity G1Jac

func init() {

	bCurveCoeff.SetUint64(4)

	// TODO @gbotrel check that
	g1Gen.X = fp.Element{6679831729115696150, 8653662730902241269, 1535610680227111361, 17342916647841752903, 17135755455211762752, 1297449291367578485}
	g1Gen.Y = fp.Element{13451288730302620273, 10097742279870053774, 15949884091978425806, 5885175747529691540, 1016841820992199104, 845620083434234474}

	g1Gen.Z.SetOne()

	// (X,Y,Z) = (1,1,0)
	g1Infinity.X.SetOne()
	g1Infinity.Y.SetOne()

}
