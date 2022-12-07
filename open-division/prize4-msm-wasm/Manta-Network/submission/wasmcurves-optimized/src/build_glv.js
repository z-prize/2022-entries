const utils = require("./utils.js");

// Supports only BLS12-381.
module.exports = function buildGLV(module, prefix, fnName) {
    const n8r = 32;
    const n8q = 48;
    const f1mField = "f1m";
    const g1mField = "g1m";

    function toMontgomery(a) {
        return BigInt(a) * (1n << BigInt(n8q * 8)) % q;
    }
    const q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaabn;

    const v0 = 1;
    const v1 = -228988810152649578064853576960394133503n;
    const u0 = 228988810152649578064853576960394133504n;
    const u1 = 1;
    const negV1 = 228988810152649578064853576960394133503n;
    const zero = 0;
    const beta = 793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350n;
    const divisor = 52435875175126190479447740508185965837690552500527637822603658699938581184513n; // v0*u1 - v1*u0
    const pU0 = module.alloc(64, utils.bigInt2BytesLE(u0, 64));
    const pU1 = module.alloc(64, utils.bigInt2BytesLE(u1, 64));
    const pV0 = module.alloc(64, utils.bigInt2BytesLE(v0, 64));
    const pV1 = module.alloc(64, utils.bigInt2BytesLE(v1, 64));
    const pNegV1 = module.alloc(64, utils.bigInt2BytesLE(negV1, 64));
    const pZero = module.alloc(64, utils.bigInt2BytesLE(zero, 64));
    const pBeta = module.alloc(64, utils.bigInt2BytesLE(toMontgomery(beta), 64));
    const pDivisor = module.alloc(64, utils.bigInt2BytesLE(divisor, 64));

    // Checks if a 512-bit scalar is positive or not.
    // Assuming 0 is positve since it should not affect msm.
    function buildIsPositive() {
        const f = module.addFunction(fnName + "_isPositive");
        // Pointer to a 512-bit scalar
        f.addParam("pScalar", "i32");
        // Returns 1 for positive and 0 for negative.
        f.setReturnType("i32");
        // Value at the highest int32 memory of pScalar
        f.addLocal("highestInt32", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal("highestInt32", c.call(prefix + "_utility_loadI32", c.getLocal("pScalar"), c.i32_const(15))),
            c.i32_xor(
                c.i32_shr_u(c.i32_and(c.getLocal("highestInt32"), c.i32_const(0x80000000)), c.i32_const(31)),
                c.i32_const(0x1)
            ),
        );
    }

    // Given a pointer `pScalar` to a 256-bit scalar, decomposes into two 128-bit scalars pointed by `pScalarRes`.
    function buildDecomposeScalar() {
        const f = module.addFunction(fnName + "_decomposeScalar");
        // Pointer to a 256-bit scalar
        f.addParam("pScalar", "i32");
        // Pointer to two 128-bit scalars. These two 128-bit scalars stores the absolute value. Each 128-bit scalar is stored in a 256-bit memory.
        f.addParam("pScalarRes", "i32");
        // Encodes the sign of two scalars. The encoding is 00...00s_1s_0 where s_0 and s_1 are the sign of k1 and k2, respectively.
        f.setReturnType("i32");
        // Pointer to a 512-bit scratch space.
        f.addLocal("pScratchSpace", "i32");
        // Pointer to a 512-bit scratch space.
        f.addLocal("pScratchSpace1", "i32");
        // Pointer to a 512-bit q1.
        f.addLocal("pQ1", "i32");
        // Pointer to a 512-bit q2.
        f.addLocal("pQ2", "i32");
        // Pointer to a 512-bit k1.
        f.addLocal("pK1", "i32");
        // Pointer to a 512-bit k2.
        f.addLocal("pK2", "i32");
        // Pointer to a 512-bit remainder.
        f.addLocal("pQr", "i32");
        // Remainder
        f.addLocal("remainder", "i32");
        // Sign
        f.addLocal("sign", "i32");
        // Stores the 256-bit pScalar in this 512-bit memory.
        f.addLocal("pScalar512", "i32");
        // Index
        f.addLocal("i", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal("pScratchSpace", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.setLocal("pScratchSpace1", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.setLocal("pQ1", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.setLocal("pQ2", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.setLocal("pK1", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.setLocal("pK2", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.setLocal("pScalar512", c.call(prefix + "_utility_allocateMemory", c.i32_const(64))),
            c.call(prefix + "_int512_zero", c.getLocal("pScalar512")),
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.i32_const(4))),
                c.call(prefix + "_utility_storeI64",
                    c.getLocal("pScalar512"),
                    c.getLocal("i"),
                    c.call(prefix + "_utility_loadI64", c.getLocal("pScalar"), c.getLocal("i")),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
            // q1 = (u1 * pScalar) / ((v0 * u1) - (v1 * u0));
            // Since u1 = 1, we have q1 = pScalar / ((v0 * u1) - (v1 * u0));
            c.call(prefix + "_int512_div", c.getLocal("pScalar512"), c.i32_const(pDivisor), c.getLocal("pQ1"), c.getLocal("pQr")),
            // q2 = (-v1 * pScalar) / ((v0 * u1) - (v1 * u0));
            c.call(prefix + "_int512_mul", c.getLocal("pScalar512"), c.i32_const(pNegV1), c.getLocal("pScratchSpace")),
            c.call(prefix + "_int512_div", c.getLocal("pScratchSpace"), c.i32_const(pDivisor), c.getLocal("pQ2"), c.getLocal("pQr")),
            // pK1 = pScalar - &q1 * v0 - &q2 * u0;
            // Since v0 is 1, we have pK1 = pScalar - &q1 - &q2 * u0;
            c.drop(c.call(prefix + "_int512_sub", c.getLocal("pScalar512"), c.getLocal("pQ1"), c.getLocal("pK1"))),
            c.call(prefix + "_int512_mul", c.getLocal("pQ2"), c.i32_const(pU0), c.getLocal("pScratchSpace")),
            c.drop(c.call(prefix + "_int512_sub", c.getLocal("pK1"), c.getLocal("pScratchSpace"), c.getLocal("pK1"))),
            // pK2 = 0 - q1 * v.1 - q2 * u.1;
            // since u.1 = 1, we have pK2 = 0 - q1 * v.1 - q2;
            c.call(prefix + "_int512_mul", c.getLocal("pQ1"), c.i32_const(pV1), c.getLocal("pScratchSpace")),
            c.drop(c.call(prefix + "_int512_sub", c.i32_const(pZero), c.getLocal("pScratchSpace"), c.getLocal("pK2"))),
            c.drop(c.call(prefix + "_int512_sub", c.getLocal("pK2"), c.getLocal("pQ2"), c.getLocal("pK2"))),
            // if pK1 > 0:
            //    sign = sign || 1
            // else:
            //    pK1 = 0 - pK1
            // if pK2 > 0:
            //    sign = sign || 2
            // else:
            //    pK2 = 0 - pK2
            c.setLocal("sign", c.i32_const(0)),
            c.if(c.call(fnName + "_isPositive", c.getLocal("pK1")),
                c.setLocal("sign", c.i32_or(c.getLocal("sign"), c.i32_const(1))),
                c.drop(c.call(prefix + "_int512_sub", c.i32_const(pZero), c.getLocal("pK1"), c.getLocal("pK1"))),
            ),
            c.if(c.call(fnName + "_isPositive", c.getLocal("pK2")),
                c.setLocal("sign", c.i32_or(c.getLocal("sign"), c.i32_const(2))),
                c.drop(c.call(prefix + "_int512_sub", c.i32_const(pZero), c.getLocal("pK2"), c.getLocal("pK2"))),
            ),
            // pScalarRes = [pK1[0], pK1[1], pK2[0], pK2[1]]
            c.call(prefix + "_int512_zero", c.getLocal("pScalarRes")),            
            c.call(prefix + "_utility_storeI64", c.getLocal("pScalarRes"), c.i32_const(0), c.call(prefix + "_utility_loadI64", c.getLocal("pK1"), c.i32_const(0))),
            c.call(prefix + "_utility_storeI64", c.getLocal("pScalarRes"), c.i32_const(1), c.call(prefix + "_utility_loadI64", c.getLocal("pK1"), c.i32_const(1))),
            c.call(prefix + "_utility_storeI64", c.getLocal("pScalarRes"), c.i32_const(4), c.call(prefix + "_utility_loadI64", c.getLocal("pK2"), c.i32_const(0))),
            c.call(prefix + "_utility_storeI64", c.getLocal("pScalarRes"), c.i32_const(5), c.call(prefix + "_utility_loadI64", c.getLocal("pK2"), c.i32_const(1))),
            c.i32_store(c.i32_const(0), c.getLocal("pScratchSpace")),
            c.getLocal("sign"),
        );
    }

    // Given a point P = (x, y) at `pPoint` and a 1-bit `isPositive`, computes a new point Q = (beta*x, y) and further converts to (beta*x, -y) if isPositive is 0.
    // The resulting point is stored at `pPointRes`.
    function buildEndomorphism() {
        const f = module.addFunction(fnName + "_endomorphism");
        f.addParam("pPoint", "i32");
        f.addParam("sign", "i32");
        f.addParam("pPointRes", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.call(f1mField + "_mul", c.getLocal("pPoint"), c.i32_const(pBeta), c.getLocal("pPointRes")),
            c.if(c.getLocal("sign"),
                c.call(f1mField + "_copy", c.i32_add(c.getLocal("pPoint"), c.i32_const(n8q)), c.i32_add(c.getLocal("pPointRes"), c.i32_const(n8q))),
                c.call(f1mField + "_neg", c.i32_add(c.getLocal("pPoint"), c.i32_const(n8q)), c.i32_add(c.getLocal("pPointRes"), c.i32_const(n8q))),
            ),
        );
    }

    // Applies endomorphism on the input point vector `pPoints` and the input scalar vector `pScalars` of length `numPoints`.
    // Input: each point is on affine curve, each scalar is n8q-bytes. Each point is 2*n8q-byte. 
    //          Each scalar is n8r-byte.
    // Output: Each point is still on affine curve, each scalar is n8q-bytes. Each point is 2*n8q-byte.
    //          Each scalar is n8r-byte but the most significant n8r/2 bytes are 0.
    //
    // Input: pPoints = [p0, p1, ..., p_{n-1}]
    //        pScalars = [s0, s1, ..., s_{n-1}]
    // Consider pi -> pi0, pi1 
    //          si -> si0, si1
    // Output:
    //        pPointsRes = [p00, p01, p10, p11, p20, p21, ..., p_{n-1}0, p_{n-1}1]
    //        pScalarsRes = [s00, s01, s10, s11, s20, s21, ..., s_{n-1}0, s_{n-1}1]
    function buildPreprocessEndomorphism() {
        const f = module.addFunction(fnName + "_preprocessEndomorphism");
        // Pointer to the input point vector. Shape: numPoints
        f.addParam("pPoints", "i32");
        // Pointer to the input scalar vector. Each scalar is n8r bytes. Shape: numPoints
        f.addParam("pScalars", "i32");
        // Number of points
        f.addParam("numPoints", "i32");
        // Pointer to the output point vector. Shape: 2 * numPoints
        f.addParam("pPointsRes", "i32");
        // Pointer to the output scalar vector. Each scalar is n8r bytes. Shape: 2 * numPoints
        f.addParam("pScalarsRes", "i32");
        // Index
        f.addLocal("i", "i32");
        // Sign
        f.addLocal("sign", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("numPoints"))),
                c.setLocal("sign",
                    c.call(fnName + "_decomposeScalar",
                        c.i32_add(
                            c.getLocal("pScalars"),
                            c.i32_mul(c.getLocal("i"), c.i32_const(n8r)),
                        ),
                        c.i32_add(
                            c.getLocal("pScalarsRes"),
                            c.i32_mul(c.getLocal("i"), c.i32_const(2*n8r)),
                        ),
                    ),
                ),
                c.call(f1mField + "_copy",
                    c.i32_add(c.getLocal("pPoints"), c.i32_mul(c.getLocal("i"), c.i32_const(2 * n8q))),
                    c.i32_add(c.getLocal("pPointsRes"), c.i32_mul(c.getLocal("i"), c.i32_const(4 * n8q))),
                ),
                c.if(c.i32_and(c.getLocal("sign"), c.i32_const(1)),
                    c.call(f1mField + "_copy",
                        c.i32_add(c.getLocal("pPoints"),
                            c.i32_add(
                                c.i32_mul(c.getLocal("i"), c.i32_const(2 * n8q)),
                                c.i32_const(n8q),
                            ),
                        ),
                        c.i32_add(c.getLocal("pPointsRes"),
                            c.i32_add(
                                c.i32_mul(c.getLocal("i"), c.i32_const(4 * n8q)),
                                c.i32_const(n8q),
                            ),
                        ),
                    ),
                    c.call(f1mField + "_neg",
                        c.i32_add(c.getLocal("pPoints"),
                            c.i32_add(
                                c.i32_mul(c.getLocal("i"), c.i32_const(2 * n8q)),
                                c.i32_const(n8q),
                            ),
                        ),
                        c.i32_add(c.getLocal("pPointsRes"),
                            c.i32_add(
                                c.i32_mul(c.getLocal("i"), c.i32_const(4 * n8q)),
                                c.i32_const(n8q),
                            ),
                        ),
                    ),
                ),
                c.call(fnName + "_endomorphism",
                    c.i32_add(
                        c.getLocal("pPoints"),
                        c.i32_mul(c.getLocal("i"), c.i32_const(n8q * 2)),
                    ),
                    c.i32_shr_u(c.getLocal("sign"), c.i32_const(1)),
                    c.i32_add(
                        c.getLocal("pPointsRes"),
                        c.i32_add(
                            c.i32_mul(c.getLocal("i"), c.i32_const(4 * n8q)),
                            c.i32_const(2 * n8q),
                        ),
                    ),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        );
    }

    buildIsPositive();
    buildDecomposeScalar();
    buildEndomorphism();
    buildPreprocessEndomorphism();
    module.exportFunction(fnName + "_isPositive");
    module.exportFunction(fnName + "_decomposeScalar");
    module.exportFunction(fnName + "_endomorphism");
    module.exportFunction(fnName + "_preprocessEndomorphism");
};
