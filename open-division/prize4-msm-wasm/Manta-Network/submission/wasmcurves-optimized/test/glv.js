const assert = require("assert");
const buildBls12381 = require("../src/bls12381/build_bls12381.js");
const ChaCha = require("ffjavascript").ChaCha;
const buildProtoboard = require("wasmbuilder").buildProtoboard;

describe("GLV Tests", function () {
    this.timeout(10000000);
    // Fq: 48 bytes = 384 bits
    const n8q = 48;
    // Fr: 32 bytes = 256 bits
    const n8r = 32;
    let pb;
    before(async () => {
        pb = await buildProtoboard((module) => {
            buildBls12381(module);
        }, n8q);
    });

    // Prints the hex representation of a single coordinates in a point
    function printHex(p) {
        pb.f1m_fromMontgomery(p, p);
        const n = pb.get(p);
        pb.f1m_toMontgomery(p, p);
        return "0x" + n.toString(16);
    }

    // Prints the (x, y, z) coordinates of a G1 point
    function printG1(s, p) {
        console.log(s + " G1(" + printHex(p) + " , " + printHex(p + n8q) + " , " + printHex(p + n8q * 2) + ")");
    }

    it("isPositive is correct.", async () => {
        const oneHandred = 100;
        const pOneHundred = pb.alloc(64);
        pb.set(pOneHundred, oneHandred, 64);
        // Note: wasmcurve cannot set negative numbers over 32, we use a workaround method to generate negtives.
        // The actual i^th inputs should be 100-inputs[i].
        const inputs = [0, 1, 100, 500, 9003405095674209932115908784230457051068760537362306482987933690960811974463n];
        const expectedOutput = [1, 1, 1, 0, 0];
        const pScalar = pb.alloc(64);
        const subtractor = pb.alloc(64);
        for (let i = 0; i < inputs.length; i++) {
            pb.set(subtractor, inputs[i], 64);
            pb.g1m_int512_sub(pOneHundred, subtractor, pScalar);
            let output = pb.g1m_glv_isPositive(pScalar);
            assert.equal(expectedOutput[i], output);
        }
    });

    it("decomposeScalar is correct.", async () => {
        const scalar = 9003405095674209932115908784230457051068760537362306482987933690960811974463n;
        const expectedOutput = [
            86900781371527243792514624323931922239n,
            39318100695279906693562908013718409681n,
        ];
        const pScalar = pb.alloc(32);
        pb.set(pScalar, scalar, 32);
        const pScalarRes = pb.alloc(32 * 2);
        let sign = pb.g1m_glv_decomposeScalar(pScalar, pScalarRes);
        let output = pb.get(pScalarRes, 2, 32);
        for (let i = 0; i < 2; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
        assert.equal(sign, 1);
    });

    it("scalarMul is correct.", async () => {
        const input = [0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bbn, 0x8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1n]
        const scalar = 9003405095674209932115908784230457051068760537362306482987933690960811974463n;
        const pExpectedOutput = pb.alloc(n8q * 3);
        const pPoints = pb.alloc(n8q * 3);
        const pScalar = pb.alloc(32);
        pb.set(pScalar, scalar, 32);
        pb.set(pPoints, input[0], 48);
        pb.set(pPoints + 48, input[1], 48);
        pb.f1m_one(pPoints + 96);
        pb.f1m_toMontgomery(pPoints, pPoints);
        pb.f1m_toMontgomery(pPoints + 48, pPoints + 48);
        pb.g1m_timesScalar(pPoints, pScalar, n8r, pExpectedOutput);
        pb.g1m_normalize(pExpectedOutput, pExpectedOutput);

        const pScalarSplit = pb.alloc(64);
        let sign = pb.g1m_glv_decomposeScalar(pScalar, pScalarSplit);
        const pConvertedPoint = pb.alloc(n8q * 3);
        const pAccumulator = pb.alloc(n8q * 3);
        const pCalculated = pb.alloc(n8q * 3);
        pb.g1m_glv_endomorphism(pPoints, sign & 1, pConvertedPoint);
        pb.f1m_one(pConvertedPoint + 96);
        pb.g1m_timesScalar(pPoints, pScalarSplit, n8r, pAccumulator);
        pb.g1m_glv_endomorphism(pPoints, sign & 2, pConvertedPoint);
        pb.f1m_one(pConvertedPoint + 96);
        pb.g1m_timesScalar(pConvertedPoint, pScalarSplit + n8r, n8r, pCalculated);
        pb.g1m_add(pAccumulator, pCalculated, pAccumulator);
        pb.g1m_normalize(pAccumulator, pAccumulator);

        let output = pb.get(pAccumulator, 2, 48);
        let expectedOutput = pb.get(pExpectedOutput, 2, 48);
        for (let i = 0; i < 2; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
    });

    it("preprocessEndomorphism is correct.", async () => {
        let inputPoints = [
            0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bbn, 0x8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1n,
            0x572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4en, 0x166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28n,
            0x9ece308f9d1f0131765212deca99697b112d61f9be9a5f1f3780a51335b3ff981747a0b2ca2179b96d2c0c9024e5224n, 0x32b80d3a6f5b09f8a84623389c5f80ca69a0cddabc3097f9d9c27310fd43be6e745256c634af45ca3473b0590ae30d1n,
            0x10e7791fb972fe014159aa33a98622da3cdc98ff707965e536d8636b5fcc5ac7a91a8c46e59a00dca575af0f18fb13dcn, 0x16ba437edcc6551e30c10512367494bfb6b01cc6681e8a4c3cd2501832ab5c4abc40b4578b85cbaffbf0bcd70d67c6e2n,
            0xc9b60d5afcbd5663a8a44b7c5a02f19e9a77ab0a35bd65809bb5c67ec582c897feb04decc694b13e08587f3ff9b5b60n, 0x143be6d078c2b79a7d4f1d1b21486a030ec93f56aa54e1de880db5a66dd833a652a95bee27c824084006cb5644cbd43fn,
            0x6e82f6da4520f85c5d27d8f329eccfa05944fd1096b20734c894966d12a9e2a9a9744529d7212d33883113a0cadb909n, 0x17d81038f7d60bee9110d9c0d6d1102fe2d998c957f28e31ec284cc04134df8e47e8f82ff3af2e60a6d9688a4563477cn,
            0x1928f3beb93519eecf0145da903b40a4c97dca00b21f12ac0df3be9116ef2ef27b2ae6bcd4c5bc2d54ef5a70627efcb7n, 0x108dadbaa4b636445639d5ae3089b3c43a8a1d47818edd1839d7383959a41c10fdc66849cfa1b08c5a11ec7e28981a1cn,
            0x85ae765588126f5e860d019c0e26235f567a9c0c0b2d8ff30f3e8d436b1082596e5e7462d20f5be3764fd473e57f9cfn, 0x19e7dfab8a794b6abb9f84e57739de172a63415273f460d1607fa6a74f0acd97d9671b801dd1fd4f18232dd1259359a1n,
            0x19cdf3807146e68e041314ca93e1fee0991224ec2a74beb2866816fd0826ce7b6263ee31e953a86d1b72cc2215a57793n, 0x7481b1f261aabacf45c6e4fc278055441bfaf99f604d1f835c0752ac9742b4522c9f5c77db40989e7da608505d48616n,
            0x19cdf3807146e68e041314ca93e1fee0991224ec2a74beb2866816fd0826ce7b6263ee31e953a86d1b72cc2215a57793n, 0x7481b1f261aabacf45c6e4fc278055441bfaf99f604d1f835c0752ac9742b4522c9f5c77db40989e7da608505d48616n,
        ];
        let inputScalars = [
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            9003405095674209932115908784230457051068760537362306482987933690960811974463n,
            0x0842,
            0x0842,
        ];
        let numPoints = 10;
        let expectedScalarOutput = [
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            86900781371527243792514624323931922239n, 39318100695279906693562908013718409681n,
            2114, 0,
            2114, 0,
        ];
        const pRes = pb.alloc(n8q * 3);
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pScalars = pb.alloc(numPoints * n8r);
        const pPreprocessedPoints = pb.alloc(numPoints * n8q * 2 * 2);
        const pPreprocessedScalars = pb.alloc(numPoints * n8r * 2);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(pScalars + n8r * i, inputScalars[i], n8r);
        }
        pb.g1m_glv_preprocessEndomorphism(pPoints, pScalars, numPoints, pPreprocessedPoints, pPreprocessedScalars);
        pb.g1m_multiexp_multiExp(
            pPreprocessedPoints,
            pPreprocessedScalars,
            numPoints * 2,
            pRes,
        );
        pb.g1m_normalize(pRes, pRes);
        let scalarOutput = pb.get(pPreprocessedScalars, numPoints * 2, n8r);
        for (let i = 0; i < 2 * numPoints; i++) {
            assert.equal(expectedScalarOutput[i], scalarOutput[i]);
        }

        // Computes expected output
        const pAccumulator = pb.alloc(n8q * 3);
        const pPointForTest = pb.alloc(n8q * 3);
        const pScalarForTest = pb.alloc(n8r);
        const pCalculated = pb.alloc(n8q * 3);
        pb.g1m_zero(pAccumulator);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pScalarForTest, inputScalars[i], 32);
            pb.set(pPointForTest, inputPoints[i * 2], 48);
            pb.set(pPointForTest + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_one(pPointForTest + 96);
            pb.f1m_toMontgomery(pPointForTest, pPointForTest);
            pb.f1m_toMontgomery(pPointForTest + 48, pPointForTest + 48);
            pb.g1m_timesScalar(pPointForTest, pScalarForTest, n8r, pCalculated);
            pb.g1m_add(pAccumulator, pCalculated, pAccumulator)
        }
        pb.g1m_normalize(pAccumulator, pAccumulator);

        let output = pb.get(pRes, 2, 48);
        let expectedOutput = pb.get(pAccumulator, 2, 48);
        for (let i = 0; i < 2; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
    });

    // // Use this code for benchmark. We comment it out since it takes several minutes to run.
    // it("Benchmark.", async () => {
    //     const scale = 18;
    //     const N = 1 << scale;
    //     console.log("Number of Points: 2^", scale);
    //     const pG1 = pb.bls12381.pG1gen;
    //     const pCalculated = pb.alloc(n8q * 3);
    //     const REPEAT = 10;
    //     const pScalars = pb.alloc(n8r * N);
    //     const rng = new ChaCha();
    //     for (let i = 0; i < N * n8r / 4; i++) {
    //         pb.i32[pScalars / 4 + i] = rng.nextU32();
    //     }
    //     const pPointCoefficients = pb.alloc(n8r * N);
    //     for (let i = 0; i < N * n8r / 4; i++) {
    //         pb.i32[pPointCoefficients / 4 + i] = rng.nextU32();
    //     }
    //     const pPoints = pb.alloc(n8q * 2 * N);
    //     for (let i = 0; i < N; i++) {
    //         pb.g1m_timesScalarAffine(pG1, pPointCoefficients + n8r * i, n8r, pCalculated);
    //         pb.g1m_toAffine(pCalculated, pPoints + i * n8q * 2);
    //     }
    //     const pPreprocessedPoints = pb.alloc(N * n8q * 2 * 2);
    //     const pPreprocessedScalars = pb.alloc(N * n8r * 2);
    //     console.log("Starting multiExp");
    //     let start, end;
    //     start = new Date().getTime();
    //     for (let i = 0; i < REPEAT; i++) {
    //         pb.g1m_multiexpAffine_wasmcurve(pPoints, pScalars, n8r, N, pCalculated);
    //     }
    //     end = new Date().getTime();
    //     time = end - start;
    //     console.log("wasmcurve msm Time (ms): " + time);
    //     const pRes = pb.alloc(n8q * 3);
    //     start = new Date().getTime();
    //     for (let i = 0; i < REPEAT; i++) {
    //         pb.g1m_multiexp_multiExp(pPoints, pScalars, N, pRes);
    //     }
    //     end = new Date().getTime();
    //     time = end - start;
    //     console.log("multiexp+batchAffine msm Time (ms): " + time);
    //     const pResWithGLV = pb.alloc(n8q * 3);
    //     start = new Date().getTime();
    //     for (let i = 0; i < REPEAT; i++) {
    //         pb.g1m_glv_preprocessEndomorphism(pPoints, pScalars, N, pPreprocessedPoints, pPreprocessedScalars);
    //         pb.g1m_multiexp_multiExp(pPreprocessedPoints, pPreprocessedScalars, N * 2, pResWithGLV);
    //     }
    //     end = new Date().getTime();
    //     time = end - start;
    //     console.log("multiexp+batchAffine+GLV msm Time (ms): " + time);
    //     pb.g1m_normalize(pRes, pRes);
    //     pb.g1m_normalize(pCalculated, pCalculated);
    //     pb.g1m_normalize(pResWithGLV, pResWithGLV);
    //     let output = pb.get(pRes, 2, 48);
    //     let wasmcurveOutput = pb.get(pCalculated, 2, 48);
    //     let outputWithGLV = pb.get(pResWithGLV, 2, 48);
    //     assert.equal(output[0], wasmcurveOutput[0]);
    //     assert.equal(output[1], wasmcurveOutput[1]);
    //     assert.equal(output[0], outputWithGLV[0]);
    //     assert.equal(output[1], outputWithGLV[1]);
    // });
});
