const assert = require("assert");
const ChaCha = require("ffjavascript").ChaCha;
const buildBls12381 = require("../src/bls12381/build_bls12381.js");
const buildProtoboard = require("wasmbuilder").buildProtoboard;

describe("Basic tests for batch affine in bls12-381", function () {
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

    // Copy the i^th affine point in inputPoints vector in javascript memory to a projective point in wasm memory and applies montgomery transformation.
    function copyPoint(inputPoints, i) {
        const p = pb.alloc(2 * n8q * 3);
        pb.set(p, inputPoints[i * 2], 48);
        pb.set(p + 48, inputPoints[i * 2 + 1], 48);
        pb.f1m_one(p + 96, p + 96);
        pb.f1m_toMontgomery(p, p);
        pb.f1m_toMontgomery(p + 48, p + 48);
        return p
    }

    it("organizeBucketsOneRound is correct.", async () => {
        let inputs = [0xffffffffffffffffn, 0x0000000100000003, 0xffffffffffffffffn, 0x0000000300000001, 0x0000000400000002, 0x0000000500000001, 0x0000000600000003];
        let expectedOutput = [0x0000000300000001, 0x0000000500000001, 0x0000000400000002, 0x0000000100000003, 0x0000000600000003, 0xffffffffffffffffn, 0xffffffffffffffffn];
        let expectedOutputBucketCount = [0, 2, 1, 2];
        let numPoints = 7;
        let numBuckets = 4;
        const pPointSchedules = pb.alloc(8 * numPoints);
        const pMetadata = pb.alloc(8 * numPoints);
        const pBucketCount = pb.alloc(4 * numBuckets);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, inputs[i], 8);
        }
        pb.g1m_multiexp_organizeBucketsOneRound(pPointSchedules, numPoints, numBuckets, pMetadata, pBucketCount);
        let output = pb.get(pMetadata, numPoints, 8);
        let outputBucketCounts = pb.get(pBucketCount, numBuckets, 4);
        for (let i = 0; i < numPoints - 2; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
        for (let i = 0; i < numBuckets; i++) {
            assert.equal(outputBucketCounts[i], expectedOutputBucketCount[i]);
        }
    });

    it("organizeBuckets is correct.", async () => {
        let inputs = [
            0xffffffffffffffffn, 0x0000000100000003, 0xffffffffffffffffn, 0x0000000300000001, 0x0000000400000002, 0x0000000500000001, 0x0000000600000003,
            0x0000000000000004, 0x0000000100000002, 0x0000000200000003, 0xffffffffffffffffn, 0x0000000400000007, 0x0000000500000006, 0x0000000600000002,
        ];
        let expectedOutput = [
            0x0000000300000001, 0x0000000500000001, 0x0000000400000002, 0x0000000100000003, 0x0000000600000003, 0xffffffffffffffffn, 0xffffffffffffffffn,
            0x0000000100000002, 0x0000000600000002, 0x0000000200000003, 0x0000000000000004, 0x0000000500000006, 0x0000000400000007, 0xffffffffffffffffn,
        ];
        let expectedOutputBucketCounts = [
            0, 2, 1, 2, 0, 0, 0, 0,
            0, 0, 2, 1, 1, 0, 1, 1,
        ];
        const numPoints = 7;
        const numBuckets = 8;
        const numChunks = 2;
        const pPointSchedules = pb.alloc(8 * numChunks * numPoints);
        const pMetadata = pb.alloc(8 * numChunks * numPoints);
        const pBucketCounts = pb.alloc(4 * numChunks * numBuckets);
        for (let i = 0; i < numChunks * numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, inputs[i], 8);
        }
        pb.g1m_multiexp_organizeBuckets(pPointSchedules, numPoints, numChunks, numBuckets, pMetadata, pBucketCounts);
        let output = pb.get(pMetadata, numChunks * numPoints, 8);
        let outputBucketCounts = pb.get(pBucketCounts, numChunks * numBuckets, 4);
        for (let i = 0; i < numChunks * numPoints; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
        for (let i = 0; i < numChunks * numBuckets; i++) {
            assert.equal(outputBucketCounts[i], expectedOutputBucketCounts[i]);
        }
    });

    it("countBits is correct.", async () => {
        let inputs = [3, 5, 2];
        let expectedOutput = [0, 2, 6, 10];
        const numBuckets = 3;
        const maxBucketBits = 3;
        const pBucketCounts = pb.alloc(4 * numBuckets);
        const pBitOffsets = pb.alloc(4 * (numBuckets + 1));
        for (let i = 0; i < numBuckets; i++) {
            pb.set(pBucketCounts + 4 * i, inputs[i], 4);
        }
        pb.g1m_multiexp_countBits(pBucketCounts, numBuckets, maxBucketBits, pBitOffsets);
        let output = pb.get(pBitOffsets, numBuckets + 1, 4);
        for (let i = 0; i < numBuckets + 1; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
    });

    it("getChunk is correct.", async () => {
        const inputScalarArr = [0xF, 0xF0, 0x70, 0xE5, 0xFFFF, 0xE51, 0x7E51,
            0x27E51, 0x927E51, 0x8927E51, 0xF8927E51, 0x1F8927E51];
        const chunkSizeArr = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
        const scalarSizeArr = [1, 1, 1, 1, 2, 2, 2, 3, 5, 4, 4, 5];
        const numChunksArr = [1, 2, 2, 2, 4, 3, 4, 4, 5, 6, 7, 8];
        const expectedOutputArr = [
            [15],
            [16, 7],
            [16, 3],
            [5, 7],
            [31, 31, 31, 1],
            [17, 18, 3],
            [17, 18, 31, 0],
            [17, 18, 31, 4],
            [17, 18, 31, 4, 9],
            [17, 18, 31, 4, 9, 4],
            [17, 18, 31, 4, 9, 28, 3],
            [17, 18, 31, 4, 9, 28, 7, 0],
        ];
        for (let i = 0; i < 12; i++) {
            let inputScalar = inputScalarArr[i];
            const chunkSize = chunkSizeArr[i];
            const scalarSize = scalarSizeArr[i];
            const numChunks = numChunksArr[i];
            let expectedOutput = expectedOutputArr[i];
            const pScalar = pb.alloc(scalarSize);
            const pChunks = pb.alloc(4 * numChunks);
            pb.set(pScalar, inputScalar, scalarSize);
            pb.g1m_multiexp_testGetChunk(pScalar, scalarSize, chunkSize, pChunks);
            let output = pb.get(pChunks, numChunks, 4);
            if (numChunks == 1) {
                assert.equal(output, expectedOutput[0]);
            } else {
                for (let i = 0; i < numChunks; i++) {
                    assert.equal(output[i], expectedOutput[i]);
                }
            }
        }
    });

    it("singlePointComputeSchedule is correct.", async () => {
        const inputScalarArr = [
            0x0000000F, 0x000000F0, 0x00000070, 0x000000E5,
            0x0000FFFF, 0x00000E51, 0x00007E51, 0x00027E51,
            0x00927E51, 0x08927E51, 0xF8927E51];
        const scalarSize = 4;
        const chunkSize = 5;
        const pointIdx = 0;
        const numChunks = 7;
        const numPoints = 1;
        const expectedOutputPointSchedulesArr = [
            [0x000000000000000F, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000010, 0x0000000000000007, 0xffffffffffffffffn, 0xffffffffffffffffn,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000010, 0x0000000000000003, 0xffffffffffffffffn, 0xffffffffffffffffn,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000005, 0x0000000000000007, 0xffffffffffffffffn, 0xffffffffffffffffn,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x000000000000001F, 0x000000000000001F, 0x000000000000001F, 0x0000000000000001,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000011, 0x0000000000000012, 0x0000000000000003, 0xffffffffffffffffn,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000011, 0x0000000000000012, 0x000000000000001F, 0xffffffffffffffffn,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000011, 0x0000000000000012, 0x000000000000001F, 0x0000000000000004,
                0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000011, 0x0000000000000012, 0x000000000000001F, 0x0000000000000004,
                0x0000000000000009, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000000000011, 0x0000000000000012, 0x000000000000001F, 0x0000000000000004,
                0x0000000000000009, 0x0000000000000004, 0xffffffffffffffffn],
            [0x0000000000000011, 0x0000000000000012, 0x000000000000001F, 0x0000000000000004,
                0x0000000000000009, 0x000000000000001C, 0x0000000000000003],
        ];
        const expectedOutputRoundCountsArr = [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ];
        for (let i = 0; i < 11; i++) {
            let inputScalar = inputScalarArr[i];
            let expectedOutputPointSchedules = expectedOutputPointSchedulesArr[i];
            let expectedOutputRoundCounts = expectedOutputRoundCountsArr[i];
            const pScalar = pb.alloc(scalarSize);
            const pPointSchedules = pb.alloc(8 * numChunks);
            const pRoundCounts = pb.alloc(4 * numChunks);
            pb.set(pScalar, inputScalar, scalarSize);
            pb.g1m_multiexp_singlePointComputeSchedule(pScalar, scalarSize, chunkSize, pointIdx, numPoints, numChunks, pPointSchedules, pRoundCounts);
            let outputPointSchedules = pb.get(pPointSchedules, numChunks, 8);
            let outputRoundCount = pb.get(pRoundCounts, numChunks, 4);
            for (let j = 0; j < numChunks; j++) {
                assert.equal(outputPointSchedules[j], expectedOutputPointSchedules[j]);
                assert.equal(outputRoundCount[j], expectedOutputRoundCounts[j]);
            }
        }
    });

    it("computeSchedule is correct.", async () => {
        const inputScalars = [
            0x0000000F, 0x000000F0, 0x00000070, 0x000000E5,
            0x0000FFFF, 0x00000E51, 0x00007E51, 0x00027E51,
            0x00927E51, 0x08927E51, 0xF8927E51];
        const scalarSize = 4;
        const chunkSize = 5;
        const numChunks = 7;
        const numPoints = 11;
        const expectedOutputPointSchedules = [
            [0x000000000000000F, 0x0000000100000010, 0x0000000200000010, 0x0000000300000005, 0x000000040000001F, 0x0000000500000011, 0x0000000600000011, 0x0000000700000011, 0x0000000800000011, 0x0000000900000011, 0x0000000A00000011], // Round1
            [0xffffffffffffffffn, 0x0000000100000007, 0x0000000200000003, 0x0000000300000007, 0x000000040000001F, 0x0000000500000012, 0x0000000600000012, 0x0000000700000012, 0x0000000800000012, 0x0000000900000012, 0x0000000A00000012], // Round2
            [0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0x000000040000001F, 0x0000000500000003, 0x000000060000001F, 0x000000070000001F, 0x000000080000001F, 0x000000090000001F, 0x0000000A0000001F], // Round3
            [0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0x0000000400000001, 0xffffffffffffffffn, 0xffffffffffffffffn, 0x0000000700000004, 0x0000000800000004, 0x0000000900000004, 0x0000000A00000004], // Round4
            [0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0x0000000800000009, 0x0000000900000009, 0x0000000A00000009], // Round5
            [0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0x0000000900000004, 0x0000000A0000001C], // Round6
            [0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0x0000000A00000003], // Round7
        ];
        const expectedOutputRoundCounts = [11, 10, 7, 5, 3, 2, 1];
        const pScalars = pb.alloc(scalarSize * numPoints);
        const pPointSchedules = pb.alloc(8 * numChunks * numPoints);
        const pRoundCounts = pb.alloc(4 * numChunks);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pScalars + scalarSize * i, inputScalars[i], scalarSize);
        }
        pb.g1m_multiexp_computeSchedule(pScalars, numPoints, scalarSize, chunkSize, numChunks, pPointSchedules, pRoundCounts);
        let outputPointSchedules = pb.get(pPointSchedules, numChunks * numPoints, 8);
        let outputRoundCounts = pb.get(pRoundCounts, numChunks, 4);
        for (let i = 0; i < numChunks; i++) {
            for (let j = 0; j < numPoints; j++) {
                assert.equal(outputPointSchedules[i * numPoints + j], expectedOutputPointSchedules[i][j]);
            }
        }
        for (let i = 0; i < numChunks; i++) {
            assert.equal(outputRoundCounts[i], expectedOutputRoundCounts[i]);
        }
    });

    it("addAffinePointsOneRound is correct.", async () => {
        // use fake point for simplicity
        let pairs = [
            0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bbn,
            0x8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1n,
            0x572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4en,
            0x166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28n,
            0x11111111, 0x11112222, 0x22221111, 0x22222222,
            0xc9b60d5afcbd5663a8a44b7c5a02f19e9a77ab0a35bd65809bb5c67ec582c897feb04decc694b13e08587f3ff9b5b60n,
            0x143be6d078c2b79a7d4f1d1b21486a030ec93f56aa54e1de880db5a66dd833a652a95bee27c824084006cb5644cbd43fn,
            0x99991111, 0x99992222, 0x44441111, 0x44442222,
            0x55551111, 0x55552222,
            0x00001113, 0x66662222, 0x00001113, 0x66662222 // same point test
        ];
        let expectedF1mOutput = []
        let expectedG1mOutput = []
        let resultTest = []
        let cleanDataTest = []
        let pointsInRound = 8;
        let numPoints = 10;
        const pPaires = pb.alloc(numPoints * n8q * 2);

        // F1m operation results
        const x1 = pb.alloc(n8q);
        const y1 = pb.alloc(n8q);
        const x2 = pb.alloc(n8q);
        const y2 = pb.alloc(n8q);
        const m = pb.alloc(n8q);
        const x1_square = pb.alloc(n8q);
        const x1_squarex1_square = pb.alloc(n8q);
        const x1_squarex1_squarex1_square = pb.alloc(n8q);
        const y1_add_y1 = pb.alloc(n8q);
        const y1_add_y1_inv = pb.alloc(n8q);

        const y2_minus_y1 = pb.alloc(n8q);
        const x2_minus_x1 = pb.alloc(n8q);
        const x2_minus_x1_inv = pb.alloc(n8q);
        const x2_add_x1 = pb.alloc(n8q);
        const m_square = pb.alloc(n8q);
        const x1_minus_x3 = pb.alloc(n8q);
        const x1_minus_x3_mul_m = pb.alloc(n8q);
        const x3 = pb.alloc(n8q);
        const y3 = pb.alloc(n8q);

        for (let start = (numPoints - pointsInRound) * 2; start < numPoints * 2; start += 4) {
            pb.set(x1, pairs[start], 48);
            pb.set(y1, pairs[start + 1], 48);
            pb.set(x2, pairs[start + 2], 48);
            pb.set(y2, pairs[start + 3], 48);
            pb.f1m_toMontgomery(x1, x1);
            pb.f1m_toMontgomery(y1, y1);
            pb.f1m_toMontgomery(x2, x2);
            pb.f1m_toMontgomery(y2, y2);

            pb.f1m_add(x2, x1, x2_add_x1);
            pb.f1m_sub(x2, x1, x2_minus_x1);

            if (pb.get(x2_minus_x1, 1, 48) == 0) {
                pb.f1m_mul(x1, x1, x1_square);
                pb.f1m_add(x1_square, x1_square, x1_squarex1_square);
                pb.f1m_add(x1_squarex1_square, x1_square, x1_squarex1_squarex1_square);
                pb.f1m_add(y1, y1, y1_add_y1);
                pb.f1m_inverse(y1_add_y1, y1_add_y1_inv);
                pb.f1m_mul(y1_add_y1_inv, x1_squarex1_squarex1_square, m);
            }
            else {
                pb.f1m_sub(y2, y1, y2_minus_y1);
                pb.f1m_inverse(x2_minus_x1, x2_minus_x1_inv);
                pb.f1m_mul(x2_minus_x1_inv, y2_minus_y1, m);
            }
            pb.f1m_mul(m, m, m_square); // m^2
            pb.f1m_sub(m_square, x2_add_x1, x3);//x3
            pb.f1m_sub(x1, x3, x1_minus_x3);
            pb.f1m_mul(x1_minus_x3, m, x1_minus_x3_mul_m);
            pb.f1m_sub(x1_minus_x3_mul_m, y1, y3);//y3
            pb.f1m_fromMontgomery(x3, x3);
            pb.f1m_fromMontgomery(y3, y3);
            expectedF1mOutput.push(pb.get(x3, 1, 48));
            expectedF1mOutput.push(pb.get(y3, 1, 48));
        }

        // addAffinePointsOneRound results
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPaires + 96 * i, pairs[i * 2], 48);
            pb.set(pPaires + 96 * i + 48, pairs[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPaires + 96 * i, pPaires + 96 * i);
            pb.f1m_toMontgomery(pPaires + 96 * i + 48, pPaires + 96 * i + 48);
        }

        pb.g1m_multiexp_addAffinePointsOneRound(numPoints, pointsInRound, pPaires);
        for (let i = 0; i < numPoints; i++) {
            pb.f1m_fromMontgomery(pPaires + 96 * i, pPaires + 96 * i);
            pb.f1m_fromMontgomery(pPaires + 96 * i + 48, pPaires + 96 * i + 48);
        }
        let output = pb.get(pPaires, numPoints * 2, 48);
        for (let i = numPoints * 2 - pointsInRound; i < numPoints * 2; i++) {
            resultTest.push(output[i]);
        }
        for (let i = 0; i < (numPoints - pointsInRound) * 2; i++) {
            cleanDataTest.push(output[i]);
        }

        // G1m add
        const expectedOutput = pb.alloc(numPoints / 2 * n8q * 3);
        const g1minput = pb.alloc(numPoints * n8q * 3);
        for (let i = 0; i < numPoints; i++) {
            pb.set(g1minput + 144 * i, pairs[i * 2], 48);
            pb.set(g1minput + 144 * i + 48, pairs[i * 2 + 1], 48);
            pb.f1m_toMontgomery(g1minput + 144 * i, g1minput + 144 * i);
            pb.f1m_toMontgomery(g1minput + 144 * i + 48, g1minput + 144 * i + 48);
            pb.f1m_one(g1minput + 144 * i + 96);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(expectedOutput + 48 * i, 0, 48);
        }
        for (let i = 0; i < numPoints / 2; i++) {
            pb.g1m_add(g1minput + 2 * i * 144, g1minput + 2 * i * 144 + 144, expectedOutput + i * 144)
            pb.g1m_normalize(expectedOutput + i * 144, expectedOutput + i * 144)
        }
        for (let i = 0; i < numPoints; i++) {
            pb.f1m_fromMontgomery(expectedOutput + 144 * i, expectedOutput + 144 * i);
            pb.f1m_fromMontgomery(expectedOutput + 144 * i + 48, expectedOutput + 144 * i + 48);
            pb.f1m_fromMontgomery(expectedOutput + 144 * i + 96, expectedOutput + 144 * i + 96);
        }
        let output2 = pb.get(expectedOutput, numPoints / 2 * 3, 48);
        for (let i = (numPoints - pointsInRound) / 2 * 3; i < numPoints / 2 * 3; i += 3) {
            expectedG1mOutput.push(output2[i]);
            expectedG1mOutput.push(output2[i + 1]);
        }

        // Test whether pPaires equals paire in index [0...pointsInRound] 
        for (let i = 0; i < (numPoints - pointsInRound) * 2; i++) {
            assert.equal(pairs[i], cleanDataTest[i]);
        }
        // Test result in two ways
        // f1m
        for (let i = 0; i < pointsInRound; i++) {
            assert.equal(expectedF1mOutput[i], resultTest[i]);
        }
        // wasmcurve g1m
        for (let i = 0; i < pointsInRound; i++) {
            assert.equal(expectedG1mOutput[i], resultTest[i]);
        }
    });

    it("addNaiveOneRound is correct.", async () => {
        let pairs = [
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
        let pointsInRound = 2;
        let numPoints = 10;

        // addNaiveOneRound results
        const pPaires = pb.alloc(numPoints * n8q * 2);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPaires + 96 * i, pairs[i * 2], 48);
            pb.set(pPaires + 96 * i + 48, pairs[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPaires + 96 * i, pPaires + 96 * i);
            pb.f1m_toMontgomery(pPaires + 96 * i + 48, pPaires + 96 * i + 48);
        }
        pb.g1m_multiexp_addNaiveOneRound(numPoints, pointsInRound, pPaires);
        for (let i = 0; i < (numPoints - pointsInRound); i++) {
            pb.f1m_fromMontgomery(pPaires + 96 * i, pPaires + 96 * i);
            pb.f1m_fromMontgomery(pPaires + 96 * i + 48, pPaires + 96 * i + 48);
        }
        for (let i = (numPoints - pointsInRound / 2); i < numPoints; i++) {
            pb.g1m_normalize(pPaires + 96 * i, pPaires + 96 * i);
            pb.f1m_fromMontgomery(pPaires + 96 * i, pPaires + 96 * i);
            pb.f1m_fromMontgomery(pPaires + 96 * i + 48, pPaires + 96 * i + 48);
        }
        let output = pb.get(pPaires, numPoints * 2, 48);

        // addAffinePointsOneRound results
        const pAffineInputPaires = pb.alloc(numPoints * n8q * 2);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pAffineInputPaires + 96 * i, pairs[i * 2], 48);
            pb.set(pAffineInputPaires + 96 * i + 48, pairs[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pAffineInputPaires + 96 * i, pAffineInputPaires + 96 * i);
            pb.f1m_toMontgomery(pAffineInputPaires + 96 * i + 48, pAffineInputPaires + 96 * i + 48);
        }
        pb.g1m_multiexp_addAffinePointsOneRound(numPoints, pointsInRound, pAffineInputPaires);
        for (let i = 0; i < numPoints; i++) {
            pb.f1m_fromMontgomery(pAffineInputPaires + 96 * i, pAffineInputPaires + 96 * i);
            pb.f1m_fromMontgomery(pAffineInputPaires + 96 * i + 48, pAffineInputPaires + 96 * i + 48);
        }
        let expectedOutput = pb.get(pAffineInputPaires, numPoints * 2, 48);

        for (let i = 0; i < (numPoints - pointsInRound); i++) {
            assert.equal(output[2 * i], expectedOutput[2 * i]);
            assert.equal(output[2 * i + 1], expectedOutput[2 * i + 1]);
        }
        for (let i = numPoints - pointsInRound / 2; i < numPoints; i++) {
            assert.equal(output[2 * i], expectedOutput[2 * i]);
            assert.equal(output[2 * i + 1], expectedOutput[2 * i + 1]);
        }
    });

    it("reorderPoints is correct.", async () => {
        // use fake point for simplicity
        let points = [
            0x000011111111111111, 0x000022222222222222,
            0x11111111, 0x11112222,
            0x22221111, 0x22222222,
            0x33331111, 0x33332222,
            0x44441111, 0x44442222,
            0x55551111, 0x55552222,
            0x66661111, 0x66662222,
            0x77771111, 0x77772222,
            0x88881111, 0x88882222,
            0x99991111, 0x99992222,
        ];
        let pointSchedules = [
            0x0000000300000002,
            0x0000000800000001, 0x0000000900000001,
            0x0000000400000002, 0x0000000500000002, 0x0000000600000002, 0x0000000700000002,
            0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn,
        ];
        let expectedOutput = [
            0x33331111, 0x33332222,
            0x88881111, 0x88882222,
            0x99991111, 0x99992222,
            0x44441111, 0x44442222,
            0x55551111, 0x55552222,
            0x66661111, 0x66662222,
            0x77771111, 0x77772222,
            // 0, 0, 0, 0, 0, 0
        ];
        let numPoints = 10;
        let countBucket0 = 3
        const pPointSchedules = pb.alloc(8 * numPoints);
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pRes = pb.alloc(numPoints * n8q * 2);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, pointSchedules[i], 8);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, points[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, points[i * 2 + 1], 48);
        }
        pb.g1m_multiexp_reorderPoints(pPoints, pPointSchedules, numPoints, countBucket0, pRes);
        let output = pb.get(pRes, numPoints * 2, 48);
        for (let i = 0; i < (numPoints - countBucket0) * 2; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
    });

    it("constructAdditionChains is correct.", async () => {
        let inputs = [
            0x0000000800000001, 0x0000000900000001, 0x0000000300000002, 0x0000000400000002,
            0x0000000500000002, 0x0000000600000002, 0x0000000700000002, 0xffffffffffffffffn,
            0xffffffffffffffffn, 0xffffffffffffffffn,
        ];
        let precomputedBucketCounts = [0, 2, 5];
        let expectedOutput = [
            0x0000000300000002, 0x0000000800000001, 0x0000000900000001, 0x0000000400000002, 0x0000000500000002,
            0x0000000600000002, 0x0000000700000002,
            0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn,
        ];
        let expectedBitOffset = [0, 1, 3, 7];
        let numPoints = 10;
        let numBuckets = 3;
        const pPointSchedules = pb.alloc(8 * numPoints);
        const pBucketCounts = pb.alloc(4 * numBuckets);
        const pBitOffsets = pb.alloc((numBuckets + 1) * 4);
        const pMetadata = pb.alloc(8 * numPoints);
        const pMaxBucketBits = pb.alloc(4);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, inputs[i], 8);
        }
        for (let i = 0; i < numBuckets; i++) {
            pb.set(pBucketCounts + 4 * i, precomputedBucketCounts[i], 4);
        }
        pb.g1m_multiexp_constructAdditionChains(pPointSchedules, numPoints, numBuckets, pBucketCounts, pBitOffsets, pMetadata, pMaxBucketBits);
        let output = pb.get(pMetadata, numPoints, 8);
        let outputBitOffset = pb.get(pBitOffsets, 4, 4);
        let maxBucketBits = pb.get(pMaxBucketBits, 1, 4);
        for (let i = 0; i < 4; i++) {
            assert.equal(outputBitOffset[i], expectedBitOffset[i]);
        }
        for (let i = 0; i < 7; i++) {
            assert.equal(output[i], expectedOutput[i]);
        }
        assert.equal(maxBucketBits, 3);
    });

    it("evaluateAdditionChains is correct.", async () => {
        let precomputedBitOffset = [0, 1, 3, 7];
        let inputPoints = [
            0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bbn, 0x8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1n,
            0x572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4en, 0x166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28n,
            0x11111111, 0x11112222,
            0x22221111, 0x22222222,
            0xc9b60d5afcbd5663a8a44b7c5a02f19e9a77ab0a35bd65809bb5c67ec582c897feb04decc694b13e08587f3ff9b5b60n, 0x143be6d078c2b79a7d4f1d1b21486a030ec93f56aa54e1de880db5a66dd833a652a95bee27c824084006cb5644cbd43fn,
            0x99991111, 0x99992222,
            0x44441111, 0x44442222,
            0x55551111, 0x55552222,
            0x00001113, 0x66662222,
            0x00001113, 0x66662222,
        ];
        let pointSchedules = [
            0x0000000300000002,
            0x0000000800000001, 0x0000000900000001,
            0x0000000400000002, 0x0000000500000002, 0x0000000600000002, 0x0000000700000002,
            0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn,
        ];
        // Computation process:
        // Round1:
        // [(p3, 2),
        //  (p8, 1), (p9, 1),
        //  (p4, 2), (p8+p9, 1), (p4+p5, 2), (p6+p7, 2),
        // ]
        // Round2:
        // [(p3, 2),
        //  (p8, 1), (p9, 1),
        //  (p4, 2), (p8+p9, 1), (p4+p5, 2), (p4+p5+p6+p7, 2),
        // ]
        let expectedOutput = [
            0x22221111, 0x22222222,
            0x00001113, 0x66662222,
            0x00001113, 0x66662222,
            0xc9b60d5afcbd5663a8a44b7c5a02f19e9a77ab0a35bd65809bb5c67ec582c897feb04decc694b13e08587f3ff9b5b60n, 0x143be6d078c2b79a7d4f1d1b21486a030ec93f56aa54e1de880db5a66dd833a652a95bee27c824084006cb5644cbd43fn,
            0x11ad74c06469ff4c6759668d8d416f9c3cdbc7951ad425c27ce02286ee1627186b7ed2c1923b18f43d7b647cc5e0f45cn, 0x6a87279419d11a51bebeec5b21ee707a5f80ca4154ce823b0b02c0fef9dbd29e7d221d4e780fc9c413033b05df5646bn,
            0xfc7a315b421f107c57fc2bcf07a39044dec34e22d206c13131a92401398b090ec80d65f2aef2eeee70c60d1f005120en, 0x99b90ad5051c9224f84079be603e4fb77dd6fe4329e870792010da8d65e0f25afcec9bba5954317318158816f549e3en,
            0x1573654bdd7ebf04cee4e365ece6c637702e57046751a5aa1550981b70bd5a5dbaa64460b11837b32e9fde2820f8f02cn, 0x149b2813ec8bbfc753a663d7ebfad6a07b7ff3cf6377dc39dc565c2bb362afb727cd269650dda513b12b8640c86fa8a4n,
        ];
        let numPoints = 10;
        let maxBucketBits = 3;
        let bucketZeroCount = 3;
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pBitOffsets = pb.alloc((maxBucketBits + 1) * 4);
        const pPointSchedules = pb.alloc(numPoints * 8);
        const pPointRes = pb.alloc(numPoints * n8q * 2);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < maxBucketBits + 1; i++) {
            pb.set(pBitOffsets + 4 * i, precomputedBitOffset[i], 4);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, pointSchedules[i], 8);
        }
        pb.g1m_multiexp_reorderPoints(
            pPoints,
            pPointSchedules,
            numPoints,
            bucketZeroCount,
            pPointRes,
        );
        pb.g1m_multiexp_evaluateAdditionChains(
            pPointRes,
            pBitOffsets,
            numPoints - bucketZeroCount,
            maxBucketBits,
        );
        for (let i = 0; i < numPoints * 2; i++) {
            pb.f1m_fromMontgomery(pPointRes + 48 * i, pPointRes + 48 * i);
        }
        let output = pb.get(pPointRes, numPoints * 2, 48);
        assert.equal(output[0], expectedOutput[0]);
        assert.equal(output[1], expectedOutput[1]);
        for (let i = 4; i < numPoints - bucketZeroCount; i++) {
            assert.equal(output[2 * i], expectedOutput[2 * i]);
            assert.equal(output[2 * i + 1], expectedOutput[2 * i + 1]);
        }
    });

    it("reduceBuckets is correct.", async () => {
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
        let pointSchedules = [
            0x0000000800000001, 0x0000000900000001, 0x0000000300000002, 0x0000000400000002,
            0x0000000500000002, 0x0000000600000002, 0x0000000700000002, 0xffffffffffffffffn,
            0xffffffffffffffffn, 0xffffffffffffffffn,
        ];
        let numPoints = 10;
        let numBuckets = 3;
        let maxBucketBits = 3;
        // Bucket counts always only set bucketRounds[0] = 0.
        let bucketCounts = [0, 2, 5];
        let bitOffset = [0, 1, 3, 7];
        // Computation process:
        // Round1 Construct Addition Chain:
        // [(p3, 2),
        //  (p8, 1), (p9, 1),
        //  (p4, 2), (p5, 2), (p6, 2), (p7, 2),
        // Round1 Evaluate Addition Chain:
        // [(p3, 2),
        //  (p8, 1), (p9, 1),
        //  (p4, 2), (p8+p9, 1), (p4+p5, 2), (p4+p5+p6+p7, 2),
        // ]
        // Round2 Construct Addition Chain:
        // [(p8+p9, 1),
        //  (p3, 2), (p4+p5+p6+p7, 2),
        // ]
        // Round2 Evaluate Addition Chain:
        // [(p8+p9, 1),
        //  (p3, 2), (p3+p4+p5+p6+p7, 2),
        // ]
        // Round3 Construct Addition Chain:
        // [(p8+p9, 1), (p3+p4+p5+p6+p7, 2)]
        let expectedOutput = [
            0x1252a4ac3529f8b2b6e8189b95a60b8865f07f9a9b73f98d5df708511d3f68632c4c7d1e2b03e6b1d1e2c01839752adan, 0x2a1bc189e36902d1a49b9965eca3cb818ab5c26dffca63ca9af032870f7bbc615ac65f21bed27bd77dd65f2e90f5358n,
            0xd84464b3966ec5bede84aa487facfca7823af383715078da03b387cc2f5d5597cdd7d025aa07db00a38b953bdeb6e3fn, 0x174a09cd44ccb04c382893c3d197578c85f48cc7c2ae8bbfc7a8cda72ee7ca7833de357666e979a5a9ebac59b5e1d15dn,
        ];
        let expectedOutputPointSchedules = [0x0000000000000001, 0x0000000200000002];
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pBucketCounts = pb.alloc(numBuckets * 4);
        const pBitOffsets = pb.alloc((maxBucketBits + 1) * 4);
        const pPointSchedules = pb.alloc(numPoints * 8);
        const pPointPairs1 = pb.alloc(numPoints * n8q * 2);
        const pPointPairs2 = pb.alloc(numPoints * n8q * 2);
        const pPointScheduleAlt = pb.alloc(numPoints * 8);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < maxBucketBits + 1; i++) {
            pb.set(pBitOffsets + 4 * i, bitOffset[i], 4);
        }
        for (let i = 0; i < numBuckets; i++) {
            pb.set(pBucketCounts + 4 * i, bucketCounts[i], 4);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, pointSchedules[i], 8);
        }
        const pPointRes = pb.g1m_multiexp_reduceBuckets(
            pPoints,
            pPointSchedules,
            numPoints,
            numBuckets,
            pBucketCounts,
            pBitOffsets,
            pPointScheduleAlt,
            pPointPairs1,
            pPointPairs2,
        );
        for (let i = 0; i < numPoints * 2; i++) {
            pb.f1m_fromMontgomery(pPointRes + 48 * i, pPointRes + 48 * i);
        }
        let output = pb.get(pPointRes, numPoints * 2, 48);
        let pointScheduleOutput = pb.get(pPointScheduleAlt, numPoints, 8);
        for (let i = 0; i < 2; i++) {
            assert.equal(output[2 * i], expectedOutput[2 * i]);
            assert.equal(output[2 * i + 1], expectedOutput[2 * i + 1]);
            assert.equal(pointScheduleOutput[i], expectedOutputPointSchedules[i]);
        }
    });

    it("g1m add is correct.", async () => {
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
        let numPoints = 10;
        const pPoints_for_test = pb.alloc(numPoints * n8q * 3);
        const point4_add_point5 = pb.alloc(n8q * 3);
        const point6_add_point7 = pb.alloc(n8q * 3);
        const point4567 = pb.alloc(n8q * 3);
        const point4567ForCompared = pb.alloc(n8q * 3);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints_for_test + 144 * i, inputPoints[i * 2], 48);
            pb.set(pPoints_for_test + 144 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_one(pPoints_for_test + 144 * i + 96);
            pb.f1m_toMontgomery(pPoints_for_test + 144 * i, pPoints_for_test + 144 * i);
            pb.f1m_toMontgomery(pPoints_for_test + 144 * i + 48, pPoints_for_test + 144 * i + 48);
        }
        pb.g1m_add(pPoints_for_test + n8q * 3 * 4, pPoints_for_test + n8q * 3 * 5, point4567);
        pb.g1m_add(point4567, pPoints_for_test + n8q * 3 * 7, point4567);
        pb.g1m_add(point4567, pPoints_for_test + n8q * 3 * 6, point4567);
        pb.g1m_normalize(point4567, point4567);
        pb.g1m_add(pPoints_for_test + n8q * 3 * 4, pPoints_for_test + n8q * 3 * 5, point4_add_point5);
        pb.g1m_add(pPoints_for_test + n8q * 3 * 6, pPoints_for_test + n8q * 3 * 7, point6_add_point7);
        pb.g1m_add(point4_add_point5, point6_add_point7, point4567ForCompared);
        pb.g1m_normalize(point4567ForCompared, point4567ForCompared);
        let output1 = pb.get(point4567, 3, 48);
        let output2 = pb.get(point4567ForCompared, 3, 48);
        for (let i = 0; i < 3; i++) {
            assert.equal(output1[i], output2[i]);
        }
    });

    it("reduceBucketsToSinglePoint is correct.", async () => {
        let pointSchedules = [0x0000000000000001, 0x0000000200000002];
        let pointBuckets = [
            0x1252a4ac3529f8b2b6e8189b95a60b8865f07f9a9b73f98d5df708511d3f68632c4c7d1e2b03e6b1d1e2c01839752adan, 0x2a1bc189e36902d1a49b9965eca3cb818ab5c26dffca63ca9af032870f7bbc615ac65f21bed27bd77dd65f2e90f5358n,
            0xd84464b3966ec5bede84aa487facfca7823af383715078da03b387cc2f5d5597cdd7d025aa07db00a38b953bdeb6e3fn, 0x174a09cd44ccb04c382893c3d197578c85f48cc7c2ae8bbfc7a8cda72ee7ca7833de357666e979a5a9ebac59b5e1d15dn,
        ];
        let numNonZeroBuckets = 2;
        const pPointSchedules = pb.alloc(numNonZeroBuckets * 8);
        const pPointBuckets = pb.alloc(numNonZeroBuckets * n8q * 2);
        const pAccumulator = pb.alloc(n8q * 3);
        const pRunningSum = pb.alloc(n8q * 3);
        for (let i = 0; i < numNonZeroBuckets; i++) {
            pb.set(pPointBuckets + 96 * i, pointBuckets[i * 2], 48);
            pb.set(pPointBuckets + 96 * i + 48, pointBuckets[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPointBuckets + 96 * i, pPointBuckets + 96 * i);
            pb.f1m_toMontgomery(pPointBuckets + 96 * i + 48, pPointBuckets + 96 * i + 48);
        }
        for (let i = 0; i < numNonZeroBuckets; i++) {
            pb.set(pPointSchedules + 8 * i, pointSchedules[i], 8);
        }
        const pExpectedOutput = pb.alloc(2 * n8q * 3);
        for (let i = 0; i < 2; i++) {
            pb.set(pExpectedOutput + 144 * i, pointBuckets[i * 2], 48);
            pb.set(pExpectedOutput + 144 * i + 48, pointBuckets[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pExpectedOutput + 144 * i, pExpectedOutput + 144 * i);
            pb.f1m_toMontgomery(pExpectedOutput + 144 * i + 48, pExpectedOutput + 144 * i + 48);
            pb.f1m_one(pExpectedOutput + 144 * i + 96, pExpectedOutput + 144 * i + 96);
        }
        const pRes = pb.alloc(n8q * 3);
        pb.g1m_add(pExpectedOutput, pExpectedOutput + n8q * 3, pRes);// p1 + p2
        pb.g1m_add(pRes, pExpectedOutput + n8q * 3, pRes);// p1 + p2 + p2
        pb.g1m_normalize(pRes, pRes);
        let expectedOutput = pb.get(pRes, 2, n8q);
        pb.g1m_multiexp_reduceBucketsToSinglePoint(
            pPointSchedules,
            pPointBuckets,
            numNonZeroBuckets,
            pAccumulator,
            pRunningSum,
        );
        pb.g1m_normalize(pAccumulator, pAccumulator);
        let output = pb.get(pAccumulator, 2, n8q);
        assert.equal(output[0], expectedOutput[0]);
        assert.equal(output[1], expectedOutput[1]);
    });

    it("accumulateAcrossChunks is correct.", async () => {
        let inputAccumulator = [
            // First test
            0x0, 0x0,
            // Second test
            0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bbn, 0x8b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1n,
        ];
        let inputAccumulatorSingleChunk = [
            // First test
            0x572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4en, 0x166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28n,
            // Second test
            0x9ece308f9d1f0131765212deca99697b112d61f9be9a5f1f3780a51335b3ff981747a0b2ca2179b96d2c0c9024e5224n, 0x32b80d3a6f5b09f8a84623389c5f80ca69a0cddabc3097f9d9c27310fd43be6e745256c634af45ca3473b0590ae30d1n,
        ];
        let inputChunkIndexArray = [0, 4];
        const numTests = 2;
        const chunkSize = 5;
        let expectedOutput = [
            // First test
            0x572cbea904d67468808c8eb50a9450c9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4en, 0x166a9d8cabc673a322fda673779d8e3822ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28n,
            // Second test
            0x60d5589316a5e16e1d9bb03db45136afb9a3d6e97d350256129ee32a8e33396907dc44d2211762967d88d3e2840f71bn, 0xf7a704c78e5a7707638e8711e923198992fbef5924d12b53d8658ff5ef1c50cf768cb5918e73c39e00f18a006237cd5n,
        ];
        // Computes expected output.
        // const pRes = pb.alloc(n8q * 3);
        // const _pAccumulatorProjective = copyPoint(inputAccumulator, 1);
        // const _pAccumulatorSingleChunkProjective = copyPoint(inputAccumulatorSingleChunk, 1);
        // pb.g1m_double(_pAccumulatorProjective, pRes);// 2*pAccumulatorProjective
        // pb.g1m_double(pRes, pRes);// 2^2*pAccumulatorProjective
        // pb.g1m_double(pRes, pRes);// 2^3*pAccumulatorProjective
        // pb.g1m_double(pRes, pRes);// 2^4*pAccumulatorProjective
        // pb.g1m_double(pRes, pRes);// 2^5*pAccumulatorProjective
        // pb.g1m_add(pRes, _pAccumulatorSingleChunkProjective, pRes);// 2^5*pAccumulatorProjective+pAccumulatorSingleChunkProjective
        // pb.g1m_normalize(pRes, pRes);
        // pb.f1m_fromMontgomery(pRes, pRes);
        // pb.f1m_fromMontgomery(pRes + 48, pRes + 48);
        // let expectedSecondOutput = pb.get(pRes, 2, n8q);
        // console.log(expectedSecondOutput[0].toString(16), expectedSecondOutput[1].toString(16));
        const pAccumulatorProjective = pb.alloc(2 * n8q * 3);
        for (let i = 0; i < numTests; i++) {
            pb.set(pAccumulatorProjective, inputAccumulator[i * 2], 48);
            pb.set(pAccumulatorProjective + 48, inputAccumulator[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pAccumulatorProjective, pAccumulatorProjective);
            pb.f1m_toMontgomery(pAccumulatorProjective + 48, pAccumulatorProjective + 48);
            if (i == 0) {
                pb.f1m_zero(pAccumulatorProjective + 96, pAccumulatorProjective + 96);
            } else {
                pb.f1m_one(pAccumulatorProjective + 96, pAccumulatorProjective + 96);
            }
            const pAccumulatorSingleChunkProjective = copyPoint(inputAccumulatorSingleChunk, i);
            pb.g1m_multiexp_accumulateAcrossChunks(
                pAccumulatorProjective,
                pAccumulatorSingleChunkProjective,
                inputChunkIndexArray[i],
                chunkSize,
                1,
            );
            pb.g1m_normalize(pAccumulatorProjective, pAccumulatorProjective);
            pb.f1m_fromMontgomery(pAccumulatorProjective, pAccumulatorProjective);
            pb.f1m_fromMontgomery(pAccumulatorProjective + 48, pAccumulatorProjective + 48);
            let output = pb.get(pAccumulatorProjective, 2, n8q);
            assert.equal(output[0], expectedOutput[i * 2]);
            assert.equal(output[1], expectedOutput[i * 2 + 1]);
        }
    });

    it("multiExpSingleChunk is correct.", async () => {
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
        let pointSchedules = [
            0x0000000800000001, 0x0000000900000001, 0x0000000300000002, 0x0000000400000002,
            0x0000000500000002, 0x0000000600000002, 0x0000000700000002, 0xffffffffffffffffn,
            0xffffffffffffffffn, 0xffffffffffffffffn,
        ];
        let numNonZeroBuckets = 2;
        let bucketCounts = [0, 2, 5];
        let numPoints = 10;
        let chunkSize = 5;
        let numBuckets = 3;
        let maxBucketBits = 3;
        let roundIdx = 0;
        let expectedOutput = [0xa44163d9f9776392ce5f29f1ecbcc177f8a91f28927f5890c672433b4a3c9b2a34830842d9396dc561348501e885afbn, 0x11fea8cc549a93debba9fad190faea8987e607ef969c38f1015bbfa5d7a6c5b50e240891d55df5f489def327e03c6e55n];
        const pRes = pb.alloc(n8q * 3);
        // Computes expected result.
        // const p3 = copyPoint(inputPoints, 3);
        // const p4 = copyPoint(inputPoints, 4);
        // const p5 = copyPoint(inputPoints, 5);
        // const p6 = copyPoint(inputPoints, 6);
        // const p7 = copyPoint(inputPoints, 7);
        // const p8 = copyPoint(inputPoints, 8);
        // const p9 = copyPoint(inputPoints, 9);
        // pb.g1m_add(p3, p4, pRes);
        // pb.g1m_add(pRes, p5, pRes);
        // pb.g1m_add(pRes, p6, pRes);
        // pb.g1m_add(pRes, p7, pRes);
        // pb.g1m_double(pRes, pRes); // 2*(p3+p4+p5+p6+p7)
        // pb.g1m_add(pRes, p8, pRes);
        // pb.g1m_add(pRes, p9, pRes);
        // pb.g1m_normalize(pRes, pRes);
        // pb.f1m_fromMontgomery(pRes, pRes);
        // pb.f1m_fromMontgomery(pRes + 48, pRes + 48);
        // let expectedOutput = pb.get(pRes, 2, n8q);
        // console.log(expectedOutput[0].toString(16), expectedOutput[1].toString(16));
        const pPointSchedules = pb.alloc(numPoints * 8);
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pBucketCounts = pb.alloc(numBuckets * 4);
        const pAccumulator = pb.alloc(n8q * 3);
        const pRunningSum = pb.alloc(n8q * 3);
        const pBitOffsets = pb.alloc((maxBucketBits + 1) * 4);
        const pPointScheduleAlt = pb.alloc(numPoints * 8);
        const pPointPairs1 = pb.alloc(numPoints * n8q * 2);
        const pPointPairs2 = pb.alloc(numPoints * n8q * 2);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPointSchedules + 8 * i, pointSchedules[i], 8);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < numBuckets; i++) {
            pb.set(pBucketCounts + 4 * i, bucketCounts[i], 4);
        }
        pb.g1m_zero(pRes);
        pb.g1m_multiexp_multiExpSingleChunk(
            pPointSchedules,
            pPoints,
            numNonZeroBuckets,
            pBucketCounts,
            numPoints,
            chunkSize,
            1,
            numBuckets,
            roundIdx,
            pAccumulator,
            pRes,
            pRunningSum,
            pBitOffsets,
            pPointScheduleAlt,
            pPointPairs1,
            pPointPairs2,
        );
        pb.g1m_normalize(pRes, pRes);
        pb.f1m_fromMontgomery(pRes, pRes);
        pb.f1m_fromMontgomery(pRes + 48, pRes + 48);
        let output = pb.get(pRes, 2, 48);
        assert.equal(output[0], expectedOutput[0]);
        assert.equal(output[1], expectedOutput[1]);
    });

    it("multiExpChunks is correct (case 1).", async () => {
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
        let pointSchedules = [
            [0x0000000800000001, 0x0000000900000001, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000800000001, 0x0000000900000001, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000800000001, 0x0000000900000001, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
        ];
        let numNonZeroBuckets = [1, 1, 1];
        let bucketCounts = [
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 0]
        ];
        let numPoints = 10;
        let chunkSize = 1;
        let numBuckets = 3;
        let expectedOutput = [0xf3f78ee37dbcbbc784fa2a75e047e02f8748af86365f3961cfc1b21055e552b46ec0377085da06914e0cffec0d3f0a4n, 0x9edb1291da66508ab53fa7654d393a8b293d19ead0305a9c20488ae6524001c4b3a1fd553590dcb5031fabba8cb3e75n];
        let numChunks = 3;
        const pRes = pb.alloc(n8q * 3);
        const ps = pb.alloc(n8r);
        pb.set(ps, 4 + 2 + 1);
        pb.g1m_timesScalar(pRes, ps, n8r, pRes);
        pb.g1m_normalize(pRes, pRes);
        const pPointSchedules = pb.alloc(numChunks * numPoints * 8);
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pBucketCounts = pb.alloc(numChunks * numBuckets * 4);
        const pNumNonZeroBuckets = pb.alloc(numChunks * 4);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < numChunks; i++) {
            pb.set(pNumNonZeroBuckets + 4 * i, numNonZeroBuckets[i], 4);
        }
        for (let i = 0; i < numChunks; i++) {
            for (let j = 0; j < numPoints; j++) {
                let idx = i * numPoints + j;
                pb.set(pPointSchedules + 8 * idx, pointSchedules[i][j], 8);
            }
        }
        for (let i = 0; i < numChunks; i++) {
            for (let j = 0; j < numBuckets; j++) {
                let idx = i * numBuckets + j;
                pb.set(pBucketCounts + 4 * idx, bucketCounts[i][j], 4);
            }
        }
        pb.g1m_zero(pRes);
        pb.g1m_multiexp_multiExpChunks(
            pPointSchedules,
            pPoints,
            pNumNonZeroBuckets,
            pBucketCounts,
            numPoints,
            chunkSize,
            numChunks,
            numBuckets,
            pRes
        );
        pb.g1m_normalize(pRes, pRes);
        pb.f1m_fromMontgomery(pRes, pRes);
        pb.f1m_fromMontgomery(pRes + 48, pRes + 48);
        let output = pb.get(pRes, 2, 48);
        assert.equal(output[0], expectedOutput[0]);
        assert.equal(output[1], expectedOutput[1]);
    });

    it("multiExpChunks is correct (case 2).", async () => {
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
        let pointSchedules = [
            [0x0000000800000001, 0x0000000900000001, 0x0000000300000002, 0x0000000400000002, 0x0000000500000002, 0x0000000600000002, 0x0000000700000002, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000800000001, 0x0000000900000001, 0x0000000300000001, 0x0000000400000002, 0x0000000500000002, 0x0000000600000002, 0x0000000700000002, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn],
            [0x0000000800000001, 0x0000000900000001, 0x0000000300000001, 0x0000000400000001, 0x0000000500000002, 0x0000000600000002, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn, 0xffffffffffffffffn]
        ];
        let numNonZeroBuckets = [2, 2, 2];
        let bucketCounts = [
            [0, 2, 5],
            [0, 3, 4],
            [0, 4, 2]
        ];
        let numPoints = 10;
        let chunkSize = 5;
        let numBuckets = 3;
        let expectedOutput = [0xab946c3839c448be221f5c9aa507b5ef74cced9a0699cbd95bcc074ebe87839c9e31a37bf69f1246691fa915052f32n, 0x191089c5a773234c4affbb0f70946b4d845287a77019de67eb142f44c0ddf51e1e60899100252bb90970730dcf399c26n];
        let numChunks = 3;
        const pRes = pb.alloc(n8q * 3);

        // Computes expected result.
        // const p3 = copyPoint(inputPoints, 3);
        // const p4 = copyPoint(inputPoints, 4);
        // const p5 = copyPoint(inputPoints, 5);
        // const p6 = copyPoint(inputPoints, 6);
        // const p7 = copyPoint(inputPoints, 7);
        // const p8 = copyPoint(inputPoints, 8);
        // const p9 = copyPoint(inputPoints, 9);
        // const pRes_chunk1 = pb.alloc(n8q * 3);
        // const pRes_chunk2 = pb.alloc(n8q * 3);
        // const pRes_chunk3 = pb.alloc(n8q * 3);
        // pb.g1m_add(p3, p4, pRes_chunk1);
        // pb.g1m_add(pRes_chunk1, p5, pRes_chunk1);
        // pb.g1m_add(pRes_chunk1, p6, pRes_chunk1);
        // pb.g1m_add(pRes_chunk1, p7, pRes_chunk1);
        // pb.g1m_double(pRes_chunk1, pRes_chunk1); // 2 * (p3+p4+p5+p6+p7)
        // pb.g1m_add(pRes_chunk1, p8, pRes_chunk1);
        // pb.g1m_add(pRes_chunk1, p9, pRes_chunk1);// 2 * (p3+p4+p5+p6+p7) + p8 + p9
        // pb.g1m_add(p4, p5, pRes_chunk2);
        // pb.g1m_add(pRes_chunk2, p6, pRes_chunk2);
        // pb.g1m_add(pRes_chunk2, p7, pRes_chunk2);
        // pb.g1m_double(pRes_chunk2, pRes_chunk2); // 2 * (p4+p5+p6+p7)
        // pb.g1m_add(pRes_chunk2, p3, pRes_chunk2);
        // pb.g1m_add(pRes_chunk2, p8, pRes_chunk2);
        // pb.g1m_add(pRes_chunk2, p9, pRes_chunk2);// 2 * (p4+p5+p6+p7) + p8 + p9 + p3
        // pb.g1m_add(p5, p6, pRes_chunk3);
        // pb.g1m_double(pRes_chunk3, pRes_chunk3); // 2 * (p5+p6)
        // pb.g1m_add(pRes_chunk3, p3, pRes_chunk3);
        // pb.g1m_add(pRes_chunk3, p4, pRes_chunk3);
        // pb.g1m_add(pRes_chunk3, p8, pRes_chunk3);
        // pb.g1m_add(pRes_chunk3, p9, pRes_chunk3);// 2 * (p5+p6) + p8 + p9 + p3 + p4
        // const ps = pb.alloc(n8r);
        // pb.set(ps, 1024);
        // pb.g1m_timesScalar(pRes_chunk3, ps, n8r, pRes_chunk3);
        // pb.set(ps, 32);
        // pb.g1m_timesScalar(pRes_chunk2, ps, n8r, pRes_chunk2);
        // pb.g1m_zero(pRes);
        // pb.g1m_add(pRes_chunk1, pRes, pRes);
        // pb.g1m_add(pRes_chunk2, pRes, pRes);
        // pb.g1m_add(pRes_chunk3, pRes, pRes);
        // pb.g1m_normalize(pRes, pRes);
        // printG1("Expected res", pRes);

        const pPointSchedules = pb.alloc(numChunks * numPoints * 8);
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pBucketCounts = pb.alloc(numChunks * numBuckets * 4);
        const pNumNonZeroBuckets = pb.alloc(numChunks * 4);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < numChunks; i++) {
            pb.set(pNumNonZeroBuckets + 4 * i, numNonZeroBuckets[i], 4);
        }
        for (let i = 0; i < numChunks; i++) {
            for (let j = 0; j < numPoints; j++) {
                let idx = i * numPoints + j;
                pb.set(pPointSchedules + 8 * idx, pointSchedules[i][j], 8);
            }
        }
        for (let i = 0; i < numChunks; i++) {
            for (let j = 0; j < numBuckets; j++) {
                let idx = i * numBuckets + j;
                pb.set(pBucketCounts + 4 * idx, bucketCounts[i][j], 4);
            }
        }
        pb.g1m_zero(pRes);
        pb.g1m_multiexp_multiExpChunks(
            pPointSchedules,
            pPoints,
            pNumNonZeroBuckets,
            pBucketCounts,
            numPoints,
            chunkSize,
            numChunks,
            numBuckets,
            pRes
        );
        pb.g1m_normalize(pRes, pRes);
        pb.f1m_fromMontgomery(pRes, pRes);
        pb.f1m_fromMontgomery(pRes + 48, pRes + 48);
        let output = pb.get(pRes, 2, 48);
        assert.equal(output[0], expectedOutput[0]);
        assert.equal(output[1], expectedOutput[1]);
    });

    it("multiExp is correct (case 1).", async () => {
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
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0842, 0x0842,
        ];
        let numPoints = 10;
        let expectedOutput = [0x8f98e551e647b5da65f222e728a590a0bb89948d03d44768c2c2b9f87f74a30db23c44009205a1aa81632166f20d284n, 0x1489ffc27fcc09dd72707b3d1d67db85d1777dcc2e27849628f528154daaf5846cf57d55119091bf7237de9732bb45f6n];
        const pRes = pb.alloc(n8q * 3);
        const pPoints = pb.alloc(numPoints * n8q * 2);
        const pScalars = pb.alloc(numPoints * n8r);
        for (let i = 0; i < numPoints; i++) {
            pb.set(pPoints + 96 * i, inputPoints[i * 2], 48);
            pb.set(pPoints + 96 * i + 48, inputPoints[i * 2 + 1], 48);
            pb.f1m_toMontgomery(pPoints + 96 * i, pPoints + 96 * i);
            pb.f1m_toMontgomery(pPoints + 96 * i + 48, pPoints + 96 * i + 48);
        }
        for (let i = 0; i < numPoints; i++) {
            pb.set(pScalars + n8r * i, inputScalars[i], n8r);
        }
        // expected res
        // pb.g1m_zero(pAccumulator);
        // for(let i = 0; i<numPoints;i++){
        //     pb.set(pScalarForTest, inputScalars[i], 32);
        //     pb.set(pPointForTest, inputPoints[i * 2], 48);
        //     pb.set(pPointForTest + 48, inputPoints[i * 2 + 1], 48);
        //     pb.f1m_one(pPointForTest + 96);
        //     pb.f1m_toMontgomery(pPointForTest, pPointForTest);
        //     pb.f1m_toMontgomery(pPointForTest+ 48, pPointForTest+ 48);
        //     pb.g1m_timesScalar(pPointForTest, pScalarForTest, n8r, pCalculated);
        //     pb.g1m_add(pAccumulator, pCalculated, pAccumulator)
        // }
        // pb.g1m_normalize(pAccumulator, pAccumulator);
        // printG1("Expected", pAccumulator);
        pb.g1m_multiexp_multiExp(
            pPoints,
            pScalars,
            numPoints,
            pRes,
        );
        // let numChunks = 52;
        // // Print pPointSchedules
        // let output = pb.get(outputPointer, numChunks * numPoints, 8);
        // for (let i = 0; i<numChunks; i++) {
        //     for(let j = 0; j< numPoints; j++) {
        //         console.log("i: ", i, ", j: ", j, ", pPointSchedules: ", output[i*numPoints + j].toString(16));
        //     }
        // }
        // // Print pRoundCounts or pNumNonZeroBuckets
        // let output = pb.get(outputPointer, numChunks, 4);
        // for (let i = 0; i < numChunks; i++) {
        //     console.log("i: ", i, ", pRoundCounts: ", output[i].toString(16));
        // }
        // // Print pBucketCounts
        // const numBuckets = 32;
        // let output = pb.get(outputPointer, numChunks * numBuckets, 4);
        // for (let i = 0; i<5; i++) {
        //     for(let j = 0; j< numBuckets; j++) {
        //         console.log("i: ", i, ", j: ", j, ", pBucketCounts: ", output[i*numBuckets + j].toString(16));
        //     }
        // }
        pb.g1m_normalize(pRes, pRes);
        pb.f1m_fromMontgomery(pRes, pRes);
        pb.f1m_fromMontgomery(pRes + 48, pRes + 48);
        let output = pb.get(pRes, 2, 48);
        assert.equal(output[0], expectedOutput[0]);
        assert.equal(output[1], expectedOutput[1]);
    });

    // Use this code for benchmark. We comment it out since it takes several minutes to run.
    // it("Compare our multiExp with wasmcurve.", async () => {
    //     const N = 1 << 16;
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
    //     console.log("Our msm Time (ms): " + time);
    //     pb.g1m_normalize(pRes, pRes);
    //     pb.g1m_normalize(pCalculated, pCalculated);
    //     let output = pb.get(pRes, 2, 48);
    //     let wasmcurveOutput = pb.get(pCalculated, 2, 48);
    //     assert.equal(output[0], wasmcurveOutput[0]);
    //     assert.equal(output[1], wasmcurveOutput[1]);
    // });
});
