import { tic, toc } from "../extra/tictoc.js";
import { p, randomBaseFieldx2, mod, beta } from "../finite-field-js.js";
import fs from "node:fs/promises";
import { webcrypto } from "node:crypto";
import {
  createFiniteField,
  benchMultiply,
  montgomeryParams,
  multiply32,
  moduleWithMemory,
  jsHelpers,
} from "../finite-field-generate.js";
import { Writer } from "../lib/wasm-generate.js";
import {
  compileFiniteFieldWasm,
  interpretWat,
  writeFile,
} from "../finite-field-compile.js";
import { bigintFromBytes } from "../util.js";
import { getPointers, writeBigint } from "../finite-field.js";
if (Number(process.version.slice(1, 3)) < 19) globalThis.crypto = webcrypto;

let N = 1e7;
for (let w of [30]) {
  let { js, wat } = await compileFiniteFieldWasm(p, w, {
    withBenchmarks: true,
    endoCubeRoot: beta,
  });
  // create extra files which we can identify by bit length
  await writeFile(`./src/wasm/finite-field.${w}.gen.wat`, wat);
  await writeFile(`./src/wasm/finite-field.${w}.wasm.js`, js);
  let wasm = await import(`../wasm/finite-field.${w}.wasm.js`);
  let ff = createFiniteField(p, w, wasm);
  // let [x, z] = testCorrectness(p, w, ff);
  let [x, z] = getPointers(2);
  let x0 = randomBaseFieldx2();
  writeBigint(x, x0);
  writeBigint(z, randomBaseFieldx2());
  tic(`multiply (w=${w}) x ${N}`);
  ff.benchMultiply(x, N);
  let timeMul = toc();
  console.log(`${(N / timeMul / 1e6).toFixed(2).padStart(5)} mio. mul / s`);

  tic();
  ff.benchSquare(x, N);
  let timeSquare = toc();
  console.log(
    `${(N / timeSquare / 1e6).toFixed(2).padStart(5)} mio. squ / s (squ = ${(
      timeSquare / timeMul
    ).toFixed(2)} mul)`
  );

  tic();
  ff.benchMultiplyUnrolled(x, N);
  let timeMulU = toc();
  console.log(
    `${(N / timeMulU / 1e6).toFixed(2).padStart(5)} mio. mun / s (mun = ${(
      timeMulU / timeMul
    ).toFixed(2)} mul)`
  );

  tic();
  ff.benchMultiplyBarrett(x, N);
  let timeMulB = toc();
  console.log(
    `${(N / timeMulB / 1e6).toFixed(2).padStart(5)} mio. mba / s (mba = ${(
      timeMulB / timeMul
    ).toFixed(2)} mul)`
  );
  let timeMulKara;
  if (w === 30) {
    tic();
    ff.benchMultiplyKaratsuba(x, N);
    timeMulKara = toc();
    console.log(
      `${(N / timeMulKara / 1e6).toFixed(2).padStart(5)} mio. mbk / s (mbk = ${(
        timeMulKara / timeMul
      ).toFixed(2)} mul)`
    );
  }
  console.log(`montgomery mul\t ${((timeMul / N) * 1e9).toFixed(0)} ns`);
  if (w === 30)
    console.log(
      `barrett-karatsuba mul\t ${((timeMulKara / N) * 1e9).toFixed(0)} ns`
    );
  console.log(
    `barrett-schoolbook mul\t ${((timeMulB / N) * 1e9).toFixed(0)} ns`
  );

  tic();
  ff.benchMultiplyDifference(x, z, N);
  let timeMulDiff = toc();
  console.log(
    `${(N / timeMulDiff / 1e6).toFixed(2).padStart(5)} mio. mud / s (mud = ${(
      timeMulDiff / timeMul
    ).toFixed(2)} mul)`
  );

  tic();
  ff.benchAdd(x, N);
  let timeAdd = toc();
  console.log(
    `${(N / timeAdd / 1e6).toFixed(2)} mio. add / s (add = ${(
      timeAdd / timeMul
    ).toFixed(2)} mul)`
  );
  tic();
  ff.benchSubtract(z, x, N);
  let timeSubtract = toc();
  console.log(
    `${(N / timeSubtract / 1e6).toFixed(2)} mio. sub / s (sub = ${(
      timeSubtract / timeMul
    ).toFixed(2)} mul)`
  );

  tic();
  benchMultiplyJs(x0, N);
  let timeJs = toc();
  console.log(
    `${(N / timeJs / 1e6).toFixed(2).padStart(5)} mio. mjs / s (mjs = ${(
      timeJs / timeMul
    ).toFixed(2)} mul)`
  );
  console.log(`bigint mul\t ${((timeJs / N) * 1e9).toFixed(0)} ns`);

  console.log();
}
{
  let w = 32;
  // for (let unrollOuter of [0, 1]) {
  for (let unrollOuter of []) {
    let { n } = montgomeryParams(p, w);
    let writer = Writer();
    moduleWithMemory(
      writer,
      `;; generated for w=${w}, n=${n}, n*w=${n * w}`,
      100,
      () => {
        multiply32(writer, p, w, { unrollOuter });
        benchMultiply(writer);
      }
    );
    await fs.writeFile("./src/wasm/finite-field.32.wat", writer.text);
    let wasm = await interpretWat(writer);
    let helpers = jsHelpers(p, w, wasm);
    let x = testCorrectness(p, w, { ...helpers, ...wasm });
    tic(`multiply (w=${w}, unrolled=${unrollOuter}) x ${N}`);
    wasm.benchMultiply(x, N);
    let time = toc();
    console.log(`${(N / time / 1e6).toFixed(2).padStart(5)} mio. mul / s`);
  }
}

function benchMultiplyJs(x, N) {
  for (let i = 0; i < N; i++) {
    x = (x * x) % p;
  }
  return x;
}

/**
 *
 * @param {bigint} p
 * @param {number} w
 * @param {import("../finite-field-generate.js").FiniteField} ff
 * @returns
 */
function testCorrectness(
  p,
  w,
  {
    multiply,
    add,
    subtract,
    reduce,
    isEqual,
    isZero,
    isGreater,
    makeOdd,
    inverse,
    R,
    writeBigint,
    readBigInt,
    getPointers,
    readBytes,
    writeBytes,
  }
) {
  let [x, y, z, R2] = getPointers(4);
  let scratch = getPointers(10);
  for (let i = 0; i < 100; i++) {
    let x0 = randomBaseFieldx2();
    let y0 = randomBaseFieldx2();
    writeBigint(x, x0);
    writeBigint(y, y0);
    writeBigint(R2, mod(R * R, p));
    multiply(z, x, y);
    let z0 = mod(x0 * y0, p);
    multiply(z, z, R2);
    let z1 = readBigInt(z);
    if (z0 !== z1 && !(z1 > p && z0 + p === z1)) {
      throw Error("bad multiplication");
    }

    if (add) {
      add(z, x, y);
      z0 = mod(x0 + y0, p);
      z1 = mod(readBigInt(z), p);
      if (z0 !== z1 && !(z1 > p && z0 + p === z1)) {
        throw Error("bad addition");
      }
    }
    if (subtract) {
      subtract(z, x, y);
      z0 = mod(x0 - y0, p);
      z1 = mod(readBigInt(z), p);
      if (z0 !== z1 && !(z1 > p && z0 + p === z1)) {
        throw Error("bad subtraction");
      }
    }
    if (reduce) {
      reduce(x);
      z0 = mod(x0, p);
      z1 = readBigInt(x);
      if (z0 !== z1) {
        throw Error("bad reduce");
      }
      writeBigint(x, x0);
    }
    if (isEqual) {
      if (isEqual(x, x) !== 1) throw Error("bad isEqual");
      if (isEqual(x, y) !== 0) throw Error("bad isEqual");
      writeBigint(z, 0n);
      if (isZero(z) !== 1) throw Error("bad isZero");
      if (isZero(x) !== 0) throw Error("bad isZero");
      writeBigint(z, 1n);
      add(z, x, z);
      if (isGreater(z, x) !== 1) throw Error("bad isGreater");
      if (isGreater(x, x) !== 0) throw Error("bad isGreater");
      if (isGreater(x, z) !== 0) throw Error("bad isGreater");
    }
    if (makeOdd) {
      // makeOdd
      let m = 117;
      writeBigint(x, 5n << BigInt(m));
      writeBigint(z, 3n);
      let k = makeOdd(x, z);
      if (k !== m) throw Error("bad makeOdd");
      if (readBigInt(x) !== 5n) throw Error("bad makeOdd");
      if (readBigInt(z) !== 3n << BigInt(m)) throw Error("bad makeOdd");
    }
    if (inverse) {
      writeBigint(x, x0);
      multiply(x, x, R2); // x -> xR
      inverse(scratch[0], z, x); // z = 1/x R
      multiply(z, z, x); // x/x R = 1R
      z1 = readBigInt(z);
      if (mod(z1, p) !== mod(R, p)) throw Error("inverse");
    }
    if (readBytes) {
      writeBigint(x, x0);
      let bytes = readBytes(scratch, x);
      let x1 = bigintFromBytes(bytes);
      if (x0 !== x1) throw Error("bad readBytes");
      writeBytes(scratch, x, bytes);
      x1 = readBigInt(x);
      if (x0 !== x1) throw Error("bad writeBytes");
    }
  }
  return [x, z];
}
