import {
  PointVectorInput,
  ScalarVectorInput,
  compute_msm,
} from "./src/extra/reference.node.js";
import { msmProjective, randomCurvePoints } from "./src/msm-projective.js";
import { tic, toc } from "./src/extra/tictoc.js";
import { load } from "./src/scripts/store-inputs.js";
import { cpus } from "node:os";
import { execSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { webcrypto } from "node:crypto";
import {
  randomBaseFieldx2,
  randomScalars,
  writeBigint,
  benchMultiply,
  benchInverse,
  getPointer,
} from "./src/finite-field.js";
import { msmAffine } from "./src/msm.js";
// web crypto compat
if (Number(process.version.slice(1, 3)) < 19) globalThis.crypto = webcrypto;

// benchmark raw mod mul
let x0 = randomBaseFieldx2();
let x = getPointer();
writeBigint(x, x0);
let nMulRaw = 1e7;
tic("raw mul x 10M");
benchMultiply(x, nMulRaw);
let timeMul = toc();
let mPerSec = Math.round(nMulRaw / timeMul);

// benchmark inverse
let nInvRaw = 5e4;
tic("raw inv x 50K");
benchInverse(nInvRaw);
let timeInv = toc();
let invPerSec = Math.round(nInvRaw / timeInv);
let mulPerInv = mPerSec / invPerSec;

let n = Number(process.argv[2] ?? 14);
console.log(`running msm with 2^${n} = ${2 ** n} inputs`);

tic("warm-up JIT compiler with fixed set of points");
{
  let { points, scalars } = await load(14);
  msmProjective(scalars, points);
  msmAffine(scalars, points);
  msmAffine(scalars, points);
  await new Promise((r) => setTimeout(r, 100));
  msmAffine(scalars, points);
}
toc();

tic("load inputs & convert to rust");
let points, scalars;
if (true) {
  let result = await load(n);
  points = result.points;
  scalars = result.scalars;
} else {
  // TODO: doesn't work
  points = randomCurvePoints(2 ** n);
  scalars = randomScalars(2 ** n);
}
let scalarVec = ScalarVectorInput.fromJsArray(scalars);
let pointVec = PointVectorInput.fromJsArray(points);
toc();

// benchmark msm + count number of muls
tic("msm (rust)");
compute_msm(pointVec, scalarVec);
let ref = toc();
{
  tic("msm (projective)");
  let { nMul1, nMul2, nMul3 } = msmProjective(scalars, points);
  toc();
  let nMul = nMul1 + nMul2 + nMul3;

  console.log(`
# muls:
  stage 1: ${(1e-6 * nMul1).toFixed(3).padStart(6)} M
  stage 2: ${(1e-6 * (nMul2 + nMul3)).toFixed(3).padStart(6)} M
  total:   ${(1e-6 * nMul).toFixed(3).padStart(6)} M
`);
}

tic("msm (ours)");
let { nMul1, nMul2, nMul3, nInv } = msmAffine(scalars, points);
let ours = toc();

let nMul = nMul1 + nMul2 + nMul3;
let nonMulOverhead = 1 - nMul / mPerSec / ours;

console.log(`
# muls:
  stage 1: ${(1e-6 * nMul1).toFixed(3).padStart(6)} M
  stage 2: ${(1e-6 * (nMul2 + nMul3)).toFixed(3).padStart(6)} M
  total:   ${(1e-6 * nMul).toFixed(3).padStart(6)} M

# inv:     ${(1e-3 * nInv).toFixed(3).padStart(6)} K
  ~= muls  ${(1e-6 * mulPerInv * nInv).toFixed(3).padStart(6)} M

~total     ${(1e-6 * (mulPerInv * nInv + nMul * ((5 + 0.8) / 6)))
  .toFixed(3)
  .padStart(6)} M

raw muls / s: ${(1e-6 * mPerSec).toFixed(2)} M
non-mul overhead: ${(100 * nonMulOverhead).toFixed(1)}%
1 inv = ${mulPerInv.toFixed(1)} mul
`);

if (n < 14) process.exit(0);

let commit = execSync("git rev-parse --short HEAD").toString().trim();
let cpu = cpus()[0].model;

let benchmark = { n, ref, ours, nMul, mPerSec, commit, cpu };

let file = "./evaluations/bench.json";
/**
 * @type {(typeof benchmark)[]}
 */
let benchmarks = JSON.parse(await readFile(file, "utf-8"));

// delete any benchmark for same commit & cpu & n
let redundant = benchmarks.findIndex(
  (b) => b.commit === commit && b.cpu === cpu && b.n === n
);
if (redundant !== -1) benchmarks.splice(redundant, 1);

benchmarks.push(benchmark);
await writeFile(file, JSON.stringify(benchmarks, null, 1), "utf-8");
