import { tic, toc } from "./src/extra/tictoc.js";
import { load } from "./src/scripts/store-inputs.js";
import { webcrypto } from "node:crypto";
import { msmAffine } from "./src/msm.js";
// web crypto compat
if (Number(process.version.slice(1, 3)) < 19) globalThis.crypto = webcrypto;

let n = process.argv[2] ?? 14;
console.log(`running msm with 2^${n} = ${2 ** n} inputs`);

tic("warm-up JIT compiler with fixed set of points");
{
  let { points, scalars } = await load(14);
  msmAffine(scalars, points);
  await new Promise((r) => setTimeout(r, 100));
  msmAffine(scalars, points);
}
toc();

tic("load inputs & convert to rust");
let points, scalars;
let result = await load(n);
points = result.points;
scalars = result.scalars;
toc();

tic("msm (ours)");
let { statistics } = msmAffine(scalars, points);
toc();

let { nMul1, nMul2, nMul3 } = statistics;
let nMul = nMul1 + nMul2 + nMul3;

console.log(`
# muls:
  stage 1: ${(1e-6 * nMul1).toFixed(3).padStart(6)} M
  stage 2: ${(1e-6 * (nMul2 + nMul3)).toFixed(3).padStart(6)} M
  total:  ${(1e-6 * nMul).toFixed(3).padStart(6)} M
`);
