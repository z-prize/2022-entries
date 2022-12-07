import {
  PointVectorInput,
  ScalarVectorInput,
  compute_msm,
} from "./src/extra/reference.node.js";
import { msmProjective } from "./src/msm-projective.js";
import { tic, toc } from "./src/extra/tictoc.js";
import { webcrypto } from "node:crypto";
import { mod, p } from "./src/finite-field.js";
import { msmAffine } from "./src/msm.js";
import { bigintFromBytes, bigintToBytes } from "./src/util.js";
import { modInverse } from "./src/finite-field-js.js";
import { msmDumbAffine } from "./src/extra/dumb-curve-affine.js";
import { load } from "./src/scripts/store-inputs.js";
// web crypto compat
if (Number(process.version.slice(1, 3)) < 19) globalThis.crypto = webcrypto;

let runSlowMsm = false;

let n = process.argv[2] ?? 14;
console.log(`running msm with 2^${n} = ${2 ** n} inputs`);

tic("load inputs & convert to rust");
let points, scalars, pointVec, scalarVec;
let loaded = await load(n);
points = loaded.points;
scalars = loaded.scalars;
// custom test data
// points = [points[0], points[1]];
// scalars = [bigintToBytes(0n, 32), bigintToBytes(0n, 32)];

scalarVec = ScalarVectorInput.fromJsArray(scalars);
pointVec = PointVectorInput.fromJsArray(points);
toc();

tic("msm (rust)");
let [xRefBytes, yRefBytes, zRefBytes] = compute_msm(pointVec, scalarVec);
toc();

let xRefProj = bigintFromBytes(xRefBytes);
let yRefProj = bigintFromBytes(yRefBytes);
let zRefProj = bigintFromBytes(zRefBytes);
let [xRef, yRef] = toAffineFromJacobi(xRefProj, yRefProj, zRefProj);

let xBigint, yBigint;
if (runSlowMsm) {
  tic("msm (dumb)");
  [xBigint, yBigint] = msmDumbAffine(scalars, points);
  toc();
}

tic("msm (projective)");
let resultProj = msmProjective(scalars, points);
toc();
let xProjProj = mod(resultProj.x, p);
let yProjProj = mod(resultProj.y, p);
let zProjProj = mod(resultProj.z, p);
let [xProj, yProj] = toAffine(xProjProj, yProjProj, zProjProj);

tic("msm (ours)");
let { result } = msmAffine(scalars, points);
toc();
let [xAffPacked, yAffPacked] = result;
let xAff = bigintFromBytes(xAffPacked);
let yAff = bigintFromBytes(yAffPacked);

if (runSlowMsm) {
  console.log("big === ref", { x: xRef === xBigint, y: yRef === yBigint });
  console.log("big === proj", { x: xBigint === xProj, y: yBigint === yProj });
  console.log("big === aff", { x: xBigint === xAff, y: yBigint === yAff });
}
console.log("ref === proj", { x: xRef === xProj, y: yRef === yProj });
console.log("ref === aff", { x: xRef === xAff, y: yRef === yAff });

console.log("proj === aff", { x: xProj === xAff, y: yProj === yAff });

function toAffine(x, y, z) {
  if (z === 0n) return [0n, 0n, true];
  let zInv = modInverse(z, p);
  return [mod(x * zInv, p), mod(y * zInv, p)];
}
function toAffineFromJacobi(x, y, z) {
  if (z === 0n) return [0n, 0n, true];
  let zInv = modInverse(z, p);
  let zInvSquared = mod(zInv * zInv, p);
  return [mod(x * zInvSquared, p), mod(y * zInvSquared * zInv, p)];
}
