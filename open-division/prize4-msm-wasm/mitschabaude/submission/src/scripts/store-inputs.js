import fs from "node:fs/promises";
import { tic, toc } from "../extra/tictoc.js";
import { log2 } from "../util.js";
import { webcrypto } from "node:crypto";
import {
  PointVectorInput,
  ScalarVectorInput,
} from "../extra/reference.node.js";
if (Number(process.version.slice(1, 3)) < 19) globalThis.crypto = webcrypto;

export { load };

let file = "./inputs.json";

Uint8Array.prototype.toJSON = function () {
  return [...this];
};

let isMain = process.argv[1] === import.meta.url.slice(7);
if (isMain) {
  let n = process.argv[2] ?? 14;
  let isLoad = process.argv[3] === "--load" || process.argv[2] === "--load";
  if (!isLoad) {
    await store(n);
  } else {
    tic("load inputs");
    let inputs = await load(n);
    toc();
    console.log(`read 2^${log2(inputs.N)} inputs`);
  }
}

async function store(n) {
  tic("create inputs (rust)");
  let points = new PointVectorInput(2 ** n).toJsArray();
  let scalars = new ScalarVectorInput(2 ** n).toJsArray();
  toc();

  let json = JSON.stringify({ scalars, points });
  await fs.writeFile(file, json, "utf-8");
  console.log(`Wrote ${(json.length * 1e-3).toFixed(2)} kB to ${file}`);
}

/**
 *
 * @param {number} n
 */
async function load(n) {
  /**
   * @type {{scalars: number[]; points: import("../msm-projective.js").CompatiblePoint[]}}
   */
  let { points, scalars } = JSON.parse(await fs.readFile(file, "utf-8"));
  let N = points.length;
  let N0 = 2 ** n;
  if (N0 > N)
    throw Error(`Cannot load 2^${n} points, only have 2^${log2(N)} stored.`);
  points = points
    .slice(0, N0)
    .map(([x, y, inf]) => [new Uint8Array(x), new Uint8Array(y), inf]);
  scalars = scalars.slice(0, N0).map((s) => new Uint8Array(s));
  return { points, scalars, N };
}
