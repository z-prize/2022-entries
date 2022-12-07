import { log2 } from "../util.js";
export { load };

let file = "./inputs.json";

/**
 *
 * @param {number} n
 */
async function load(n) {
  let response = await fetch(file);
  /**
   * @type {{scalars: number[]; points: import("../msm-projective.js").CompatiblePoint[]}}
   */
  let { points, scalars } = await response.json();
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
