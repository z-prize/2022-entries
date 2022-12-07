import { tic, toc } from "./src/extra/tictoc.web.js";
import { load } from "./src/scripts/store-inputs.web.js";
import { msmAffine } from "./src/msm.js";

let n = 16;
console.log(`running msm with 2^${n} inputs`);

tic("load inputs & convert to rust");
let { points, scalars } = await load(n);
toc();

tic("warm-up JIT compiler with fixed points");
msmAffine(scalars.slice(0, 2 ** 14), points.slice(0, 2 ** 14));
toc();
await new Promise((r) => setTimeout(r, 100));

tic("msm (ours)");
msmAffine(scalars, points);
toc();
