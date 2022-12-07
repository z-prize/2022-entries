import init from "./reference-impl-web/wasm_zkp_challenge.js";
import wasmFile from "./reference-impl-web/wasm_zkp_challenge_bg.wasm";

await init(wasmFile);

export {
  PointVectorInput,
  ScalarVectorInput,
  compute_msm,
} from "./reference-impl-web/wasm_zkp_challenge.js";
