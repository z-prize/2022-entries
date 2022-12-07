import fs from "node:fs/promises";
import { createFiniteFieldWat, createGLVWat } from "./finite-field-generate.js";
import { toBase64 } from "fast-base64";
import Wabt from "wabt";

export { compileFiniteFieldWasm, compileWat, interpretWat, writeFile };

let isMain = process.argv[1] === import.meta.url.slice(7);
if (isMain) {
  let p =
    0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaabn;
  let w = 30;
  let q = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001n;
  let lambda = 0xd201000000010000n ** 2n - 1n;
  let beta =
    0x1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaacn;
  compileFiniteFieldWasm(p, w, { withBenchmarks: true, endoCubeRoot: beta });
  compileGLVWasm(q, lambda, w, { withBenchmarks: true });
}

async function compileFiniteFieldWasm(
  p,
  w,
  { withBenchmarks = false, endoCubeRoot } = {}
) {
  let writer = await createFiniteFieldWat(p, w, {
    withBenchmarks,
    endoCubeRoot,
  });
  await writeFile(`./src/wasm/finite-field.wat`, writer.text);
  let { js, wasm } = await compileWat(writer);
  await writeFile(`./src/wasm/finite-field.wasm.js`, js);
  await writeFile("./src/wasm/finite-field.wasm", wasm);
  return { js, wasm, wat: writer.text };
}

async function compileGLVWasm(q, lambda, w, { withBenchmarks = false } = {}) {
  let writer = await createGLVWat(q, lambda, w, { withBenchmarks });
  await writeFile(`./src/wasm/scalar-glv.wat`, writer.text);
  let { js, wasm } = await compileWat(writer);
  await writeFile(`./src/wasm/scalar-glv.wasm.js`, js);
  await writeFile("./src/wasm/scalar-glv.wasm", wasm);
  return { js, wasm, wat: writer.text };
}

// --- general wat2wasm functionality ---
let wabt;

async function writeFile(fileName, content) {
  if (typeof content === "string") {
    await fs.writeFile(fileName, content, "utf8");
  } else {
    await fs.writeFile(fileName, content);
  }
  console.log(`wrote ${(content.length / 1e3).toFixed(1)}kB to ${fileName}`);
}

async function compileWat({ text, exports, imports }) {
  // TODO: imports
  // console.log({ imports });
  let wat = text;
  wabt ??= await Wabt();
  let wabtModule = wabt.parseWat("", wat, wasmFeatures);
  let wasmBytes = new Uint8Array(
    wabtModule.toBinary({ write_debug_names: true }).buffer
  );
  let base64 = await toBase64(wasmBytes);
  return {
    wasm: wasmBytes,
    js: `// compiled from wat
import { toBytes } from 'fast-base64';
let wasmBytes = await toBytes("${base64}");
let { instance } = await WebAssembly.instantiate(
  wasmBytes
);
let { ${[...exports].join(", ")} } = instance.exports;
export { ${[...exports].join(", ")} };
`,
  };
}

async function interpretWat({ text }) {
  // TODO: imports
  let wat = text;
  wabt ??= await Wabt();
  let wabtModule = wabt.parseWat("", wat, wasmFeatures);
  let wasmBytes = new Uint8Array(
    wabtModule.toBinary({ write_debug_names: true }).buffer
  );
  let { instance } = await WebAssembly.instantiate(wasmBytes, {});
  return instance.exports;
}

const wasmFeatures = {
  exceptions: true,
  mutable_globals: true,
  sat_float_to_int: true,
  sign_extension: true,
  simd: true,
  threads: true,
  multi_value: true,
  tail_call: true,
  bulk_memory: true,
  reference_types: true,
  annotations: true,
  gc: true,
};
