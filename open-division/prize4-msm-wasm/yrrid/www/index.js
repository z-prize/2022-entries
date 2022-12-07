import * as reference from "reference";
import * as submission from "./submission.wasm"
import * as utility from "./utility";

const REPEAT = 5;

// Please modify the following for your submission.
// point_arr: a js array of points
// scalar_arr: a js array of scalars
function submission_compute_msm(point_arr, scalar_arr) {
  let size = scalar_arr.length;
  
  var windowBits;
  
  if (size > 128 * 1024) {
    windowBits = 16;
  } else if (size > 96 * 1024) {
    windowBits = 15;
  } else {
    windowBits = 13;
  }

  submission.msmInitialize(size, windowBits, 1024, 128); //nPoints,windowbits as first 2 params
  
  const wasm_buffer = new Uint8Array(submission.memory.buffer);

  var ptr = submission.msmScalarsOffset();
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < 32; j++) {
      wasm_buffer[ptr] = scalar_arr[i][j];
      ptr = ptr + 1;
    }
  }
  
  ptr = submission.msmPointsOffset();
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < 48; j++) {
      wasm_buffer[ptr] = point_arr[i][0][j];
      ptr = ptr + 1;
    }
    for (let j = 0; j < 48; j++) {
      wasm_buffer[ptr] = point_arr[i][1][j];
      ptr = ptr + 1;
    }
  }
  
  let res_ptr = submission.msmRun();
  return [wasm_buffer.slice(res_ptr, res_ptr + 48), wasm_buffer.slice(res_ptr + 48, res_ptr + 96), false];
}

/*********************************************************************************************************
 * Do NOT Edit any code below.
 *********************************************************************************************************/

const median = arr => {
  const mid = Math.floor(arr.length / 2),
    nums = [...arr].sort((a, b) => a - b);
  return arr.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
};

function arraysEqual(arr1, arr2) {
  if (arr1 === arr2) return true;
  if (arr1 == null || arr2 == null) return false;
  if (arr1.length !== arr2.length) return false;
  for (var i = 0; i < arr1.length; ++i) {
    if (arr1[i] !== arr2[i]) return false;
  }
  return true;
}

function check_correctness() {
  for (let repeat = 0; repeat <= 100; repeat++) {
    for (let size = 6; size <= 6; size += 2) { // Note: This size will be updated during evaluation
      const point_vec = new reference.PointVectorInput(Math.pow(2, size));
      const scalar_vec = new reference.ScalarVectorInput(Math.pow(2, size));
      const reference_result = reference.compute_msm(point_vec, scalar_vec).toJsArray();
      const js_point_vec = point_vec.toJsArray();
      const js_scalar_vec = scalar_vec.toJsArray();
      const submission_result = submission_compute_msm(js_point_vec, js_scalar_vec);
      if (!arraysEqual(submission_result[0], reference_result[0])
        || !arraysEqual(submission_result[1], reference_result[1])) {
        return `Correctness check failed.\n\
submission_result: ${submission_result}\n\
reference_result: ${reference_result}\n`;
      }
    }
  }
  return "Correctness check passed.\n";
}

function benchmark_submission() {
  let out_text = "Submission performance.\n";
  for (let size = 6; size <= 6; size += 2) { // Note: This size will be updated during evaluation
    const point_vec = new reference.PointVectorInput(Math.pow(2, size));
    const scalar_vec = new reference.ScalarVectorInput(Math.pow(2, size));
    const js_point_vec = point_vec.toJsArray();
    const js_scalar_vec = scalar_vec.toJsArray();
    const perf = Array.from(
      { length: REPEAT },
      (_, i) => {
        const t0 = performance.now();
        submission_compute_msm(js_point_vec, js_scalar_vec);
        const t1 = performance.now();
        return t1 - t0;
      }
    );
    let cur_res = `Input vector length: 2^${size}, latency: ${median(perf)} ms \n`;
    out_text = out_text.concat(cur_res);
  }
  return out_text;
}

function benchmark_reference() {
  let out_text = "Reference performance.\n";
  for (let size = 6; size <= 6; size += 2) { // Note: This size will be updated during evaluation
    const point_vec = new reference.PointVectorInput(Math.pow(2, size));
    const scalar_vec = new reference.ScalarVectorInput(Math.pow(2, size));
    const perf = Array.from(
      { length: REPEAT },
      (_, i) => {
        const t0 = performance.now();
        reference.compute_msm(point_vec, scalar_vec);
        const t1 = performance.now();
        return t1 - t0;
      }
    );
    let cur_res = `Input vector length: 2^${size}, latency: ${median(perf)} ms \n`;
    out_text = out_text.concat(cur_res);
  }
  return out_text;
}

const correctness_result = check_correctness();
const benchmark_submission_result = benchmark_submission();
const benchmark_reference_result = benchmark_reference();
const pre = document.getElementById("wasm-msm");
pre.textContent = correctness_result + "\n" + benchmark_submission_result + "\n" + benchmark_reference_result;
