import { compute_msm_baseline, compute_msm, compute_msm_with_c, generate_msm_inputs, deserialize_msm_inputs, ScalarVectorInput, PointVectorInput, minicov_capture_coverage } from "wasm-prover";

const outputPre = document.getElementById("wasm-prover");
const instanceInput = document.getElementById("instance-file");
const runButtonOpt = document.getElementById("run-button-opt");
const runButtonBase = document.getElementById("run-button-baseline");
const profileButton = document.getElementById("profile-button");

// Parameters for generated MSM inputs.
const MSM_GENERATE_NUM = 10;
const MSM_GENERATE_SIZE = 8;

const ADJUST_C = false;
const MSM_C_MIN = 12;
const MSM_C_STEP = 1;
const MSM_C_MAX = 14;

const MARK_START_DESERIALIZE = () => `Start deserialize input`;
const MARK_STOP_DESERIALIZE = () => `Stop deserialize input`;
const MEASURE_DESERIALIZE = () => `Input deserialize time`;

const MARK_START_GENERATE = (size, c) => `MSM 2^${size}: Start generate input`;
const MARK_STOP_GENERATE = (size, c) => `MSM 2^${size}: Stop generate input`;
const MEASURE_GENERATE = (size, c) => `MSM 2^${size}: Input generation time`;

const MARK_START_MSM = (size, c) => `MSM 2^${size} {c: ${c}}: Start calculation`;
const MARK_STOP_MSM = (size, c) => `MSM 2^${size} {c: ${c}}: Stop calculation`;
const MEASURE_MSM = (size, c) => `MSM 2^${size} {c: ${c}}: Calculation time`;

// compute the median of an array
const median = arr => {
  const mid = Math.floor(arr.length / 2),
    nums = [...arr].sort((a, b) => a - b);
  return arr.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
};

// compute the average of an array
const average = arr => {
  let sum = 0;
  for (const val of arr) {
    sum += val;
  }
  sum /= arr.length
  return sum
};

function buffer2hex(buffer) {
  const array = Array.from(new Uint8Array(buffer))
  const hexarray = array.map(b => b.toString(16).padStart(2, '0'))
  return hexarray.join('')
}

async function deserialize_file_input() {
  // If there is no file input, return undefined.
  if (instanceInput.files.length == 0) {
    return undefined
  }

  const file = instanceInput.files.item(0)
  const data = await file.arrayBuffer()
  // const hash = await crypto.subtle.digest('SHA-256', data)
  // console.log(`Instance input file ${file.name} is of length ${file.size} and hash: ${buffer2hex(hash)}`)
  // Note that this returns an InstanceObjectVector.

  performance.mark(MARK_START_DESERIALIZE())
  const deserialized = deserialize_msm_inputs(new Uint8Array(data))
  performance.mark(MARK_STOP_DESERIALIZE())
  performance.measure(MEASURE_DESERIALIZE(), MARK_START_DESERIALIZE(), MARK_STOP_DESERIALIZE())

  return deserialized
}

async function load_or_generate_msm_inputs() {
  // First check for a file input and deserialize it if one is provided.
  const deserialized = await deserialize_file_input()
  if (deserialized !== undefined) {
    return deserialized
  }

  // No file was provided, so we should generate new inputs.
  const generated = Array.from({ length: MSM_GENERATE_NUM }, () => {
    // Generating the input itself is actually a rather time consuming operation.
    performance.mark(MARK_START_GENERATE(MSM_GENERATE_SIZE))
    const instance = generate_msm_inputs(Math.pow(2, MSM_GENERATE_SIZE))
    performance.mark(MARK_STOP_GENERATE(MSM_GENERATE_SIZE))
    performance.measure(MEASURE_GENERATE(MSM_GENERATE_SIZE), MARK_START_GENERATE(MSM_GENERATE_SIZE), MARK_STOP_GENERATE(MSM_GENERATE_SIZE))
    return instance
  })
  return generated
}

async function wasm_bench_msm_with_c(instances, opt, c) {
  const size = Math.floor(Math.log2(instances.at(0).length)) // Assume all instances as same size.

  for (let j = 0; j < 1; j++) {
    for (let i = 0; i < instances.length; i++) {
      console.log(`Running benchmark with instance ${j}/${i} {c: ${c}}`)
      const instance = instances.at(i)
      const points = instance.points()
      const scalars = instance.scalars()

      // Measure the actual MSM computation.
      performance.mark(MARK_START_MSM(size, c));
      let result;
      if (opt && c !== undefined) {
        result = compute_msm_with_c(points, scalars, c)
      } else if (opt) {
        result = compute_msm(points, scalars)
      } else {
        result = compute_msm_baseline(points, scalars)
      };
      performance.mark(MARK_STOP_MSM(size, c));
      performance.measure(MEASURE_MSM(size, c), MARK_START_MSM(size, c), MARK_STOP_MSM(size, c));
    }
  }

  // Extract the performance markers and format the aggregate result from all instances.
  const measures = performance.getEntriesByName(MEASURE_MSM(size, c), "measure");
  let durations = measures.map(({ duration }) => duration);
  let cur_res = `\nbench_msm(). input vector length: 2^${size} {c: ${c}},\n  median performance: ${median(durations)} ms,\n  average performance: ${average(durations)} ms`;
  outputPre.textContent += cur_res;
  return cur_res;
}

async function wasm_bench_msm(opt) {
  let out_text = "";

  // Clear marks and measures previously written.
  performance.clearMarks();
  performance.clearMeasures();

  const instances = await load_or_generate_msm_inputs()
  const size = Math.floor(Math.log2(instances.at(0).length)) // Assume all instances as same size.
  console.log(`Running benchmark with ${instances.length} instances of size 2^${size}`)

  // Note: Using a classic for loop because the Rust object, InstanceObjectVector, does not support
  // the iterator interface.
  if (opt && ADJUST_C) {
    for (let c = MSM_C_MIN; c <= MSM_C_MAX; c += MSM_C_STEP) {
      out_text += await wasm_bench_msm_with_c(instances, opt, c);
    }
  } else {
    out_text += await wasm_bench_msm_with_c(instances, opt, undefined);
  }
  console.log(`Finished running benchmark`)

  return out_text;
}

function saveByteArray(filename, bytes) {
    var blob = new Blob([bytes], {type: "application/profraw"});
    var link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = filename;
    link.click();
};

// benchmarking msm opt
runButtonOpt.onclick = async () => {
  outputPre.textContent = `running (opt)...`
  outputPre.textContent = await wasm_bench_msm(true)
}

// benchmarking msm baseline
runButtonBase.onclick = async () => {
  outputPre.textContent = `running (baseline)...`
  outputPre.textContent = await wasm_bench_msm(false)
}

// benchmarking msm baseline
profileButton.onclick = async () => {
  const coverage = minicov_capture_coverage()
  saveByteArray("profile.profraw", coverage)
}
