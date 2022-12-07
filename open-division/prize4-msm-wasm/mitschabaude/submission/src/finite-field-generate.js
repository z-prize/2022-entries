import { bigintToBits, bigintToBytes, bigintToLegs, log2 } from "./util.js";
import { mod, modInverse, randomBaseFieldx2 } from "./finite-field-js.js";
import {
  addCodeImport,
  addExport,
  addFuncExport,
  block,
  forLoop1,
  forLoop8,
  func,
  ifElse,
  if_,
  loop,
  module,
  ops,
  Writer,
} from "./lib/wasm-generate.js";
import { barrett, karatsuba30, multiply } from "./finite-field-multiply.js";

// main API
export {
  createFiniteField,
  createFiniteFieldWat,
  createGLVWat,
  jsHelpers,
  montgomeryParams,
};

// for w=32 benchmark
export { benchMultiply, multiply32, moduleWithMemory };

// TODO this should NOT be a global constant
const mulInputFactor = 8n;

/**
 * @typedef {ReturnType<typeof createFiniteField> extends Promise<infer T> ? T : never} FiniteField
 */

/**
 * Creates arithmetic functions built on top of Wasm, for any p & w
 *
 * @param {bigint} p
 * @param {number} w
 * @param {import('./wasm/finite-field.wasm')} wasm
 */
function createFiniteField(p, w, wasm) {
  let {
    multiply,
    addNoReduce,
    subtractNoReduce,
    reduce,
    isZero,
    isGreater,
    makeOdd,
    copy,
    isEqual,
    leftShift,
    almostInverseMontgomery,
  } = wasm;
  let helpers = jsHelpers(p, w, wasm);
  let { writeBigint, getPointers, getStablePointers } = helpers;

  // put some constants in wasm memory
  let { K, R } = montgomeryParams(p, w);
  let N = log2(p);

  let constantsBigint = {
    zero: 0n,
    one: 1n,
    p,
    R: mod(R, p),
    R2: mod(R * R, p),
    R2corr: mod(1n << BigInt(4 * K - 2 * N + 1), p),
    // common numbers in montgomery representation
    mg1: mod(1n * R, p),
    mg2: mod(2n * R, p),
    mg4: mod(4n * R, p),
    mg8: mod(8n * R, p),
  };
  let constantsKeys = Object.keys(constantsBigint);
  let constantsPointers = getStablePointers(constantsKeys.length);

  /**
   * @type {Record<keyof typeof constantsBigint, number>}
   */
  let constants = Object.fromEntries(
    constantsKeys.map((key, i) => {
      let pointer = constantsPointers[i];
      writeBigint(pointer, constantsBigint[key]);
      return [key, pointer];
    })
  );

  let pPlus1Div4 = bigintToBits((p + 1n) / 4n, 381);
  let numberOfInversions = 0;

  /**
   * montgomery inverse, a 2^K -> a^(-1) 2^K (mod p)
   *
   * @param {number[]} scratch
   * @param {number} r
   * @param {number} a
   */
  function inverse_(scratch, r, a) {
    if (isZero(a)) throw Error("cannot invert 0");
    numberOfInversions++;
    // TODO: make more efficient
    for (let i = 0; i < Number(mulInputFactor) * 2 - 1; i++) {
      reduce(a);
    }
    // reduce(a);

    let k = almostInverseMontgomery(scratch[0], r, a);
    // TODO: negation -- special case which is simpler
    // don't have to reduce r here, because it's already < p
    subtractNoReduce(r, constants.p, r);

    // mutliply by 2^(2N - k), where N = 381 = bit length of p
    // TODO: efficient multiplication by power-of-2?
    // we use k+1 here because that's the value the theory is about:
    // N <= k+1 <= 2N, so that 0 <= 2N-(k+1) <= N, so that
    // 1 <= 2^(2N-(k+1)) <= 2^N < 2p
    // (in practice, k seems to be normally distributed around ~1.4N and never reach either N or 2N)
    leftShift(r, r, 2 * N - (k + 1)); // * 2^(2N - (k+1)) * 2^(-K)

    // now we multiply by 2^(2(K + K-N) + 1))
    multiply(r, r, constants.R2corr); // * 2^(2K + 2(K-n) + 1) * 2^(-K)
    // = * 2 ^ (2n - k - 1 + 2(K-n) + 1)) = 2^(2*K - k)
    // ^^^ transforms (a * 2^K)^(-1)*2^k = a^(-1) 2^(-K+k)
    //     to a^(-1) 2^(-K+k + 2K -k) = a^(-1) 2^K = the montgomery representation of a^(-1)
  }

  // this is modified from the algorithms in papers in that it
  // * returns k-1 instead of k
  // * returns r < p unconditionally
  // * allows to batch left- / right-shifts
  /**
   *
   * @param {number[]} scratch
   * @param {number} r
   * @param {number} a
   * @returns
   */
  function almostInverseMontgomery_([u, v, s], r, a) {
    // u = p, v = a, r = 0, s = 1
    copy(u, constants.p);
    copy(v, a);
    copy(r, constants.zero);
    copy(s, constants.one);
    let k = 0;
    k += makeOdd(u, s);
    k += makeOdd(v, r);
    while (true) {
      if (isGreater(u, v)) {
        subtractNoReduce(u, u, v);
        addNoReduce(r, r, s);
        k += makeOdd(u, s);
      } else {
        subtractNoReduce(v, v, u);
        addNoReduce(s, r, s);
        if (isZero(v)) break;
        k += makeOdd(v, r);
      }
    }
    // TODO: this works without r << 1 at the end because k is also not incremented
    // so the invariant a*r = 2^k (mod p) is still true with a factor 2 less on both sides
    return k;
  }

  /**
   * sqrt(x)
   *
   * @param {number[]} scratch
   * @param {number} root
   * @param {number} x
   * @returns boolean indicating whether taking the root was successful
   */
  function sqrt([tmp], root, x) {
    pow([tmp], root, x, pPlus1Div4);
    multiply(tmp, root, root);
    reduce(tmp);
    reduce(x);
    if (!isEqual(tmp, x)) return false;
    return true;
  }
  /**
   * montgomery modular exponentiation, a^n
   *
   * @param {number[]} scratch
   * @param {number} x a^n
   * @param {number} a
   * @param {boolean[]} nBits bits of n
   */
  function pow([a], x, a0, nBits) {
    copy(x, constants.mg1);
    copy(a, a0);
    for (let bit of nBits) {
      if (bit) multiply(x, x, a);
      multiply(a, a, a);
    }
  }

  /**
   * benchmark inverse, by doing N*(inv + add)
   * (add is negligible; done to re-randomize, to avoid unrealistic compiler optimizations)
   * @param {number} N
   */
  function benchInverse(N) {
    let scratch = getPointers(10);
    let [x, y] = getPointers(2);
    let x0 = randomBaseFieldx2();
    let y0 = randomBaseFieldx2();
    writeBigint(x, x0);
    writeBigint(y, y0);
    wasm.benchInverse(scratch[0], y, x, N);
  }

  return {
    ...wasm,
    ...helpers,
    constants,
    /**
     * montgomery modular exponentiation, a^n
     */
    pow,
    /**
     * sqrt(x)
     */
    sqrt,
    benchInverse,
    getInversions() {
      return numberOfInversions;
    },
    getAndResetOpCounts() {
      let nMul = wasm.multiplyCount.valueOf();
      let nInv = wasm.inverseCount.valueOf();
      wasm.resetMultiplyCount();
      wasm.resetInverseCount();
      return [nMul, nInv];
    },
  };
}

/**
 *
 * @param {bigint} p
 * @param {number} w
 * @param {{withBenchmarks?: boolean, endoCubeRoot?: bigint}}
 */
async function createFiniteFieldWat(
  p,
  w,
  { withBenchmarks = false, endoCubeRoot = undefined } = {}
) {
  let { n } = montgomeryParams(p, w);
  let writer = Writer();
  moduleWithMemory(
    writer,
    `generated for w=${w}, n=${n}, n*w=${n * w}`,
    // this is the number of "pages" of 2^16 bytes each
    // max # pages is 2^16, which gives us 2^16*2^16 = 2^32 bytes = 4 GB
    (1 << 15) + (1 << 14),
    () => {
      addAffine(writer, p, w);
      wasmInverse(writer, p, w, { countOperations: !!withBenchmarks });

      multiply(writer, p, w, { countMultiplications: !!withBenchmarks });

      add(writer, p, w);
      subtract(writer, p, w);

      reduce(writer, p, w);
      makeOdd(writer, p, w);
      finiteFieldHelpers(writer, p, w);

      barrett(writer, p, w, { withBenchmarks });

      if (w === 30) {
        karatsuba30(writer, p, w, { withBenchmarks });
      }

      if (endoCubeRoot !== undefined) {
        endomorphism(writer, p, w, { beta: endoCubeRoot });
      }

      if (withBenchmarks) {
        benchMultiply(writer);
        benchAdd(writer);
        benchSubtract(writer);
      }
    }
    // {
    //   "console.log": [
    //     ops.func("log", ops.param32()),
    //     ops.func("log64", ops.param64()),
    //   ],
    // }
  );
  return writer;
}

/**
 *
 * @param {bigint} lambda
 * @param {number} w
 */
async function createGLVWat(q, lambda, w, { withBenchmarks = false } = {}) {
  let { n } = montgomeryParams(lambda, w);
  let writer = Writer();
  moduleWithMemory(
    writer,
    `generated for w=${w}, n=${n}, n*w=${n * w}`,
    1 << 10,
    () => {
      barrett(writer, lambda, w, { withBenchmarks });
      glv(writer, q, lambda, w);
    }
  );
  return writer;
}

/**
 * affine EC addition, G3 = G1 + G2
 *
 * assuming d = 1/(x2 - x1) is given, and inputs aren't zero, and x1 !== x2
 * (edge cases are handled one level higher, before batching)
 *
 * this supports addition with assignment where G3 === G1 (but not G3 === G2)
 */
function addAffine(writer, p, w) {
  let { n } = montgomeryParams(p, w);
  let sizeField = 8 * n;
  let { line, lines, comment } = writer;
  let { i32, local, local32, param32, call, return_, br_if } = ops;

  let [x3, x1, x2, y3, y1, y2] = ["$x3", "$x1", "$x2", "$y3", "$y1", "$y2"];
  let [m, tmp, d] = ["$m", "$tmp", "$d"];

  addFuncExport(writer, "addAffine");
  func(
    writer,
    "addAffine",
    [param32(m), param32(x3), param32(x1), param32(x2), param32(d)],
    () => {
      line(local32(y3), local32(y1), local32(y2), local32(tmp));
      lines(
        // compute other pointers from inputs
        local.set(y1, i32.add(x1, sizeField)),
        local.set(y2, i32.add(x2, sizeField)),
        local.set(y3, i32.add(x3, sizeField)),
        local.set(tmp, i32.add(m, sizeField)),
        ";; mark output point as non-zero", // mark output point as non-zero
        i32.store8(x3, 1, { offset: 2 * sizeField }),
        ";; m = (y2 - y1)*d", // m = (y2 - y1)*d
        call("multiplyDifference", m, d, y2, y1),
        ";; x3 = m^2 - x1 - x2", // x3 = m^2 - x1 - x2
        call("square", tmp, m),
        call("subtract", x3, tmp, x1),
        call("subtract", x3, x3, x2),
        ";; y3 = (x2 - x3)*m - y2", // y3 = (x2 - x3)*m - y2
        call("multiplyDifference", y3, m, x2, x3),
        call("subtract", y3, y3, y2)
      );
    }
  );

  let [scratch, S, G, H] = ["$scratch", "$S", "$G", "$H"];
  let [I, x, $n, $i, $j, $N] = ["$I", "$x", "$n", "$i", "$j", "$N"];

  addFuncExport(writer, "batchAddUnsafe");
  func(
    writer,
    "batchAddUnsafe",
    [
      param32(scratch),
      param32(d),
      param32(x),
      param32(S),
      param32(G),
      param32(H),
      param32($n),
    ],
    () => {
      line(local32($i), local32($j), local32(I), local32($N));
      lines(
        local.set(I, scratch),
        local.set(scratch, i32.add(scratch, sizeField)),
        local.set($N, i32.mul($n, sizeField))
      );
      comment("return early if n = 0 or 1");
      line(i32.eqz($n));
      if_(writer, () => {
        line(return_());
      });
      line(i32.eq($n, 1));
      if_(writer, () => {
        lines(
          call("subtractPositive", x, i32.load(H), i32.load(G)),
          call("inverse", scratch, d, x),
          call("addAffine", scratch, i32.load(S), i32.load(G), i32.load(H), d),
          return_()
        );
      });

      comment("create products di = x0*...*xi, where xi = Hi_x - Gi_x");
      lines(
        call("subtractPositive", x, i32.load(H), i32.load(G)),
        call(
          "subtractPositive",
          i32.add(x, sizeField),
          i32.load(H, { offset: 4 }),
          i32.load(G, { offset: 4 })
        ),
        call("multiply", i32.add(d, sizeField), i32.add(x, sizeField), x),
        i32.eq($n, 2)
      );
      if_(writer, () => {
        lines(
          call("inverse", scratch, I, i32.add(d, sizeField)),
          call("multiply", i32.add(d, sizeField), x, I),
          call(
            "addAffine",
            scratch,
            i32.load(S, { offset: 4 }),
            i32.load(G, { offset: 4 }),
            i32.load(H, { offset: 4 }),
            i32.add(d, sizeField)
          ),
          call("multiply", d, i32.add(x, sizeField), I),
          call("addAffine", scratch, i32.load(S), i32.load(G), i32.load(H), d),
          return_()
        );
      });
      line(local.set($i, i32.const(2 * sizeField)));
      line(local.set($j, i32.const(2 * 4)));
      loop(writer, () => {
        lines(
          call(
            "subtractPositive",
            i32.add(x, $i),
            i32.load(i32.add(H, $j)),
            i32.load(i32.add(G, $j))
          ),
          call(
            "multiply",
            i32.add(d, $i),
            i32.add(d, i32.sub($i, sizeField)),
            i32.add(x, $i)
          ),
          local.set($j, i32.add($j, 4)),
          br_if(0, i32.ne($N, local.tee($i, i32.add($i, sizeField))))
        );
        line();
      });
      comment("inverse I = 1/(x0*...*x(n-1))");
      line(call("inverse", scratch, I, i32.add(d, i32.sub($N, sizeField))));
      comment("create inverses 1/x(n-1), ..., 1/x2");
      line(local.set($i, i32.sub($N, sizeField)));
      line(local.set($j, i32.sub($j, 4)));
      loop(writer, () => {
        lines(
          call(
            "multiply",
            i32.add(d, $i),
            i32.add(d, i32.sub($i, sizeField)),
            I
          ),
          call(
            "addAffine",
            scratch,
            i32.load(i32.add(S, $j)),
            i32.load(i32.add(G, $j)),
            i32.load(i32.add(H, $j)),
            i32.add(d, $i)
          ),
          call("multiply", I, I, i32.add(x, $i)),
          local.set($j, i32.sub($j, 4)),
          br_if(0, i32.ne(sizeField, local.tee($i, i32.sub($i, sizeField))))
        );
      });
      comment("1/x1, 1/x0");
      lines(
        call("multiply", i32.add(d, sizeField), x, I),
        call(
          "addAffine",
          scratch,
          i32.load(S, { offset: 4 }),
          i32.load(G, { offset: 4 }),
          i32.load(H, { offset: 4 }),
          i32.add(d, sizeField)
        ),
        call("multiply", d, i32.add(x, sizeField), I),
        call("addAffine", scratch, i32.load(S), i32.load(G), i32.load(H), d)
      );
    }
  );
}

function wasmInverse(writer, p, w, { countOperations = false } = {}) {
  let { n, K, lengthP } = montgomeryParams(p, w);
  let N = lengthP;
  // constants
  let sizeField = 8 * n;

  let { line, lines, comment } = writer;
  let {
    i64,
    i32,
    local,
    local32,
    param32,
    global32Mut,
    global,
    result32,
    call,
    br_if,
    br,
    return_,
  } = ops;

  // count multiplications to analyze higher-level algorithms
  let inverseCount = "$inverseCount";
  if (countOperations) {
    addExport(writer, "inverseCount", global(inverseCount));
    addFuncExport(writer, "resetInverseCount");
    line(global32Mut(inverseCount, 0));
    func(writer, "resetInverseCount", [], () => {
      line(global.set(inverseCount, i32.const(0)));
    });
  }

  // constants we store as global pointers
  let [r2corrGlobal, pGlobal] = ["$r2corr", "$p"];

  let R2corr = mod(1n << BigInt(4 * K - 2 * N + 1), p);
  let P = bigintToLegs(p, w, n);
  dataInt64(writer, r2corrGlobal, bigintToLegs(R2corr, w, n));
  dataInt64(writer, pGlobal, P);

  let [r, a, scratch] = ["$r", "$a", "$scratch"];
  let [u, v, s, k] = ["$u", "$v", "$s", "$k"];

  // this is modified from the algorithms in papers in that it
  // * returns k-1 instead of k
  // * returns r < p unconditionally
  // * allows to batch left- / right-shifts
  addFuncExport(writer, "almostInverse");
  func(
    writer,
    "almostInverse",
    [param32(u), param32(r), param32(a), result32],
    () => {
      line(local32(v), local32(s), local32(k));
      if (countOperations) {
        line(global.set(inverseCount, i32.add(global.get(inverseCount), 1)));
      }
      lines(
        // setup locals
        local.set(v, i32.add(u, sizeField)),
        local.set(s, i32.add(v, sizeField))
      );
      // u = p, v = a, r = 0, s = 1
      for (let i = 0; i < n; i++) {
        line(i64.store(u, P[i], { offset: 8 * i }));
      }
      line(call("copy", v, a));
      for (let i = 0; i < n; i++) {
        line(i64.store(r, 0, { offset: 8 * i }));
      }
      let one = bigintToLegs(1n, w, n);
      for (let i = 0; i < n; i++) {
        line(i64.store(s, one[i], { offset: 8 * i }));
      }
      lines(
        call("makeOdd", u, s),
        call("makeOdd", v, r),
        i32.add(),
        local.set(k)
      );
      block(writer, () => {
        loop(writer, () => {
          line(call("isGreater", u, v));
          ifElse(
            writer,
            () => {
              lines(
                call("subtractNoReduce", u, u, v),
                call("addNoReduce", r, r, s),
                local.set(k, i32.add(k, call("makeOdd", u, s)))
              );
            },
            () => {
              lines(
                call("subtractNoReduce", v, v, u),
                call("addNoReduce", s, s, r),
                br_if(2, call("isZero", v)),
                local.set(k, i32.add(k, call("makeOdd", v, r)))
              );
            }
          );
          line(br(0));
        });
      });
      line(local.get(k));
    }
  );

  addFuncExport(writer, "inverse");
  func(writer, "inverse", [param32(scratch), param32(r), param32(a)], () => {
    line(local32(k));

    for (let i = 0; i < Number(mulInputFactor) * 2 - 1; i++) {
      line(call("reduce", a));
    }
    // for debugging
    // line(call("isZero", a));
    // if_(writer, () => {
    //   lines(call("log", i32.const(500)), return_());
    // });
    lines(
      call("almostInverse", scratch, r, a),
      local.set(k),
      // don't have to reduce r here, because it's already < p
      call("subtractNoReduce", r, global.get(pGlobal), r),
      // multiply by 2^(2N - k), where N = 381 = bit length of p
      // TODO: efficient multiplication by power-of-2?
      // we use k+1 here because that's the value the theory is about:
      // N <= k+1 <= 2N, so that 0 <= 2N-(k+1) <= N, so that
      // 1 <= 2^(2N-(k+1)) <= 2^N < 2p
      // (in practice, k seems to be normally distributed around ~1.4N and never reach either N or 2N)
      call("leftShift", r, r, i32.sub(i32.const(2 * N - 1), k)), // * 2^(2N - (k+1)) * 2^(-K)
      // now we multiply by 2^(2(K + K-N) + 1))
      call("multiply", r, r, global.get(r2corrGlobal)) // * 2^(2K + 2(K-n) + 1) * 2^(-K)
      // = * 2 ^ (2n - k - 1 + 2(K-n) + 1)) = 2^(2*K - k)
      // ^^^ transforms (a * 2^K)^(-1)*2^k = a^(-1) 2^(-K+k)
      //     to a^(-1) 2^(-K+k + 2K -k) = a^(-1) 2^K = the montgomery representation of a^(-1)
    );
  });

  let [I, z, x, $n, $i, $N] = ["$I", "$z", "$x", "$n", "$i", "$N"];

  addFuncExport(writer, "batchInverse");
  func(
    writer,
    "batchInverse",
    [param32(scratch), param32(z), param32(x), param32($n)],
    () => {
      line(local32($i), local32(I), local32($N));
      line(local.set(I, scratch));
      line(local.set(scratch, i32.add(scratch, sizeField)));
      line(local.set($N, i32.mul($n, sizeField)));
      comment("return early if n = 0 or 1");
      line(i32.eqz($n));
      if_(writer, () => {
        line(return_());
      });
      line(i32.eq($n, 1));
      if_(writer, () => {
        lines(call("inverse", scratch, z, x), return_());
      });
      comment("create products x0*x1, ..., x0*...*x(n-1)");
      line(call("multiply", i32.add(z, sizeField), i32.add(x, sizeField), x));
      line(i32.eq($n, 2));
      if_(writer, () => {
        lines(
          call("inverse", scratch, I, i32.add(z, sizeField)),
          call("multiply", i32.add(z, sizeField), x, I),
          call("multiply", z, i32.add(x, sizeField), I),
          return_()
        );
      });
      line(local.set($i, i32.const(2 * sizeField)));
      loop(writer, () => {
        lines(
          call(
            "multiply",
            i32.add(z, $i),
            i32.add(z, i32.sub($i, sizeField)),
            i32.add(x, $i)
          )
        );
        line(br_if(0, i32.ne($N, local.tee($i, i32.add($i, sizeField)))));
      });
      comment("inverse I = 1/(x0*...*x(n-1))");
      line(call("inverse", scratch, I, i32.add(z, i32.sub($N, sizeField))));
      comment("create inverses 1/x(n-1), ..., 1/x2");
      line(local.set($i, i32.sub($N, sizeField)));
      loop(writer, () => {
        lines(
          call(
            "multiply",
            i32.add(z, $i),
            i32.add(z, i32.sub($i, sizeField)),
            I
          ),
          call("multiply", I, I, i32.add(x, $i))
        );
        line(
          br_if(0, i32.ne(sizeField, local.tee($i, i32.sub($i, sizeField))))
        );
      });
      comment("1/x1, 1/x0");
      lines(
        call("multiply", i32.add(z, sizeField), x, I),
        call("multiply", z, i32.add(x, sizeField), I)
      );
    }
  );

  if (countOperations) {
    let [i, N] = ["$i", "$N"];
    addFuncExport(writer, "benchInverse");
    func(
      writer,
      "benchInverse",
      [param32(scratch), param32(a), param32(u), param32(N)],
      () => {
        line(local32(i));
        forLoop1(writer, i, 0, local.get(N), () => {
          lines(
            // x <- x + y
            // y <- 1/x
            call("inverse", scratch, u, a),
            call("add", a, a, u)
          );
        });
      }
    );
  }
}

/**
 * addition modulo 2p
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 */
function add(writer, p, w) {
  let { n, wordMax } = montgomeryParams(p, w);
  // constants

  let P2 = bigintToLegs(mulInputFactor * 2n * p, w, n);
  let { line, lines, comment, join } = writer;
  let { i64, local, local64, param32, br_if } = ops;

  let [x, y, out] = ["$x", "$y", "$out"];
  let [tmp, carry] = ["$tmp", "$carry"];

  function addition({ doReduce }) {
    line(local64(tmp), local64(carry));

    // first loop: x + y
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = x[i] + y[i] + carry;
        i64.load(x, { offset: 8 * i }),
        i64.load(y, { offset: 8 * i }),
        join(i64.add(), local.get(carry), i64.add()),
        // split result
        join(local.tee(tmp), i64.const(w), i64.shr_s(), local.set(carry)),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i })
      );
    }
    if (!doReduce) return;
    // second loop: check if we overflowed by checking x + y < 2p
    block(writer, () => {
      for (let i = n - 1; i >= 0; i--) {
        lines(
          // if (out[i] < 2p[i]) return
          local.set(tmp, i64.load(out, { offset: 8 * i })),
          br_if(1, i64.lt_u(tmp, P2[i])),
          // if (out[i] !== 2p[i]) break;
          br_if(0, i64.ne(tmp, P2[i]))
        );
      }
    });
    // third loop
    // if we're here, t >= 2p, so do t - 2p to get back in 0,..,2p-1
    line(local.set(carry, i64.const(0)));
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = out[i] - 2p[i] + carry;
        i64.load(out, { offset: 8 * i }),
        i64.const(P2[i]),
        i64.sub(),
        local.get(carry),
        i64.add(),
        local.set(tmp),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i }),
        local.set(carry, i64.shr_s(tmp, w))
      );
    }
  }

  addFuncExport(writer, "add");
  func(writer, "add", [param32(out), param32(x), param32(y)], () =>
    addition({ doReduce: true })
  );

  addFuncExport(writer, "addNoReduce");
  func(writer, "addNoReduce", [param32(out), param32(x), param32(y)], () =>
    addition({ doReduce: false })
  );
}

/**
 * subtraction modulo 2p
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 */
function subtract(writer, p, w) {
  let { n, wordMax } = montgomeryParams(p, w);
  // constants
  let dP2 = bigintToLegs(mulInputFactor * 2n * p, w, n);
  let { line, lines, comment, join } = writer;
  let { i64, local, local64, param32 } = ops;

  let [x, y, out] = ["$x", "$y", "$out"];
  let [tmp] = ["$tmp"];

  function subtraction({ doReduce }) {
    line(local64(tmp));

    // first loop: x - y
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = x[i] - y[i] + carry;
        i64.load(x, { offset: 8 * i }),
        i > 0 && i64.add(),
        i64.load(y, { offset: 8 * i }),
        i64.sub(),
        local.set(tmp),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i }),
        (i < n - 1 || doReduce) && i64.shr_s(tmp, w) // put carry on the stack
      );
    }
    if (!doReduce) return;
    // check if we underflowed by checking carry === 0 (in that case, we didn't and can return)
    lines(join(i64.const(0), i64.eq()), `if return end`);
    // second loop
    // if we're here, y > x and out = x - y + R, while we want x - y + 2p
    // so do (out += 2p) and ignore the known overflow of R
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = (2*p)[i] + out[i] + carry;
        i64.const(dP2[i]),
        i > 0 && i64.add(),
        i64.load(out, { offset: 8 * i }),
        i64.add(),
        local.set(tmp),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i }),
        i < n - 1 && i64.shr_s(tmp, w)
      );
    }
  }

  addFuncExport(writer, "subtract");
  func(writer, "subtract", [param32(out), param32(x), param32(y)], () =>
    subtraction({ doReduce: true })
  );

  // x - y for x > y, where we can avoid conditional reducing
  addFuncExport(writer, "subtractNoReduce");
  func(writer, "subtractNoReduce", [param32(out), param32(x), param32(y)], () =>
    subtraction({ doReduce: false })
  );

  // x - y + f*2p -- subtraction that's guaranteed to stay positive if y < f*2p,
  // so there's no conditional branch. (fy are parameters to be tweaked)
  // => output is < x + f*2p but not necessarily <2p
  // this is often fine for inputs to multiplications, which e.g. contract <8p inputs to <2p outputs
  /**
   *
   * @param {number | bigint} f
   */
  function subtractPositive(f) {
    let f2P = bigintToLegs(BigInt(f) * 2n * p, w, n);

    line(local64(tmp));
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = 2p + x[i] - y[i] + carry;
        i64.const(f2P[i]),
        i > 0 && i64.add(),
        i64.load(x, { offset: 8 * i }),
        i64.add(),
        i64.load(y, { offset: 8 * i }),
        i64.sub(),
        local.set(tmp),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i }),
        i < n - 1 && i64.shr_s(tmp, w)
      );
    }
  }

  addFuncExport(writer, "subtractPositive");
  func(
    writer,
    "subtractPositive",
    [param32(out), param32(x), param32(y)],
    () => {
      subtractPositive(mulInputFactor);
    }
  );
}

/**
 * reduce in place from modulo 2*d*p to modulo d*p, i.e.
 * if (x > d*p) x -= d*p
 * (condition: d*p < R = 2^(n*w); we always have d=1 for now but different ones could be used
 * once we try supporting less reductions in add/sub)
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 */
function reduce(writer, p, w, d = 1) {
  let { n, wordMax } = montgomeryParams(p, w);
  // constants
  let dp = bigintToLegs(BigInt(d) * p, w, n);
  let { line, lines, comment } = writer;
  let { i64, local, local64, param32, br_if } = ops;

  let [x] = ["$x"];
  let [tmp, carry] = ["$tmp", "$carry"];

  addFuncExport(writer, "reduce");
  func(writer, "reduce", [param32(x)], () => {
    line(local64(tmp), local64(carry));
    // check if x < p
    block(writer, () => {
      for (let i = n - 1; i >= 0; i--) {
        lines(
          // if (x[i] < d*p[i]) return
          local.set(tmp, i64.load(x, { offset: 8 * i })),
          br_if(1, i64.lt_u(tmp, dp[i])),
          // if (x[i] !== d*p[i]) break;
          br_if(0, i64.ne(tmp, dp[i]))
        );
      }
    });
    // if we're here, t >= dp but we assume t < 2dp, so do t - dp
    line(local.set(carry, i64.const(0)));
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, x[i]) = x[i] - dp[i] + carry;
        i64.load(x, { offset: 8 * i }),
        i64.const(dp[i]),
        i64.sub(),
        local.get(carry),
        i64.add(),
        local.set(tmp),
        i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
        local.set(carry, i64.shr_s(tmp, w))
      );
    }
  });
}

/**
 * a core building block for montgomery inversion
 *
 * takes u, s < p. sets k=0. while u is even, update u /= 2 and s *= 2 and increment k++
 * at the end, u <- u/2^k, s <- s*2^k and the new u is odd
 * returns k
 * (the implementation shifts u >> k and s << k at once if k < w, and shifts by whole words until k < w)
 *
 * in the inversion algorithm it's guaranteed that s << k will remain < p,
 * so everything holds modulo p
 *
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 */
function makeOdd(writer, p, w) {
  let { n, wordMax } = montgomeryParams(p, w);
  let { line, lines, comment } = writer;
  let {
    i64,
    i32,
    local,
    local64,
    local32,
    param32,
    result32,
    memory,
    return_,
    br_if,
    br,
  } = ops;

  let [u, s, k, l, tmp, k0] = ["$u", "$s", "$k", "$l", "$tmp", "$k0"];

  addFuncExport(writer, "makeOdd");
  // (the most common case)
  func(writer, "makeOdd", [param32(u), param32(s), result32], () => {
    line(local64(k), local32(k0), local64(l), local64(tmp));

    // k = count_trailing_zeros(u[0])
    lines(local.set(k, i64.ctz(i64.load(u))), i64.eqz(k));
    if_(writer, () => {
      lines(i32.const(0), return_());
    });
    block(writer, () => {
      // while k === 64 (i.e., u[0] === 0), shift by whole words
      // (note: u is not supposed to be 0, so u[0] = 0 implies that u is divisible by 2^w)
      loop(writer, () => {
        lines(
          br_if(1, i64.ne(k, 64)),

          // copy u[1],...,u[n-1] --> u[0],...,u[n-2]
          memory.copy(local.get(u), i32.add(u, 8), i32.const((n - 1) * 8)),
          // u[n-1] = 0
          i64.store(u, 0, { offset: 8 * (n - 1) }),
          // copy s[0],...,u[n-2] --> s[1],...,s[n-1]
          memory.copy(i32.add(s, 8), local.get(s), i32.const((n - 1) * 8)),
          // s[0] = 0
          i64.store(s, 0),

          local.set(k0, i32.add(k0, w)),

          local.set(k, i64.ctz(i64.load(u))),
          br(0)
        );
      });
    });

    // here we know that k \in 0,...,w-1
    // l = w - k
    line(local.set(l, i64.sub(w, k)));
    comment("u >> k");
    // for (let i = 0; i < n-1; i++) {
    //   u[i] = (u[i] >> k) | ((u[i + 1] << l) & wordMax);
    // }
    // u[n-1] = u[n-1] >> k;
    line(local.set(tmp, i64.load(u)));
    for (let i = 0; i < n - 1; i++) {
      lines(
        local.get(u),
        i64.shr_u(tmp, k),
        i64.and(
          i64.shl(local.tee(tmp, i64.load(u, { offset: 8 * (i + 1) })), l),
          wordMax
        ),
        i64.or(),
        i64.store("", "", { offset: 8 * i })
      );
    }
    line(i64.store(u, i64.shr_u(tmp, k), { offset: 8 * (n - 1) }));
    comment("s << k");
    // for (let i = 10; i >= 0; i--) {
    //   s[i+1] = (s[i] >> l) | ((s[i+1] << k) & wordMax);
    // }
    // s[0] = (s[0] << k) & wordMax;
    line(local.set(tmp, i64.load(s, { offset: 8 * (n - 1) })));
    for (let i = n - 2; i >= 0; i--) {
      lines(
        local.get(s),
        i64.and(i64.shl(tmp, k), wordMax),
        i64.shr_u(local.tee(tmp, i64.load(s, { offset: 8 * i })), l),
        i64.or(),
        i64.store(null, null, { offset: 8 * (i + 1) })
      );
    }
    line(i64.store(s, i64.and(i64.shl(tmp, k), wordMax)));
    comment("return k");
    line(i32.add(k0, i32.wrap_i64(local.get(k))));
  });

  // doing the constant 1 shift + a variable shift (which then is often a no-op)
  // turns out to be slower than just doing the variable shift right away
  // addFuncExport(writer, "shiftTogether1");
  // func(writer, "shiftTogether1", [param32(u), param32(s)], () => {
  //   line(local64(tmp));
  //   let k = 1;
  //   let l = w - 1;
  //   comment("u >> 1");
  //   // for (let i = 0; i < n-1; i++) {
  //   //   u[i] = (u[i] >> 1) | ((u[i + 1] << l) & wordMax);
  //   // }
  //   // u[n-1] = u[n-1] >> k;
  //   line(local.set(tmp, i64.load(u)));
  //   for (let i = 0; i < n - 1; i++) {
  //     lines(
  //       local.get(u),
  //       i64.shr_u(tmp, k),
  //       i64.and(
  //         i64.shl(local.tee(tmp, i64.load(u, { offset: 8 * (i + 1) })), l),
  //         wordMax
  //       ),
  //       i64.or(),
  //       i64.store("", "", { offset: 8 * i })
  //     );
  //   }
  //   line(i64.store(u, i64.shr_u(tmp, k), { offset: 8 * (n - 1) }));
  //   comment("s << 1");
  //   // for (let i = 10; i >= 0; i--) {
  //   //   s[i+1] = (s[i] >> l) | ((s[i+1] << k) & wordMax);
  //   // }
  //   // s[0] = (s[0] << k) & wordMax;
  //   line(local.set(tmp, i64.load(s, { offset: 8 * (n - 1) })));
  //   for (let i = n - 2; i >= 0; i--) {
  //     lines(
  //       local.get(s),
  //       i64.and(i64.shl(tmp, k), wordMax),
  //       i64.shr_u(local.tee(tmp, i64.load(s, { offset: 8 * i })), l),
  //       i64.or(),
  //       i64.store(null, null, { offset: 8 * (i + 1) })
  //     );
  //   }
  //   line(i64.store(s, i64.and(i64.shl(tmp, k), wordMax)));
  // });
}

/**
 * various helpers for finite field arithmetic:
 * isEqual, isZero, isGreater, copy
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 */
function finiteFieldHelpers(writer, p, w) {
  let { n, wordMax, lengthP } = montgomeryParams(p, w);
  let { line, lines, comment } = writer;
  let { i64, i32, local, local64, param32, result32, return_, br_if, memory } =
    ops;

  let [x, y, xi, yi, bytes, tmp] = ["$x", "$y", "$xi", "$yi", "$bytes", "$tmp"];

  // x === y
  addFuncExport(writer, "isEqual");
  func(writer, "isEqual", [param32(x), param32(y), result32], () => {
    for (let i = 0; i < n; i++) {
      line(
        i64.ne(i64.load(x, { offset: 8 * i }), i64.load(y, { offset: 8 * i }))
      );
      if_(writer, () => {
        line(return_(i32.const(0)));
      });
    }
    line(i32.const(1));
  });

  // x === -y
  let P = bigintToLegs(p, w, n);
  addFuncExport(writer, "isEqualNegative");
  func(writer, "isEqualNegative", [param32(x), param32(y), result32], () => {
    line(local64(tmp));
    for (let i = 0; i < n; i++) {
      lines(
        // x[i] + y[i] ?= P[i]
        i64.load(x, { offset: 8 * i }),
        i > 0 && i64.add(),
        i64.load(y, { offset: 8 * i }),
        i64.add(),
        local.set(tmp),
        i64.and(tmp, wordMax),
        i64.const(P[i]),
        i64.ne()
      );
      if_(writer, () => {
        line(return_(i32.const(0)));
      });
      i < n - 1 && line(i64.shr_u(tmp, w));
    }
    line(i32.const(1));
  });

  // x === 0
  addFuncExport(writer, "isZero");
  func(writer, "isZero", [param32(x), result32], () => {
    for (let i = 0; i < n; i++) {
      line(i64.ne(i64.load(x, { offset: 8 * i }), 0));
      if_(writer, () => {
        line(return_(i32.const(0)));
      });
    }
    line(i32.const(1));
  });

  // x > y
  addFuncExport(writer, "isGreater");
  func(writer, "isGreater", [param32(x), param32(y), result32], () => {
    line(local64(xi), local64(yi));
    block(writer, () => {
      for (let i = n - 1; i >= 0; i--) {
        lines(
          local.tee(xi, i64.load(x, { offset: 8 * i })),
          local.tee(yi, i64.load(y, { offset: 8 * i })),
          i64.gt_u()
        );
        if_(writer, () => {
          line(return_(i32.const(1)));
        });
        line(br_if(0, i64.ne(xi, yi)));
      }
    });
    line(i32.const(0));
  });

  // copy contents of y into x
  // this should just be inlined if possible
  addFuncExport(writer, "copy");
  func(writer, "copy", [param32(x), param32(y)], () => {
    line(memory.copy(local.get(x), local.get(y), i32.const(8 * n)));
  });

  // convert between internal format and I/O-friendly, packed byte format
  // method: just pack all the n*w bits into memory contiguously
  let nPackedBytes = Math.ceil(lengthP / 8);
  addFuncExport(writer, "toPackedBytes");
  comment(
    `converts ${n}x${w}-bit representation (1 int64 per ${w}-bit limb) to packed ${nPackedBytes}-byte representation`
  );
  func(writer, "toPackedBytes", [param32(bytes), param32(x)], () => {
    let offset = 0; // memory offset
    let nRes = 0; // residual bits to write from last iteration

    line(local64(tmp)); // holds bits that aren't written yet
    // write bytes word by word
    for (let i = 0; i < n; i++) {
      // how many bytes to write in this iteration
      let nBytes = Math.floor((nRes + w) / 8); // max number of bytes we can get from residual + this word
      let bytesMask = (1n << (8n * BigInt(nBytes))) - 1n;
      lines(
        // tmp = tmp | (x[i] >> nr)  where nr is the bit length of tmp (nr < 8)
        i64.shl(i64.load(x, { offset: 8 * i }), nRes),
        local.get(tmp),
        i64.or(),
        local.set(tmp),
        // store bytes at current offset
        i64.store(bytes, i64.and(tmp, bytesMask), { offset }),
        // keep residual bits for next iteration
        local.set(tmp, i64.shr_u(tmp, 8 * nBytes))
      );
      offset += nBytes;
      nRes = nRes + w - 8 * nBytes;
    }
    // final round: write residual bits, if there are any
    if (offset < nPackedBytes) line(i64.store(bytes, tmp, { offset }));
  });

  let chunk = "$chunk";

  addFuncExport(writer, "fromPackedBytes");
  comment(
    `recovers ${n}x${w}-bit representation (1 int64 per ${w}-bit limb) from packed ${nPackedBytes}-byte representation`
  );
  func(writer, "fromPackedBytes", [param32(x), param32(bytes)], () => {
    let offset = 0; // bytes offset
    let nRes = 0; // residual bits read in the last iteration

    line(local64(tmp), local64(chunk));
    lines(local.set(tmp, i64.const(0)));
    // read bytes word by word
    for (let i = 0; i < n; i++) {
      // if we can't fill up w bits with the current residual, load a full i64 from bytes
      // (some of that i64 could be garbage, but we'll only use the parts that aren't)
      if (nRes < w) {
        lines(
          // tmp = (bytes << nRes) | tmp
          i64.shl(
            // load 8 bytes at current offset
            // due to the left shift, we lose nRes of them
            local.tee(chunk, i64.load(bytes, { offset })),
            nRes
          ),
          local.get(tmp),
          i64.or(),
          local.set(tmp),
          // store what fits in next word
          i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
          // keep residual bits for next iteration
          local.set(tmp, i64.shr_u(chunk, w - nRes))
        );
        offset += 8;
        nRes = nRes - w + 64;
      } else {
        // otherwise, the current tmp is just what we want!
        lines(
          i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
          local.set(tmp, i64.shr_u(tmp, w))
        );
        nRes = nRes - w;
      }
    }
  });
}

/**
 *
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 * @param {{beta: bigint}} options
 */
function endomorphism(writer, p, w, { beta }) {
  let { n, R } = montgomeryParams(p, w);
  let sizeField = 8 * n;

  let { line, lines } = writer;
  let { i32, local, local32, param32, global, call } = ops;

  // store beta as global pointer
  let [betaGlobal] = ["$beta"];
  let betaMontgomery = mod(beta * R, p);
  dataInt64(writer, betaGlobal, bigintToLegs(betaMontgomery, w, n));

  let [x, xOut, y, yOut] = ["$x", "$x_out", "$y", "$y_out"];

  addFuncExport(writer, "endomorphism");
  func(writer, "endomorphism", [param32(xOut), param32(x)], () => {
    line(local32(y), local32(yOut));
    lines(
      // compute other pointers from inputs
      local.set(y, i32.add(x, sizeField)),
      local.set(yOut, i32.add(xOut, sizeField)),
      // x_out = x * beta
      call("multiply", xOut, x, global.get(betaGlobal)),
      // y_out = y
      call("copy", yOut, y)
    );
  });
}

/**
 * functions for glv decompositions of scalars
 *
 * @param {any} writer
 * @param {bigint} lambda
 * @param {number} w
 */
function glv(writer, q, lambda, w) {
  let { n, wordMax, lengthP } = montgomeryParams(lambda, w);
  let { line, lines, comment } = writer;
  let {
    i64,
    i32,
    local,
    local64,
    local32,
    param32,
    result32,
    br_if,
    call,
    return_,
  } = ops;

  let k = lengthP - 1;
  let N = n * w;
  let m = 2n ** BigInt(k + N) / lambda;
  let LAMBDA = bigintToLegs(lambda, w, n);

  // let's compute the maximum error in barrett reduction
  // scalars are < q, which is slightly larger than lambda^2
  function barrettError(dSquare) {
    let errNumerator =
      m * 2n ** BigInt(k) * lambda +
      BigInt(dSquare) * lambda ** 2n * (2n ** BigInt(k + N) - m * lambda);
    let errDenominator = lambda * 2n ** BigInt(k + N);
    let lengthErr = BigInt(errDenominator.toString().length);
    let err =
      Number(errNumerator / 10n ** (lengthErr - 5n)) /
      Number(errDenominator / 10n ** (lengthErr - 5n));
    return err;
  }
  let dSquare = q / lambda ** 2n + 1n;
  let e = Math.ceil(barrettError(dSquare));
  if (e > 1) {
    console.warn("WARNING: barrett error of approximating l can be > 1");
  }
  // e is how often we have to reduce by lambda if we want a decomposition x = x0 + lambda * x1 with x0 < lambda

  let [x, bytes, tmp, carry] = ["$x", "$bytes", "$tmp", "$carry"];
  let [r, l] = ["$r", "$l"];

  addFuncExport(writer, "decompose");
  func(writer, "decompose", [param32(x)], () => {
    line(call("barrett", x));
    for (let i = 0; i < e; i++) {
      line(call("reduceByOne", x));
    }
  });

  addFuncExport(writer, "reduceByOne");
  func(writer, "reduceByOne", [param32(r)], () => {
    line(local64(tmp), local64(carry), local32(l));
    line(local.set(l, i32.add(r, n * 8)));
    // check if r < lambda
    block(writer, () => {
      for (let i = n - 1; i >= 0; i--) {
        lines(
          // if (r[i] < lambda[i]) return
          local.set(tmp, i64.load(r, { offset: 8 * i })),
          br_if(1, i64.lt_u(tmp, LAMBDA[i])),
          // if (r[i] !== lambda[i]) break;
          br_if(0, i64.ne(tmp, LAMBDA[i]))
        );
      }
    });
    // if we're here, r >= lambda so do r -= lambda and also l += 1
    line(local.set(carry, i64.const(0)));
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, r[i]) = r[i] - lambda[i] + carry;
        i64.add(i64.load(r, { offset: 8 * i }), carry),
        i64.const(LAMBDA[i]),
        i64.sub(),
        local.set(tmp),
        i64.store(r, i64.and(tmp, wordMax), { offset: 8 * i }),
        local.set(carry, i64.shr_s(tmp, w))
      );
    }
    line(local.set(carry, i64.const(1)));
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, l[i]) = l[i] + carry;
        i64.add(i64.load(l, { offset: 8 * i }), carry),
        local.set(tmp),
        i64.store(l, i64.and(tmp, wordMax), { offset: 8 * i }),
        local.set(carry, i64.shr_s(tmp, w))
      );
    }
  });

  // convert between internal format and I/O-friendly, packed byte format
  // method: just pack all the n*w bits into memory contiguously
  let nPackedBytes = Math.ceil(lengthP / 8);
  addFuncExport(writer, "toPackedBytes");
  comment(
    `converts ${n}x${w}-bit representation (1 int64 per ${w}-bit limb) to packed ${nPackedBytes}-byte representation`
  );
  func(writer, "toPackedBytes", [param32(bytes), param32(x)], () => {
    let offset = 0; // memory offset
    let nRes = 0; // residual bits to write from last iteration

    line(local64(tmp)); // holds bits that aren't written yet
    // write bytes word by word
    for (let i = 0; i < n; i++) {
      // how many bytes to write in this iteration
      let nBytes = Math.floor((nRes + w) / 8); // max number of bytes we can get from residual + this word
      let bytesMask = (1n << (8n * BigInt(nBytes))) - 1n;
      lines(
        // tmp = tmp | (x[i] >> nr)  where nr is the bit length of tmp (nr < 8)
        i64.shl(i64.load(x, { offset: 8 * i }), nRes),
        local.get(tmp),
        i64.or(),
        local.set(tmp),
        // store bytes at current offset
        i64.store(bytes, i64.and(tmp, bytesMask), { offset }),
        // keep residual bits for next iteration
        local.set(tmp, i64.shr_u(tmp, 8 * nBytes))
      );
      offset += nBytes;
      nRes = nRes + w - 8 * nBytes;
    }
    // final round: write residual bits, if there are any
    if (offset < nPackedBytes) line(i64.store(bytes, tmp, { offset }));
  });

  let chunk = "$chunk";

  addFuncExport(writer, "fromPackedBytes");
  comment(
    `recovers ${n}x${w}-bit representation (1 int64 per ${w}-bit limb) from packed ${nPackedBytes}-byte representation`
  );
  func(writer, "fromPackedBytes", [param32(x), param32(bytes)], () => {
    let offset = 0; // bytes offset
    let nRes = 0; // residual bits read in the last iteration

    line(local64(tmp), local64(chunk));
    lines(local.set(tmp, i64.const(0)));
    // read bytes word by word
    for (let i = 0; i < n; i++) {
      // if we can't fill up w bits with the current residual, load a full i64 from bytes
      // (some of that i64 could be garbage, but we'll only use the parts that aren't)
      if (nRes < w) {
        lines(
          // tmp = (bytes << nRes) | tmp
          i64.shl(
            // load 8 bytes at current offset
            // due to the left shift, we lose nRes of them
            local.tee(chunk, i64.load(bytes, { offset })),
            nRes
          ),
          local.get(tmp),
          i64.or(),
          local.set(tmp),
          // store what fits in next word
          i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
          // keep residual bits for next iteration
          local.set(tmp, i64.shr_u(chunk, w - nRes))
        );
        offset += 8;
        nRes = nRes - w + 64;
      } else {
        // otherwise, the current tmp is just what we want!
        lines(
          i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
          local.set(tmp, i64.shr_u(tmp, w))
        );
        nRes = nRes - w;
      }
    }
  });

  addFuncExport(writer, "fromPackedBytesDouble");
  comment(
    `recovers 2x${n}x${w}-bit representation (1 int64 per ${w}-bit limb) from packed 2x${nPackedBytes}-byte representation of a full scalar`
  );
  func(writer, "fromPackedBytesDouble", [param32(x), param32(bytes)], () => {
    let offset = 0; // bytes offset
    let nRes = 0; // residual bits read in the last iteration

    line(local64(tmp), local64(chunk));
    lines(local.set(tmp, i64.const(0)));
    // read bytes word by word
    for (let i = 0; i < 2 * n; i++) {
      // if we can't fill up w bits with the current residual, load a full i64 from bytes
      // (some of that i64 could be garbage, but we'll only use the parts that aren't)
      if (nRes < w) {
        lines(
          // tmp = (bytes << nRes) | tmp
          i64.shl(
            // load 8 bytes at current offset
            // due to the left shift, we lose nRes of them
            local.tee(chunk, i64.load(bytes, { offset })),
            nRes
          ),
          local.get(tmp),
          i64.or(),
          local.set(tmp),
          // store what fits in next word
          i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
          // keep residual bits for next iteration
          local.set(tmp, i64.shr_u(chunk, w - nRes))
        );
        offset += 8;
        nRes = nRes - w + 64;
      } else {
        // otherwise, the current tmp is just what we want!
        lines(
          i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
          local.set(tmp, i64.shr_u(tmp, w))
        );
        nRes = nRes - w;
      }
    }
  });

  let [startBit, bitLength, endBit, startLimb, endLimb] = [
    "$startBit",
    "$bitLength",
    "$endBit",
    "$startLimb",
    "$endLimb",
  ];
  addFuncExport(writer, "extractBitSlice");
  // implicit assumption: we need at most two limbs to extract a length-c bit slice
  // <==> w+1 >= bitLength = c
  // w+1 is currently 31, and c is about log(N)-1, so this assumption is valid until we do MSMs with > 2^30 inputs
  // we also assume that the startLimb can not be out of bounds
  // this implies that after truncation of the startBit, we have
  // startBit + bitLength <= w-1 + w+1 <= 2w < 64
  func(
    writer,
    "extractBitSlice",
    [param32(x), param32(startBit), param32(bitLength), result32],
    () => {
      line(local32(endBit), local32(startLimb), local32(endLimb));
      lines(
        local.set(endBit, i32.add(startBit, bitLength)),
        local.set(startLimb, i32.div_u(startBit, w)),
        local.set(startBit, i32.sub(startBit, i32.mul(startLimb, w))),
        local.set(endLimb, i32.div_u(endBit, w)),
        local.set(endBit, i32.sub(endBit, i32.mul(endLimb, w))),
        // check for overflow of endLimb
        i32.gt_u(endLimb, n - 1)
      );
      if_(writer, () => {
        // in that case, truncate endBit = w and endLimb = startLimb = n-1
        lines(
          local.set(endBit, i32.const(w)),
          local.set(endLimb, i32.const(n - 1))
        );
      });
      lines(i32.eq(startLimb, endLimb));
      if_(writer, () => {
        lines(
          // load scalar limb
          i64.load(i32.add(x, i32.shl(startLimb, 3))),
          i32.wrap_i64(),
          // take bits < endBit
          i32.sub(i32.shl(1, endBit), 1),
          i32.and(),
          // truncate bits < startBit
          local.get(startBit),
          i32.shr_u(),
          return_()
        );
      });
      // if we're here, endLimb = startLimb + 1 according to our assumptions
      lines(
        // load first limb
        i64.load(i32.add(x, i32.shl(startLimb, 3))),
        i32.wrap_i64(),
        // truncate bits < startBit (and leave on the stack)
        local.get(startBit),
        i32.shr_u(),
        // load second limb,
        i64.load(i32.add(x, i32.shl(i32.add(startLimb, 1), 3))),
        i32.wrap_i64(),
        // take bits < endBit
        i32.sub(i32.shl(1, endBit), 1),
        i32.and(),
        // stitch together with first half, and return
        i32.shl(i32.sub(w, startBit)),
        i32.or()
      );
    }
  );
}

/**
 * alternative addition with a much more efficient overflow check
 * at the cost of n `i64.add`s in first loop
 * -) compute z = x + y - 2p
 * -) z underflows <==> x + y < 2p (this check is just a single i64.eq)
 * -) if z doesn't underflow, return z = x + y - 2p
 * -) if z underflows, compute z + 2p = x + y and return it
 * performance is very similar to `add`
 */
function add2(writer, p, w) {
  let { n, wordMax, R } = montgomeryParams(p, w);
  // constants
  let dP2 = bigintToLegs(mulInputFactor * 2n * p, w, n);
  let { line, lines, comment, join } = writer;
  let { i64, local, local64, param32 } = ops;

  let [x, y, out] = ["$x", "$y", "$out"];
  let [tmp, carry] = ["$tmp", "$carry"];

  function addition({ doReduce }) {
    line(local64(tmp), local64(carry));

    // first loop: x + y
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = x[i] + y[i] - 2p[i] + carry;
        i64.load(x, { offset: 8 * i }),
        i64.load(y, { offset: 8 * i }),
        i64.add(),
        i64.const(dP2[i]),
        i64.sub(),
        local.get(carry),
        i64.add(),
        // split result
        join(local.tee(tmp), i64.const(w), i64.shr_s(), local.set(carry)),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i })
      );
    }
    if (!doReduce) return;
    // check if we underflowed by checking carry === 0 (in that case, we didn't and can return)
    lines(i64.eq(carry, 0), `if return end`);
    // second loop
    // if we're here, x + y < 2p and out = x + y + R - 2p, while we want x + y
    // so do (out - (R - 2p))
    line(local.set(carry, i64.const(0)));
    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      lines(
        // (carry, out[i]) = out[i] - 2p[i] + carry;
        i64.load(out, { offset: 8 * i }),
        i64.const(dP2[i]),
        i64.sub(),
        local.get(carry),
        i64.add(),
        local.set(tmp),
        i64.store(out, i64.and(tmp, wordMax), { offset: 8 * i }),
        local.set(carry, i64.shr_s(tmp, w))
      );
    }
  }

  addFuncExport(writer, "add");
  func(writer, "add", [param32(out), param32(x), param32(y)], () =>
    addition({ doReduce: true })
  );

  addFuncExport(writer, "addNoReduce");
  func(writer, "addNoReduce", [param32(out), param32(x), param32(y)], () =>
    addition({ doReduce: false })
  );
}

/**
 * MOSTLY OBSOLETE
 * montgomery product
 *
 * this is specific to w=32, in that two carry variables are needed
 * to efficiently stay within 64 bits
 *
 * @param {bigint} p modulus
 * @param {number} w word size in bits
 */
function multiply32(writer, p, w, { unrollOuter }) {
  let { n, wn, wordMax } = montgomeryParams(p, w);

  // constants
  let mu = modInverse(-p, 1n << wn);
  let P = bigintToLegs(p, w, n);

  let { line, lines, comment, join } = writer;
  let { i64, i32, local, local32, local64, param32 } = ops;

  let [x, y, xy] = ["$x", "$y", "$xy"];

  addFuncExport(writer, "multiply");
  func(writer, "multiply", [param32(xy), param32(x), param32(y)], () => {
    let [tmp, carry1, carry2, qi] = ["$tmp", "$carry1", "$carry2", "$qi"];
    let [xi, i] = ["$xi", "$i"];

    // tmp locals
    line(local64(tmp), local64(carry1), local64(carry2), local64(qi));
    line(local64(xi), local32(i));
    line();
    // locals for input y and output xy
    let Y = defineLocals(writer, "y", n);
    let T = defineLocals(writer, "t", n);
    // load y
    for (let i = 0; i < n; i++) {
      line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
    }
    line();
    function innerLoop() {
      // j=0 step, where m = m[i] is computed and we neglect t[0]
      comment(`j = 0`);
      comment("(A, tmp) = t[0] + x[i]*y[0]");
      lines(
        local.get(T[0]),
        i64.mul(xi, Y[0]),
        i64.add(),
        local.set(tmp),
        i64.shr_u(tmp, w),
        local.set(carry1),
        i64.and(tmp, wordMax),
        local.set(tmp)
      );
      comment("m = tmp * mu (mod 2^w)");
      lines(
        i64.mul(tmp, mu),
        join(i64.const(wordMax), i64.and()),
        local.set(qi)
      );
      comment("carry = (tmp + m * p[0]) >> w");
      lines(
        local.get(tmp),
        i64.mul(qi, P[0]),
        i64.add(),
        join(i64.const(w), i64.shr_u(), local.set(carry2))
      );
      line();

      for (let j = 1; j < n; j++) {
        comment(`j = ${j}`);
        // NB: this can't overflow 64 bits, because (2^32 - 1)^2 + 2*(2^32 - 1) = 2^64 - 1
        comment("tmp = t[j] + x[i] * y[j] + A");
        lines(
          local.get(T[j]),
          local.get(xi),
          local.get(Y[j]),
          join(i64.mul(), local.get(carry1), i64.add(), i64.add()),
          local.set(tmp)
        );
        comment("A = tmp >> w");
        line(local.set(carry1, i64.shr_u(tmp, w)));
        comment("tmp = (tmp & 0xffffffffn) + m * p[j] + C");
        lines(
          i64.and(tmp, wordMax),
          i64.mul(qi, P[j]),
          join(local.get(carry2), i64.add(), i64.add()),
          local.set(tmp)
        );
        comment("(C, t[j - 1]) = tmp");
        lines(
          local.set(T[j - 1], i64.and(tmp, wordMax)),
          local.set(carry2, i64.shr_u(tmp, w))
        );
        line();
      }
      comment("t[11] = A + C");
      line(local.set(T[n - 1], i64.add(carry1, carry2)));
    }
    if (unrollOuter) {
      for (let i = 0; i < n; i++) {
        comment(`i = ${i}`);
        line(local.set(xi, i64.load(x, { offset: i * 8 })));
        innerLoop();
        line();
      }
    } else {
      forLoop8(writer, i, 0, n, () => {
        line(local.set(xi, i64.load(i32.add(x, i))));
        innerLoop();
      });
    }
    for (let i = 0; i < n; i++) {
      line(i64.store(xy, T[i], { offset: 8 * i }));
    }
  });
}

function benchMultiply(W) {
  let { line } = W;
  let { local, local32, param32, call } = ops;
  let [x, y, N, i] = ["$x", "$y", "$N", "$i"];
  addFuncExport(W, "benchMultiply");
  func(W, "benchMultiply", [param32(x), param32(N)], () => {
    line(local32(i));
    forLoop1(W, i, 0, local.get(N), () => {
      line(call("multiply", local.get(x), local.get(x), local.get(x)));
    });
  });

  addFuncExport(W, "benchSquare");
  func(W, "benchSquare", [param32(x), param32(N)], () => {
    line(local32(i));
    forLoop1(W, i, 0, local.get(N), () => {
      line(call("square", local.get(x), local.get(x)));
    });
  });

  addFuncExport(W, "benchMultiplyUnrolled");
  func(W, "benchMultiplyUnrolled", [param32(x), param32(N)], () => {
    line(local32(i));
    forLoop1(W, i, 0, local.get(N), () => {
      line(call("multiplyUnrolled", local.get(x), local.get(x), local.get(x)));
    });
  });

  addFuncExport(W, "benchMultiplyDifference");
  func(
    W,
    "benchMultiplyDifference",
    [param32(x), param32(y), param32(N)],
    () => {
      line(local32(i));
      forLoop1(W, i, 0, local.get(N), () => {
        line(
          call(
            "multiplyDifference",
            local.get(x),
            local.get(x),
            local.get(x),
            local.get(y)
          )
        );
      });
    }
  );
}
function benchAdd(W) {
  let { line } = W;
  let { local, local32, param32, call } = ops;
  let [x, N, i] = ["$x", "$N", "$i"];
  addFuncExport(W, "benchAdd");
  func(W, "benchAdd", [param32(x), param32(N)], () => {
    line(local32(i));
    forLoop1(W, i, 0, local.get(N), () => {
      line(call("add", local.get(x), local.get(x), local.get(x)));
    });
  });
}
function benchSubtract(W) {
  let { line } = W;
  let { local, local32, param32, call } = ops;
  let [x, N, i, z] = ["$x", "$N", "$i", "$z"];
  addFuncExport(W, "benchSubtract");
  func(W, "benchSubtract", [param32(z), param32(x), param32(N)], () => {
    line(local32(i));
    forLoop1(W, i, 0, local.get(N), () => {
      line(call("subtract", local.get(z), local.get(z), local.get(x)));
    });
  });
}

function moduleWithMemory(writer, comment_, memSize, callback, imports = {}) {
  let { line, comment } = writer;
  comment(comment_);
  module(writer, () => {
    for (let code in imports) {
      let spec = imports[code];
      if (Array.isArray(spec)) {
        for (let s of spec) {
          addCodeImport(writer, code, s);
        }
      } else {
        addCodeImport(writer, code, spec);
      }
    }
    addExport(writer, "memory", ops.memory("memory"));
    line(ops.memory("memory", memSize));
    // global for the initial data offset
    addExport(writer, "dataOffset", ops.global("$dataOffset"));
    callback(writer);
    line(ops.global32("$dataOffset", writer.dataOffset));
  });
}

/**
 *
 * @param {any} writer
 * @param {string} globalName
 * @param {BigUint64Array} data
 */
function dataInt64(writer, globalName, data) {
  let dataOffset = writer.dataOffset;
  writer.dataOffset = dataOffset + data.length * 8;
  let strings = [...data].map((x) => {
    let bytes = [...bigintToBytes(x, 8)];
    return (
      `"` +
      bytes.map((byte) => `\\${byte.toString(16).padStart(2, "0")}`).join("") +
      `"`
    );
  });
  writer.lines(
    ops.global32(globalName, dataOffset),
    `(data ${ops.i32.const(dataOffset)}`,
    ...strings.map((s) => "  " + s),
    ")"
  );
}

function defineLocals(t, name, n) {
  let locals = [];
  for (let i = 0; i < n; ) {
    for (let j = 0; j < 4 && i < n; j++, i++) {
      let x = "$" + name + String(i).padStart(2, "0");
      t.write(ops.local64(x) + " ");
      locals.push(x);
    }
    t.line();
  }
  return locals;
}

/**
 * Compute the montgomery radix R=2^K and number of legs n
 * @param {bigint} p modulus
 * @param {number} w word size in bits
 */
function montgomeryParams(p, w) {
  // word size has to be <= 32, to be able to multiply 2 words as i64
  if (w > 32) {
    throw Error("word size has to be <= 32 for efficient multiplication");
  }
  // montgomery radix R should be R = 2^K > 2p,
  // where K is exactly divisible by the word size w
  // i.e., K = n*w, where n is the number of legs our field elements are stored in
  let lengthP = log2(p);
  let minK = lengthP + 1; // want 2^K > 2p bc montgomery mult. is modulo 2p
  // number of legs is smallest n such that K := n*w >= minK
  let n = Math.ceil(minK / w);
  let K = n * w;
  let R = 1n << BigInt(K);
  let wn = BigInt(w);
  return { n, K, R, wn, wordMax: (1n << wn) - 1n, lengthP };
}

/**
 *
 * @param {bigint} p modulus
 * @param {number} w word size
 * @param {import('./wasm/finite-field.wasm')} wasm
 */
function jsHelpers(
  p,
  w,
  { memory, toPackedBytes, fromPackedBytes, dataOffset }
) {
  let { n, wn, wordMax, R, lengthP } = montgomeryParams(p, w);
  let nPackedBytes = Math.ceil(lengthP / 8);
  let memoryBytes = new Uint8Array(memory.buffer);
  let initialOffset = dataOffset.valueOf();
  let obj = {
    n,
    R,
    bitLength: lengthP,
    fieldSizeBytes: 8 * n,
    packedSizeBytes: nPackedBytes,

    /**
     * @param {number} x
     * @param {bigint} x0
     */
    writeBigint(x, x0) {
      let arr = new BigUint64Array(memory.buffer, x, n);
      for (let i = 0; i < n; i++) {
        arr[i] = x0 & wordMax;
        x0 >>= wn;
      }
    },

    /**
     * @param {number} x
     */
    readBigInt(x, length = 1) {
      let arr = new BigUint64Array(memory.buffer.slice(x, x + n * 8 * length));
      let x0 = 0n;
      let bitPosition = 0n;
      for (let i = 0; i < arr.length; i++) {
        x0 += arr[i] << bitPosition;
        bitPosition += wn;
      }
      return x0;
    },

    /**
     * @type {number}
     */
    initial: initialOffset,
    /**
     * @type {number}
     */
    offset: initialOffset,

    /**
     * @param {number} size size of pointer (default: one field element)
     */
    getPointer(size = n * 8) {
      let pointer = obj.offset;
      obj.offset += size;
      return pointer;
    },

    /**
     * @param {number} N
     * @param {number} size size per pointer (default: one field element)
     */
    getPointers(N, size = n * 8) {
      /**
       * @type {number[]}
       */
      let pointers = Array(N);
      let offset = obj.offset;
      for (let i = 0; i < N; i++) {
        pointers[i] = offset;
        offset += size;
      }
      obj.offset = offset;
      return pointers;
    },

    /**
     * @param {number} N
     */
    getStablePointers(N) {
      let pointers = obj.getPointers(N);
      obj.initial = obj.offset;
      return pointers;
    },

    /**
     * @param {number} size size of pointer (default: one field element)
     */
    getZeroPointer(size = n * 8) {
      let offset = obj.offset;
      let pointer = obj.offset;
      memoryBytes.fill(0, offset, offset + size);
      obj.offset = offset + size;
      return pointer;
    },

    /**
     * @param {number} N
     * @param {number} size size per pointer (default: one field element)
     */
    getZeroPointers(N, size = n * 8) {
      /**
       * @type {number[]}
       */
      let pointers = Array(N);
      let offset = obj.offset;
      new Uint8Array(memory.buffer, offset, N * size).fill(0);
      for (let i = 0; i < N; i++) {
        pointers[i] = offset;
        offset += size;
      }
      obj.offset = offset;
      return pointers;
    },

    /**
     * store pointers to memory in memory themselves
     *
     * @param {number} N
     * @param {number} size size per pointer (default: one field element)
     * @returns {[Uint32Array, number]}
     */
    getPointersInMemory(N, size = n * 8) {
      let offset = obj.offset;
      // memory addresses must be multiples of 8 for BigInt64Arrays
      let length = ((N + 1) >> 1) << 1;
      let pointerPtr = offset;
      let pointers = new Uint32Array(memory.buffer, pointerPtr, length);
      offset += length * 4;
      for (let i = 0; i < N; i++) {
        pointers[i] = offset;
        offset += size;
      }
      obj.offset = offset;
      return [pointers, pointerPtr];
    },

    /**
     *
     * @param {number} N
     * @returns {[Uint32Array, number]}
     */
    getEmptyPointersInMemory(N) {
      let offset = obj.offset;
      // memory addresses must be multiples of 8 for BigInt64Arrays
      let length = ((N + 1) >> 1) << 1;
      let pointerPtr = offset;
      let pointers = new Uint32Array(memory.buffer, pointerPtr, length);
      obj.offset += length * 4;
      return [pointers, pointerPtr];
    },

    resetPointers() {
      obj.offset = obj.initial;
    },

    getOffset() {
      return obj.offset;
    },

    /**
     * @param {number[]} scratch
     * @param {number} pointer
     * @param {Uint8Array} bytes
     */
    writeBytes([bytesPtr], pointer, bytes) {
      let arr = new Uint8Array(memory.buffer, bytesPtr, 8 * n);
      arr.fill(0);
      arr.set(bytes);
      fromPackedBytes(pointer, bytesPtr);
    },
    /**
     * read field element into packed bytes representation
     *
     * @param {number[]} scratch
     * @param {number} pointer
     */
    readBytes([bytesPtr], pointer) {
      toPackedBytes(bytesPtr, pointer);
      return new Uint8Array(
        memory.buffer.slice(bytesPtr, bytesPtr + nPackedBytes)
      );
    },
  };
  return obj;
}
