import { montgomeryParams } from "./finite-field-generate.js";
import { modInverse } from "./finite-field-js.js";
import { bigintFromLegs, bigintToLegs } from "./util.js";
import {
  addExport,
  addFuncExport,
  forLoop1,
  forLoop8,
  func,
  ops,
} from "./lib/wasm-generate.js";

export { multiply, barrett, karatsuba30 };

/**
 * montgomery product
 *
 * ideas of the algorithm:
 *
 * - we compute x*y*2^(-n*w) mod p
 * - x, y and p are represented as arrays of size n, with w-bit legs/digits, each stored as int64
 * - in math, x = Sum_i=0...(n-1)( x_i*2^(i*w) ), where x_i \in [0,2^w)
 * - w <= 32 so that we can just multiply two elements x_i*y_j as int64
 * - important: be flexible w.r.t. w; the literature says w=32, but that's not ideal here
 *
 * to compute x*y*2^(-n*w) mod p, we expand x = Sum_i( x_i*2^(i*w) ), so we get
 *   S := x*y*2^(-n*w) = Sum_i( x_i*y*2^(i*w) ) * 2^(-n*w) =
 *      = Sum_i( x_i*y*2^(-(n-i)*w) ) mod p
 * this sum mod p can be computed iteratively:
 * - initialize S = 0
 * - for i=0,...,n-1 : S = (S + x_i*y) * 2^(-w) mod p
 * - note: earlier terms in the sum get multiplied by more 2^(-w) factors!
 * in each step, compute (S + x_i*y) * 2^(-w) mod p by doing the montgomery reduction trick:
 * add a multiple of p which makes the result divisible by 2^w, so *2^(-w) becomes a normal division!
 *
 * so in each step we want (S + x_i*y + q_i*p) * 2^(-w), where q_i is such that S + x_i*y + q_i*p = 0 mod 2^w
 * that's true if q_i = (-p)^(-1) * (S + x_i*y) mod 2^w
 * since the equality is mod 2^w, we can take all the parts mod 2^w -- which means taking the lowest word of S and y!
 * ==> q_i = mu * (S_0 + x_i*y_0) mod 2^w, where mu = (-p)^(-1) mod 2^w is precomputed, and is a single word
 * (of this expression, S_0 + x_i*y_0 needs to be computed anyway as part of S + x_i*y + q_i*p)
 *
 * in detail, S + x_i*y + q_i*p is computed by computing up terms like
 *   S_j + x_i*y_j + q_i*p_j
 * and, when needed, carrying over to the next term. which means the term we compute are more like
 *   carry_(j-1) + S_j + x_i*y_j + q_i*p_j
 *
 * multiplying by 2^(-w) just means shifting the array S_0,...,S_(n-1) by one term, so that e.g. (S_1 + ...) becomes S_0
 * so we get something like
 *   (carry_0, _) = S_0 + x_i*y_0 + q_i*p_0
 *   (carry_j, S_(j-1)) = carry_(j-1) + S_j + x_i*y_j + q_i*p_j    for j=1,...,(n-1)
 *   S_(n-1) = carry_(n-1)
 *
 * this is the gist, but in fact the number of carry operations depends on the bit length w
 * the w=32 case needs more carry operations than shown above, since x_i*y_j + q_i*p_j would have 65 bits already
 * on the other hand, w<32 doesn't need a carry in every j step
 * so, by making w < 32, we get more S_j + x_i*y_j + q_i*p_j terms, but (much) less carries
 */
function multiply(writer, p, w, { countMultiplications = false } = {}) {
  let { n, wn, wordMax } = montgomeryParams(p, w);
  // constants
  let mu = modInverse(-p, 1n << wn);
  let P = bigintToLegs(p, w, n);
  let P4 = bigintToLegs(4n * p, w, n);
  // how much terms we can add before a carry
  let nSafeTerms = 2 ** (64 - 2 * w);
  // how much j steps we can do before a carry:
  let nSafeSteps = 2 ** (64 - 2 * w - 1);
  let nSafeStepsSquare = Math.floor(2 ** (64 - 2 * w) / 3); // three terms per step
  // strategy is to use a carry at j=0, plus whenever we reach nSafeSteps
  // (and finally at the end)
  // how many carry variables we need
  let nCarry = 1 + Math.floor(n / nSafeSteps);

  let { line, lines, comment, join } = writer;
  let { i64, i32, local, local32, local64, param32, global32Mut, global } = ops;

  // count multiplications to analyze higher-level algorithms
  let multiplyCount = "$multiplyCount";
  if (countMultiplications) {
    addExport(writer, "multiplyCount", global(multiplyCount));
    addFuncExport(writer, "resetMultiplyCount");
    line(global32Mut(multiplyCount, 0));
    func(writer, "resetMultiplyCount", [], () => {
      line(global.set(multiplyCount, i32.const(0)));
    });
  }

  let [x, y, xy, z] = ["$x", "$y", "$xy", "$z"];
  let [tmp] = ["$tmp"];
  let [i, xi, qi] = ["$i", "$xi", "$qi"];

  addFuncExport(writer, "multiply");
  func(writer, "multiply", [param32(xy), param32(x), param32(y)], () => {
    // locals
    line(local64(tmp));
    line(local64(qi), local64(xi), local32(i));
    let Y = defineLocals(writer, "y", n);
    let S = defineLocals(writer, "t", n);

    if (countMultiplications) {
      line(global.set(multiplyCount, i32.add(global.get(multiplyCount), 1)));
    }

    // load y
    for (let i = 0; i < n; i++) {
      line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
    }

    forLoop8(writer, i, 0, n, () => {
      // load x[i]
      line(local.set(xi, i64.load(i32.add(x, i))));

      // j=0, compute q_i
      let didCarry = false;
      let doCarry = 0 % nSafeSteps === 0;
      comment("j = 0, do carry, ignore result below carry");
      lines(
        // tmp = S[0] + x[i]*y[0]
        local.get(S[0]),
        i64.mul(xi, Y[0]),
        i64.add(),
        // qi = mu * (tmp & wordMax) & wordMax
        local.set(tmp),
        local.set(qi, i64.and(i64.mul(mu, i64.and(tmp, wordMax)), wordMax)),
        local.get(tmp),
        // (stack, _) = tmp + qi*p[0]
        i64.mul(qi, P[0]),
        i64.add(),
        join(i64.const(w), i64.shr_u()) // we just put carry on the stack, use it later
      );

      for (let j = 1; j < n - 1; j++) {
        // S[j] + x[i]*y[j] + qi*p[j], or
        // stack + S[j] + x[i]*y[j] + qi*p[j]
        // ... = S[j-1], or  = (stack, S[j-1])
        didCarry = doCarry;
        doCarry = j % nSafeSteps === 0;
        comment(`j = ${j}${doCarry ? ", do carry" : ""}`);
        lines(
          local.get(S[j]),
          didCarry && i64.add(), // add carry from stack
          i64.mul(xi, Y[j]),
          i64.add(),
          i64.mul(qi, P[j]),
          i64.add(),
          doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
          doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
          local.set(S[j - 1])
        );
      }
      let j = n - 1;
      didCarry = doCarry;
      doCarry = j % nSafeSteps === 0;
      comment(`j = ${j}${doCarry ? ", do carry" : ""}`);
      if (doCarry) {
        lines(
          local.get(S[j]),
          didCarry && i64.add(), // add carry from stack
          i64.mul(xi, Y[j]),
          i64.add(),
          i64.mul(qi, P[j]),
          i64.add(),
          doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
          doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
          local.set(S[j - 1])
        );
        // if the last iteration does a carry, S[n-1] is set to it
        lines(local.set(S[j]));
      } else {
        // if the last iteration doesn't do a carry, then S[n-1] is never set,
        // so we also don't have to get it & can save 1 addition
        lines(
          i64.mul(xi, Y[j]),
          didCarry && i64.add(), // add carry from stack
          i64.mul(qi, P[j]),
          i64.add(),
          local.set(S[j - 1])
        );
      }
    });
    // outside i loop: final pass of collecting carries
    comment("final carrying & storing");
    for (let j = 1; j < n; j++) {
      lines(
        i64.store(xy, i64.and(S[j - 1], wordMax), { offset: 8 * (j - 1) }),
        local.set(S[j], i64.add(S[j], i64.shr_u(S[j - 1], w)))
      );
    }
    line(i64.store(xy, S[n - 1], { offset: 8 * (n - 1) }));
  });

  const mulInputFactor = 8n;
  let d2P = bigintToLegs(mulInputFactor * 2n * p, w, n);

  addFuncExport(writer, "multiplyDifference");
  func(
    writer,
    "multiplyDifference",
    [param32(xy), param32(x), param32(y), param32(z)],
    () => {
      // locals
      line(local64(tmp));
      line(local64(qi), local64(xi), local32(i));
      let Y = defineLocals(writer, "y", n);
      let S = defineLocals(writer, "t", n);

      if (countMultiplications) {
        line(global.set(multiplyCount, i32.add(global.get(multiplyCount), 1)));
      }

      // compute y-z
      // we can avoid carries here by making the terms w+1+eps bits
      // => implies x*y terms are 2w+1+eps bits
      // => similar case as for squaring, number of safe terms gets slightly smaller
      // (but here, we had to move carries to the term level to prove that this won't overflow)
      // for w=30, we get nSafeTerms=9, which adds 1 carry into the loop below
      // empirically it might be of similar speed or slightly faster if the carry is in the loop
      let nTerms;
      for (nTerms = 1; nTerms < nSafeTerms; nTerms++) {
        // compute ~tight upper bound on sum of n steps
        let upperBound = 0n;
        for (let i = 0; i <= 2 * n - nTerms; i++) {
          // compute upper bound for nTerms starting at i
          let bound = 0n;
          for (let j = i; j < i + nTerms; j++) {
            if (j % 2 === 0) {
              let y =
                d2P[j >> 1] +
                (j >> 1 < n - 1 ? wordMax + 1n : 0n) +
                (j >> 1 === 0 ? 0n : -1n) +
                wordMax;
              bound += wordMax * y;
            } else {
              bound += P[j >> 1] * wordMax;
            }
          }
          if (bound > upperBound) upperBound = bound;
        }
        if (upperBound >= 2n ** 64n) break;
      }
      let nSafeTermsSpecial = nTerms - 1;
      for (let i = 0; i < n; i++) {
        lines(
          i64.const(
            d2P[i] + (i < n - 1 ? wordMax + 1n : 0n) + (i === 0 ? 0n : -1n)
          ),
          // i > 0 && i64.add(),
          i64.load(y, { offset: 8 * i }),
          i64.add(),
          i64.load(z, { offset: 8 * i }),
          i64.sub(),
          // i64.and(local.tee(tmp), wordMax),
          local.set(Y[i])
          // i < n - 1 && i64.shr_s(tmp, w)
        );
      }

      forLoop8(writer, i, 0, n, () => {
        // load x[i]
        line(local.set(xi, i64.load(i32.add(x, i))));

        // j=0, compute q_i
        let doCarry = true;
        comment("j = 0, do carry, ignore result below carry");
        lines(
          // tmp = S[0] + x[i]*y[0]
          local.get(S[0]),
          i64.mul(xi, Y[0]),
          i64.add(),
          // qi = mu * (tmp & wordMax) & wordMax
          local.set(tmp),
          local.set(qi, i64.and(i64.mul(mu, i64.and(tmp, wordMax)), wordMax)),
          local.get(tmp),
          // (stack, _) = tmp + qi*p[0]
          i64.mul(qi, P[0]),
          i64.add(),
          join(i64.const(w), i64.shr_u()) // we just put carry on the stack, use it later
        );
        let didCarry = true;

        for (let j = 1; j < n - 1; j++) {
          // S[j] + x[i]*y[j] + qi*p[j], or
          // stack + S[j] + x[i]*y[j] + qi*p[j]
          // ... = S[j-1], or  = (stack, S[j-1])
          doCarry = (2 * j - 1) % nSafeTermsSpecial === 0;
          comment(`j = ${j}`);
          lines(
            local.get(S[j]),
            didCarry && i64.add(),
            i64.mul(xi, Y[j]),
            i64.add(),
            doCarry && `;; carry after ${nSafeTermsSpecial} terms`,
            doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()),
            doCarry && i64.and(tmp, wordMax)
          );
          didCarry = doCarry;
          doCarry = (2 * j) % nSafeTermsSpecial === 0;
          lines(
            i64.mul(qi, P[j]),
            i64.add(),
            doCarry && `;; carry after ${nSafeTermsSpecial} terms`,
            doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
            doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
            local.set(S[j - 1])
          );
          didCarry = doCarry || didCarry;
        }
        let j = n - 1;
        doCarry = (2 * j - 1) % nSafeTermsSpecial === 0;
        comment(`j = ${j}`);
        if (doCarry) {
          lines(
            local.get(S[j]),
            didCarry && i64.add(), // add carry from stack
            i64.mul(xi, Y[j]),
            i64.add(),
            doCarry && `;; carry after ${nSafeTermsSpecial} terms`,
            doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()),
            doCarry && i64.and(tmp, wordMax)
          );
          lines(
            i64.mul(qi, P[j]),
            i64.add(),
            doCarry && `;; carry after ${nSafeTermsSpecial} terms`,
            doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
            doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
            local.set(S[j - 1])
          );
          // if the last iteration does a carry, S[n-1] is set to it
          lines(local.set(S[j]));
        } else {
          // if the last iteration doesn't do a carry, then S[n-1] is never set,
          // so we also don't have to get it & can save 1 addition
          lines(
            i64.mul(xi, Y[j]),
            didCarry && i64.add(), // add carry from stack
            i64.mul(qi, P[j]),
            i64.add(),
            local.set(S[j - 1])
          );
        }
      });
      // outside i loop: final pass of collecting carries
      comment("final carrying & storing");
      for (let j = 1; j < n; j++) {
        lines(
          i64.store(xy, i64.and(S[j - 1], wordMax), { offset: 8 * (j - 1) }),
          local.set(S[j], i64.add(S[j], i64.shr_u(S[j - 1], w)))
        );
      }
      line(i64.store(xy, S[n - 1], { offset: 8 * (n - 1) }));
    }
  );

  let carry = "$carry";

  addFuncExport(writer, "multiplyUnrolled");
  func(
    writer,
    "multiplyUnrolled",
    [param32(xy), param32(x), param32(y)],
    () => {
      // locals
      line(local64(tmp), local64(carry));
      let X = defineLocals(writer, "x", n);
      let Y = defineLocals(writer, "y", n);
      let Q = defineLocals(writer, "q", n);

      if (countMultiplications) {
        line(global.set(multiplyCount, i32.add(global.get(multiplyCount), 1)));
      }

      // load x, y
      for (let i = 0; i < n; i++) {
        line(local.set(X[i], i64.load(local.get(x), { offset: i * 8 })));
      }
      for (let i = 0; i < n; i++) {
        line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
      }
      // for (let j = Math.max(0, i - n + 1); j < Math.min(i + 1, n); j++) {
      for (let i = 0; i < 2 * n - 1; i++) {
        comment(`i = ${i}`);
        let didCarry = false;
        let doCarry = false;
        let j0 = Math.max(0, i - n + 1);
        let jend = Math.min(i + 1, n);
        for (let j = j0, terms = 0; j < jend; j++) {
          comment(`> j = ${j}`);
          lines(
            //
            i64.mul(X[j], Y[i - j]),
            i > 0 && i64.add()
          );
          doCarry = ++terms % nSafeTerms === 0;
          if (doCarry) {
            comment(`> carry after term # ${terms}`);
            lines(
              local.set(tmp),
              i64.shr_u(tmp, w),
              didCarry && local.get(carry),
              didCarry && i64.add(),
              local.set(carry), // save carry for next i
              i64.and(tmp, wordMax) // mod 2^w the current result
            );
            didCarry = true;
          }
          if ((i < n && j < i) || (i >= n && j < n - 1)) {
            lines(
              //
              i64.mul(Q[j], P[i - j]),
              i64.add()
            );
            doCarry = ++terms % nSafeTerms === 0;
            if (doCarry) {
              comment(`> carry after term # ${terms}`);
              lines(
                local.set(tmp),
                i64.shr_u(tmp, w),
                didCarry && local.get(carry),
                didCarry && i64.add(),
                local.set(carry), // save carry for next i
                i64.and(tmp, wordMax) // mod 2^w the current result
              );
              didCarry = true;
            }
          }
          if (i < n && j === i) {
            lines(
              local.set(tmp),
              local.set(
                Q[i],
                // TODO: don't need to AND here if we carried just before
                i64.and(i64.mul(mu, i64.and(tmp, wordMax)), wordMax)
              ),
              local.get(tmp),
              i64.mul(Q[j], P[i - j]),
              i64.add(), // the low part of this is zero; the high part is put as carry on the stack
              join(i64.const(w), i64.shr_u()),
              didCarry && local.get(carry), // if we have another carry, add that as well
              didCarry && i64.add()
            );
          }
          if (i >= n && j === n - 1) {
            lines(
              //
              i64.mul(Q[j], P[i - j]),
              i64.add()
            );
            lines(
              local.set(tmp),
              i64.store(xy, i64.and(tmp, wordMax), { offset: 8 * (i - n) }),
              i64.shr_u(tmp, w),
              didCarry && local.get(carry),
              didCarry && i64.add()
            );
          }
        }
      }
      lines(
        local.set(tmp),
        i64.store(xy, i64.and(tmp, wordMax), { offset: 8 * (n - 1) })
      );
    }
  );

  let [out] = ["$out"];
  function square() {
    // locals
    line(local64(tmp));
    line(local64(qi));
    let X = defineLocals(writer, "x", n);
    let S = defineLocals(writer, "t", n);

    if (countMultiplications) {
      line(global.set(multiplyCount, i32.add(global.get(multiplyCount), 1)));
    }

    // load x
    for (let i = 0; i < n; i++) {
      line(local.set(X[i], i64.load(local.get(x), { offset: i * 8 })));
    }

    for (let i = 0; i < n; i++) {
      comment(`i = ${i}`);
      // j=0, compute q_i
      let j = 0;
      let didCarry = false;
      let doCarry = 0 % nSafeStepsSquare === 0;
      comment("j = 0, do carry, ignore result below carry");
      lines(
        // tmp = S[i] + 2*x[0]*x[i] + 4p[i] + y'[i] + z'[i]
        ...(i === 0
          ? [i64.mul(X[0], X[0])]
          : [local.get(S[0]), i64.shl(i64.mul(X[i], X[0]), 1), i64.add()]),
        // qi = mu * (tmp & wordMax) & wordMax
        local.set(tmp),
        local.set(qi, i64.and(i64.mul(mu, i64.and(tmp, wordMax)), wordMax)),
        local.get(tmp),
        // (stack, _) = tmp + qi*p[0]
        i64.mul(qi, P[0]),
        i64.add(),
        join(i64.const(w), i64.shr_u()) // we just put carry on the stack, use it later
      );

      for (let j = 1; j < n - 1; j++) {
        // S[j] + 2*x[i]*x[j] + qi*p[j], or
        // stack + S[j] + 2*x[i]*x[j] + qi*p[j]
        // ... = S[j-1], or  = (stack, S[j-1])
        didCarry = doCarry;
        doCarry = j % nSafeStepsSquare === 0;
        comment(`j = ${j}${doCarry ? ", do carry" : ""}`);
        lines(
          local.get(S[j]),
          didCarry && i64.add(), // add carry from stack
          // TODO these savings seem to make no difference in benchmarks
          // i > 0 && local.get(S[j]),
          // i > 0 && didCarry && i64.add(), // add carry from stack
          j <= i && i64.mul(X[i], X[j]),
          j < i && join(i64.const(1), i64.shl()),
          j <= i && i64.add(),
          i64.mul(qi, P[j]),
          i64.add(),
          // (i > 0 || didCarry) && i64.add(),
          doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
          doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
          local.set(S[j - 1])
        );
      }
      j = n - 1;
      didCarry = doCarry;
      doCarry = j % nSafeSteps === 0;
      comment(`j = ${j}${doCarry ? ", do carry" : ""}`);
      if (doCarry) {
        lines(
          local.get(S[j]),
          didCarry && i64.add(), // add carry from stack
          i === j && i64.mul(X[i], X[j]),
          i === j && i64.add(),
          i64.mul(qi, P[j]),
          i64.add(),
          doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
          doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
          local.set(S[j - 1]),
          // if the last iteration does a carry, S[n-1] is set to it
          local.set(S[j])
        );
      } else {
        // if the last iteration doesn't do a carry, then S[n-1] is never set,
        // so we also don't have to get it & can save 1 addition
        lines(
          i === j && i64.mul(X[i], X[j]),
          i64.mul(qi, P[j]),
          didCarry && i64.add(), // add carry from stack
          i === j && i64.add(),
          local.set(S[j - 1])
        );
      }
    }
    // outside i loop: final pass of collecting carries
    comment("final carrying & storing");
    for (let j = 1; j < n; j++) {
      lines(
        i64.store(out, i64.and(S[j - 1], wordMax), { offset: 8 * (j - 1) }),
        local.set(S[j], i64.add(S[j], i64.shr_u(S[j - 1], w)))
      );
    }
    line(i64.store(out, S[n - 1], { offset: 8 * (n - 1) }));
  }

  addFuncExport(writer, "square");
  func(writer, "square", [param32(out), param32(x)], () => square(false));

  let [k] = ["$k"];

  // multiplication by 2^k < 2p
  // TODO: this could be at least 50% faster, but probably not worth it
  // (all the multiplications by 0 and corresponding adds / carries can be saved,
  // the if loop should only go to (w*n-k) // n, and just do one final round
  // of flexible reduction by 2^(w*n-k % n))
  addFuncExport(writer, "leftShift");
  func(writer, "leftShift", [param32(xy), param32(y), param32(k)], () => {
    let [xi0, i0] = ["$xi0", "$i0"];

    // locals
    line(local64(tmp));
    line(local64(qi), local64(xi), local32(i), local32(i0), local32(xi0));
    let Y = defineLocals(writer, "y", n);
    let S = defineLocals(writer, "t", n);

    // load y
    for (let i = 0; i < n; i++) {
      line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
    }
    // figure out the value of i0, xi0 where 2^k has its bit set
    lines(
      // i0 = k // w, xi0 = 2^(k % w) = 2^(k - i0*w)
      local.set(i0, i32.div_u(k, w)),
      local.set(xi0, i32.shl(1, i32.rem_u(k, w))),
      // local.set(xi0, i32.shl(1, i32.sub(k, i32.mul(i0, w)))),
      local.set(i0, i32.mul(i0, 8))
    );

    forLoop8(writer, i, 0, n, () => {
      // compute x[i]
      line(local.set(xi, i64.extend_i32_u(i32.mul(i32.eq(i, i0), xi0))));

      // j=0, compute q_i
      let didCarry = false;
      let doCarry = 0 % nSafeSteps === 0;
      comment("j = 0, do carry, ignore result below carry");
      lines(
        // tmp = S[0] + x[i]*y[0]
        local.get(S[0]),
        i64.mul(xi, Y[0]),
        i64.add(),
        // qi = mu * (tmp & wordMax) & wordMax
        local.set(tmp),
        local.set(qi, i64.and(i64.mul(mu, i64.and(tmp, wordMax)), wordMax)),
        local.get(tmp),
        // (stack, _) = tmp + qi*p[0]
        i64.mul(qi, P[0]),
        i64.add(),
        join(i64.const(w), i64.shr_u()) // we just put carry on the stack, use it later
      );

      for (let j = 1; j < n - 1; j++) {
        // S[j] + x[i]*y[j] + qi*p[j], or
        // stack + S[j] + x[i]*y[j] + qi*p[j]
        // ... = S[j-1], or  = (stack, S[j-1])
        didCarry = doCarry;
        doCarry = j % nSafeSteps === 0;
        comment(`j = ${j}${doCarry ? ", do carry" : ""}`);
        lines(
          local.get(S[j]),
          didCarry && i64.add(), // add carry from stack
          i64.mul(xi, Y[j]),
          i64.add(),
          i64.mul(qi, P[j]),
          i64.add(),
          doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
          doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
          local.set(S[j - 1])
        );
      }
      let j = n - 1;
      didCarry = doCarry;
      doCarry = j % nSafeSteps === 0;
      comment(`j = ${j}${doCarry ? ", do carry" : ""}`);
      if (doCarry) {
        lines(
          local.get(S[j]),
          didCarry && i64.add(), // add carry from stack
          i64.mul(xi, Y[j]),
          i64.add(),
          i64.mul(qi, P[j]),
          i64.add(),
          doCarry && join(local.tee(tmp), i64.const(w), i64.shr_u()), // put carry on the stack
          doCarry && i64.and(tmp, wordMax), // mod 2^w the current result
          local.set(S[j - 1])
        );
        // if the last iteration does a carry, S[n-1] is set to it
        lines(local.set(S[j]));
      } else {
        // if the last iteration doesn't do a carry, then S[n-1] is never set,
        // so we also don't have to get it & can save 1 addition
        lines(
          i64.mul(xi, Y[j]),
          didCarry && i64.add(), // add carry from stack
          i64.mul(qi, P[j]),
          i64.add(),
          local.set(S[j - 1])
        );
      }
    });
    // outside i loop: final pass of collecting carries
    comment("final carrying & storing");
    for (let j = 1; j < n; j++) {
      lines(
        i64.store(xy, i64.and(S[j - 1], wordMax), { offset: 8 * (j - 1) }),
        local.set(S[j], i64.add(S[j], i64.shr_u(S[j - 1], w)))
      );
    }
    line(i64.store(xy, S[n - 1], { offset: 8 * (n - 1) }));
  });
}

/**
 * karatsuba method
 *
 * l := 2^w = limb size; multiplication by l is shift by 1 limb
 * L = number of limbs in lower half, m < n, should be about n/2
 *
 * x = x0 + l^m*x1, where x0 has m limbs and x1 has n - m
 *
 * x*y = (x0 + l^m*x1)(y0 + l^m*y1) =
 *     = (l^m - 1)(l^m*x1*y1 - x0*y0) + l^m(x0 + x1)(y0 + y1)
 *
 * let's assume n-m >= m (on the upper end, we have more free bits bc p << 2^N)
 * => x0 + x1 has length n-m as well, with +1 bits in the limbs
 *
 *
 * -) compute z = x0*y0, w = x1*y1
 *       => 1 m*m MUL, 1 (n-m)*(n-m) MUL
 * -) compute z = l^m*w - z
 *       => active bits: [m..(2(n-m)+m)] vs [0..2m] => overlap at [m..2m]
 *       => 1 length-m ADD
 *       => z has active bits [0..(2n-m)]
 * -) compute z = l^m*z - z
 *       => active bits: [m..2n] vs [0..(2n-m)] => overlap at [m..(2n-m)]
 *       => 1 length-2(n-m) ADD
 *       => result has active bits [0..2n]
 * -) compute x0 = x0 + x1, y0 = y0 + y1 => 2 length-(n-m) ADD
 * -) compute w = (x0 + x1)*(y0 + y1)
 *       => 1 (n-m)*(n-m) MUL
 * -) compute x*y = z = z + l^m*w
 *       => active bits [0..2n] vs [m..(2n-m)]
 *       => 1 length-2(n-m) ADD
 * additions have combined length (n-m) + (n-m) + m + 2(n-m) + 2(n-m) = 6(n-m) + m = about 3.5n
 *
 * 3.5n ADD, m^2 + 2(n-m)^2 ~ 3*(n/2)^2 MUL (where a MUL contains the addition to sum products)
 *
 * @param {any} writer
 * @param {bigint} p
 * @param {number} w
 * @param {{withBenchmarks?: boolean}} options
 */
function karatsuba30(writer, p, w, { withBenchmarks = false }) {
  if (w !== 30) throw Error("karatsuba mul only designed for w=30");
  let { n, lengthP: b } = montgomeryParams(p, w);
  let k = b - 1;
  let N = n * w;
  let wn = BigInt(w);
  let wordMax = (1n << wn) - 1n;

  let { n0, e0 } = findMsbCutoff(p, w);
  let m_ = 2n ** BigInt(k + N) / p;
  let M = bigintToLegs(m_, w, n);
  let P = bigintToLegs(p, w, n);

  // number of limbs in lower half
  let m = n >> 1;
  console.assert(m <= n - m, "m <= n-m");

  let { line, lines, comment, join } = writer;
  let { i64, local, local64, param32, local32, call, param64, result64 } = ops;

  let [x, y, xy] = ["$x", "$y", "$xy"];
  let [tmp, carry] = ["$tmp", "$carry"];
  let [xi] = ["$xi"];
  let [$i, $N] = ["$i", "$N"];
  /* 
  function multiplySchoolbookSmall(n) {
    addFuncExport(writer, `multiplySchoolbook${n}`);
    func(
      writer,
      `multiplySchoolbook${n}`,
      [param32(xy), param32(x), param32(y)],
      () => {
        let X = defineLocals(writer, "x", n);
        let Y = defineLocals(writer, "y", n);
        for (let i = 0; i < n; i++) {
          line(local.set(X[i], i64.load(local.get(x), { offset: i * 8 })));
        }
        for (let i = 0; i < n; i++) {
          line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
        }
        for (let i = 0; i < 2 * n; i++) {
          comment(`k = ${i}`);
          let j0 = Math.max(0, i - n + 1);
          for (let j = j0; j < Math.min(i + 1, n); j++) {
            lines(
              //
              i64.mul(X[j], Y[i - j]),
              i > 0 && j > j0 && i64.add()
            );
          }
          let isLast = i === 2 * n - 1;
          !isLast && line(i64.store(xy, tmp, { offset: 8 * i }));
        }
      }
    );
  }
  function multiplySchoolbookSmallLocals(n) {
    let X = Array(n)
      .fill(0)
      .map((_, i) => "$x" + i);
    let Y = Array(n)
      .fill(0)
      .map((_, i) => "$y" + i);
    addFuncExport(writer, `multiplySchoolbook${n}`);
    func(
      writer,
      `multiplySchoolbook${n}`,
      [
        ...X.map((x) => param64(x)),
        ...Y.map((x) => param64(x)),
        ...Array(2 * n)
          .fill(0)
          .map(() => result64),
      ],
      () => {
        let XY = defineLocals(writer, "xy", 2 * n);
        for (let i = 0; i < 2 * n; i++) {
          comment(`k = ${i}`);
          let j0 = Math.max(0, i - n + 1);
          for (let j = j0; j < Math.min(i + 1, n); j++) {
            lines(
              //
              i64.mul(X[j], Y[i - j]),
              i > 0 && j > j0 && i64.add()
            );
          }
          let isLast = i === 2 * n - 1;
          !isLast && line(local.set(XY[i]));
        }
        for (let i = 0; i < 2 * n; i++) {
          line(local.get(XY[i]));
        }
      }
    );
  }
  multiplySchoolbookSmallLocals(7);
  multiplySchoolbookSmallLocals(6);
  addFuncExport(writer, "multiplyKaratsuba1");
  func(
    writer,
    "multiplyKaratsuba1",
    [param32(xy), param32(x), param32(y)],
    () => {
      line(local64(tmp), local64(carry));
      let X = defineLocals(writer, "x", n);
      let Y = defineLocals(writer, "y", n);
      let Z = defineLocals(writer, "z", 2 * n);
      let W = defineLocals(writer, "w", 2 * n);

      // load x, y
      for (let i = 0; i < n; i++) {
        line(local.set(X[i], i64.load(local.get(x), { offset: i * 8 })));
      }
      for (let i = 0; i < n; i++) {
        line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
      }
      // split up inputs into two halfs
      let X0 = X.slice(0, m);
      let X1 = X.slice(m);
      let Y0 = Y.slice(0, m);
      let Y1 = Y.slice(m);

      function multiplySchoolbook(Z, X, Y, n) {
        for (let i = 0; i < n; i++) {
          line(local.get(X[i]));
        }
        for (let i = 0; i < n; i++) {
          line(local.get(Y[i]));
        }
        line(call(`multiplySchoolbook${n}`));
        for (let i = 0; i < 2 * n; i++) {
          line(local.set(Z[i]));
        }
      }
      comment(`multiply z = x0*x0 in ${m}x${m} steps`);
      multiplySchoolbook(Z, X0, Y0, m);
      comment(`multiply w = x1*x1 in ${n - m}x${n - m} steps`);
      multiplySchoolbook(W, X1, Y1, n - m);

      comment("compute z = l^m*x1*x1 - x0*x0 = l^m*w - z");
      for (let i = m; i < m + 2 * (n - m); i++) {
        lines(local.set(Z[i], i64.sub(W[i - m], Z[i])));
      }
      // z has now length m + 2(n - m) = 2n - m
      comment("compute w = l^m*z - z = (l^m - 1)(l^m*x1*x1 - x0*x0)");
      for (let i = 0; i < m; i++) {
        lines(local.set(W[i], local.get(Z[i])));
      }
      for (let i = m; i < 2 * m; i++) {
        lines(local.set(W[i], i64.add(Z[i - m], Z[i])));
      }
      for (let i = 2 * m; i < 2 * n - m; i++) {
        lines(local.set(W[i], i64.sub(Z[i - m], Z[i])));
      }
      for (let i = 2 * n - m; i < 2 * n; i++) {
        lines(local.set(W[i], local.get(Z[i - m])));
      }

      comment("x1 += x0, y1 += y0");
      for (let i = 0; i < m; i++) {
        lines(
          local.set(X1[i], i64.add(X1[i], X0[i])),
          local.set(Y1[i], i64.add(Y1[i], Y0[i]))
        );
      }
      comment(`multiply z = (x0 + x1)*(y0 + y1) in ${n - m}x${n - m} steps`);
      multiplySchoolbook(Z, X1, Y1, n - m);

      comment("compute w = w + l^m*z = x*y");
      for (let i = m; i < 2 * n - m; i++) {
        lines(local.set(W[i], i64.add(W[i], Z[i - m])));
      }
      comment(`xy = carry(z)`);
      // note: here we must do signed shifts, because we allowed negative limbs
      for (let i = 0; i < 2 * n - 1; i++) {
        lines(
          local.set(tmp, i64.shr_s(W[i], w)),
          i64.store(xy, i64.and(W[i], wordMax), { offset: 8 * i }),
          local.set(W[i + 1], i64.add(W[i + 1], tmp))
        );
      }
      lines(
        i64.store(xy, i64.and(W[2 * n - 1], wordMax), {
          offset: 8 * (2 * n - 1),
        })
      );

      line(call("barrett", xy));
    }
  );
  */

  addFuncExport(writer, "multiplyKaratsuba");
  func(
    writer,
    "multiplyKaratsuba",
    [param32(xy), param32(x), param32(y)],
    () => {
      line(local64(tmp), local64(carry));
      let X = defineLocals(writer, "x", n);
      let Y = defineLocals(writer, "y", n);
      let Z = defineLocals(writer, "z", 2 * n);
      let W = defineLocals(writer, "w", 2 * n);

      // load x, y
      for (let i = 0; i < n; i++) {
        line(local.set(X[i], i64.load(local.get(x), { offset: i * 8 })));
      }
      for (let i = 0; i < n; i++) {
        line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
      }
      // split up inputs into two halfs
      let X0 = X.slice(0, m);
      let X1 = X.slice(m);
      let Y0 = Y.slice(0, m);
      let Y1 = Y.slice(m);

      // TODO: need to find out when we can leave out carry
      // note: in here, we use unsigned shift to allow maximal # of terms => assuming positive inputs x, y
      // this assumption might be relaxed
      function multiplySchoolbook(Z, X, Y, n) {
        for (let i = 0; i < 2 * n; i++) {
          comment(`k = ${i}`);
          let j0 = Math.max(0, i - n + 1);
          for (let j = j0; j < Math.min(i + 1, n); j++) {
            lines(
              //
              i64.mul(X[j], Y[i - j]),
              i > 0 && j > j0 && i64.add()
            );
          }
          let isLast = i === 2 * n - 1;
          !isLast && line(local.set(Z[i]));
          // lines(
          //   local.set(tmp),
          //   isLast && local.set(Z[i], local.get(tmp)),
          //   !isLast && local.set(Z[i], i64.and(tmp, wordMax)),
          //   !isLast && i64.shr_u(tmp, w)
          // );
        }
      }
      comment(`multiply z = x0*x0 in ${m}x${m} steps`);
      multiplySchoolbook(Z, X0, Y0, m);
      comment(`multiply w = x1*x1 in ${n - m}x${n - m} steps`);
      multiplySchoolbook(W, X1, Y1, n - m);

      comment("compute z = l^m*x1*x1 - x0*x0 = l^m*w - z");
      for (let i = m; i < m + 2 * (n - m); i++) {
        lines(local.set(Z[i], i64.sub(W[i - m], Z[i])));
      }
      // z has now length m + 2(n - m) = 2n - m
      comment("compute w = l^m*z - z = (l^m - 1)(l^m*x1*x1 - x0*x0)");
      for (let i = 0; i < m; i++) {
        lines(local.set(W[i], local.get(Z[i])));
      }
      for (let i = m; i < 2 * m; i++) {
        lines(local.set(W[i], i64.add(Z[i - m], Z[i])));
      }
      for (let i = 2 * m; i < 2 * n - m; i++) {
        lines(local.set(W[i], i64.sub(Z[i - m], Z[i])));
      }
      for (let i = 2 * n - m; i < 2 * n; i++) {
        lines(local.set(W[i], local.get(Z[i - m])));
      }

      comment("x1 += x0, y1 += y0");
      for (let i = 0; i < m; i++) {
        lines(
          local.set(X1[i], i64.add(X1[i], X0[i])),
          local.set(Y1[i], i64.add(Y1[i], Y0[i]))
        );
      }
      comment(`multiply z = (x0 + x1)*(y0 + y1) in ${n - m}x${n - m} steps`);
      multiplySchoolbook(Z, X1, Y1, n - m);

      comment("compute w = w + l^m*z = x*y");
      for (let i = m; i < 2 * n - m; i++) {
        lines(local.set(W[i], i64.add(W[i], Z[i - m])));
      }
      comment(`xy = carry(z)`);
      // note: here we must do signed shifts, because we allowed negative limbs
      for (let i = 0; i < 2 * n - 1; i++) {
        lines(
          local.set(tmp, i64.shr_s(W[i], w)),
          i64.store(xy, i64.and(W[i], wordMax), { offset: 8 * i }),
          local.set(W[i + 1], i64.add(W[i + 1], tmp))
        );
      }
      lines(
        i64.store(xy, i64.and(W[2 * n - 1], wordMax), {
          offset: 8 * (2 * n - 1),
        })
      );

      line(call("barrett", xy));
    }
  );

  if (withBenchmarks) {
    addFuncExport(writer, "benchMultiplyKaratsuba");
    func(writer, "benchMultiplyKaratsuba", [param32(x), param32($N)], () => {
      line(local32($i));
      forLoop1(writer, $i, 0, local.get($N), () => {
        lines(
          call("multiplyKaratsuba", local.get(x), local.get(x), local.get(x))
        );
      });
    });
  }
}

/**
 * barrett reduction modulo p (may be non-prime)
 *
 * given x, p, find l, r s.t. x = l*p + r
 *
 * l = [x/p] ~= l* = [x*m / 2^K] = x*m >> K, where m = [2^K / p]
 *
 * we always have l* <= l
 *
 * for estimating the error in the other direction,
 * let's assume we can guarantee x < 2^K
 * we have
 * 2^K / p <= [2^K / p] + 1 = m + 1
 * multiplying by (x/2^K) yields
 * x/p <= m*x/2^K + x/2^K < m*x/2^K + 1
 * so we get
 * l = [x/p] <= x/p <= m*x/2^K + 1
 * since l is an integer, we can round down the rhs:
 * l <= l* + 1
 *
 * let w be the limb size, and let x be represented as 2*n*w limbs, and p as n*w limbs
 * set N = n*w
 *
 * write b := Math.ceil(log_2(p)) = number of bits needed to represent p
 *
 * write K = k + N, where we want two conditions:
 * - 2^k < p. so, k < b < N. for example, k = b-1
 *   => m = [2^(k + N) / p] < 2^N, so we can represent m using the same # of limbs
 * - we can guarantee x < 2^(k + N)
 *   => this was the condition above which led to l <= l* + 1
 *
 * for example, if x = a*b, where both a, b < 2^s * p, then x < 2^(2s) * p^2
 *   => taking logs gives a condition on k: 2s + 2b <= k + N
 *   => take k = b-1
 *   => we get 2s + 2b <= b - 1 + N
 *   => condition b + 2s + 1 <= N
 *   typically we can leave a bit of room for s, since N will be a bit larger than b
 *   means a,b don't have to be fully reduced
 *
 * split up x into two unequal parts
 *   x = 2^k (x_hi) + x_lo, where x_lo has the low k bits and x_hi the rest
 * => splitting needs ~2n bitwise ops
 *
 * let's ignore x_lo and just compute l** = (x_hi * m) >> N = [x_hi * m / 2^N] <= l* <= l
 * => x_hi < 2^N, so both m and x_hi have N limbs
 *
 * this gives an approximation error
 * l* <= x*m / 2^(k + N) = x_hi * m / 2^N + (x_lo/2^k)*(m/2^N) < x_hi * m / 2^N + 1
 * and since l* is an integer,
 * l* <= [x_hi * m / 2^N] + 1 = l** + 1
 * so
 * l <= l** + 2
 *
 * to further optimize, note that in (x_hi * m) >> N, we end up ignoring the lower half of the product's 2N limbs
 * so, we can compute the product x_hi * m by ignoring all lower limb combinations which (in combination) are < 2^N.
 * => takes only ~60% of effort compared to a full multiplication
 * => the second multiplication to compute x - l*p can ignore the entire upper half => takes ~50%
 * => so in summary, barrett reduction takes ~1.1 full multiplications
 *
 * if l~ = l - e where e~ <= e, then
 * r~ = x - (l~)p = x - lp + (e~)p = r + (e~)p, with e~ in { 0, ..., e }
 * so, the result r~ is correct modulo p, but is only in [0, (1 + e)p) instead of [0, p).
 * in fewer words, this algorithm computes (x mod e*p).
 * this is similar to montgomery multiplication, and in line with earlier our assumption that x = a*b, where both a, b < 2^s * p.
 *
 * for e = 3 we can choose s=2, so assuming a, b < 4p
 * => c = (a*b mod 3p) < 4p, so again of the same form
 * => can be used without reduction steps in further products (or addition / subtraction steps which accept inputs < 4p)
 *
 * with k = b-1, our previous condition b + 2s + 1 <= N with s=2 implies
 * b + 5 <= N
 */
function barrett(writer, p, w, { withBenchmarks = false } = {}) {
  let { n, lengthP: b } = montgomeryParams(p, w);
  let k = b - 1;
  let N = n * w;
  let wn = BigInt(w);
  let wordMax = (1n << wn) - 1n;
  let { n0, e0 } = findMsbCutoff(p, w);
  let m = 2n ** BigInt(k + N) / p;
  let M = bigintToLegs(m, w, n);
  let P = bigintToLegs(p, w, n);

  let { line, lines, comment, join } = writer;
  let { i64, i32, local, local64, param32, local32, call } = ops;

  let [x, y, xy] = ["$x", "$y", "$xy"];
  let [tmp, carry] = ["$tmp", "$carry"];
  let [xi] = ["$xi"];
  let [$i, $N] = ["$i", "$N"];

  addFuncExport(writer, "barrett");
  func(writer, "barrett", [param32(x)], () => {
    line(local64(tmp), local64(carry));
    let L = defineLocals(writer, "l", n);
    let LP = defineLocals(writer, "lp", n);

    // extract x_hi := highest k bits of x
    comment(`extract l := highest ${k} bits of x = x >> ${k}`);
    // x_hi = x.slice(n-1, 2*n) <==> x >> 2^((n-1)*w)
    comment(`load l := x[${n - 1}..${2 * n}] = (x >> ${n - 1}*${w})`);
    // now we only have to do x_hi >>= k - (n - 1)*w
    let k0 = BigInt(k - (n - 1) * w);
    let l0 = wn - k0;
    comment(`then do l >>= ${k0} (because ${k0} = ${k} - ${n - 1}*${w})`);
    lines(local.set(tmp, i64.load(x, { offset: 8 * (n - 1) })));
    for (let i = 0; i < n; i++) {
      // x_hi[i] = (x_hi[i] >> k0) | ((x_hi[i + 1] << l) & wordMax);
      lines(
        i64.shr_u(tmp, k0),
        i64.and(
          i64.shl(local.tee(tmp, i64.load(x, { offset: 8 * (i + n) })), l0),
          wordMax
        ),
        i64.or(),
        local.set(L[i])
      );
    }

    // l = multiplyMsb(x_hi, m) = [x_hi * m / 2^N]
    comment(`l = [l * m / 2^N]; the first ${n0} output limbs are neglected`);
    // compute (x_hi*m) >> 2^N, where x_hi,m < 2^N,
    // by neglecting the first n0 output limbs (which we checked don't contribute in the worst case)
    for (let i = n0; i < 2 * n - 1; i++) {
      for (let j = Math.max(0, i - n + 1); j < Math.min(i + 1, n); j++) {
        lines(
          //
          i64.mul(L[j], M[i - j]),
          !(i === n0 && j === 0) && i64.add()
        );
      }
      lines(
        i < n && join(i64.const(w), i64.shr_u()),
        i >= n && join(local.tee(tmp), i64.const(wordMax), i64.and()),
        i >= n && local.set(L[i - n]),
        i >= n && i64.shr_u(tmp, w)
      );
    }
    line(local.set(L[n - 1]));

    // lp = multiplyLsb(l, p) = (l*p)[0..n], i.e. just compute the lower half
    comment("(l*p)[0..n]");
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        lines(
          //
          i64.mul(L[j], P[i - j]),
          !(j === 0) && i64.add()
        );
      }
      line(local.set(LP[i]));
    }
    // now overwrite the low n limbs with x = x - lp
    comment("x|lo = x - l*p to the low n limbs of x");
    // and ignore the possible overflow bit because we know the result fits in N bits
    for (let i = 0; i < n; i++) {
      lines(
        // (carry, x[i]) = x[i] - LP[i] + carry;
        i64.load(x, { offset: 8 * i }),
        i > 0 && i64.add(),
        local.get(LP[i]),
        i64.sub(),
        local.set(tmp),
        i64.store(x, i64.and(tmp, wordMax), { offset: 8 * i }),
        i !== n - 1 && i64.shr_s(tmp, w)
      );
    }
    // overwrite the high n limbs with l
    comment("x|hi = l");
    for (let i = n; i < 2 * n; i++) {
      lines(i64.store(x, L[i - n], { offset: 8 * i }));
    }
  });

  addFuncExport(writer, "multiplySchoolbook");
  func(
    writer,
    "multiplySchoolbook",
    [param32(xy), param32(x), param32(y)],
    () => {
      line(local64(tmp));
      // let XY = defineLocals(writer, "xy", 2 * n);
      let X = defineLocals(writer, "x", n);
      let Y = defineLocals(writer, "y", n);
      // load x, y
      for (let i = 0; i < n; i++) {
        line(local.set(X[i], i64.load(local.get(x), { offset: i * 8 })));
      }
      for (let i = 0; i < n; i++) {
        line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
      }
      // multiply
      comment(`multiply in ${n}x${n} steps`);
      // TODO: find out if time could be saved with loop
      // forLoop8(writer, $i, 0, n, () => {});
      for (let i = 0; i < 2 * n; i++) {
        comment(`k = ${i}`);
        // line(local.set(xi, i64.load(x, { offset: 8 * i })));
        for (let j = Math.max(0, i - n + 1); j < Math.min(i + 1, n); j++) {
          lines(
            // mul
            i64.mul(X[j], Y[i - j]),
            i > 0 && i64.add()
          );
        }
        lines(
          local.set(tmp),
          i < 2 * n - 1
            ? i64.store(xy, i64.and(tmp, wordMax), { offset: 8 * i })
            : i64.store(xy, tmp, { offset: 8 * i }),
          i < 2 * n - 1 && i64.shr_u(tmp, w)
        );
      }
      line(call("barrett", xy));
    }
  );
  /* 
  addFuncExport(writer, "multiplySchoolbookRegular");
  func(
    writer,
    "multiplySchoolbookRegular",
    [param32(xy), param32(x), param32(y)],
    () => {
      line(local64(tmp), local64(xi), local32($i));
      let XY = defineLocals(writer, "xy", n);
      let Y = defineLocals(writer, "y", n);
      // load y
      for (let i = 0; i < n; i++) {
        line(local.set(Y[i], i64.load(local.get(y), { offset: i * 8 })));
      }
      comment(`multiply in ${n}x${n} steps`);
      forLoop8(writer, $i, 0, n, () => {
        lines(
          local.set(xi, i64.load(i32.add(x, $i))),
          local.get(XY[0]),
          i64.mul(xi, Y[0]),
          i64.add(),
          local.set(tmp),
          i64.store(i32.add(xy, $i), i64.and(tmp, wordMax)),
          i64.shr_u(tmp, w),
          local.get(XY[1]),
          i64.add(),
          i64.mul(xi, Y[1]),
          i64.add(),
          local.set(XY[0])
        );
        for (let j = 2; j < n; j++) {
          lines(
            local.get(XY[j]),
            i64.mul(xi, Y[j]),
            i64.add(),
            local.set(XY[j - 1])
          );
        }
      });
      for (let i = n; i < 2 * n; i++) {
        lines(
          local.set(tmp, local.get(XY[i - n])),
          i64.store(xy, i64.and(tmp, wordMax), { offset: 8 * i }),
          i < 2 * n - 1 &&
            local.set(XY[i - n + 1], i64.add(XY[i - n + 1], i64.shr_u(tmp, w)))
        );
      }
      line(call("barrett", xy));
    }
  );
 */
  if (withBenchmarks) {
    addFuncExport(writer, "benchMultiplyBarrett");
    func(writer, "benchMultiplyBarrett", [param32(x), param32($N)], () => {
      line(local32($i));
      forLoop1(writer, $i, 0, local.get($N), () => {
        lines(
          call("multiplySchoolbook", local.get(x), local.get(x), local.get(x))
        );
      });
    });
  }
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

function findMsbCutoff(p, w) {
  let { n, lengthP: b } = montgomeryParams(p, w);
  let k = b - 1;
  let N = n * w;
  let K = k + N;
  let s = 2;
  console.assert(b + 2 * s + 1 <= N);

  let m = 2n ** BigInt(K) / p; // this is bigint division => rounding down

  // let's construct a conservatively bad x_hi (with large lower limbs)
  let x_hi = 2n ** BigInt(2 * b + 2 * s - k) - 1n;

  let m_vec = bigintToLegs(m, w, n);
  let x_vec = bigintToLegs(x_hi, w, n);

  // construct the length 2N schoolbook multiplication output, without carries
  let t = schoolbook(m_vec, x_vec, { n });

  // find the maximal n0 <= n so that t[0..n0] (when interpreted as an integer) is smaller than 2^N
  let n0 = 0;
  for (let sum = 0n; n0 < 2 * n; n0++) {
    sum += t[n0] << BigInt(n0 * w);
    if (sum >= 1n << BigInt(N)) break;
  }

  // confirm the approximation is fine
  let l = (m * x_hi) >> BigInt(N);
  let l0 = bigintFromLegs(multiplyMsb(m_vec, x_vec, { n0, n, w }), w, n);

  if (l - l0 > 1n)
    console.warn(
      `WARNING: for n=${n}, w=${w} the max cutoff error is ${l - l0}`
    );
  return { n0, e0: Number(l - l0) };
}

// compute approx. to (x*y) >> 2^N, where x,y < 2^N,
// by neglecting the first n0 output limbs
function multiplyMsb(x, y, { n0, n, w }) {
  let t = new BigUint64Array(2 * n - n0);
  for (let i = 0; i < n; i++) {
    // i + j >= n0 ==> j >= n0 - i
    for (let j = Math.max(0, n0 - i); j < n; j++) {
      t[i + j - n0] += x[i] * y[j];
    }
  }
  carry(t, { w, n: 2 * n - n0 });
  return t.slice(n - n0, 2 * n - n0);
}

function schoolbook(x, y, { n }) {
  let t = new BigUint64Array(2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      t[i + j] += x[i] * y[j];
    }
  }
  return t;
}

function carry(t, { w, n }) {
  let wn = BigInt(w);
  let wordMax = (1n << wn) - 1n;
  for (let i = 0; i < n - 1; i++) {
    let carry = t[i] >> wn;
    t[i] &= wordMax;
    t[i + 1] += carry;
  }
  t[n - 1] &= wordMax;
}
