;; generated for w=30, n=13, n*w=390
(module 
  (export "memory" (memory $memory))
  (memory $memory 49152)
  (export "dataOffset" (global $dataOffset))
  (export "addAffine" (func $addAffine))
  (func $addAffine (param $m i32) (param $x3 i32) (param $x1 i32) (param $x2 i32) (param $d i32)
    (local $y3 i32) (local $y1 i32) (local $y2 i32) (local $tmp i32)
    (local.set $y1 (i32.add (local.get $x1) (i32.const 104)))
    (local.set $y2 (i32.add (local.get $x2) (i32.const 104)))
    (local.set $y3 (i32.add (local.get $x3) (i32.const 104)))
    (local.set $tmp (i32.add (local.get $m) (i32.const 104)))
    ;; mark output point as non-zero
    (i32.store8 offset=208 (local.get $x3) (i32.const 1))
    ;; m = (y2 - y1)*d
    (call $multiplyDifference (local.get $m) (local.get $d) (local.get $y2) (local.get $y1))
    ;; x3 = m^2 - x1 - x2
    (call $square (local.get $tmp) (local.get $m))
    (call $subtract (local.get $x3) (local.get $tmp) (local.get $x1))
    (call $subtract (local.get $x3) (local.get $x3) (local.get $x2))
    ;; y3 = (x2 - x3)*m - y2
    (call $multiplyDifference (local.get $y3) (local.get $m) (local.get $x2) (local.get $x3))
    (call $subtract (local.get $y3) (local.get $y3) (local.get $y2))
  )
  (export "batchAddUnsafe" (func $batchAddUnsafe))
  (func $batchAddUnsafe (param $scratch i32) (param $d i32) (param $x i32) (param $S i32) (param $G i32) (param $H i32) (param $n i32)
    (local $i i32) (local $j i32) (local $I i32) (local $N i32)
    (local.set $I (local.get $scratch))
    (local.set $scratch (i32.add (local.get $scratch) (i32.const 104)))
    (local.set $N (i32.mul (local.get $n) (i32.const 104)))
    ;; return early if n = 0 or 1
    (i32.eqz (local.get $n))
    if return end
    (i32.eq (local.get $n) (i32.const 1))
    if
      (call $subtractPositive (local.get $x) (i32.load offset=0 (local.get $H)) (i32.load offset=0 (local.get $G)))
      (call $inverse (local.get $scratch) (local.get $d) (local.get $x))
      (call $addAffine (local.get $scratch) (i32.load offset=0 (local.get $S)) (i32.load offset=0 (local.get $G)) (i32.load offset=0 (local.get $H)) (local.get $d))
      return
    end
    ;; create products di = x0*...*xi, where xi = Hi_x - Gi_x
    (call $subtractPositive (local.get $x) (i32.load offset=0 (local.get $H)) (i32.load offset=0 (local.get $G)))
    (call $subtractPositive (i32.add (local.get $x) (i32.const 104)) (i32.load offset=4 (local.get $H)) (i32.load offset=4 (local.get $G)))
    (call $multiply (i32.add (local.get $d) (i32.const 104)) (i32.add (local.get $x) (i32.const 104)) (local.get $x))
    (i32.eq (local.get $n) (i32.const 2))
    if
      (call $inverse (local.get $scratch) (local.get $I) (i32.add (local.get $d) (i32.const 104)))
      (call $multiply (i32.add (local.get $d) (i32.const 104)) (local.get $x) (local.get $I))
      (call $addAffine (local.get $scratch) (i32.load offset=4 (local.get $S)) (i32.load offset=4 (local.get $G)) (i32.load offset=4 (local.get $H)) (i32.add (local.get $d) (i32.const 104)))
      (call $multiply (local.get $d) (i32.add (local.get $x) (i32.const 104)) (local.get $I))
      (call $addAffine (local.get $scratch) (i32.load offset=0 (local.get $S)) (i32.load offset=0 (local.get $G)) (i32.load offset=0 (local.get $H)) (local.get $d))
      return
    end
    (local.set $i (i32.const 208))
    (local.set $j (i32.const 8))
    (loop 
      (call $subtractPositive (i32.add (local.get $x) (local.get $i)) (i32.load offset=0 (i32.add (local.get $H) (local.get $j))) (i32.load offset=0 (i32.add (local.get $G) (local.get $j))))
      (call $multiply (i32.add (local.get $d) (local.get $i)) (i32.add (local.get $d) (i32.sub (local.get $i) (i32.const 104))) (i32.add (local.get $x) (local.get $i)))
      (local.set $j (i32.add (local.get $j) (i32.const 4)))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 104)))))
      
    )
    ;; inverse I = 1/(x0*...*x(n-1))
    (call $inverse (local.get $scratch) (local.get $I) (i32.add (local.get $d) (i32.sub (local.get $N) (i32.const 104))))
    ;; create inverses 1/x(n-1), ..., 1/x2
    (local.set $i (i32.sub (local.get $N) (i32.const 104)))
    (local.set $j (i32.sub (local.get $j) (i32.const 4)))
    (loop 
      (call $multiply (i32.add (local.get $d) (local.get $i)) (i32.add (local.get $d) (i32.sub (local.get $i) (i32.const 104))) (local.get $I))
      (call $addAffine (local.get $scratch) (i32.load offset=0 (i32.add (local.get $S) (local.get $j))) (i32.load offset=0 (i32.add (local.get $G) (local.get $j))) (i32.load offset=0 (i32.add (local.get $H) (local.get $j))) (i32.add (local.get $d) (local.get $i)))
      (call $multiply (local.get $I) (local.get $I) (i32.add (local.get $x) (local.get $i)))
      (local.set $j (i32.sub (local.get $j) (i32.const 4)))
      (br_if 0 (i32.ne (i32.const 104) (local.tee $i (i32.sub (local.get $i) (i32.const 104)))))
    )
    ;; 1/x1, 1/x0
    (call $multiply (i32.add (local.get $d) (i32.const 104)) (local.get $x) (local.get $I))
    (call $addAffine (local.get $scratch) (i32.load offset=4 (local.get $S)) (i32.load offset=4 (local.get $G)) (i32.load offset=4 (local.get $H)) (i32.add (local.get $d) (i32.const 104)))
    (call $multiply (local.get $d) (i32.add (local.get $x) (i32.const 104)) (local.get $I))
    (call $addAffine (local.get $scratch) (i32.load offset=0 (local.get $S)) (i32.load offset=0 (local.get $G)) (i32.load offset=0 (local.get $H)) (local.get $d))
  )
  (export "inverseCount" (global $inverseCount))
  (export "resetInverseCount" (func $resetInverseCount))
  (global $inverseCount (mut i32) (i32.const 0))
  (func $resetInverseCount
    (global.set $inverseCount (i32.const 0))
  )
  (global $r2corr i32 (i32.const 0))
  (data (i32.const 0)
    "\68\6b\0e\0d\00\00\00\00"
    "\07\a2\88\2f\00\00\00\00"
    "\b7\21\ec\07\00\00\00\00"
    "\e1\d8\cc\30\00\00\00\00"
    "\83\a6\03\32\00\00\00\00"
    "\13\0c\c4\3e\00\00\00\00"
    "\fa\c0\67\36\00\00\00\00"
    "\b5\43\35\20\00\00\00\00"
    "\07\8b\d1\2a\00\00\00\00"
    "\e9\4c\56\19\00\00\00\00"
    "\3c\47\35\21\00\00\00\00"
    "\df\33\26\3f\00\00\00\00"
    "\84\49\14\00\00\00\00\00"
  )
  (global $p i32 (i32.const 104))
  (data (i32.const 104)
    "\ab\aa\ff\3f\00\00\00\00"
    "\ff\ff\fb\27\00\00\00\00"
    "\fb\ff\3f\15\00\00\00\00"
    "\ac\ff\ff\2a\00\00\00\00"
    "\1e\24\f6\30\00\00\00\00"
    "\da\83\4a\03\00\00\00\00"
    "\73\f6\2b\11\00\00\00\00"
    "\e1\3c\e1\12\00\00\00\00"
    "\77\64\d7\2c\00\00\00\00"
    "\2e\0d\d9\1e\00\00\00\00"
    "\ba\b1\a4\29\00\00\00\00"
    "\f9\5f\8e\3a\00\00\00\00"
    "\11\01\1a\00\00\00\00\00"
  )
  (export "almostInverse" (func $almostInverse))
  (func $almostInverse (param $u i32) (param $r i32) (param $a i32) (result i32)
    (local $v i32) (local $s i32) (local $k i32)
    (global.set $inverseCount (i32.add (global.get $inverseCount) (i32.const 1)))
    (local.set $v (i32.add (local.get $u) (i32.const 104)))
    (local.set $s (i32.add (local.get $v) (i32.const 104)))
    (i64.store offset=0 (local.get $u) (i64.const 0x3fffaaab))
    (i64.store offset=8 (local.get $u) (i64.const 0x27fbffff))
    (i64.store offset=16 (local.get $u) (i64.const 0x153ffffb))
    (i64.store offset=24 (local.get $u) (i64.const 0x2affffac))
    (i64.store offset=32 (local.get $u) (i64.const 0x30f6241e))
    (i64.store offset=40 (local.get $u) (i64.const 0x34a83da))
    (i64.store offset=48 (local.get $u) (i64.const 0x112bf673))
    (i64.store offset=56 (local.get $u) (i64.const 0x12e13ce1))
    (i64.store offset=64 (local.get $u) (i64.const 0x2cd76477))
    (i64.store offset=72 (local.get $u) (i64.const 0x1ed90d2e))
    (i64.store offset=80 (local.get $u) (i64.const 0x29a4b1ba))
    (i64.store offset=88 (local.get $u) (i64.const 0x3a8e5ff9))
    (i64.store offset=96 (local.get $u) (i64.const 0x1a0111))
    (call $copy (local.get $v) (local.get $a))
    (i64.store offset=0 (local.get $r) (i64.const 0))
    (i64.store offset=8 (local.get $r) (i64.const 0))
    (i64.store offset=16 (local.get $r) (i64.const 0))
    (i64.store offset=24 (local.get $r) (i64.const 0))
    (i64.store offset=32 (local.get $r) (i64.const 0))
    (i64.store offset=40 (local.get $r) (i64.const 0))
    (i64.store offset=48 (local.get $r) (i64.const 0))
    (i64.store offset=56 (local.get $r) (i64.const 0))
    (i64.store offset=64 (local.get $r) (i64.const 0))
    (i64.store offset=72 (local.get $r) (i64.const 0))
    (i64.store offset=80 (local.get $r) (i64.const 0))
    (i64.store offset=88 (local.get $r) (i64.const 0))
    (i64.store offset=96 (local.get $r) (i64.const 0))
    (i64.store offset=0 (local.get $s) (i64.const 1))
    (i64.store offset=8 (local.get $s) (i64.const 0))
    (i64.store offset=16 (local.get $s) (i64.const 0))
    (i64.store offset=24 (local.get $s) (i64.const 0))
    (i64.store offset=32 (local.get $s) (i64.const 0))
    (i64.store offset=40 (local.get $s) (i64.const 0))
    (i64.store offset=48 (local.get $s) (i64.const 0))
    (i64.store offset=56 (local.get $s) (i64.const 0))
    (i64.store offset=64 (local.get $s) (i64.const 0))
    (i64.store offset=72 (local.get $s) (i64.const 0))
    (i64.store offset=80 (local.get $s) (i64.const 0))
    (i64.store offset=88 (local.get $s) (i64.const 0))
    (i64.store offset=96 (local.get $s) (i64.const 0))
    (call $makeOdd (local.get $u) (local.get $s))
    (call $makeOdd (local.get $v) (local.get $r))
    i32.add
    (local.set $k)
    (block 
      (loop 
        (call $isGreater (local.get $u) (local.get $v))
        if
          (call $subtractNoReduce (local.get $u) (local.get $u) (local.get $v))
          (call $addNoReduce (local.get $r) (local.get $r) (local.get $s))
          (local.set $k (i32.add (local.get $k) (call $makeOdd (local.get $u) (local.get $s))))
        else
          (call $subtractNoReduce (local.get $v) (local.get $v) (local.get $u))
          (call $addNoReduce (local.get $s) (local.get $s) (local.get $r))
          (br_if 2 (call $isZero (local.get $v)))
          (local.set $k (i32.add (local.get $k) (call $makeOdd (local.get $v) (local.get $r))))
        end
        (br 0)
      )
    )
    (local.get $k)
  )
  (export "inverse" (func $inverse))
  (func $inverse (param $scratch i32) (param $r i32) (param $a i32)
    (local $k i32)
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $reduce (local.get $a))
    (call $almostInverse (local.get $scratch) (local.get $r) (local.get $a))
    (local.set $k)
    (call $subtractNoReduce (local.get $r) (global.get $p) (local.get $r))
    (call $leftShift (local.get $r) (local.get $r) (i32.sub (i32.const 0x2f9) (local.get $k)))
    (call $multiply (local.get $r) (local.get $r) (global.get $r2corr))
  )
  (export "batchInverse" (func $batchInverse))
  (func $batchInverse (param $scratch i32) (param $z i32) (param $x i32) (param $n i32)
    (local $i i32) (local $I i32) (local $N i32)
    (local.set $I (local.get $scratch))
    (local.set $scratch (i32.add (local.get $scratch) (i32.const 104)))
    (local.set $N (i32.mul (local.get $n) (i32.const 104)))
    ;; return early if n = 0 or 1
    (i32.eqz (local.get $n))
    if return end
    (i32.eq (local.get $n) (i32.const 1))
    if
      (call $inverse (local.get $scratch) (local.get $z) (local.get $x))
      return
    end
    ;; create products x0*x1, ..., x0*...*x(n-1)
    (call $multiply (i32.add (local.get $z) (i32.const 104)) (i32.add (local.get $x) (i32.const 104)) (local.get $x))
    (i32.eq (local.get $n) (i32.const 2))
    if
      (call $inverse (local.get $scratch) (local.get $I) (i32.add (local.get $z) (i32.const 104)))
      (call $multiply (i32.add (local.get $z) (i32.const 104)) (local.get $x) (local.get $I))
      (call $multiply (local.get $z) (i32.add (local.get $x) (i32.const 104)) (local.get $I))
      return
    end
    (local.set $i (i32.const 208))
    (loop 
      (call $multiply (i32.add (local.get $z) (local.get $i)) (i32.add (local.get $z) (i32.sub (local.get $i) (i32.const 104))) (i32.add (local.get $x) (local.get $i)))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 104)))))
    )
    ;; inverse I = 1/(x0*...*x(n-1))
    (call $inverse (local.get $scratch) (local.get $I) (i32.add (local.get $z) (i32.sub (local.get $N) (i32.const 104))))
    ;; create inverses 1/x(n-1), ..., 1/x2
    (local.set $i (i32.sub (local.get $N) (i32.const 104)))
    (loop 
      (call $multiply (i32.add (local.get $z) (local.get $i)) (i32.add (local.get $z) (i32.sub (local.get $i) (i32.const 104))) (local.get $I))
      (call $multiply (local.get $I) (local.get $I) (i32.add (local.get $x) (local.get $i)))
      (br_if 0 (i32.ne (i32.const 104) (local.tee $i (i32.sub (local.get $i) (i32.const 104)))))
    )
    ;; 1/x1, 1/x0
    (call $multiply (i32.add (local.get $z) (i32.const 104)) (local.get $x) (local.get $I))
    (call $multiply (local.get $z) (i32.add (local.get $x) (i32.const 104)) (local.get $I))
  )
  (export "benchInverse" (func $benchInverse))
  (func $benchInverse (param $scratch i32) (param $a i32) (param $u i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $inverse (local.get $scratch) (local.get $u) (local.get $a))
      (call $add (local.get $a) (local.get $a) (local.get $u))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "multiplyCount" (global $multiplyCount))
  (export "resetMultiplyCount" (func $resetMultiplyCount))
  (global $multiplyCount (mut i32) (i32.const 0))
  (func $resetMultiplyCount
    (global.set $multiplyCount (i32.const 0))
  )
  (export "multiply" (func $multiply))
  (func $multiply (param $xy i32) (param $x i32) (param $y i32)
    (local $tmp i64)
    (local $qi i64) (local $xi i64) (local $i i32)
    (local $y00 i64) (local $y01 i64) (local $y02 i64) (local $y03 i64) 
    (local $y04 i64) (local $y05 i64) (local $y06 i64) (local $y07 i64) 
    (local $y08 i64) (local $y09 i64) (local $y10 i64) (local $y11 i64) 
    (local $y12 i64) 
    (local $t00 i64) (local $t01 i64) (local $t02 i64) (local $t03 i64) 
    (local $t04 i64) (local $t05 i64) (local $t06 i64) (local $t07 i64) 
    (local $t08 i64) (local $t09 i64) (local $t10 i64) (local $t11 i64) 
    (local $t12 i64) 
    (global.set $multiplyCount (i32.add (global.get $multiplyCount) (i32.const 1)))
    (local.set $y00 (i64.load offset=0 (local.get $y)))
    (local.set $y01 (i64.load offset=8 (local.get $y)))
    (local.set $y02 (i64.load offset=16 (local.get $y)))
    (local.set $y03 (i64.load offset=24 (local.get $y)))
    (local.set $y04 (i64.load offset=32 (local.get $y)))
    (local.set $y05 (i64.load offset=40 (local.get $y)))
    (local.set $y06 (i64.load offset=48 (local.get $y)))
    (local.set $y07 (i64.load offset=56 (local.get $y)))
    (local.set $y08 (i64.load offset=64 (local.get $y)))
    (local.set $y09 (i64.load offset=72 (local.get $y)))
    (local.set $y10 (i64.load offset=80 (local.get $y)))
    (local.set $y11 (i64.load offset=88 (local.get $y)))
    (local.set $y12 (i64.load offset=96 (local.get $y)))
    (local.set $i (i32.const 0))
    (loop 
      (local.set $xi (i64.load offset=0 (i32.add (local.get $x) (local.get $i))))
      ;; j = 0, do carry, ignore result below carry
      (local.get $t00)
      (i64.mul (local.get $xi) (local.get $y00))
      i64.add
      (local.set $tmp)
      (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
      (local.get $tmp)
      (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
      i64.add
      (i64.const 30) i64.shr_u
      ;; j = 1
      (local.get $t01)
      i64.add
      (i64.mul (local.get $xi) (local.get $y01))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x27fbffff))
      i64.add
      (local.set $t00)
      ;; j = 2
      (local.get $t02)
      (i64.mul (local.get $xi) (local.get $y02))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x153ffffb))
      i64.add
      (local.set $t01)
      ;; j = 3
      (local.get $t03)
      (i64.mul (local.get $xi) (local.get $y03))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x2affffac))
      i64.add
      (local.set $t02)
      ;; j = 4
      (local.get $t04)
      (i64.mul (local.get $xi) (local.get $y04))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x30f6241e))
      i64.add
      (local.set $t03)
      ;; j = 5
      (local.get $t05)
      (i64.mul (local.get $xi) (local.get $y05))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x34a83da))
      i64.add
      (local.set $t04)
      ;; j = 6
      (local.get $t06)
      (i64.mul (local.get $xi) (local.get $y06))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x112bf673))
      i64.add
      (local.set $t05)
      ;; j = 7
      (local.get $t07)
      (i64.mul (local.get $xi) (local.get $y07))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
      i64.add
      (local.set $t06)
      ;; j = 8, do carry
      (local.get $t08)
      (i64.mul (local.get $xi) (local.get $y08))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x2cd76477))
      i64.add
      (local.tee $tmp) (i64.const 30) i64.shr_u
      (i64.and (local.get $tmp) (i64.const 0x3fffffff))
      (local.set $t07)
      ;; j = 9
      (local.get $t09)
      i64.add
      (i64.mul (local.get $xi) (local.get $y09))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
      i64.add
      (local.set $t08)
      ;; j = 10
      (local.get $t10)
      (i64.mul (local.get $xi) (local.get $y10))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
      i64.add
      (local.set $t09)
      ;; j = 11
      (local.get $t11)
      (i64.mul (local.get $xi) (local.get $y11))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
      i64.add
      (local.set $t10)
      ;; j = 12
      (i64.mul (local.get $xi) (local.get $y12))
      (i64.mul (local.get $qi) (i64.const 0x1a0111))
      i64.add
      (local.set $t11)
      (br_if 0 (i32.ne (i32.const 104) (local.tee $i (i32.add (local.get $i) (i32.const 8)))))
    )
    ;; final carrying & storing
    (i64.store offset=0 (local.get $xy) (i64.and (local.get $t00) (i64.const 0x3fffffff)))
    (local.set $t01 (i64.add (local.get $t01) (i64.shr_u (local.get $t00) (i64.const 30))))
    (i64.store offset=8 (local.get $xy) (i64.and (local.get $t01) (i64.const 0x3fffffff)))
    (local.set $t02 (i64.add (local.get $t02) (i64.shr_u (local.get $t01) (i64.const 30))))
    (i64.store offset=16 (local.get $xy) (i64.and (local.get $t02) (i64.const 0x3fffffff)))
    (local.set $t03 (i64.add (local.get $t03) (i64.shr_u (local.get $t02) (i64.const 30))))
    (i64.store offset=24 (local.get $xy) (i64.and (local.get $t03) (i64.const 0x3fffffff)))
    (local.set $t04 (i64.add (local.get $t04) (i64.shr_u (local.get $t03) (i64.const 30))))
    (i64.store offset=32 (local.get $xy) (i64.and (local.get $t04) (i64.const 0x3fffffff)))
    (local.set $t05 (i64.add (local.get $t05) (i64.shr_u (local.get $t04) (i64.const 30))))
    (i64.store offset=40 (local.get $xy) (i64.and (local.get $t05) (i64.const 0x3fffffff)))
    (local.set $t06 (i64.add (local.get $t06) (i64.shr_u (local.get $t05) (i64.const 30))))
    (i64.store offset=48 (local.get $xy) (i64.and (local.get $t06) (i64.const 0x3fffffff)))
    (local.set $t07 (i64.add (local.get $t07) (i64.shr_u (local.get $t06) (i64.const 30))))
    (i64.store offset=56 (local.get $xy) (i64.and (local.get $t07) (i64.const 0x3fffffff)))
    (local.set $t08 (i64.add (local.get $t08) (i64.shr_u (local.get $t07) (i64.const 30))))
    (i64.store offset=64 (local.get $xy) (i64.and (local.get $t08) (i64.const 0x3fffffff)))
    (local.set $t09 (i64.add (local.get $t09) (i64.shr_u (local.get $t08) (i64.const 30))))
    (i64.store offset=72 (local.get $xy) (i64.and (local.get $t09) (i64.const 0x3fffffff)))
    (local.set $t10 (i64.add (local.get $t10) (i64.shr_u (local.get $t09) (i64.const 30))))
    (i64.store offset=80 (local.get $xy) (i64.and (local.get $t10) (i64.const 0x3fffffff)))
    (local.set $t11 (i64.add (local.get $t11) (i64.shr_u (local.get $t10) (i64.const 30))))
    (i64.store offset=88 (local.get $xy) (i64.and (local.get $t11) (i64.const 0x3fffffff)))
    (local.set $t12 (i64.add (local.get $t12) (i64.shr_u (local.get $t11) (i64.const 30))))
    (i64.store offset=96 (local.get $xy) (local.get $t12))
  )
  (export "multiplyDifference" (func $multiplyDifference))
  (func $multiplyDifference (param $xy i32) (param $x i32) (param $y i32) (param $z i32)
    (local $tmp i64)
    (local $qi i64) (local $xi i64) (local $i i32)
    (local $y00 i64) (local $y01 i64) (local $y02 i64) (local $y03 i64) 
    (local $y04 i64) (local $y05 i64) (local $y06 i64) (local $y07 i64) 
    (local $y08 i64) (local $y09 i64) (local $y10 i64) (local $y11 i64) 
    (local $y12 i64) 
    (local $t00 i64) (local $t01 i64) (local $t02 i64) (local $t03 i64) 
    (local $t04 i64) (local $t05 i64) (local $t06 i64) (local $t07 i64) 
    (local $t08 i64) (local $t09 i64) (local $t10 i64) (local $t11 i64) 
    (local $t12 i64) 
    (global.set $multiplyCount (i32.add (global.get $multiplyCount) (i32.const 1)))
    (i64.const 0x7ffaaab0)
    (i64.load offset=0 (local.get $y))
    i64.add
    (i64.load offset=0 (local.get $z))
    i64.sub
    (local.set $y00)
    (i64.const 0x7fbffffe)
    (i64.load offset=8 (local.get $y))
    i64.add
    (i64.load offset=8 (local.get $z))
    i64.sub
    (local.set $y01)
    (i64.const 0x53ffffb8)
    (i64.load offset=16 (local.get $y))
    i64.add
    (i64.load offset=16 (local.get $z))
    i64.sub
    (local.set $y02)
    (i64.const 0x6ffffac4)
    (i64.load offset=24 (local.get $y))
    i64.add
    (i64.load offset=24 (local.get $z))
    i64.sub
    (local.set $y03)
    (i64.const 0x4f6241e9)
    (i64.load offset=32 (local.get $y))
    i64.add
    (i64.load offset=32 (local.get $z))
    i64.sub
    (local.set $y04)
    (i64.const 0x74a83dab)
    (i64.load offset=40 (local.get $y))
    i64.add
    (i64.load offset=40 (local.get $z))
    i64.sub
    (local.set $y05)
    (i64.const 0x52bf672f)
    (i64.load offset=48 (local.get $y))
    i64.add
    (i64.load offset=48 (local.get $z))
    i64.sub
    (local.set $y06)
    (i64.const 0x6e13ce13)
    (i64.load offset=56 (local.get $y))
    i64.add
    (i64.load offset=56 (local.get $z))
    i64.sub
    (local.set $y07)
    (i64.const 0x4d764773)
    (i64.load offset=64 (local.get $y))
    i64.add
    (i64.load offset=64 (local.get $z))
    i64.sub
    (local.set $y08)
    (i64.const 0x6d90d2ea)
    (i64.load offset=72 (local.get $y))
    i64.add
    (i64.load offset=72 (local.get $z))
    i64.sub
    (local.set $y09)
    (i64.const 0x5a4b1ba6)
    (i64.load offset=80 (local.get $y))
    i64.add
    (i64.load offset=80 (local.get $z))
    i64.sub
    (local.set $y10)
    (i64.const 0x68e5ff99)
    (i64.load offset=88 (local.get $y))
    i64.add
    (i64.load offset=88 (local.get $z))
    i64.sub
    (local.set $y11)
    (i64.const 0x1a0111d)
    (i64.load offset=96 (local.get $y))
    i64.add
    (i64.load offset=96 (local.get $z))
    i64.sub
    (local.set $y12)
    (local.set $i (i32.const 0))
    (loop 
      (local.set $xi (i64.load offset=0 (i32.add (local.get $x) (local.get $i))))
      ;; j = 0, do carry, ignore result below carry
      (local.get $t00)
      (i64.mul (local.get $xi) (local.get $y00))
      i64.add
      (local.set $tmp)
      (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
      (local.get $tmp)
      (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
      i64.add
      (i64.const 30) i64.shr_u
      ;; j = 1
      (local.get $t01)
      i64.add
      (i64.mul (local.get $xi) (local.get $y01))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x27fbffff))
      i64.add
      (local.set $t00)
      ;; j = 2
      (local.get $t02)
      (i64.mul (local.get $xi) (local.get $y02))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x153ffffb))
      i64.add
      (local.set $t01)
      ;; j = 3
      (local.get $t03)
      (i64.mul (local.get $xi) (local.get $y03))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x2affffac))
      i64.add
      (local.set $t02)
      ;; j = 4
      (local.get $t04)
      (i64.mul (local.get $xi) (local.get $y04))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x30f6241e))
      i64.add
      (local.set $t03)
      ;; j = 5
      (local.get $t05)
      (i64.mul (local.get $xi) (local.get $y05))
      i64.add
      ;; carry after 9 terms
      (local.tee $tmp) (i64.const 30) i64.shr_u
      (i64.and (local.get $tmp) (i64.const 0x3fffffff))
      (i64.mul (local.get $qi) (i64.const 0x34a83da))
      i64.add
      (local.set $t04)
      ;; j = 6
      (local.get $t06)
      i64.add
      (i64.mul (local.get $xi) (local.get $y06))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x112bf673))
      i64.add
      (local.set $t05)
      ;; j = 7
      (local.get $t07)
      (i64.mul (local.get $xi) (local.get $y07))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
      i64.add
      (local.set $t06)
      ;; j = 8
      (local.get $t08)
      (i64.mul (local.get $xi) (local.get $y08))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x2cd76477))
      i64.add
      (local.set $t07)
      ;; j = 9
      (local.get $t09)
      (i64.mul (local.get $xi) (local.get $y09))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
      i64.add
      ;; carry after 9 terms
      (local.tee $tmp) (i64.const 30) i64.shr_u
      (i64.and (local.get $tmp) (i64.const 0x3fffffff))
      (local.set $t08)
      ;; j = 10
      (local.get $t10)
      i64.add
      (i64.mul (local.get $xi) (local.get $y10))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
      i64.add
      (local.set $t09)
      ;; j = 11
      (local.get $t11)
      (i64.mul (local.get $xi) (local.get $y11))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
      i64.add
      (local.set $t10)
      ;; j = 12
      (i64.mul (local.get $xi) (local.get $y12))
      (i64.mul (local.get $qi) (i64.const 0x1a0111))
      i64.add
      (local.set $t11)
      (br_if 0 (i32.ne (i32.const 104) (local.tee $i (i32.add (local.get $i) (i32.const 8)))))
    )
    ;; final carrying & storing
    (i64.store offset=0 (local.get $xy) (i64.and (local.get $t00) (i64.const 0x3fffffff)))
    (local.set $t01 (i64.add (local.get $t01) (i64.shr_u (local.get $t00) (i64.const 30))))
    (i64.store offset=8 (local.get $xy) (i64.and (local.get $t01) (i64.const 0x3fffffff)))
    (local.set $t02 (i64.add (local.get $t02) (i64.shr_u (local.get $t01) (i64.const 30))))
    (i64.store offset=16 (local.get $xy) (i64.and (local.get $t02) (i64.const 0x3fffffff)))
    (local.set $t03 (i64.add (local.get $t03) (i64.shr_u (local.get $t02) (i64.const 30))))
    (i64.store offset=24 (local.get $xy) (i64.and (local.get $t03) (i64.const 0x3fffffff)))
    (local.set $t04 (i64.add (local.get $t04) (i64.shr_u (local.get $t03) (i64.const 30))))
    (i64.store offset=32 (local.get $xy) (i64.and (local.get $t04) (i64.const 0x3fffffff)))
    (local.set $t05 (i64.add (local.get $t05) (i64.shr_u (local.get $t04) (i64.const 30))))
    (i64.store offset=40 (local.get $xy) (i64.and (local.get $t05) (i64.const 0x3fffffff)))
    (local.set $t06 (i64.add (local.get $t06) (i64.shr_u (local.get $t05) (i64.const 30))))
    (i64.store offset=48 (local.get $xy) (i64.and (local.get $t06) (i64.const 0x3fffffff)))
    (local.set $t07 (i64.add (local.get $t07) (i64.shr_u (local.get $t06) (i64.const 30))))
    (i64.store offset=56 (local.get $xy) (i64.and (local.get $t07) (i64.const 0x3fffffff)))
    (local.set $t08 (i64.add (local.get $t08) (i64.shr_u (local.get $t07) (i64.const 30))))
    (i64.store offset=64 (local.get $xy) (i64.and (local.get $t08) (i64.const 0x3fffffff)))
    (local.set $t09 (i64.add (local.get $t09) (i64.shr_u (local.get $t08) (i64.const 30))))
    (i64.store offset=72 (local.get $xy) (i64.and (local.get $t09) (i64.const 0x3fffffff)))
    (local.set $t10 (i64.add (local.get $t10) (i64.shr_u (local.get $t09) (i64.const 30))))
    (i64.store offset=80 (local.get $xy) (i64.and (local.get $t10) (i64.const 0x3fffffff)))
    (local.set $t11 (i64.add (local.get $t11) (i64.shr_u (local.get $t10) (i64.const 30))))
    (i64.store offset=88 (local.get $xy) (i64.and (local.get $t11) (i64.const 0x3fffffff)))
    (local.set $t12 (i64.add (local.get $t12) (i64.shr_u (local.get $t11) (i64.const 30))))
    (i64.store offset=96 (local.get $xy) (local.get $t12))
  )
  (export "multiplyUnrolled" (func $multiplyUnrolled))
  (func $multiplyUnrolled (param $xy i32) (param $x i32) (param $y i32)
    (local $tmp i64) (local $carry i64)
    (local $x00 i64) (local $x01 i64) (local $x02 i64) (local $x03 i64) 
    (local $x04 i64) (local $x05 i64) (local $x06 i64) (local $x07 i64) 
    (local $x08 i64) (local $x09 i64) (local $x10 i64) (local $x11 i64) 
    (local $x12 i64) 
    (local $y00 i64) (local $y01 i64) (local $y02 i64) (local $y03 i64) 
    (local $y04 i64) (local $y05 i64) (local $y06 i64) (local $y07 i64) 
    (local $y08 i64) (local $y09 i64) (local $y10 i64) (local $y11 i64) 
    (local $y12 i64) 
    (local $q00 i64) (local $q01 i64) (local $q02 i64) (local $q03 i64) 
    (local $q04 i64) (local $q05 i64) (local $q06 i64) (local $q07 i64) 
    (local $q08 i64) (local $q09 i64) (local $q10 i64) (local $q11 i64) 
    (local $q12 i64) 
    (global.set $multiplyCount (i32.add (global.get $multiplyCount) (i32.const 1)))
    (local.set $x00 (i64.load offset=0 (local.get $x)))
    (local.set $x01 (i64.load offset=8 (local.get $x)))
    (local.set $x02 (i64.load offset=16 (local.get $x)))
    (local.set $x03 (i64.load offset=24 (local.get $x)))
    (local.set $x04 (i64.load offset=32 (local.get $x)))
    (local.set $x05 (i64.load offset=40 (local.get $x)))
    (local.set $x06 (i64.load offset=48 (local.get $x)))
    (local.set $x07 (i64.load offset=56 (local.get $x)))
    (local.set $x08 (i64.load offset=64 (local.get $x)))
    (local.set $x09 (i64.load offset=72 (local.get $x)))
    (local.set $x10 (i64.load offset=80 (local.get $x)))
    (local.set $x11 (i64.load offset=88 (local.get $x)))
    (local.set $x12 (i64.load offset=96 (local.get $x)))
    (local.set $y00 (i64.load offset=0 (local.get $y)))
    (local.set $y01 (i64.load offset=8 (local.get $y)))
    (local.set $y02 (i64.load offset=16 (local.get $y)))
    (local.set $y03 (i64.load offset=24 (local.get $y)))
    (local.set $y04 (i64.load offset=32 (local.get $y)))
    (local.set $y05 (i64.load offset=40 (local.get $y)))
    (local.set $y06 (i64.load offset=48 (local.get $y)))
    (local.set $y07 (i64.load offset=56 (local.get $y)))
    (local.set $y08 (i64.load offset=64 (local.get $y)))
    (local.set $y09 (i64.load offset=72 (local.get $y)))
    (local.set $y10 (i64.load offset=80 (local.get $y)))
    (local.set $y11 (i64.load offset=88 (local.get $y)))
    (local.set $y12 (i64.load offset=96 (local.get $y)))
    ;; i = 0
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y00))
    (local.set $tmp)
    (local.set $q00 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q00) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 1
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y01))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q01 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q01) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 2
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y02))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y01))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q02 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q02) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 3
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y03))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x2affffac))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y02))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y01))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q03 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q03) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 4
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y04))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y03))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x2affffac))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y02))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y01))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q04 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q04) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 5
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y05))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x34a83da))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y04))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y03))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x2affffac))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y02))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y01))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q05 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q05) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 6
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y06))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x112bf673))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y05))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x34a83da))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y04))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y03))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x2affffac))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y02))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y01))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q06 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q06) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 7
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y07))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y06))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x112bf673))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y05))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x34a83da))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y04))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y03))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x2affffac))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y02))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y01))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q07 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q07) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; i = 8
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y08))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y07))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y06))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x112bf673))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y05))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x34a83da))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y04))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y03))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x2affffac))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y02))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y01))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x27fbffff))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q08 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q08) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    (local.get $carry)
    i64.add
    ;; i = 9
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y09))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y08))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y07))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y06))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x112bf673))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y05))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x34a83da))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y04))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y03))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x2affffac))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y02))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x153ffffb))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y01))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q09 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q09) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    (local.get $carry)
    i64.add
    ;; i = 10
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y10))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y09))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y08))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y07))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y06))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x112bf673))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y05))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x34a83da))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y04))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y03))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x2affffac))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y02))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y01))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q10 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q10) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    (local.get $carry)
    i64.add
    ;; i = 11
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y11))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y10))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y09))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y08))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y07))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y06))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x112bf673))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y05))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x34a83da))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y04))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x30f6241e))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y03))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x2affffac))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y02))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y01))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q11 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q11) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    (local.get $carry)
    i64.add
    ;; i = 12
    ;; > j = 0
    (i64.mul (local.get $x00) (local.get $y12))
    i64.add
    (i64.mul (local.get $q00) (i64.const 0x1a0111))
    i64.add
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y11))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y10))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y09))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y08))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y07))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y06))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x112bf673))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y05))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x34a83da))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y04))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y03))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x2affffac))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y02))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y01))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x27fbffff))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y00))
    i64.add
    (local.set $tmp)
    (local.set $q12 (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $q12) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    (local.get $carry)
    i64.add
    ;; i = 13
    ;; > j = 1
    (i64.mul (local.get $x01) (local.get $y12))
    i64.add
    (i64.mul (local.get $q01) (i64.const 0x1a0111))
    i64.add
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y11))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y10))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y09))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y08))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y07))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y06))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x112bf673))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y05))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x34a83da))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y04))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y03))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x2affffac))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y02))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x153ffffb))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y01))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x27fbffff))
    i64.add
    (local.set $tmp)
    (i64.store offset=0 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.get $carry)
    i64.add
    ;; i = 14
    ;; > j = 2
    (i64.mul (local.get $x02) (local.get $y12))
    i64.add
    (i64.mul (local.get $q02) (i64.const 0x1a0111))
    i64.add
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y11))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y10))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y09))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y08))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y07))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y06))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x112bf673))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y05))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x34a83da))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y04))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y03))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x2affffac))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y02))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x153ffffb))
    i64.add
    (local.set $tmp)
    (i64.store offset=8 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.get $carry)
    i64.add
    ;; i = 15
    ;; > j = 3
    (i64.mul (local.get $x03) (local.get $y12))
    i64.add
    (i64.mul (local.get $q03) (i64.const 0x1a0111))
    i64.add
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y11))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y10))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y09))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y08))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y07))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y06))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x112bf673))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y05))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x34a83da))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y04))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x30f6241e))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y03))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x2affffac))
    i64.add
    (local.set $tmp)
    (i64.store offset=16 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.get $carry)
    i64.add
    ;; i = 16
    ;; > j = 4
    (i64.mul (local.get $x04) (local.get $y12))
    i64.add
    (i64.mul (local.get $q04) (i64.const 0x1a0111))
    i64.add
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y11))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y10))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y09))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y08))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y07))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y06))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x112bf673))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y05))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x34a83da))
    i64.add
    ;; > carry after term # 16
    (local.set $tmp)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $carry)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y04))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x30f6241e))
    i64.add
    (local.set $tmp)
    (i64.store offset=24 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.get $carry)
    i64.add
    ;; i = 17
    ;; > j = 5
    (i64.mul (local.get $x05) (local.get $y12))
    i64.add
    (i64.mul (local.get $q05) (i64.const 0x1a0111))
    i64.add
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y11))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y10))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y09))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y08))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y07))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y06))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x112bf673))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y05))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x34a83da))
    i64.add
    (local.set $tmp)
    (i64.store offset=32 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 18
    ;; > j = 6
    (i64.mul (local.get $x06) (local.get $y12))
    i64.add
    (i64.mul (local.get $q06) (i64.const 0x1a0111))
    i64.add
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y11))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y10))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y09))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y08))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y07))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x12e13ce1))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y06))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x112bf673))
    i64.add
    (local.set $tmp)
    (i64.store offset=40 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 19
    ;; > j = 7
    (i64.mul (local.get $x07) (local.get $y12))
    i64.add
    (i64.mul (local.get $q07) (i64.const 0x1a0111))
    i64.add
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y11))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y10))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y09))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y08))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x2cd76477))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y07))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x12e13ce1))
    i64.add
    (local.set $tmp)
    (i64.store offset=48 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 20
    ;; > j = 8
    (i64.mul (local.get $x08) (local.get $y12))
    i64.add
    (i64.mul (local.get $q08) (i64.const 0x1a0111))
    i64.add
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y11))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y10))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y09))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x1ed90d2e))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y08))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x2cd76477))
    i64.add
    (local.set $tmp)
    (i64.store offset=56 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 21
    ;; > j = 9
    (i64.mul (local.get $x09) (local.get $y12))
    i64.add
    (i64.mul (local.get $q09) (i64.const 0x1a0111))
    i64.add
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y11))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y10))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x29a4b1ba))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y09))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $tmp)
    (i64.store offset=64 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 22
    ;; > j = 10
    (i64.mul (local.get $x10) (local.get $y12))
    i64.add
    (i64.mul (local.get $q10) (i64.const 0x1a0111))
    i64.add
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y11))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x3a8e5ff9))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y10))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x29a4b1ba))
    i64.add
    (local.set $tmp)
    (i64.store offset=72 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 23
    ;; > j = 11
    (i64.mul (local.get $x11) (local.get $y12))
    i64.add
    (i64.mul (local.get $q11) (i64.const 0x1a0111))
    i64.add
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y11))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $tmp)
    (i64.store offset=80 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; i = 24
    ;; > j = 12
    (i64.mul (local.get $x12) (local.get $y12))
    i64.add
    (i64.mul (local.get $q12) (i64.const 0x1a0111))
    i64.add
    (local.set $tmp)
    (i64.store offset=88 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $tmp)
    (i64.store offset=96 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
  )
  (export "square" (func $square))
  (func $square (param $out i32) (param $x i32)
    (local $tmp i64)
    (local $qi i64)
    (local $x00 i64) (local $x01 i64) (local $x02 i64) (local $x03 i64) 
    (local $x04 i64) (local $x05 i64) (local $x06 i64) (local $x07 i64) 
    (local $x08 i64) (local $x09 i64) (local $x10 i64) (local $x11 i64) 
    (local $x12 i64) 
    (local $t00 i64) (local $t01 i64) (local $t02 i64) (local $t03 i64) 
    (local $t04 i64) (local $t05 i64) (local $t06 i64) (local $t07 i64) 
    (local $t08 i64) (local $t09 i64) (local $t10 i64) (local $t11 i64) 
    (local $t12 i64) 
    (global.set $multiplyCount (i32.add (global.get $multiplyCount) (i32.const 1)))
    (local.set $x00 (i64.load offset=0 (local.get $x)))
    (local.set $x01 (i64.load offset=8 (local.get $x)))
    (local.set $x02 (i64.load offset=16 (local.get $x)))
    (local.set $x03 (i64.load offset=24 (local.get $x)))
    (local.set $x04 (i64.load offset=32 (local.get $x)))
    (local.set $x05 (i64.load offset=40 (local.get $x)))
    (local.set $x06 (i64.load offset=48 (local.get $x)))
    (local.set $x07 (i64.load offset=56 (local.get $x)))
    (local.set $x08 (i64.load offset=64 (local.get $x)))
    (local.set $x09 (i64.load offset=72 (local.get $x)))
    (local.set $x10 (i64.load offset=80 (local.get $x)))
    (local.set $x11 (i64.load offset=88 (local.get $x)))
    (local.set $x12 (i64.load offset=96 (local.get $x)))
    ;; i = 0
    ;; j = 0, do carry, ignore result below carry
    (i64.mul (local.get $x00) (local.get $x00))
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 1
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x01) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x01) (local.get $x01))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 2
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x02) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x02) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x02) (local.get $x02))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 3
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x03) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x03) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x03) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x03) (local.get $x03))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 4
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x04) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x04) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x04) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x04) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x04) (local.get $x04))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 5
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x05) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x05) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x05) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x05) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x05) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x05) (local.get $x05))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 6
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x06) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x06) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x06) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x06) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x06) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x06) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x06) (local.get $x06))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 7
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x07) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x07) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x07) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x07) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x07) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x07) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x07) (local.get $x06))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $x07) (local.get $x07))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 8
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x08) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x08) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x08) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x08) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x08) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x08) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x08) (local.get $x06))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $x08) (local.get $x07))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $x08) (local.get $x08))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 9
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x09) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x09) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x09) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x09) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x09) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x09) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x09) (local.get $x06))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $x09) (local.get $x07))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $x09) (local.get $x08))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $x09) (local.get $x09))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 10
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x10) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x10) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x10) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x10) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x10) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x10) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x10) (local.get $x06))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $x10) (local.get $x07))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $x10) (local.get $x08))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $x10) (local.get $x09))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $x10) (local.get $x10))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 11
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x11) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x11) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x11) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x11) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x11) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x11) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x11) (local.get $x06))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $x11) (local.get $x07))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $x11) (local.get $x08))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $x11) (local.get $x09))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $x11) (local.get $x10))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $x11) (local.get $x11))
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    (local.set $t11)
    ;; i = 12
    ;; j = 0, do carry, ignore result below carry
    (local.get $t00)
    (i64.shl (i64.mul (local.get $x12) (local.get $x00)) (i64.const 1))
    i64.add
    (local.set $tmp)
    (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
    (local.get $tmp)
    (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
    i64.add
    (i64.const 30) i64.shr_u
    ;; j = 1
    (local.get $t01)
    i64.add
    (i64.mul (local.get $x12) (local.get $x01))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x27fbffff))
    i64.add
    (local.set $t00)
    ;; j = 2
    (local.get $t02)
    (i64.mul (local.get $x12) (local.get $x02))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x153ffffb))
    i64.add
    (local.set $t01)
    ;; j = 3
    (local.get $t03)
    (i64.mul (local.get $x12) (local.get $x03))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2affffac))
    i64.add
    (local.set $t02)
    ;; j = 4
    (local.get $t04)
    (i64.mul (local.get $x12) (local.get $x04))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x30f6241e))
    i64.add
    (local.set $t03)
    ;; j = 5, do carry
    (local.get $t05)
    (i64.mul (local.get $x12) (local.get $x05))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x34a83da))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t04)
    ;; j = 6
    (local.get $t06)
    i64.add
    (i64.mul (local.get $x12) (local.get $x06))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x112bf673))
    i64.add
    (local.set $t05)
    ;; j = 7
    (local.get $t07)
    (i64.mul (local.get $x12) (local.get $x07))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
    i64.add
    (local.set $t06)
    ;; j = 8
    (local.get $t08)
    (i64.mul (local.get $x12) (local.get $x08))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x2cd76477))
    i64.add
    (local.set $t07)
    ;; j = 9
    (local.get $t09)
    (i64.mul (local.get $x12) (local.get $x09))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
    i64.add
    (local.set $t08)
    ;; j = 10, do carry
    (local.get $t10)
    (i64.mul (local.get $x12) (local.get $x10))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
    i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_u
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (local.set $t09)
    ;; j = 11
    (local.get $t11)
    i64.add
    (i64.mul (local.get $x12) (local.get $x11))
    (i64.const 1) i64.shl
    i64.add
    (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
    i64.add
    (local.set $t10)
    ;; j = 12
    (i64.mul (local.get $x12) (local.get $x12))
    (i64.mul (local.get $qi) (i64.const 0x1a0111))
    i64.add
    (local.set $t11)
    ;; final carrying & storing
    (i64.store offset=0 (local.get $out) (i64.and (local.get $t00) (i64.const 0x3fffffff)))
    (local.set $t01 (i64.add (local.get $t01) (i64.shr_u (local.get $t00) (i64.const 30))))
    (i64.store offset=8 (local.get $out) (i64.and (local.get $t01) (i64.const 0x3fffffff)))
    (local.set $t02 (i64.add (local.get $t02) (i64.shr_u (local.get $t01) (i64.const 30))))
    (i64.store offset=16 (local.get $out) (i64.and (local.get $t02) (i64.const 0x3fffffff)))
    (local.set $t03 (i64.add (local.get $t03) (i64.shr_u (local.get $t02) (i64.const 30))))
    (i64.store offset=24 (local.get $out) (i64.and (local.get $t03) (i64.const 0x3fffffff)))
    (local.set $t04 (i64.add (local.get $t04) (i64.shr_u (local.get $t03) (i64.const 30))))
    (i64.store offset=32 (local.get $out) (i64.and (local.get $t04) (i64.const 0x3fffffff)))
    (local.set $t05 (i64.add (local.get $t05) (i64.shr_u (local.get $t04) (i64.const 30))))
    (i64.store offset=40 (local.get $out) (i64.and (local.get $t05) (i64.const 0x3fffffff)))
    (local.set $t06 (i64.add (local.get $t06) (i64.shr_u (local.get $t05) (i64.const 30))))
    (i64.store offset=48 (local.get $out) (i64.and (local.get $t06) (i64.const 0x3fffffff)))
    (local.set $t07 (i64.add (local.get $t07) (i64.shr_u (local.get $t06) (i64.const 30))))
    (i64.store offset=56 (local.get $out) (i64.and (local.get $t07) (i64.const 0x3fffffff)))
    (local.set $t08 (i64.add (local.get $t08) (i64.shr_u (local.get $t07) (i64.const 30))))
    (i64.store offset=64 (local.get $out) (i64.and (local.get $t08) (i64.const 0x3fffffff)))
    (local.set $t09 (i64.add (local.get $t09) (i64.shr_u (local.get $t08) (i64.const 30))))
    (i64.store offset=72 (local.get $out) (i64.and (local.get $t09) (i64.const 0x3fffffff)))
    (local.set $t10 (i64.add (local.get $t10) (i64.shr_u (local.get $t09) (i64.const 30))))
    (i64.store offset=80 (local.get $out) (i64.and (local.get $t10) (i64.const 0x3fffffff)))
    (local.set $t11 (i64.add (local.get $t11) (i64.shr_u (local.get $t10) (i64.const 30))))
    (i64.store offset=88 (local.get $out) (i64.and (local.get $t11) (i64.const 0x3fffffff)))
    (local.set $t12 (i64.add (local.get $t12) (i64.shr_u (local.get $t11) (i64.const 30))))
    (i64.store offset=96 (local.get $out) (local.get $t12))
  )
  (export "leftShift" (func $leftShift))
  (func $leftShift (param $xy i32) (param $y i32) (param $k i32)
    (local $tmp i64)
    (local $qi i64) (local $xi i64) (local $i i32) (local $i0 i32) (local $xi0 i32)
    (local $y00 i64) (local $y01 i64) (local $y02 i64) (local $y03 i64) 
    (local $y04 i64) (local $y05 i64) (local $y06 i64) (local $y07 i64) 
    (local $y08 i64) (local $y09 i64) (local $y10 i64) (local $y11 i64) 
    (local $y12 i64) 
    (local $t00 i64) (local $t01 i64) (local $t02 i64) (local $t03 i64) 
    (local $t04 i64) (local $t05 i64) (local $t06 i64) (local $t07 i64) 
    (local $t08 i64) (local $t09 i64) (local $t10 i64) (local $t11 i64) 
    (local $t12 i64) 
    (local.set $y00 (i64.load offset=0 (local.get $y)))
    (local.set $y01 (i64.load offset=8 (local.get $y)))
    (local.set $y02 (i64.load offset=16 (local.get $y)))
    (local.set $y03 (i64.load offset=24 (local.get $y)))
    (local.set $y04 (i64.load offset=32 (local.get $y)))
    (local.set $y05 (i64.load offset=40 (local.get $y)))
    (local.set $y06 (i64.load offset=48 (local.get $y)))
    (local.set $y07 (i64.load offset=56 (local.get $y)))
    (local.set $y08 (i64.load offset=64 (local.get $y)))
    (local.set $y09 (i64.load offset=72 (local.get $y)))
    (local.set $y10 (i64.load offset=80 (local.get $y)))
    (local.set $y11 (i64.load offset=88 (local.get $y)))
    (local.set $y12 (i64.load offset=96 (local.get $y)))
    (local.set $i0 (i32.div_u (local.get $k) (i32.const 30)))
    (local.set $xi0 (i32.shl (i32.const 1) (i32.rem_u (local.get $k) (i32.const 30))))
    (local.set $i0 (i32.mul (local.get $i0) (i32.const 8)))
    (local.set $i (i32.const 0))
    (loop 
      (local.set $xi (i64.extend_i32_u (i32.mul (i32.eq (local.get $i) (local.get $i0)) (local.get $xi0))))
      ;; j = 0, do carry, ignore result below carry
      (local.get $t00)
      (i64.mul (local.get $xi) (local.get $y00))
      i64.add
      (local.set $tmp)
      (local.set $qi (i64.and (i64.mul (i64.const 0x3ffcfffd) (i64.and (local.get $tmp) (i64.const 0x3fffffff))) (i64.const 0x3fffffff)))
      (local.get $tmp)
      (i64.mul (local.get $qi) (i64.const 0x3fffaaab))
      i64.add
      (i64.const 30) i64.shr_u
      ;; j = 1
      (local.get $t01)
      i64.add
      (i64.mul (local.get $xi) (local.get $y01))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x27fbffff))
      i64.add
      (local.set $t00)
      ;; j = 2
      (local.get $t02)
      (i64.mul (local.get $xi) (local.get $y02))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x153ffffb))
      i64.add
      (local.set $t01)
      ;; j = 3
      (local.get $t03)
      (i64.mul (local.get $xi) (local.get $y03))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x2affffac))
      i64.add
      (local.set $t02)
      ;; j = 4
      (local.get $t04)
      (i64.mul (local.get $xi) (local.get $y04))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x30f6241e))
      i64.add
      (local.set $t03)
      ;; j = 5
      (local.get $t05)
      (i64.mul (local.get $xi) (local.get $y05))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x34a83da))
      i64.add
      (local.set $t04)
      ;; j = 6
      (local.get $t06)
      (i64.mul (local.get $xi) (local.get $y06))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x112bf673))
      i64.add
      (local.set $t05)
      ;; j = 7
      (local.get $t07)
      (i64.mul (local.get $xi) (local.get $y07))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x12e13ce1))
      i64.add
      (local.set $t06)
      ;; j = 8, do carry
      (local.get $t08)
      (i64.mul (local.get $xi) (local.get $y08))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x2cd76477))
      i64.add
      (local.tee $tmp) (i64.const 30) i64.shr_u
      (i64.and (local.get $tmp) (i64.const 0x3fffffff))
      (local.set $t07)
      ;; j = 9
      (local.get $t09)
      i64.add
      (i64.mul (local.get $xi) (local.get $y09))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x1ed90d2e))
      i64.add
      (local.set $t08)
      ;; j = 10
      (local.get $t10)
      (i64.mul (local.get $xi) (local.get $y10))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x29a4b1ba))
      i64.add
      (local.set $t09)
      ;; j = 11
      (local.get $t11)
      (i64.mul (local.get $xi) (local.get $y11))
      i64.add
      (i64.mul (local.get $qi) (i64.const 0x3a8e5ff9))
      i64.add
      (local.set $t10)
      ;; j = 12
      (i64.mul (local.get $xi) (local.get $y12))
      (i64.mul (local.get $qi) (i64.const 0x1a0111))
      i64.add
      (local.set $t11)
      (br_if 0 (i32.ne (i32.const 104) (local.tee $i (i32.add (local.get $i) (i32.const 8)))))
    )
    ;; final carrying & storing
    (i64.store offset=0 (local.get $xy) (i64.and (local.get $t00) (i64.const 0x3fffffff)))
    (local.set $t01 (i64.add (local.get $t01) (i64.shr_u (local.get $t00) (i64.const 30))))
    (i64.store offset=8 (local.get $xy) (i64.and (local.get $t01) (i64.const 0x3fffffff)))
    (local.set $t02 (i64.add (local.get $t02) (i64.shr_u (local.get $t01) (i64.const 30))))
    (i64.store offset=16 (local.get $xy) (i64.and (local.get $t02) (i64.const 0x3fffffff)))
    (local.set $t03 (i64.add (local.get $t03) (i64.shr_u (local.get $t02) (i64.const 30))))
    (i64.store offset=24 (local.get $xy) (i64.and (local.get $t03) (i64.const 0x3fffffff)))
    (local.set $t04 (i64.add (local.get $t04) (i64.shr_u (local.get $t03) (i64.const 30))))
    (i64.store offset=32 (local.get $xy) (i64.and (local.get $t04) (i64.const 0x3fffffff)))
    (local.set $t05 (i64.add (local.get $t05) (i64.shr_u (local.get $t04) (i64.const 30))))
    (i64.store offset=40 (local.get $xy) (i64.and (local.get $t05) (i64.const 0x3fffffff)))
    (local.set $t06 (i64.add (local.get $t06) (i64.shr_u (local.get $t05) (i64.const 30))))
    (i64.store offset=48 (local.get $xy) (i64.and (local.get $t06) (i64.const 0x3fffffff)))
    (local.set $t07 (i64.add (local.get $t07) (i64.shr_u (local.get $t06) (i64.const 30))))
    (i64.store offset=56 (local.get $xy) (i64.and (local.get $t07) (i64.const 0x3fffffff)))
    (local.set $t08 (i64.add (local.get $t08) (i64.shr_u (local.get $t07) (i64.const 30))))
    (i64.store offset=64 (local.get $xy) (i64.and (local.get $t08) (i64.const 0x3fffffff)))
    (local.set $t09 (i64.add (local.get $t09) (i64.shr_u (local.get $t08) (i64.const 30))))
    (i64.store offset=72 (local.get $xy) (i64.and (local.get $t09) (i64.const 0x3fffffff)))
    (local.set $t10 (i64.add (local.get $t10) (i64.shr_u (local.get $t09) (i64.const 30))))
    (i64.store offset=80 (local.get $xy) (i64.and (local.get $t10) (i64.const 0x3fffffff)))
    (local.set $t11 (i64.add (local.get $t11) (i64.shr_u (local.get $t10) (i64.const 30))))
    (i64.store offset=88 (local.get $xy) (i64.and (local.get $t11) (i64.const 0x3fffffff)))
    (local.set $t12 (i64.add (local.get $t12) (i64.shr_u (local.get $t11) (i64.const 30))))
    (i64.store offset=96 (local.get $xy) (local.get $t12))
  )
  (export "add" (func $add))
  (func $add (param $out i32) (param $x i32) (param $y i32)
    (local $tmp i64) (local $carry i64)
    ;; i = 0
    (i64.load offset=0 (local.get $x))
    (i64.load offset=0 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 1
    (i64.load offset=8 (local.get $x))
    (i64.load offset=8 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 2
    (i64.load offset=16 (local.get $x))
    (i64.load offset=16 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 3
    (i64.load offset=24 (local.get $x))
    (i64.load offset=24 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 4
    (i64.load offset=32 (local.get $x))
    (i64.load offset=32 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 5
    (i64.load offset=40 (local.get $x))
    (i64.load offset=40 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 6
    (i64.load offset=48 (local.get $x))
    (i64.load offset=48 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 7
    (i64.load offset=56 (local.get $x))
    (i64.load offset=56 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 8
    (i64.load offset=64 (local.get $x))
    (i64.load offset=64 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 9
    (i64.load offset=72 (local.get $x))
    (i64.load offset=72 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 10
    (i64.load offset=80 (local.get $x))
    (i64.load offset=80 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 11
    (i64.load offset=88 (local.get $x))
    (i64.load offset=88 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 12
    (i64.load offset=96 (local.get $x))
    (i64.load offset=96 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (block 
      (local.set $tmp (i64.load offset=96 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x1a0111e)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x1a0111e)))
      (local.set $tmp (i64.load offset=88 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x28e5ff9a)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x28e5ff9a)))
      (local.set $tmp (i64.load offset=80 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x1a4b1ba7)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x1a4b1ba7)))
      (local.set $tmp (i64.load offset=72 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x2d90d2eb)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x2d90d2eb)))
      (local.set $tmp (i64.load offset=64 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0xd764774)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0xd764774)))
      (local.set $tmp (i64.load offset=56 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x2e13ce14)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x2e13ce14)))
      (local.set $tmp (i64.load offset=48 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x12bf6730)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x12bf6730)))
      (local.set $tmp (i64.load offset=40 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x34a83dac)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x34a83dac)))
      (local.set $tmp (i64.load offset=32 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0xf6241ea)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0xf6241ea)))
      (local.set $tmp (i64.load offset=24 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x2ffffac5)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x2ffffac5)))
      (local.set $tmp (i64.load offset=16 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x13ffffb9)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x13ffffb9)))
      (local.set $tmp (i64.load offset=8 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x3fbfffff)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x3fbfffff)))
      (local.set $tmp (i64.load offset=0 (local.get $out)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x3ffaaab0)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x3ffaaab0)))
    )
    (local.set $carry (i64.const 0))
    ;; i = 0
    (i64.load offset=0 (local.get $out))
    (i64.const 0x3ffaaab0)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 1
    (i64.load offset=8 (local.get $out))
    (i64.const 0x3fbfffff)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 2
    (i64.load offset=16 (local.get $out))
    (i64.const 0x13ffffb9)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 3
    (i64.load offset=24 (local.get $out))
    (i64.const 0x2ffffac5)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 4
    (i64.load offset=32 (local.get $out))
    (i64.const 0xf6241ea)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 5
    (i64.load offset=40 (local.get $out))
    (i64.const 0x34a83dac)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 6
    (i64.load offset=48 (local.get $out))
    (i64.const 0x12bf6730)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 7
    (i64.load offset=56 (local.get $out))
    (i64.const 0x2e13ce14)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 8
    (i64.load offset=64 (local.get $out))
    (i64.const 0xd764774)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 9
    (i64.load offset=72 (local.get $out))
    (i64.const 0x2d90d2eb)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 10
    (i64.load offset=80 (local.get $out))
    (i64.const 0x1a4b1ba7)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 11
    (i64.load offset=88 (local.get $out))
    (i64.const 0x28e5ff9a)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 12
    (i64.load offset=96 (local.get $out))
    (i64.const 0x1a0111e)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
  )
  (export "addNoReduce" (func $addNoReduce))
  (func $addNoReduce (param $out i32) (param $x i32) (param $y i32)
    (local $tmp i64) (local $carry i64)
    ;; i = 0
    (i64.load offset=0 (local.get $x))
    (i64.load offset=0 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 1
    (i64.load offset=8 (local.get $x))
    (i64.load offset=8 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 2
    (i64.load offset=16 (local.get $x))
    (i64.load offset=16 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 3
    (i64.load offset=24 (local.get $x))
    (i64.load offset=24 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 4
    (i64.load offset=32 (local.get $x))
    (i64.load offset=32 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 5
    (i64.load offset=40 (local.get $x))
    (i64.load offset=40 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 6
    (i64.load offset=48 (local.get $x))
    (i64.load offset=48 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 7
    (i64.load offset=56 (local.get $x))
    (i64.load offset=56 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 8
    (i64.load offset=64 (local.get $x))
    (i64.load offset=64 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 9
    (i64.load offset=72 (local.get $x))
    (i64.load offset=72 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 10
    (i64.load offset=80 (local.get $x))
    (i64.load offset=80 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 11
    (i64.load offset=88 (local.get $x))
    (i64.load offset=88 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; i = 12
    (i64.load offset=96 (local.get $x))
    (i64.load offset=96 (local.get $y))
    i64.add (local.get $carry) i64.add
    (local.tee $tmp) (i64.const 30) i64.shr_s (local.set $carry)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
  )
  (export "subtract" (func $subtract))
  (func $subtract (param $out i32) (param $x i32) (param $y i32)
    (local $tmp i64)
    ;; i = 0
    (i64.load offset=0 (local.get $x))
    (i64.load offset=0 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 1
    (i64.load offset=8 (local.get $x))
    i64.add
    (i64.load offset=8 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 2
    (i64.load offset=16 (local.get $x))
    i64.add
    (i64.load offset=16 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 3
    (i64.load offset=24 (local.get $x))
    i64.add
    (i64.load offset=24 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 4
    (i64.load offset=32 (local.get $x))
    i64.add
    (i64.load offset=32 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 5
    (i64.load offset=40 (local.get $x))
    i64.add
    (i64.load offset=40 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 6
    (i64.load offset=48 (local.get $x))
    i64.add
    (i64.load offset=48 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 7
    (i64.load offset=56 (local.get $x))
    i64.add
    (i64.load offset=56 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 8
    (i64.load offset=64 (local.get $x))
    i64.add
    (i64.load offset=64 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 9
    (i64.load offset=72 (local.get $x))
    i64.add
    (i64.load offset=72 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 10
    (i64.load offset=80 (local.get $x))
    i64.add
    (i64.load offset=80 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 11
    (i64.load offset=88 (local.get $x))
    i64.add
    (i64.load offset=88 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 12
    (i64.load offset=96 (local.get $x))
    i64.add
    (i64.load offset=96 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.const 0) i64.eq
    if return end
    ;; i = 0
    (i64.const 0x3ffaaab0)
    (i64.load offset=0 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 1
    (i64.const 0x3fbfffff)
    i64.add
    (i64.load offset=8 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 2
    (i64.const 0x13ffffb9)
    i64.add
    (i64.load offset=16 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 3
    (i64.const 0x2ffffac5)
    i64.add
    (i64.load offset=24 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 4
    (i64.const 0xf6241ea)
    i64.add
    (i64.load offset=32 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 5
    (i64.const 0x34a83dac)
    i64.add
    (i64.load offset=40 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 6
    (i64.const 0x12bf6730)
    i64.add
    (i64.load offset=48 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 7
    (i64.const 0x2e13ce14)
    i64.add
    (i64.load offset=56 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 8
    (i64.const 0xd764774)
    i64.add
    (i64.load offset=64 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 9
    (i64.const 0x2d90d2eb)
    i64.add
    (i64.load offset=72 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 10
    (i64.const 0x1a4b1ba7)
    i64.add
    (i64.load offset=80 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 11
    (i64.const 0x28e5ff9a)
    i64.add
    (i64.load offset=88 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 12
    (i64.const 0x1a0111e)
    i64.add
    (i64.load offset=96 (local.get $out))
    i64.add
    (local.set $tmp)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
  )
  (export "subtractNoReduce" (func $subtractNoReduce))
  (func $subtractNoReduce (param $out i32) (param $x i32) (param $y i32)
    (local $tmp i64)
    ;; i = 0
    (i64.load offset=0 (local.get $x))
    (i64.load offset=0 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 1
    (i64.load offset=8 (local.get $x))
    i64.add
    (i64.load offset=8 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 2
    (i64.load offset=16 (local.get $x))
    i64.add
    (i64.load offset=16 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 3
    (i64.load offset=24 (local.get $x))
    i64.add
    (i64.load offset=24 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 4
    (i64.load offset=32 (local.get $x))
    i64.add
    (i64.load offset=32 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 5
    (i64.load offset=40 (local.get $x))
    i64.add
    (i64.load offset=40 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 6
    (i64.load offset=48 (local.get $x))
    i64.add
    (i64.load offset=48 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 7
    (i64.load offset=56 (local.get $x))
    i64.add
    (i64.load offset=56 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 8
    (i64.load offset=64 (local.get $x))
    i64.add
    (i64.load offset=64 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 9
    (i64.load offset=72 (local.get $x))
    i64.add
    (i64.load offset=72 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 10
    (i64.load offset=80 (local.get $x))
    i64.add
    (i64.load offset=80 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 11
    (i64.load offset=88 (local.get $x))
    i64.add
    (i64.load offset=88 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 12
    (i64.load offset=96 (local.get $x))
    i64.add
    (i64.load offset=96 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
  )
  (export "subtractPositive" (func $subtractPositive))
  (func $subtractPositive (param $out i32) (param $x i32) (param $y i32)
    (local $tmp i64)
    ;; i = 0
    (i64.const 0x3ffaaab0)
    (i64.load offset=0 (local.get $x))
    i64.add
    (i64.load offset=0 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=0 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 1
    (i64.const 0x3fbfffff)
    i64.add
    (i64.load offset=8 (local.get $x))
    i64.add
    (i64.load offset=8 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=8 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 2
    (i64.const 0x13ffffb9)
    i64.add
    (i64.load offset=16 (local.get $x))
    i64.add
    (i64.load offset=16 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=16 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 3
    (i64.const 0x2ffffac5)
    i64.add
    (i64.load offset=24 (local.get $x))
    i64.add
    (i64.load offset=24 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=24 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 4
    (i64.const 0xf6241ea)
    i64.add
    (i64.load offset=32 (local.get $x))
    i64.add
    (i64.load offset=32 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=32 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 5
    (i64.const 0x34a83dac)
    i64.add
    (i64.load offset=40 (local.get $x))
    i64.add
    (i64.load offset=40 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=40 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 6
    (i64.const 0x12bf6730)
    i64.add
    (i64.load offset=48 (local.get $x))
    i64.add
    (i64.load offset=48 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=48 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 7
    (i64.const 0x2e13ce14)
    i64.add
    (i64.load offset=56 (local.get $x))
    i64.add
    (i64.load offset=56 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=56 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 8
    (i64.const 0xd764774)
    i64.add
    (i64.load offset=64 (local.get $x))
    i64.add
    (i64.load offset=64 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=64 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 9
    (i64.const 0x2d90d2eb)
    i64.add
    (i64.load offset=72 (local.get $x))
    i64.add
    (i64.load offset=72 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=72 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 10
    (i64.const 0x1a4b1ba7)
    i64.add
    (i64.load offset=80 (local.get $x))
    i64.add
    (i64.load offset=80 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=80 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 11
    (i64.const 0x28e5ff9a)
    i64.add
    (i64.load offset=88 (local.get $x))
    i64.add
    (i64.load offset=88 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=88 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    ;; i = 12
    (i64.const 0x1a0111e)
    i64.add
    (i64.load offset=96 (local.get $x))
    i64.add
    (i64.load offset=96 (local.get $y))
    i64.sub
    (local.set $tmp)
    (i64.store offset=96 (local.get $out) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
  )
  (export "reduce" (func $reduce))
  (func $reduce (param $x i32)
    (local $tmp i64) (local $carry i64)
    (block 
      (local.set $tmp (i64.load offset=96 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x1a0111)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x1a0111)))
      (local.set $tmp (i64.load offset=88 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x3a8e5ff9)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x3a8e5ff9)))
      (local.set $tmp (i64.load offset=80 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x29a4b1ba)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x29a4b1ba)))
      (local.set $tmp (i64.load offset=72 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x1ed90d2e)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x1ed90d2e)))
      (local.set $tmp (i64.load offset=64 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x2cd76477)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x2cd76477)))
      (local.set $tmp (i64.load offset=56 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x12e13ce1)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x12e13ce1)))
      (local.set $tmp (i64.load offset=48 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x112bf673)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x112bf673)))
      (local.set $tmp (i64.load offset=40 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x34a83da)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x34a83da)))
      (local.set $tmp (i64.load offset=32 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x30f6241e)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x30f6241e)))
      (local.set $tmp (i64.load offset=24 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x2affffac)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x2affffac)))
      (local.set $tmp (i64.load offset=16 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x153ffffb)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x153ffffb)))
      (local.set $tmp (i64.load offset=8 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x27fbffff)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x27fbffff)))
      (local.set $tmp (i64.load offset=0 (local.get $x)))
      (br_if 1 (i64.lt_u (local.get $tmp) (i64.const 0x3fffaaab)))
      (br_if 0 (i64.ne (local.get $tmp) (i64.const 0x3fffaaab)))
    )
    (local.set $carry (i64.const 0))
    ;; i = 0
    (i64.load offset=0 (local.get $x))
    (i64.const 0x3fffaaab)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=0 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 1
    (i64.load offset=8 (local.get $x))
    (i64.const 0x27fbffff)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=8 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 2
    (i64.load offset=16 (local.get $x))
    (i64.const 0x153ffffb)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=16 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 3
    (i64.load offset=24 (local.get $x))
    (i64.const 0x2affffac)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=24 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 4
    (i64.load offset=32 (local.get $x))
    (i64.const 0x30f6241e)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=32 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 5
    (i64.load offset=40 (local.get $x))
    (i64.const 0x34a83da)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=40 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 6
    (i64.load offset=48 (local.get $x))
    (i64.const 0x112bf673)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=48 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 7
    (i64.load offset=56 (local.get $x))
    (i64.const 0x12e13ce1)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=56 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 8
    (i64.load offset=64 (local.get $x))
    (i64.const 0x2cd76477)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=64 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 9
    (i64.load offset=72 (local.get $x))
    (i64.const 0x1ed90d2e)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=72 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 10
    (i64.load offset=80 (local.get $x))
    (i64.const 0x29a4b1ba)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=80 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 11
    (i64.load offset=88 (local.get $x))
    (i64.const 0x3a8e5ff9)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=88 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
    ;; i = 12
    (i64.load offset=96 (local.get $x))
    (i64.const 0x1a0111)
    i64.sub
    (local.get $carry)
    i64.add
    (local.set $tmp)
    (i64.store offset=96 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $carry (i64.shr_s (local.get $tmp) (i64.const 30)))
  )
  (export "makeOdd" (func $makeOdd))
  (func $makeOdd (param $u i32) (param $s i32) (result i32)
    (local $k i64) (local $k0 i32) (local $l i64) (local $tmp i64)
    (local.set $k (i64.ctz (i64.load offset=0 (local.get $u))))
    (i64.eqz (local.get $k))
    if
      (i32.const 0)
      return
    end
    (block 
      (loop 
        (br_if 1 (i64.ne (local.get $k) (i64.const 64)))
        (memory.copy (local.get $u) (i32.add (local.get $u) (i32.const 8)) (i32.const 96))
        (i64.store offset=96 (local.get $u) (i64.const 0))
        (memory.copy (i32.add (local.get $s) (i32.const 8)) (local.get $s) (i32.const 96))
        (i64.store offset=0 (local.get $s) (i64.const 0))
        (local.set $k0 (i32.add (local.get $k0) (i32.const 30)))
        (local.set $k (i64.ctz (i64.load offset=0 (local.get $u))))
        (br 0)
      )
    )
    (local.set $l (i64.sub (i64.const 30) (local.get $k)))
    ;; u >> k
    (local.set $tmp (i64.load offset=0 (local.get $u)))
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=8 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=0)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=16 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=8)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=24 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=16)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=32 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=24)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=40 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=32)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=48 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=40)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=56 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=48)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=64 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=56)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=72 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=64)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=80 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=72)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=88 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=80)
    (local.get $u)
    (i64.shr_u (local.get $tmp) (local.get $k))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=96 (local.get $u))) (local.get $l)) (i64.const 0x3fffffff))
    i64.or
    (i64.store offset=88)
    (i64.store offset=96 (local.get $u) (i64.shr_u (local.get $tmp) (local.get $k)))
    ;; s << k
    (local.set $tmp (i64.load offset=96 (local.get $s)))
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=88 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=96)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=80 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=88)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=72 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=80)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=64 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=72)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=56 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=64)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=48 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=56)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=40 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=48)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=32 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=40)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=24 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=32)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=16 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=24)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=8 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=16)
    (local.get $s)
    (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff))
    (i64.shr_u (local.tee $tmp (i64.load offset=0 (local.get $s))) (local.get $l))
    i64.or
    (i64.store offset=8)
    (i64.store offset=0 (local.get $s) (i64.and (i64.shl (local.get $tmp) (local.get $k)) (i64.const 0x3fffffff)))
    ;; return k
    (i32.add (local.get $k0) (i32.wrap_i64 (local.get $k)))
  )
  (export "isEqual" (func $isEqual))
  (func $isEqual (param $x i32) (param $y i32) (result i32)
    (i64.ne (i64.load offset=0 (local.get $x)) (i64.load offset=0 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=8 (local.get $x)) (i64.load offset=8 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=16 (local.get $x)) (i64.load offset=16 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=24 (local.get $x)) (i64.load offset=24 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=32 (local.get $x)) (i64.load offset=32 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=40 (local.get $x)) (i64.load offset=40 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=48 (local.get $x)) (i64.load offset=48 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=56 (local.get $x)) (i64.load offset=56 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=64 (local.get $x)) (i64.load offset=64 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=72 (local.get $x)) (i64.load offset=72 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=80 (local.get $x)) (i64.load offset=80 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=88 (local.get $x)) (i64.load offset=88 (local.get $y)))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=96 (local.get $x)) (i64.load offset=96 (local.get $y)))
    if (return (i32.const 0)) end
    (i32.const 1)
  )
  (export "isEqualNegative" (func $isEqualNegative))
  (func $isEqualNegative (param $x i32) (param $y i32) (result i32)
    (local $tmp i64)
    (i64.load offset=0 (local.get $x))
    (i64.load offset=0 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x3fffaaab)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=8 (local.get $x))
    i64.add
    (i64.load offset=8 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x27fbffff)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=16 (local.get $x))
    i64.add
    (i64.load offset=16 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x153ffffb)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=24 (local.get $x))
    i64.add
    (i64.load offset=24 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x2affffac)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=32 (local.get $x))
    i64.add
    (i64.load offset=32 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x30f6241e)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=40 (local.get $x))
    i64.add
    (i64.load offset=40 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x34a83da)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=48 (local.get $x))
    i64.add
    (i64.load offset=48 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x112bf673)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=56 (local.get $x))
    i64.add
    (i64.load offset=56 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x12e13ce1)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=64 (local.get $x))
    i64.add
    (i64.load offset=64 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x2cd76477)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=72 (local.get $x))
    i64.add
    (i64.load offset=72 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x1ed90d2e)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=80 (local.get $x))
    i64.add
    (i64.load offset=80 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x29a4b1ba)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=88 (local.get $x))
    i64.add
    (i64.load offset=88 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x3a8e5ff9)
    i64.ne
    if (return (i32.const 0)) end
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.load offset=96 (local.get $x))
    i64.add
    (i64.load offset=96 (local.get $y))
    i64.add
    (local.set $tmp)
    (i64.and (local.get $tmp) (i64.const 0x3fffffff))
    (i64.const 0x1a0111)
    i64.ne
    if (return (i32.const 0)) end
    (i32.const 1)
  )
  (export "isZero" (func $isZero))
  (func $isZero (param $x i32) (result i32)
    (i64.ne (i64.load offset=0 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=8 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=16 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=24 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=32 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=40 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=48 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=56 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=64 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=72 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=80 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=88 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i64.ne (i64.load offset=96 (local.get $x)) (i64.const 0))
    if (return (i32.const 0)) end
    (i32.const 1)
  )
  (export "isGreater" (func $isGreater))
  (func $isGreater (param $x i32) (param $y i32) (result i32)
    (local $xi i64) (local $yi i64)
    (block 
      (local.tee $xi (i64.load offset=96 (local.get $x)))
      (local.tee $yi (i64.load offset=96 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=88 (local.get $x)))
      (local.tee $yi (i64.load offset=88 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=80 (local.get $x)))
      (local.tee $yi (i64.load offset=80 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=72 (local.get $x)))
      (local.tee $yi (i64.load offset=72 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=64 (local.get $x)))
      (local.tee $yi (i64.load offset=64 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=56 (local.get $x)))
      (local.tee $yi (i64.load offset=56 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=48 (local.get $x)))
      (local.tee $yi (i64.load offset=48 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=40 (local.get $x)))
      (local.tee $yi (i64.load offset=40 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=32 (local.get $x)))
      (local.tee $yi (i64.load offset=32 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=24 (local.get $x)))
      (local.tee $yi (i64.load offset=24 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=16 (local.get $x)))
      (local.tee $yi (i64.load offset=16 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=8 (local.get $x)))
      (local.tee $yi (i64.load offset=8 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
      (local.tee $xi (i64.load offset=0 (local.get $x)))
      (local.tee $yi (i64.load offset=0 (local.get $y)))
      i64.gt_u
      if (return (i32.const 1)) end
      (br_if 0 (i64.ne (local.get $xi) (local.get $yi)))
    )
    (i32.const 0)
  )
  (export "copy" (func $copy))
  (func $copy (param $x i32) (param $y i32)
    (memory.copy (local.get $x) (local.get $y) (i32.const 104))
  )
  (export "toPackedBytes" (func $toPackedBytes))
  ;; converts 13x30-bit representation (1 int64 per 30-bit limb) to packed 48-byte representation
  (func $toPackedBytes (param $bytes i32) (param $x i32)
    (local $tmp i64)
    (i64.shl (i64.load offset=0 (local.get $x)) (i64.const 0))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=0 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 24)))
    (i64.shl (i64.load offset=8 (local.get $x)) (i64.const 6))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=3 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=16 (local.get $x)) (i64.const 4))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=7 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=24 (local.get $x)) (i64.const 2))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=11 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=32 (local.get $x)) (i64.const 0))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=15 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 24)))
    (i64.shl (i64.load offset=40 (local.get $x)) (i64.const 6))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=18 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=48 (local.get $x)) (i64.const 4))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=22 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=56 (local.get $x)) (i64.const 2))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=26 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=64 (local.get $x)) (i64.const 0))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=30 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 24)))
    (i64.shl (i64.load offset=72 (local.get $x)) (i64.const 6))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=33 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=80 (local.get $x)) (i64.const 4))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=37 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=88 (local.get $x)) (i64.const 2))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=41 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 32)))
    (i64.shl (i64.load offset=96 (local.get $x)) (i64.const 0))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=45 (local.get $bytes) (i64.and (local.get $tmp) (i64.const 0xffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 24)))
  )
  (export "fromPackedBytes" (func $fromPackedBytes))
  ;; recovers 13x30-bit representation (1 int64 per 30-bit limb) from packed 48-byte representation
  (func $fromPackedBytes (param $x i32) (param $bytes i32)
    (local $tmp i64) (local $chunk i64)
    (local.set $tmp (i64.const 0))
    (i64.shl (local.tee $chunk (i64.load offset=0 (local.get $bytes))) (i64.const 0))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=0 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 30)))
    (i64.store offset=8 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 30)))
    (i64.shl (local.tee $chunk (i64.load offset=8 (local.get $bytes))) (i64.const 4))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=16 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 26)))
    (i64.store offset=24 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 30)))
    (i64.shl (local.tee $chunk (i64.load offset=16 (local.get $bytes))) (i64.const 8))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=32 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 22)))
    (i64.store offset=40 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 30)))
    (i64.shl (local.tee $chunk (i64.load offset=24 (local.get $bytes))) (i64.const 12))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=48 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 18)))
    (i64.store offset=56 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 30)))
    (i64.shl (local.tee $chunk (i64.load offset=32 (local.get $bytes))) (i64.const 16))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=64 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 14)))
    (i64.store offset=72 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 30)))
    (i64.shl (local.tee $chunk (i64.load offset=40 (local.get $bytes))) (i64.const 20))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=80 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 10)))
    (i64.store offset=88 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $tmp) (i64.const 30)))
    (i64.shl (local.tee $chunk (i64.load offset=48 (local.get $bytes))) (i64.const 24))
    (local.get $tmp)
    i64.or
    (local.set $tmp)
    (i64.store offset=96 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (local.set $tmp (i64.shr_u (local.get $chunk) (i64.const 6)))
  )
  (export "barrett" (func $barrett))
  (func $barrett (param $x i32)
    (local $tmp i64) (local $carry i64)
    (local $l00 i64) (local $l01 i64) (local $l02 i64) (local $l03 i64) 
    (local $l04 i64) (local $l05 i64) (local $l06 i64) (local $l07 i64) 
    (local $l08 i64) (local $l09 i64) (local $l10 i64) (local $l11 i64) 
    (local $l12 i64) 
    (local $lp00 i64) (local $lp01 i64) (local $lp02 i64) (local $lp03 i64) 
    (local $lp04 i64) (local $lp05 i64) (local $lp06 i64) (local $lp07 i64) 
    (local $lp08 i64) (local $lp09 i64) (local $lp10 i64) (local $lp11 i64) 
    (local $lp12 i64) 
    ;; extract l := highest 380 bits of x = x >> 380
    ;; load l := x[12..26] = (x >> 12*30)
    ;; then do l >>= 20 (because 20 = 380 - 12*30)
    (local.set $tmp (i64.load offset=96 (local.get $x)))
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=104 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l00)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=112 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l01)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=120 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l02)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=128 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l03)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=136 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l04)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=144 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l05)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=152 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l06)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=160 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l07)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=168 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l08)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=176 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l09)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=184 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l10)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=192 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l11)
    (i64.shr_u (local.get $tmp) (i64.const 20))
    (i64.and (i64.shl (local.tee $tmp (i64.load offset=200 (local.get $x))) (i64.const 10)) (i64.const 0x3fffffff))
    i64.or
    (local.set $l12)
    ;; l = [l * m / 2^N]; the first 11 output limbs are neglected
    (i64.mul (local.get $l00) (i64.const 0x33cc9e45))
    (i64.mul (local.get $l01) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x371e0286))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x1de74e65))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x3167a058))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x3c701ec4))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x3e207f56))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x1646e8ba))
    i64.add
    (i64.const 30) i64.shr_u
    (i64.mul (local.get $l00) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l01) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x371e0286))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x1de74e65))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x3167a058))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x3c701ec4))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x3e207f56))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x1646e8ba))
    i64.add
    (i64.const 30) i64.shr_u
    (i64.mul (local.get $l01) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x371e0286))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x1de74e65))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x3167a058))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x3c701ec4))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x3e207f56))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l00)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l02) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x371e0286))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x1de74e65))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x3167a058))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x3c701ec4))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l01)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l03) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x371e0286))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x1de74e65))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x3167a058))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l02)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l04) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x371e0286))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x1de74e65))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l03)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l05) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x2a52f7d1))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x371e0286))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l04)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l06) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x341ff6a0))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x2a52f7d1))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l05)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l07) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x28a46e09))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x341ff6a0))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l06)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l08) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x30cc7a6b))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x28a46e09))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l07)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l09) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x338a0406))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x30cc7a6b))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l08)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l10) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x33cc9e45))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x338a0406))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l09)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l11) (i64.const 0x2760d74b))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x33cc9e45))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l10)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (i64.mul (local.get $l12) (i64.const 0x2760d74b))
    i64.add
    (local.tee $tmp) (i64.const 0x3fffffff) i64.and
    (local.set $l11)
    (i64.shr_u (local.get $tmp) (i64.const 30))
    (local.set $l12)
    ;; (l*p)[0..n]
    (i64.mul (local.get $l00) (i64.const 0x3fffaaab))
    (local.set $lp00)
    (i64.mul (local.get $l00) (i64.const 0x27fbffff))
    (i64.mul (local.get $l01) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp01)
    (i64.mul (local.get $l00) (i64.const 0x153ffffb))
    (i64.mul (local.get $l01) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp02)
    (i64.mul (local.get $l00) (i64.const 0x2affffac))
    (i64.mul (local.get $l01) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp03)
    (i64.mul (local.get $l00) (i64.const 0x30f6241e))
    (i64.mul (local.get $l01) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp04)
    (i64.mul (local.get $l00) (i64.const 0x34a83da))
    (i64.mul (local.get $l01) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp05)
    (i64.mul (local.get $l00) (i64.const 0x112bf673))
    (i64.mul (local.get $l01) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp06)
    (i64.mul (local.get $l00) (i64.const 0x12e13ce1))
    (i64.mul (local.get $l01) (i64.const 0x112bf673))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp07)
    (i64.mul (local.get $l00) (i64.const 0x2cd76477))
    (i64.mul (local.get $l01) (i64.const 0x12e13ce1))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x112bf673))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp08)
    (i64.mul (local.get $l00) (i64.const 0x1ed90d2e))
    (i64.mul (local.get $l01) (i64.const 0x2cd76477))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x12e13ce1))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x112bf673))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp09)
    (i64.mul (local.get $l00) (i64.const 0x29a4b1ba))
    (i64.mul (local.get $l01) (i64.const 0x1ed90d2e))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x2cd76477))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x12e13ce1))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x112bf673))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp10)
    (i64.mul (local.get $l00) (i64.const 0x3a8e5ff9))
    (i64.mul (local.get $l01) (i64.const 0x29a4b1ba))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x1ed90d2e))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x2cd76477))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x12e13ce1))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x112bf673))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp11)
    (i64.mul (local.get $l00) (i64.const 0x1a0111))
    (i64.mul (local.get $l01) (i64.const 0x3a8e5ff9))
    i64.add
    (i64.mul (local.get $l02) (i64.const 0x29a4b1ba))
    i64.add
    (i64.mul (local.get $l03) (i64.const 0x1ed90d2e))
    i64.add
    (i64.mul (local.get $l04) (i64.const 0x2cd76477))
    i64.add
    (i64.mul (local.get $l05) (i64.const 0x12e13ce1))
    i64.add
    (i64.mul (local.get $l06) (i64.const 0x112bf673))
    i64.add
    (i64.mul (local.get $l07) (i64.const 0x34a83da))
    i64.add
    (i64.mul (local.get $l08) (i64.const 0x30f6241e))
    i64.add
    (i64.mul (local.get $l09) (i64.const 0x2affffac))
    i64.add
    (i64.mul (local.get $l10) (i64.const 0x153ffffb))
    i64.add
    (i64.mul (local.get $l11) (i64.const 0x27fbffff))
    i64.add
    (i64.mul (local.get $l12) (i64.const 0x3fffaaab))
    i64.add
    (local.set $lp12)
    ;; x|lo = x - l*p to the low n limbs of x
    (i64.load offset=0 (local.get $x))
    (local.get $lp00)
    i64.sub
    (local.set $tmp)
    (i64.store offset=0 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=8 (local.get $x))
    i64.add
    (local.get $lp01)
    i64.sub
    (local.set $tmp)
    (i64.store offset=8 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=16 (local.get $x))
    i64.add
    (local.get $lp02)
    i64.sub
    (local.set $tmp)
    (i64.store offset=16 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=24 (local.get $x))
    i64.add
    (local.get $lp03)
    i64.sub
    (local.set $tmp)
    (i64.store offset=24 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=32 (local.get $x))
    i64.add
    (local.get $lp04)
    i64.sub
    (local.set $tmp)
    (i64.store offset=32 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=40 (local.get $x))
    i64.add
    (local.get $lp05)
    i64.sub
    (local.set $tmp)
    (i64.store offset=40 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=48 (local.get $x))
    i64.add
    (local.get $lp06)
    i64.sub
    (local.set $tmp)
    (i64.store offset=48 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=56 (local.get $x))
    i64.add
    (local.get $lp07)
    i64.sub
    (local.set $tmp)
    (i64.store offset=56 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=64 (local.get $x))
    i64.add
    (local.get $lp08)
    i64.sub
    (local.set $tmp)
    (i64.store offset=64 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=72 (local.get $x))
    i64.add
    (local.get $lp09)
    i64.sub
    (local.set $tmp)
    (i64.store offset=72 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=80 (local.get $x))
    i64.add
    (local.get $lp10)
    i64.sub
    (local.set $tmp)
    (i64.store offset=80 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=88 (local.get $x))
    i64.add
    (local.get $lp11)
    i64.sub
    (local.set $tmp)
    (i64.store offset=88 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_s (local.get $tmp) (i64.const 30))
    (i64.load offset=96 (local.get $x))
    i64.add
    (local.get $lp12)
    i64.sub
    (local.set $tmp)
    (i64.store offset=96 (local.get $x) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    ;; x|hi = l
    (i64.store offset=104 (local.get $x) (local.get $l00))
    (i64.store offset=112 (local.get $x) (local.get $l01))
    (i64.store offset=120 (local.get $x) (local.get $l02))
    (i64.store offset=128 (local.get $x) (local.get $l03))
    (i64.store offset=136 (local.get $x) (local.get $l04))
    (i64.store offset=144 (local.get $x) (local.get $l05))
    (i64.store offset=152 (local.get $x) (local.get $l06))
    (i64.store offset=160 (local.get $x) (local.get $l07))
    (i64.store offset=168 (local.get $x) (local.get $l08))
    (i64.store offset=176 (local.get $x) (local.get $l09))
    (i64.store offset=184 (local.get $x) (local.get $l10))
    (i64.store offset=192 (local.get $x) (local.get $l11))
    (i64.store offset=200 (local.get $x) (local.get $l12))
  )
  (export "multiplySchoolbook" (func $multiplySchoolbook))
  (func $multiplySchoolbook (param $xy i32) (param $x i32) (param $y i32)
    (local $tmp i64)
    (local $x00 i64) (local $x01 i64) (local $x02 i64) (local $x03 i64) 
    (local $x04 i64) (local $x05 i64) (local $x06 i64) (local $x07 i64) 
    (local $x08 i64) (local $x09 i64) (local $x10 i64) (local $x11 i64) 
    (local $x12 i64) 
    (local $y00 i64) (local $y01 i64) (local $y02 i64) (local $y03 i64) 
    (local $y04 i64) (local $y05 i64) (local $y06 i64) (local $y07 i64) 
    (local $y08 i64) (local $y09 i64) (local $y10 i64) (local $y11 i64) 
    (local $y12 i64) 
    (local.set $x00 (i64.load offset=0 (local.get $x)))
    (local.set $x01 (i64.load offset=8 (local.get $x)))
    (local.set $x02 (i64.load offset=16 (local.get $x)))
    (local.set $x03 (i64.load offset=24 (local.get $x)))
    (local.set $x04 (i64.load offset=32 (local.get $x)))
    (local.set $x05 (i64.load offset=40 (local.get $x)))
    (local.set $x06 (i64.load offset=48 (local.get $x)))
    (local.set $x07 (i64.load offset=56 (local.get $x)))
    (local.set $x08 (i64.load offset=64 (local.get $x)))
    (local.set $x09 (i64.load offset=72 (local.get $x)))
    (local.set $x10 (i64.load offset=80 (local.get $x)))
    (local.set $x11 (i64.load offset=88 (local.get $x)))
    (local.set $x12 (i64.load offset=96 (local.get $x)))
    (local.set $y00 (i64.load offset=0 (local.get $y)))
    (local.set $y01 (i64.load offset=8 (local.get $y)))
    (local.set $y02 (i64.load offset=16 (local.get $y)))
    (local.set $y03 (i64.load offset=24 (local.get $y)))
    (local.set $y04 (i64.load offset=32 (local.get $y)))
    (local.set $y05 (i64.load offset=40 (local.get $y)))
    (local.set $y06 (i64.load offset=48 (local.get $y)))
    (local.set $y07 (i64.load offset=56 (local.get $y)))
    (local.set $y08 (i64.load offset=64 (local.get $y)))
    (local.set $y09 (i64.load offset=72 (local.get $y)))
    (local.set $y10 (i64.load offset=80 (local.get $y)))
    (local.set $y11 (i64.load offset=88 (local.get $y)))
    (local.set $y12 (i64.load offset=96 (local.get $y)))
    ;; multiply in 13x13 steps
    ;; k = 0
    (i64.mul (local.get $x00) (local.get $y00))
    (local.set $tmp)
    (i64.store offset=0 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 1
    (i64.mul (local.get $x00) (local.get $y01))
    i64.add
    (i64.mul (local.get $x01) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=8 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 2
    (i64.mul (local.get $x00) (local.get $y02))
    i64.add
    (i64.mul (local.get $x01) (local.get $y01))
    i64.add
    (i64.mul (local.get $x02) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=16 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 3
    (i64.mul (local.get $x00) (local.get $y03))
    i64.add
    (i64.mul (local.get $x01) (local.get $y02))
    i64.add
    (i64.mul (local.get $x02) (local.get $y01))
    i64.add
    (i64.mul (local.get $x03) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=24 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 4
    (i64.mul (local.get $x00) (local.get $y04))
    i64.add
    (i64.mul (local.get $x01) (local.get $y03))
    i64.add
    (i64.mul (local.get $x02) (local.get $y02))
    i64.add
    (i64.mul (local.get $x03) (local.get $y01))
    i64.add
    (i64.mul (local.get $x04) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=32 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 5
    (i64.mul (local.get $x00) (local.get $y05))
    i64.add
    (i64.mul (local.get $x01) (local.get $y04))
    i64.add
    (i64.mul (local.get $x02) (local.get $y03))
    i64.add
    (i64.mul (local.get $x03) (local.get $y02))
    i64.add
    (i64.mul (local.get $x04) (local.get $y01))
    i64.add
    (i64.mul (local.get $x05) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=40 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 6
    (i64.mul (local.get $x00) (local.get $y06))
    i64.add
    (i64.mul (local.get $x01) (local.get $y05))
    i64.add
    (i64.mul (local.get $x02) (local.get $y04))
    i64.add
    (i64.mul (local.get $x03) (local.get $y03))
    i64.add
    (i64.mul (local.get $x04) (local.get $y02))
    i64.add
    (i64.mul (local.get $x05) (local.get $y01))
    i64.add
    (i64.mul (local.get $x06) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=48 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 7
    (i64.mul (local.get $x00) (local.get $y07))
    i64.add
    (i64.mul (local.get $x01) (local.get $y06))
    i64.add
    (i64.mul (local.get $x02) (local.get $y05))
    i64.add
    (i64.mul (local.get $x03) (local.get $y04))
    i64.add
    (i64.mul (local.get $x04) (local.get $y03))
    i64.add
    (i64.mul (local.get $x05) (local.get $y02))
    i64.add
    (i64.mul (local.get $x06) (local.get $y01))
    i64.add
    (i64.mul (local.get $x07) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=56 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 8
    (i64.mul (local.get $x00) (local.get $y08))
    i64.add
    (i64.mul (local.get $x01) (local.get $y07))
    i64.add
    (i64.mul (local.get $x02) (local.get $y06))
    i64.add
    (i64.mul (local.get $x03) (local.get $y05))
    i64.add
    (i64.mul (local.get $x04) (local.get $y04))
    i64.add
    (i64.mul (local.get $x05) (local.get $y03))
    i64.add
    (i64.mul (local.get $x06) (local.get $y02))
    i64.add
    (i64.mul (local.get $x07) (local.get $y01))
    i64.add
    (i64.mul (local.get $x08) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=64 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 9
    (i64.mul (local.get $x00) (local.get $y09))
    i64.add
    (i64.mul (local.get $x01) (local.get $y08))
    i64.add
    (i64.mul (local.get $x02) (local.get $y07))
    i64.add
    (i64.mul (local.get $x03) (local.get $y06))
    i64.add
    (i64.mul (local.get $x04) (local.get $y05))
    i64.add
    (i64.mul (local.get $x05) (local.get $y04))
    i64.add
    (i64.mul (local.get $x06) (local.get $y03))
    i64.add
    (i64.mul (local.get $x07) (local.get $y02))
    i64.add
    (i64.mul (local.get $x08) (local.get $y01))
    i64.add
    (i64.mul (local.get $x09) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=72 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 10
    (i64.mul (local.get $x00) (local.get $y10))
    i64.add
    (i64.mul (local.get $x01) (local.get $y09))
    i64.add
    (i64.mul (local.get $x02) (local.get $y08))
    i64.add
    (i64.mul (local.get $x03) (local.get $y07))
    i64.add
    (i64.mul (local.get $x04) (local.get $y06))
    i64.add
    (i64.mul (local.get $x05) (local.get $y05))
    i64.add
    (i64.mul (local.get $x06) (local.get $y04))
    i64.add
    (i64.mul (local.get $x07) (local.get $y03))
    i64.add
    (i64.mul (local.get $x08) (local.get $y02))
    i64.add
    (i64.mul (local.get $x09) (local.get $y01))
    i64.add
    (i64.mul (local.get $x10) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=80 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 11
    (i64.mul (local.get $x00) (local.get $y11))
    i64.add
    (i64.mul (local.get $x01) (local.get $y10))
    i64.add
    (i64.mul (local.get $x02) (local.get $y09))
    i64.add
    (i64.mul (local.get $x03) (local.get $y08))
    i64.add
    (i64.mul (local.get $x04) (local.get $y07))
    i64.add
    (i64.mul (local.get $x05) (local.get $y06))
    i64.add
    (i64.mul (local.get $x06) (local.get $y05))
    i64.add
    (i64.mul (local.get $x07) (local.get $y04))
    i64.add
    (i64.mul (local.get $x08) (local.get $y03))
    i64.add
    (i64.mul (local.get $x09) (local.get $y02))
    i64.add
    (i64.mul (local.get $x10) (local.get $y01))
    i64.add
    (i64.mul (local.get $x11) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=88 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 12
    (i64.mul (local.get $x00) (local.get $y12))
    i64.add
    (i64.mul (local.get $x01) (local.get $y11))
    i64.add
    (i64.mul (local.get $x02) (local.get $y10))
    i64.add
    (i64.mul (local.get $x03) (local.get $y09))
    i64.add
    (i64.mul (local.get $x04) (local.get $y08))
    i64.add
    (i64.mul (local.get $x05) (local.get $y07))
    i64.add
    (i64.mul (local.get $x06) (local.get $y06))
    i64.add
    (i64.mul (local.get $x07) (local.get $y05))
    i64.add
    (i64.mul (local.get $x08) (local.get $y04))
    i64.add
    (i64.mul (local.get $x09) (local.get $y03))
    i64.add
    (i64.mul (local.get $x10) (local.get $y02))
    i64.add
    (i64.mul (local.get $x11) (local.get $y01))
    i64.add
    (i64.mul (local.get $x12) (local.get $y00))
    i64.add
    (local.set $tmp)
    (i64.store offset=96 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 13
    (i64.mul (local.get $x01) (local.get $y12))
    i64.add
    (i64.mul (local.get $x02) (local.get $y11))
    i64.add
    (i64.mul (local.get $x03) (local.get $y10))
    i64.add
    (i64.mul (local.get $x04) (local.get $y09))
    i64.add
    (i64.mul (local.get $x05) (local.get $y08))
    i64.add
    (i64.mul (local.get $x06) (local.get $y07))
    i64.add
    (i64.mul (local.get $x07) (local.get $y06))
    i64.add
    (i64.mul (local.get $x08) (local.get $y05))
    i64.add
    (i64.mul (local.get $x09) (local.get $y04))
    i64.add
    (i64.mul (local.get $x10) (local.get $y03))
    i64.add
    (i64.mul (local.get $x11) (local.get $y02))
    i64.add
    (i64.mul (local.get $x12) (local.get $y01))
    i64.add
    (local.set $tmp)
    (i64.store offset=104 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 14
    (i64.mul (local.get $x02) (local.get $y12))
    i64.add
    (i64.mul (local.get $x03) (local.get $y11))
    i64.add
    (i64.mul (local.get $x04) (local.get $y10))
    i64.add
    (i64.mul (local.get $x05) (local.get $y09))
    i64.add
    (i64.mul (local.get $x06) (local.get $y08))
    i64.add
    (i64.mul (local.get $x07) (local.get $y07))
    i64.add
    (i64.mul (local.get $x08) (local.get $y06))
    i64.add
    (i64.mul (local.get $x09) (local.get $y05))
    i64.add
    (i64.mul (local.get $x10) (local.get $y04))
    i64.add
    (i64.mul (local.get $x11) (local.get $y03))
    i64.add
    (i64.mul (local.get $x12) (local.get $y02))
    i64.add
    (local.set $tmp)
    (i64.store offset=112 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 15
    (i64.mul (local.get $x03) (local.get $y12))
    i64.add
    (i64.mul (local.get $x04) (local.get $y11))
    i64.add
    (i64.mul (local.get $x05) (local.get $y10))
    i64.add
    (i64.mul (local.get $x06) (local.get $y09))
    i64.add
    (i64.mul (local.get $x07) (local.get $y08))
    i64.add
    (i64.mul (local.get $x08) (local.get $y07))
    i64.add
    (i64.mul (local.get $x09) (local.get $y06))
    i64.add
    (i64.mul (local.get $x10) (local.get $y05))
    i64.add
    (i64.mul (local.get $x11) (local.get $y04))
    i64.add
    (i64.mul (local.get $x12) (local.get $y03))
    i64.add
    (local.set $tmp)
    (i64.store offset=120 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 16
    (i64.mul (local.get $x04) (local.get $y12))
    i64.add
    (i64.mul (local.get $x05) (local.get $y11))
    i64.add
    (i64.mul (local.get $x06) (local.get $y10))
    i64.add
    (i64.mul (local.get $x07) (local.get $y09))
    i64.add
    (i64.mul (local.get $x08) (local.get $y08))
    i64.add
    (i64.mul (local.get $x09) (local.get $y07))
    i64.add
    (i64.mul (local.get $x10) (local.get $y06))
    i64.add
    (i64.mul (local.get $x11) (local.get $y05))
    i64.add
    (i64.mul (local.get $x12) (local.get $y04))
    i64.add
    (local.set $tmp)
    (i64.store offset=128 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 17
    (i64.mul (local.get $x05) (local.get $y12))
    i64.add
    (i64.mul (local.get $x06) (local.get $y11))
    i64.add
    (i64.mul (local.get $x07) (local.get $y10))
    i64.add
    (i64.mul (local.get $x08) (local.get $y09))
    i64.add
    (i64.mul (local.get $x09) (local.get $y08))
    i64.add
    (i64.mul (local.get $x10) (local.get $y07))
    i64.add
    (i64.mul (local.get $x11) (local.get $y06))
    i64.add
    (i64.mul (local.get $x12) (local.get $y05))
    i64.add
    (local.set $tmp)
    (i64.store offset=136 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 18
    (i64.mul (local.get $x06) (local.get $y12))
    i64.add
    (i64.mul (local.get $x07) (local.get $y11))
    i64.add
    (i64.mul (local.get $x08) (local.get $y10))
    i64.add
    (i64.mul (local.get $x09) (local.get $y09))
    i64.add
    (i64.mul (local.get $x10) (local.get $y08))
    i64.add
    (i64.mul (local.get $x11) (local.get $y07))
    i64.add
    (i64.mul (local.get $x12) (local.get $y06))
    i64.add
    (local.set $tmp)
    (i64.store offset=144 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 19
    (i64.mul (local.get $x07) (local.get $y12))
    i64.add
    (i64.mul (local.get $x08) (local.get $y11))
    i64.add
    (i64.mul (local.get $x09) (local.get $y10))
    i64.add
    (i64.mul (local.get $x10) (local.get $y09))
    i64.add
    (i64.mul (local.get $x11) (local.get $y08))
    i64.add
    (i64.mul (local.get $x12) (local.get $y07))
    i64.add
    (local.set $tmp)
    (i64.store offset=152 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 20
    (i64.mul (local.get $x08) (local.get $y12))
    i64.add
    (i64.mul (local.get $x09) (local.get $y11))
    i64.add
    (i64.mul (local.get $x10) (local.get $y10))
    i64.add
    (i64.mul (local.get $x11) (local.get $y09))
    i64.add
    (i64.mul (local.get $x12) (local.get $y08))
    i64.add
    (local.set $tmp)
    (i64.store offset=160 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 21
    (i64.mul (local.get $x09) (local.get $y12))
    i64.add
    (i64.mul (local.get $x10) (local.get $y11))
    i64.add
    (i64.mul (local.get $x11) (local.get $y10))
    i64.add
    (i64.mul (local.get $x12) (local.get $y09))
    i64.add
    (local.set $tmp)
    (i64.store offset=168 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 22
    (i64.mul (local.get $x10) (local.get $y12))
    i64.add
    (i64.mul (local.get $x11) (local.get $y11))
    i64.add
    (i64.mul (local.get $x12) (local.get $y10))
    i64.add
    (local.set $tmp)
    (i64.store offset=176 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 23
    (i64.mul (local.get $x11) (local.get $y12))
    i64.add
    (i64.mul (local.get $x12) (local.get $y11))
    i64.add
    (local.set $tmp)
    (i64.store offset=184 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 24
    (i64.mul (local.get $x12) (local.get $y12))
    i64.add
    (local.set $tmp)
    (i64.store offset=192 (local.get $xy) (i64.and (local.get $tmp) (i64.const 0x3fffffff)))
    (i64.shr_u (local.get $tmp) (i64.const 30))
    ;; k = 25
    (local.set $tmp)
    (i64.store offset=200 (local.get $xy) (local.get $tmp))
    (call $barrett (local.get $xy))
  )
  (export "benchMultiplyBarrett" (func $benchMultiplyBarrett))
  (func $benchMultiplyBarrett (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $multiplySchoolbook (local.get $x) (local.get $x) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "multiplyKaratsuba" (func $multiplyKaratsuba))
  (func $multiplyKaratsuba (param $xy i32) (param $x i32) (param $y i32)
    (local $tmp i64) (local $carry i64)
    (local $x00 i64) (local $x01 i64) (local $x02 i64) (local $x03 i64) 
    (local $x04 i64) (local $x05 i64) (local $x06 i64) (local $x07 i64) 
    (local $x08 i64) (local $x09 i64) (local $x10 i64) (local $x11 i64) 
    (local $x12 i64) 
    (local $y00 i64) (local $y01 i64) (local $y02 i64) (local $y03 i64) 
    (local $y04 i64) (local $y05 i64) (local $y06 i64) (local $y07 i64) 
    (local $y08 i64) (local $y09 i64) (local $y10 i64) (local $y11 i64) 
    (local $y12 i64) 
    (local $z00 i64) (local $z01 i64) (local $z02 i64) (local $z03 i64) 
    (local $z04 i64) (local $z05 i64) (local $z06 i64) (local $z07 i64) 
    (local $z08 i64) (local $z09 i64) (local $z10 i64) (local $z11 i64) 
    (local $z12 i64) (local $z13 i64) (local $z14 i64) (local $z15 i64) 
    (local $z16 i64) (local $z17 i64) (local $z18 i64) (local $z19 i64) 
    (local $z20 i64) (local $z21 i64) (local $z22 i64) (local $z23 i64) 
    (local $z24 i64) (local $z25 i64) 
    (local $w00 i64) (local $w01 i64) (local $w02 i64) (local $w03 i64) 
    (local $w04 i64) (local $w05 i64) (local $w06 i64) (local $w07 i64) 
    (local $w08 i64) (local $w09 i64) (local $w10 i64) (local $w11 i64) 
    (local $w12 i64) (local $w13 i64) (local $w14 i64) (local $w15 i64) 
    (local $w16 i64) (local $w17 i64) (local $w18 i64) (local $w19 i64) 
    (local $w20 i64) (local $w21 i64) (local $w22 i64) (local $w23 i64) 
    (local $w24 i64) (local $w25 i64) 
    (local.set $x00 (i64.load offset=0 (local.get $x)))
    (local.set $x01 (i64.load offset=8 (local.get $x)))
    (local.set $x02 (i64.load offset=16 (local.get $x)))
    (local.set $x03 (i64.load offset=24 (local.get $x)))
    (local.set $x04 (i64.load offset=32 (local.get $x)))
    (local.set $x05 (i64.load offset=40 (local.get $x)))
    (local.set $x06 (i64.load offset=48 (local.get $x)))
    (local.set $x07 (i64.load offset=56 (local.get $x)))
    (local.set $x08 (i64.load offset=64 (local.get $x)))
    (local.set $x09 (i64.load offset=72 (local.get $x)))
    (local.set $x10 (i64.load offset=80 (local.get $x)))
    (local.set $x11 (i64.load offset=88 (local.get $x)))
    (local.set $x12 (i64.load offset=96 (local.get $x)))
    (local.set $y00 (i64.load offset=0 (local.get $y)))
    (local.set $y01 (i64.load offset=8 (local.get $y)))
    (local.set $y02 (i64.load offset=16 (local.get $y)))
    (local.set $y03 (i64.load offset=24 (local.get $y)))
    (local.set $y04 (i64.load offset=32 (local.get $y)))
    (local.set $y05 (i64.load offset=40 (local.get $y)))
    (local.set $y06 (i64.load offset=48 (local.get $y)))
    (local.set $y07 (i64.load offset=56 (local.get $y)))
    (local.set $y08 (i64.load offset=64 (local.get $y)))
    (local.set $y09 (i64.load offset=72 (local.get $y)))
    (local.set $y10 (i64.load offset=80 (local.get $y)))
    (local.set $y11 (i64.load offset=88 (local.get $y)))
    (local.set $y12 (i64.load offset=96 (local.get $y)))
    ;; multiply z = x0*x0 in 6x6 steps
    ;; k = 0
    (i64.mul (local.get $x00) (local.get $y00))
    (local.set $z00)
    ;; k = 1
    (i64.mul (local.get $x00) (local.get $y01))
    (i64.mul (local.get $x01) (local.get $y00))
    i64.add
    (local.set $z01)
    ;; k = 2
    (i64.mul (local.get $x00) (local.get $y02))
    (i64.mul (local.get $x01) (local.get $y01))
    i64.add
    (i64.mul (local.get $x02) (local.get $y00))
    i64.add
    (local.set $z02)
    ;; k = 3
    (i64.mul (local.get $x00) (local.get $y03))
    (i64.mul (local.get $x01) (local.get $y02))
    i64.add
    (i64.mul (local.get $x02) (local.get $y01))
    i64.add
    (i64.mul (local.get $x03) (local.get $y00))
    i64.add
    (local.set $z03)
    ;; k = 4
    (i64.mul (local.get $x00) (local.get $y04))
    (i64.mul (local.get $x01) (local.get $y03))
    i64.add
    (i64.mul (local.get $x02) (local.get $y02))
    i64.add
    (i64.mul (local.get $x03) (local.get $y01))
    i64.add
    (i64.mul (local.get $x04) (local.get $y00))
    i64.add
    (local.set $z04)
    ;; k = 5
    (i64.mul (local.get $x00) (local.get $y05))
    (i64.mul (local.get $x01) (local.get $y04))
    i64.add
    (i64.mul (local.get $x02) (local.get $y03))
    i64.add
    (i64.mul (local.get $x03) (local.get $y02))
    i64.add
    (i64.mul (local.get $x04) (local.get $y01))
    i64.add
    (i64.mul (local.get $x05) (local.get $y00))
    i64.add
    (local.set $z05)
    ;; k = 6
    (i64.mul (local.get $x01) (local.get $y05))
    (i64.mul (local.get $x02) (local.get $y04))
    i64.add
    (i64.mul (local.get $x03) (local.get $y03))
    i64.add
    (i64.mul (local.get $x04) (local.get $y02))
    i64.add
    (i64.mul (local.get $x05) (local.get $y01))
    i64.add
    (local.set $z06)
    ;; k = 7
    (i64.mul (local.get $x02) (local.get $y05))
    (i64.mul (local.get $x03) (local.get $y04))
    i64.add
    (i64.mul (local.get $x04) (local.get $y03))
    i64.add
    (i64.mul (local.get $x05) (local.get $y02))
    i64.add
    (local.set $z07)
    ;; k = 8
    (i64.mul (local.get $x03) (local.get $y05))
    (i64.mul (local.get $x04) (local.get $y04))
    i64.add
    (i64.mul (local.get $x05) (local.get $y03))
    i64.add
    (local.set $z08)
    ;; k = 9
    (i64.mul (local.get $x04) (local.get $y05))
    (i64.mul (local.get $x05) (local.get $y04))
    i64.add
    (local.set $z09)
    ;; k = 10
    (i64.mul (local.get $x05) (local.get $y05))
    (local.set $z10)
    ;; k = 11
    ;; multiply w = x1*x1 in 7x7 steps
    ;; k = 0
    (i64.mul (local.get $x06) (local.get $y06))
    (local.set $w00)
    ;; k = 1
    (i64.mul (local.get $x06) (local.get $y07))
    (i64.mul (local.get $x07) (local.get $y06))
    i64.add
    (local.set $w01)
    ;; k = 2
    (i64.mul (local.get $x06) (local.get $y08))
    (i64.mul (local.get $x07) (local.get $y07))
    i64.add
    (i64.mul (local.get $x08) (local.get $y06))
    i64.add
    (local.set $w02)
    ;; k = 3
    (i64.mul (local.get $x06) (local.get $y09))
    (i64.mul (local.get $x07) (local.get $y08))
    i64.add
    (i64.mul (local.get $x08) (local.get $y07))
    i64.add
    (i64.mul (local.get $x09) (local.get $y06))
    i64.add
    (local.set $w03)
    ;; k = 4
    (i64.mul (local.get $x06) (local.get $y10))
    (i64.mul (local.get $x07) (local.get $y09))
    i64.add
    (i64.mul (local.get $x08) (local.get $y08))
    i64.add
    (i64.mul (local.get $x09) (local.get $y07))
    i64.add
    (i64.mul (local.get $x10) (local.get $y06))
    i64.add
    (local.set $w04)
    ;; k = 5
    (i64.mul (local.get $x06) (local.get $y11))
    (i64.mul (local.get $x07) (local.get $y10))
    i64.add
    (i64.mul (local.get $x08) (local.get $y09))
    i64.add
    (i64.mul (local.get $x09) (local.get $y08))
    i64.add
    (i64.mul (local.get $x10) (local.get $y07))
    i64.add
    (i64.mul (local.get $x11) (local.get $y06))
    i64.add
    (local.set $w05)
    ;; k = 6
    (i64.mul (local.get $x06) (local.get $y12))
    (i64.mul (local.get $x07) (local.get $y11))
    i64.add
    (i64.mul (local.get $x08) (local.get $y10))
    i64.add
    (i64.mul (local.get $x09) (local.get $y09))
    i64.add
    (i64.mul (local.get $x10) (local.get $y08))
    i64.add
    (i64.mul (local.get $x11) (local.get $y07))
    i64.add
    (i64.mul (local.get $x12) (local.get $y06))
    i64.add
    (local.set $w06)
    ;; k = 7
    (i64.mul (local.get $x07) (local.get $y12))
    (i64.mul (local.get $x08) (local.get $y11))
    i64.add
    (i64.mul (local.get $x09) (local.get $y10))
    i64.add
    (i64.mul (local.get $x10) (local.get $y09))
    i64.add
    (i64.mul (local.get $x11) (local.get $y08))
    i64.add
    (i64.mul (local.get $x12) (local.get $y07))
    i64.add
    (local.set $w07)
    ;; k = 8
    (i64.mul (local.get $x08) (local.get $y12))
    (i64.mul (local.get $x09) (local.get $y11))
    i64.add
    (i64.mul (local.get $x10) (local.get $y10))
    i64.add
    (i64.mul (local.get $x11) (local.get $y09))
    i64.add
    (i64.mul (local.get $x12) (local.get $y08))
    i64.add
    (local.set $w08)
    ;; k = 9
    (i64.mul (local.get $x09) (local.get $y12))
    (i64.mul (local.get $x10) (local.get $y11))
    i64.add
    (i64.mul (local.get $x11) (local.get $y10))
    i64.add
    (i64.mul (local.get $x12) (local.get $y09))
    i64.add
    (local.set $w09)
    ;; k = 10
    (i64.mul (local.get $x10) (local.get $y12))
    (i64.mul (local.get $x11) (local.get $y11))
    i64.add
    (i64.mul (local.get $x12) (local.get $y10))
    i64.add
    (local.set $w10)
    ;; k = 11
    (i64.mul (local.get $x11) (local.get $y12))
    (i64.mul (local.get $x12) (local.get $y11))
    i64.add
    (local.set $w11)
    ;; k = 12
    (i64.mul (local.get $x12) (local.get $y12))
    (local.set $w12)
    ;; k = 13
    ;; compute z = l^m*x1*x1 - x0*x0 = l^m*w - z
    (local.set $z06 (i64.sub (local.get $w00) (local.get $z06)))
    (local.set $z07 (i64.sub (local.get $w01) (local.get $z07)))
    (local.set $z08 (i64.sub (local.get $w02) (local.get $z08)))
    (local.set $z09 (i64.sub (local.get $w03) (local.get $z09)))
    (local.set $z10 (i64.sub (local.get $w04) (local.get $z10)))
    (local.set $z11 (i64.sub (local.get $w05) (local.get $z11)))
    (local.set $z12 (i64.sub (local.get $w06) (local.get $z12)))
    (local.set $z13 (i64.sub (local.get $w07) (local.get $z13)))
    (local.set $z14 (i64.sub (local.get $w08) (local.get $z14)))
    (local.set $z15 (i64.sub (local.get $w09) (local.get $z15)))
    (local.set $z16 (i64.sub (local.get $w10) (local.get $z16)))
    (local.set $z17 (i64.sub (local.get $w11) (local.get $z17)))
    (local.set $z18 (i64.sub (local.get $w12) (local.get $z18)))
    (local.set $z19 (i64.sub (local.get $w13) (local.get $z19)))
    ;; compute w = l^m*z - z = (l^m - 1)(l^m*x1*x1 - x0*x0)
    (local.set $w00 (local.get $z00))
    (local.set $w01 (local.get $z01))
    (local.set $w02 (local.get $z02))
    (local.set $w03 (local.get $z03))
    (local.set $w04 (local.get $z04))
    (local.set $w05 (local.get $z05))
    (local.set $w06 (i64.add (local.get $z00) (local.get $z06)))
    (local.set $w07 (i64.add (local.get $z01) (local.get $z07)))
    (local.set $w08 (i64.add (local.get $z02) (local.get $z08)))
    (local.set $w09 (i64.add (local.get $z03) (local.get $z09)))
    (local.set $w10 (i64.add (local.get $z04) (local.get $z10)))
    (local.set $w11 (i64.add (local.get $z05) (local.get $z11)))
    (local.set $w12 (i64.sub (local.get $z06) (local.get $z12)))
    (local.set $w13 (i64.sub (local.get $z07) (local.get $z13)))
    (local.set $w14 (i64.sub (local.get $z08) (local.get $z14)))
    (local.set $w15 (i64.sub (local.get $z09) (local.get $z15)))
    (local.set $w16 (i64.sub (local.get $z10) (local.get $z16)))
    (local.set $w17 (i64.sub (local.get $z11) (local.get $z17)))
    (local.set $w18 (i64.sub (local.get $z12) (local.get $z18)))
    (local.set $w19 (i64.sub (local.get $z13) (local.get $z19)))
    (local.set $w20 (local.get $z14))
    (local.set $w21 (local.get $z15))
    (local.set $w22 (local.get $z16))
    (local.set $w23 (local.get $z17))
    (local.set $w24 (local.get $z18))
    (local.set $w25 (local.get $z19))
    ;; x1 += x0, y1 += y0
    (local.set $x06 (i64.add (local.get $x06) (local.get $x00)))
    (local.set $y06 (i64.add (local.get $y06) (local.get $y00)))
    (local.set $x07 (i64.add (local.get $x07) (local.get $x01)))
    (local.set $y07 (i64.add (local.get $y07) (local.get $y01)))
    (local.set $x08 (i64.add (local.get $x08) (local.get $x02)))
    (local.set $y08 (i64.add (local.get $y08) (local.get $y02)))
    (local.set $x09 (i64.add (local.get $x09) (local.get $x03)))
    (local.set $y09 (i64.add (local.get $y09) (local.get $y03)))
    (local.set $x10 (i64.add (local.get $x10) (local.get $x04)))
    (local.set $y10 (i64.add (local.get $y10) (local.get $y04)))
    (local.set $x11 (i64.add (local.get $x11) (local.get $x05)))
    (local.set $y11 (i64.add (local.get $y11) (local.get $y05)))
    ;; multiply z = (x0 + x1)*(y0 + y1) in 7x7 steps
    ;; k = 0
    (i64.mul (local.get $x06) (local.get $y06))
    (local.set $z00)
    ;; k = 1
    (i64.mul (local.get $x06) (local.get $y07))
    (i64.mul (local.get $x07) (local.get $y06))
    i64.add
    (local.set $z01)
    ;; k = 2
    (i64.mul (local.get $x06) (local.get $y08))
    (i64.mul (local.get $x07) (local.get $y07))
    i64.add
    (i64.mul (local.get $x08) (local.get $y06))
    i64.add
    (local.set $z02)
    ;; k = 3
    (i64.mul (local.get $x06) (local.get $y09))
    (i64.mul (local.get $x07) (local.get $y08))
    i64.add
    (i64.mul (local.get $x08) (local.get $y07))
    i64.add
    (i64.mul (local.get $x09) (local.get $y06))
    i64.add
    (local.set $z03)
    ;; k = 4
    (i64.mul (local.get $x06) (local.get $y10))
    (i64.mul (local.get $x07) (local.get $y09))
    i64.add
    (i64.mul (local.get $x08) (local.get $y08))
    i64.add
    (i64.mul (local.get $x09) (local.get $y07))
    i64.add
    (i64.mul (local.get $x10) (local.get $y06))
    i64.add
    (local.set $z04)
    ;; k = 5
    (i64.mul (local.get $x06) (local.get $y11))
    (i64.mul (local.get $x07) (local.get $y10))
    i64.add
    (i64.mul (local.get $x08) (local.get $y09))
    i64.add
    (i64.mul (local.get $x09) (local.get $y08))
    i64.add
    (i64.mul (local.get $x10) (local.get $y07))
    i64.add
    (i64.mul (local.get $x11) (local.get $y06))
    i64.add
    (local.set $z05)
    ;; k = 6
    (i64.mul (local.get $x06) (local.get $y12))
    (i64.mul (local.get $x07) (local.get $y11))
    i64.add
    (i64.mul (local.get $x08) (local.get $y10))
    i64.add
    (i64.mul (local.get $x09) (local.get $y09))
    i64.add
    (i64.mul (local.get $x10) (local.get $y08))
    i64.add
    (i64.mul (local.get $x11) (local.get $y07))
    i64.add
    (i64.mul (local.get $x12) (local.get $y06))
    i64.add
    (local.set $z06)
    ;; k = 7
    (i64.mul (local.get $x07) (local.get $y12))
    (i64.mul (local.get $x08) (local.get $y11))
    i64.add
    (i64.mul (local.get $x09) (local.get $y10))
    i64.add
    (i64.mul (local.get $x10) (local.get $y09))
    i64.add
    (i64.mul (local.get $x11) (local.get $y08))
    i64.add
    (i64.mul (local.get $x12) (local.get $y07))
    i64.add
    (local.set $z07)
    ;; k = 8
    (i64.mul (local.get $x08) (local.get $y12))
    (i64.mul (local.get $x09) (local.get $y11))
    i64.add
    (i64.mul (local.get $x10) (local.get $y10))
    i64.add
    (i64.mul (local.get $x11) (local.get $y09))
    i64.add
    (i64.mul (local.get $x12) (local.get $y08))
    i64.add
    (local.set $z08)
    ;; k = 9
    (i64.mul (local.get $x09) (local.get $y12))
    (i64.mul (local.get $x10) (local.get $y11))
    i64.add
    (i64.mul (local.get $x11) (local.get $y10))
    i64.add
    (i64.mul (local.get $x12) (local.get $y09))
    i64.add
    (local.set $z09)
    ;; k = 10
    (i64.mul (local.get $x10) (local.get $y12))
    (i64.mul (local.get $x11) (local.get $y11))
    i64.add
    (i64.mul (local.get $x12) (local.get $y10))
    i64.add
    (local.set $z10)
    ;; k = 11
    (i64.mul (local.get $x11) (local.get $y12))
    (i64.mul (local.get $x12) (local.get $y11))
    i64.add
    (local.set $z11)
    ;; k = 12
    (i64.mul (local.get $x12) (local.get $y12))
    (local.set $z12)
    ;; k = 13
    ;; compute w = w + l^m*z = x*y
    (local.set $w06 (i64.add (local.get $w06) (local.get $z00)))
    (local.set $w07 (i64.add (local.get $w07) (local.get $z01)))
    (local.set $w08 (i64.add (local.get $w08) (local.get $z02)))
    (local.set $w09 (i64.add (local.get $w09) (local.get $z03)))
    (local.set $w10 (i64.add (local.get $w10) (local.get $z04)))
    (local.set $w11 (i64.add (local.get $w11) (local.get $z05)))
    (local.set $w12 (i64.add (local.get $w12) (local.get $z06)))
    (local.set $w13 (i64.add (local.get $w13) (local.get $z07)))
    (local.set $w14 (i64.add (local.get $w14) (local.get $z08)))
    (local.set $w15 (i64.add (local.get $w15) (local.get $z09)))
    (local.set $w16 (i64.add (local.get $w16) (local.get $z10)))
    (local.set $w17 (i64.add (local.get $w17) (local.get $z11)))
    (local.set $w18 (i64.add (local.get $w18) (local.get $z12)))
    (local.set $w19 (i64.add (local.get $w19) (local.get $z13)))
    ;; xy = carry(z)
    (local.set $tmp (i64.shr_s (local.get $w00) (i64.const 30)))
    (i64.store offset=0 (local.get $xy) (i64.and (local.get $w00) (i64.const 0x3fffffff)))
    (local.set $w01 (i64.add (local.get $w01) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w01) (i64.const 30)))
    (i64.store offset=8 (local.get $xy) (i64.and (local.get $w01) (i64.const 0x3fffffff)))
    (local.set $w02 (i64.add (local.get $w02) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w02) (i64.const 30)))
    (i64.store offset=16 (local.get $xy) (i64.and (local.get $w02) (i64.const 0x3fffffff)))
    (local.set $w03 (i64.add (local.get $w03) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w03) (i64.const 30)))
    (i64.store offset=24 (local.get $xy) (i64.and (local.get $w03) (i64.const 0x3fffffff)))
    (local.set $w04 (i64.add (local.get $w04) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w04) (i64.const 30)))
    (i64.store offset=32 (local.get $xy) (i64.and (local.get $w04) (i64.const 0x3fffffff)))
    (local.set $w05 (i64.add (local.get $w05) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w05) (i64.const 30)))
    (i64.store offset=40 (local.get $xy) (i64.and (local.get $w05) (i64.const 0x3fffffff)))
    (local.set $w06 (i64.add (local.get $w06) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w06) (i64.const 30)))
    (i64.store offset=48 (local.get $xy) (i64.and (local.get $w06) (i64.const 0x3fffffff)))
    (local.set $w07 (i64.add (local.get $w07) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w07) (i64.const 30)))
    (i64.store offset=56 (local.get $xy) (i64.and (local.get $w07) (i64.const 0x3fffffff)))
    (local.set $w08 (i64.add (local.get $w08) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w08) (i64.const 30)))
    (i64.store offset=64 (local.get $xy) (i64.and (local.get $w08) (i64.const 0x3fffffff)))
    (local.set $w09 (i64.add (local.get $w09) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w09) (i64.const 30)))
    (i64.store offset=72 (local.get $xy) (i64.and (local.get $w09) (i64.const 0x3fffffff)))
    (local.set $w10 (i64.add (local.get $w10) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w10) (i64.const 30)))
    (i64.store offset=80 (local.get $xy) (i64.and (local.get $w10) (i64.const 0x3fffffff)))
    (local.set $w11 (i64.add (local.get $w11) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w11) (i64.const 30)))
    (i64.store offset=88 (local.get $xy) (i64.and (local.get $w11) (i64.const 0x3fffffff)))
    (local.set $w12 (i64.add (local.get $w12) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w12) (i64.const 30)))
    (i64.store offset=96 (local.get $xy) (i64.and (local.get $w12) (i64.const 0x3fffffff)))
    (local.set $w13 (i64.add (local.get $w13) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w13) (i64.const 30)))
    (i64.store offset=104 (local.get $xy) (i64.and (local.get $w13) (i64.const 0x3fffffff)))
    (local.set $w14 (i64.add (local.get $w14) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w14) (i64.const 30)))
    (i64.store offset=112 (local.get $xy) (i64.and (local.get $w14) (i64.const 0x3fffffff)))
    (local.set $w15 (i64.add (local.get $w15) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w15) (i64.const 30)))
    (i64.store offset=120 (local.get $xy) (i64.and (local.get $w15) (i64.const 0x3fffffff)))
    (local.set $w16 (i64.add (local.get $w16) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w16) (i64.const 30)))
    (i64.store offset=128 (local.get $xy) (i64.and (local.get $w16) (i64.const 0x3fffffff)))
    (local.set $w17 (i64.add (local.get $w17) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w17) (i64.const 30)))
    (i64.store offset=136 (local.get $xy) (i64.and (local.get $w17) (i64.const 0x3fffffff)))
    (local.set $w18 (i64.add (local.get $w18) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w18) (i64.const 30)))
    (i64.store offset=144 (local.get $xy) (i64.and (local.get $w18) (i64.const 0x3fffffff)))
    (local.set $w19 (i64.add (local.get $w19) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w19) (i64.const 30)))
    (i64.store offset=152 (local.get $xy) (i64.and (local.get $w19) (i64.const 0x3fffffff)))
    (local.set $w20 (i64.add (local.get $w20) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w20) (i64.const 30)))
    (i64.store offset=160 (local.get $xy) (i64.and (local.get $w20) (i64.const 0x3fffffff)))
    (local.set $w21 (i64.add (local.get $w21) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w21) (i64.const 30)))
    (i64.store offset=168 (local.get $xy) (i64.and (local.get $w21) (i64.const 0x3fffffff)))
    (local.set $w22 (i64.add (local.get $w22) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w22) (i64.const 30)))
    (i64.store offset=176 (local.get $xy) (i64.and (local.get $w22) (i64.const 0x3fffffff)))
    (local.set $w23 (i64.add (local.get $w23) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w23) (i64.const 30)))
    (i64.store offset=184 (local.get $xy) (i64.and (local.get $w23) (i64.const 0x3fffffff)))
    (local.set $w24 (i64.add (local.get $w24) (local.get $tmp)))
    (local.set $tmp (i64.shr_s (local.get $w24) (i64.const 30)))
    (i64.store offset=192 (local.get $xy) (i64.and (local.get $w24) (i64.const 0x3fffffff)))
    (local.set $w25 (i64.add (local.get $w25) (local.get $tmp)))
    (i64.store offset=200 (local.get $xy) (i64.and (local.get $w25) (i64.const 0x3fffffff)))
    (call $barrett (local.get $xy))
  )
  (export "benchMultiplyKaratsuba" (func $benchMultiplyKaratsuba))
  (func $benchMultiplyKaratsuba (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $multiplyKaratsuba (local.get $x) (local.get $x) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (global $beta i32 (i32.const 208))
  (data (i32.const 208)
    "\81\71\90\1c\00\00\00\00"
    "\86\e4\bd\3c\00\00\00\00"
    "\3e\4c\57\26\00\00\00\00"
    "\ec\75\24\33\00\00\00\00"
    "\1b\bc\3e\1c\00\00\00\00"
    "\64\68\ee\39\00\00\00\00"
    "\56\a8\ff\16\00\00\00\00"
    "\ff\99\34\2c\00\00\00\00"
    "\16\bd\50\05\00\00\00\00"
    "\30\ac\cb\14\00\00\00\00"
    "\86\8c\d1\17\00\00\00\00"
    "\f7\59\59\21\00\00\00\00"
    "\d4\c6\09\00\00\00\00\00"
  )
  (export "endomorphism" (func $endomorphism))
  (func $endomorphism (param $x_out i32) (param $x i32)
    (local $y i32) (local $y_out i32)
    (local.set $y (i32.add (local.get $x) (i32.const 104)))
    (local.set $y_out (i32.add (local.get $x_out) (i32.const 104)))
    (call $multiply (local.get $x_out) (local.get $x) (global.get $beta))
    (call $copy (local.get $y_out) (local.get $y))
  )
  (export "benchMultiply" (func $benchMultiply))
  (func $benchMultiply (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $multiply (local.get $x) (local.get $x) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "benchSquare" (func $benchSquare))
  (func $benchSquare (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $square (local.get $x) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "benchMultiplyUnrolled" (func $benchMultiplyUnrolled))
  (func $benchMultiplyUnrolled (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $multiplyUnrolled (local.get $x) (local.get $x) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "benchMultiplyDifference" (func $benchMultiplyDifference))
  (func $benchMultiplyDifference (param $x i32) (param $y i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $multiplyDifference (local.get $x) (local.get $x) (local.get $x) (local.get $y))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "benchAdd" (func $benchAdd))
  (func $benchAdd (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $add (local.get $x) (local.get $x) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (export "benchSubtract" (func $benchSubtract))
  (func $benchSubtract (param $z i32) (param $x i32) (param $N i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (loop 
      (call $subtract (local.get $z) (local.get $z) (local.get $x))
      (br_if 0 (i32.ne (local.get $N) (local.tee $i (i32.add (local.get $i) (i32.const 1)))))
    )
  )
  (global $dataOffset i32 (i32.const 0x138))
)
