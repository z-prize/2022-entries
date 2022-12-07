module.exports = function buildUtility(module, fnName) {
    // Loads an i64 scalar pArr[index].
    function buildLoadI64() {
        const f = module.addFunction(fnName + "_loadI64");
        // Pointer to a 1-d array with i64 elements
        f.addParam("pArr", "i32");
        // Index
        f.addParam("index", "i32");
        f.setReturnType("i64");
        const c = f.getCodeBuilder();
        f.addCode(
            c.i64_load(
                c.i32_add(
                    c.getLocal("pArr"),
                    c.i32_shl(
                        c.getLocal("index"),
                        c.i32_const(3),
                    ),
                ),
            ),
        )
    }

    // Stores an i64 scalar at pArr[index].
    function buildStoreI64() {
        const f = module.addFunction(fnName + "_storeI64");
        // Pointer to a 1-d array with i64 elements
        f.addParam("pArr", "i32");
        // Index
        f.addParam("index", "i32");
        // Value
        f.addParam("value", "i64");
        const c = f.getCodeBuilder();
        f.addCode(
            c.i64_store(
                c.i32_add(
                    c.getLocal("pArr"),
                    c.i32_shl(
                        c.getLocal("index"),
                        c.i32_const(3),
                    ),
                ),
                c.getLocal("value"),
            ),
        )
    }

    // Loads an i32 scalar pArr[index].
    function buildLoadI32() {
        const f = module.addFunction(fnName + "_loadI32");
        // Pointer to a 1-d array with i32 elements
        f.addParam("pArr", "i32");
        // Index
        f.addParam("index", "i32");
        f.setReturnType("i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.i32_load(c.i32_add(
                c.getLocal("pArr"),
                c.i32_shl(
                    c.getLocal("index"),
                    c.i32_const(2),
                ),
            )),
        )
    }

    // Stores an i32 scalar at pArr[index].
    function buildStoreI32() {
        const f = module.addFunction(fnName + "_storeI32");
        // Pointer to a 1-d array with i32 elements
        f.addParam("pArr", "i32");
        // Index
        f.addParam("index", "i32");
        // Value
        f.addParam("value", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.i32_store(
                c.i32_add(
                    c.getLocal("pArr"),
                    c.i32_shl(
                        c.getLocal("index"),
                        c.i32_const(2),
                    ),
                ),
                c.getLocal("value"),
            ),
        )
    }

    // Allocates a memory of `size` that are pointed to by `pointer`.
    function buildAllocateMemory() {
        const f = module.addFunction(fnName + "_allocateMemory");
        // Number of bytes to be allocated
        f.addParam("size", "i32");
        // An empty pointer
        f.addLocal("pointer", "i32");
        f.setReturnType("i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal("pointer", c.i32_load(c.i32_const(0))),
            c.i32_store(
                c.i32_const(0),
                c.i32_add(
                    c.getLocal("pointer"),
                    c.getLocal("size"),
                ),
            ),
            c.getLocal("pointer"),
        );
    }

    // Computes `pArr[i] += v` given a pointer `pArr` to an array of `i32`, index `i`, and an i32 value `v`.
    //
    // Note
    // This function does not check if `i` is out-of-bound for `pArr`.
    function buildAddAssignI32InMemoryUncheck() {
        const f = module.addFunction(fnName + "_addAssignI32InMemoryUncheck");
        // A pointer to an array of `i32`.
        f.addParam("pArr", "i32");
        // Index
        f.addParam("i", "i32");
        // Value
        f.addParam("v", "i32");
        // pointer to `pArr[i]`
        f.addLocal("pArrI", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal(
                "pArrI",
                c.i32_add(
                    c.getLocal("pArr"),
                    c.i32_shl(
                        c.getLocal("i"),
                        c.i32_const(2),
                    ),
                ),
            ),
            c.i32_store(
                c.getLocal("pArrI"),
                c.i32_add(
                    c.i32_load(c.getLocal("pArrI")),
                    c.getLocal("v"),
                ),
            ),
        );
    }

    // Initiates an array to a default value.
    function buildInitializeI32() {
        const f = module.addFunction(fnName + "_initializeI32");
        // Pointer to an array
        f.addParam("pArr", "i32");
        // Length of the array
        f.addParam("length", "i32");
        // Default value
        f.addParam("default", "i32");
        // Index
        f.addLocal("i", "i32");
        const c = f.getCodeBuilder();
        // pArr[i] = 0 for all 0 <= i < length
        f.addCode(
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.i32_store(
                    c.i32_add(
                        c.getLocal("pArr"),
                        c.i32_shl(
                            c.getLocal("i"),
                            c.i32_const(2),
                        ),
                    ),
                    c.getLocal("default"),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        )
    }

    // Initiates an array to a default value.
    function buildInitializeI64() {
        const f = module.addFunction(fnName + "_initializeI64");
        // Pointer to an array
        f.addParam("pArr", "i32");
        // Length of the array
        f.addParam("length", "i32");
        // default value
        f.addParam("default", "i64");
        // index
        f.addLocal("i", "i32");
        const c = f.getCodeBuilder();
        // pArr[i] = 0 for all 0 <= i < length
        f.addCode(
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.i64_store(
                    c.i32_add(
                        c.getLocal("pArr"),
                        c.i32_shl(
                            c.getLocal("i"),
                            c.i32_const(3),
                        ),
                    ),
                    c.getLocal("default"),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        )
    }

    // Gets the maximum in an i32 array pointed by `pArr` with length `length`.
    function buildMaxArrayValue() {
        const f = module.addFunction(fnName + "_maxArrayValue");
        // Pointer to an array
        f.addParam("pArr", "i32");
        // Length of the array
        f.addParam("length", "i32");
        f.setReturnType("i32");
        // Max value
        f.addLocal("max", "i32");
        // Index
        f.addLocal("i", "i32");
        // Temporary value
        f.addLocal("tmp", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            // max = 0
            // for (i = 0; i < length; i++) {
            //      if(pArr[i] > max) {
            //          max = pArr[i]
            //      }
            // }
            c.setLocal("max", c.i32_const(0)),
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.setLocal("tmp",
                    c.call(fnName + "_loadI32",
                        c.getLocal("pArr"),
                        c.getLocal("i"),
                    ),
                ),
                c.if(
                    c.i32_gt_s(
                        c.getLocal("tmp"),
                        c.getLocal("max"),
                    ),
                    c.setLocal("max", c.getLocal("tmp")),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
            c.getLocal("max"),
        );
    }

    // Copies data from `pInputArr` to `pOutputArr`
    function buildCopyArray() {
        const f = module.addFunction(fnName + "_copyArray");
        // Pointer to the input array
        f.addParam("pInputArr", "i32");
        // Length of the array
        f.addParam("length", "i32");
        // Pointer to the output array
        f.addParam("pOutputArr", "i32");
        // Index
        f.addLocal("i", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            // for (i = 0; i < length; i++) {
            //      pOutputArr[i] = pInputArr[i]
            // }
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.call(fnName + "_storeI32",
                    c.getLocal("pOutputArr"),
                    c.getLocal("i"),
                    c.call(fnName + "_loadI32",
                        c.getLocal("pInputArr"),
                        c.getLocal("i"),
                    ),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        );
    }

    // Given a number `n`, counts the number of significant bits.
    // For example, if n = 5 (i.e., 00000000000000000000000000000101), the output is 3
    function buildGetMSB() {
        const f = module.addFunction(fnName + "_getMsb");
        f.addParam("n", "i32");
        f.setReturnType("i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.i32_sub(
                c.i32_const(32),
                c.i32_clz(c.getLocal("n")),
            ),
        );
    }

    // Given a point pArr to a 2-d array, returns a 1-d array `pCounts` where the i^th element is
    //  the number of non-zero elements in the i^th row of pArr.
    function buildCountNonZero() {
        const f = module.addFunction(fnName + "_countNonZero");
        // Pointer to the input array. Shape: numRow * numCol
        f.addParam("pArr", "i32");
        // Number of rows
        f.addParam("numRow", "i32");
        // Number of columns
        f.addParam("numCol", "i32");
        // Pointer to a 1-d array. Shape: numRow
        f.addParam("pCounts", "i32");
        // Index
        f.addLocal("i", "i32");
        // Index
        f.addLocal("j", "i32");
        // Counts
        f.addLocal("count", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("numRow"))),
                c.setLocal("count", c.i32_const(0)),
                c.setLocal("j", c.i32_const(0)),
                c.block(c.loop(
                    c.br_if(1, c.i32_eq(c.getLocal("j"), c.getLocal("numCol"))),
                    c.if(
                        c.i32_ne(
                            c.call(fnName + "_loadI32",
                                c.getLocal("pArr"),
                                c.i32_add(
                                    c.i32_mul(
                                        c.getLocal("i"),
                                        c.getLocal("numCol"),
                                    ),
                                    c.getLocal("j"),
                                ),
                            ),
                            c.i32_const(0),
                        ),
                        c.setLocal("count", c.i32_add(c.getLocal("count"), c.i32_const(1))),
                    ),
                    c.setLocal("j", c.i32_add(c.getLocal("j"), c.i32_const(1))),
                    c.br(0),
                )),
                c.call(fnName + "_storeI32",
                    c.getLocal("pCounts"),
                    c.getLocal("i"),
                    c.getLocal("count"),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        );
    }

    // Tests if storeI32 and loadI32 is correct.
    function buildTestStoreLoadI32() {
        const f = module.addFunction(fnName + "_testLoadStoreI32");
        // Pointer to a 1-d array with i32 elements
        f.addParam("pArr", "i32");
        // Length of the input vector
        f.addParam("length", "i32");
        // Index
        f.addLocal("i", "i32");
        // Temporary value
        f.addLocal("tmp", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            // for(i=0; i<length; i++) {
            //      tmp = pArr[i];
            //      tmp += i;
            //      pArr[i] = tmp;
            // }
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.setLocal("tmp",
                    c.call(fnName + "_loadI32",
                        c.getLocal("pArr"),
                        c.getLocal("i"),
                    ),
                ),
                c.setLocal("tmp",
                    c.i32_add(
                        c.getLocal("tmp"),
                        c.getLocal("i"),
                    ),
                ),
                c.call(fnName + "_storeI32",
                    c.getLocal("pArr"),
                    c.getLocal("i"),
                    c.getLocal("tmp"),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        )
    }

    // Tests if storeI64 and loadI64 is correct.
    function buildTestStoreLoadI64() {
        const f = module.addFunction(fnName + "_testLoadStoreI64");
        // Pointer to a 1-d array with i64 elements
        f.addParam("pArr", "i32");
        // Length of the input vector
        f.addParam("length", "i32");
        // Index
        f.addLocal("i", "i32");
        // Temporary value
        f.addLocal("tmp", "i64");
        const c = f.getCodeBuilder();
        f.addCode(
            // for(i=0; i<length; i++) {
            //      tmp = pArr[i];
            //      tmp += i;
            //      pArr[i] = tmp;
            // }
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.setLocal("tmp",
                    c.call(fnName + "_loadI64",
                        c.getLocal("pArr"),
                        c.getLocal("i"),
                    ),
                ),
                c.setLocal("tmp",
                    c.i64_add(
                        c.getLocal("tmp"),
                        c.i64_extend_i32_u(c.getLocal("i"))
                    ),
                ),
                c.call(fnName + "_storeI64",
                    c.getLocal("pArr"),
                    c.getLocal("i"),
                    c.getLocal("tmp"),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        )
    }

    // Tests if maxArrayValue is correct.
    function buildTestMaxArrayValue() {
        const f = module.addFunction(fnName + "_testMaxArrayValue");
        // Pointer to an array
        f.addParam("pArr", "i32");
        // Length of the array
        f.addParam("length", "i32");
        // Max value
        f.addParam("pMax", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.call(fnName + "_storeI32",
                c.getLocal("pMax"),
                c.i32_const(0),
                c.call(fnName + "_maxArrayValue",
                    c.getLocal("pArr"),
                    c.getLocal("length"),
                ),
            ),
        );
    }

    // Tests if maxArrayValue is correct.
    function buildTestGetMSB() {
        const f = module.addFunction(fnName + "_testGetMsb");
        // Pointer to an array
        f.addParam("pArr", "i32");
        // Length of the array
        f.addParam("length", "i32");
        // Pointer to an array of Msb
        f.addParam("pMsb", "i32");
        // Index
        f.addLocal("i", "i32");
        const c = f.getCodeBuilder();
        f.addCode(
            c.setLocal("i", c.i32_const(0)),
            c.block(c.loop(
                c.br_if(1, c.i32_eq(c.getLocal("i"), c.getLocal("length"))),
                c.call(fnName + "_storeI32",
                    c.getLocal("pMsb"),
                    c.getLocal("i"),
                    c.call(fnName + "_getMsb",
                        c.call(fnName + "_loadI32",
                            c.getLocal("pArr"),
                            c.getLocal("i"),
                        ),
                    ),
                ),
                c.setLocal("i", c.i32_add(c.getLocal("i"), c.i32_const(1))),
                c.br(0),
            )),
        );
    }

    buildLoadI64();
    buildStoreI64();
    buildLoadI32();
    buildStoreI32();
    buildAddAssignI32InMemoryUncheck();
    buildAllocateMemory();
    buildInitializeI32();
    buildInitializeI64();
    buildMaxArrayValue();
    buildCopyArray();
    buildGetMSB();
    buildCountNonZero();
    buildTestGetMSB();
    buildTestMaxArrayValue();
    buildTestStoreLoadI32();
    buildTestStoreLoadI64();
    module.exportFunction(fnName + "_testGetMsb");
    module.exportFunction(fnName + "_testLoadStoreI32");
    module.exportFunction(fnName + "_testLoadStoreI64");
    module.exportFunction(fnName + "_testMaxArrayValue");
    module.exportFunction(fnName + "_countNonZero");
}
