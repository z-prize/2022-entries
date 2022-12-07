# lib

This folder is for parts of the project which can logically stand on their own and could be usable for different projects as well.

## wasm-generate.js

This is the "library" which helps us generate all the Wasm code for this project.
It superceded my initial approach of writing the Wasm by hand in WAT (WebAssembly text) format.

The APIs imitate WAT syntax, with some sugar to make it easier to use, and return strings of WAT code.

For example, this is how you create the code for adding two integers:

```js
import { Writer, ops } from "./wasm-generate.js";

let writer = Writer(); // an object which holds a WAT string and various supporting data, like a list of created exports
let { line, comment } = writer;
let { i64, local64 } = ops; // a global which contains most operations

let a = "$a"; // variables are strings that begin with a "$"

// ...

lines(i64.add(a, 10)); // add a line of wasm code!
```

Tn WAT, the equivalent code would be written

```wat
(i64.add (local.get $a) (i64.const 10))
```

Note that the `i64.add` from our library accepts plain numbers as input (wrapping them with `i64.const`)
and automatically treats strings starting with "$" as variables (and wraps them with `local.get`).
Code has to be explicitly added to the writer with functions like `line`.

All in all, this is obviously not great yet, but, due to the metaprogramming ability,
has been far better for creating large amounts of wasm code than writing the wasm by hand.
