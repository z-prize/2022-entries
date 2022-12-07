submission.wasm: main.go bls12-381/bls12-381.go bls12-381/g1.go bls12-381/multiexp.go bls12-381/fp/arith.go bls12-381/fp/element.go bls12-381/fr/arith.go bls12-381/fr/element.go
	tinygo build -o ./submission.wasm -target wasm -gc=leaking -opt=2 -scheduler=none -panic=trap -no-debug ./main.go
