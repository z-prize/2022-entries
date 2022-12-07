# Compiles reference into a `.wasm` module
cd reference
wasm-pack build --release
cd ..

# No need to copy submission anywhere.

# Evaluation
cd ./www
npm install
npm run start
