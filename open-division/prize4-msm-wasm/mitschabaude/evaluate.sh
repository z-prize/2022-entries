# Compiles reference into a `.wasm` module
cd reference
wasm-pack build --release
cd ..

# Compiles submission into a `.wasm` module
# Feel free to change
cd submission
npm i
npm run build
cd ..

# Evaluation
cd ./www
npm install
npm run start
