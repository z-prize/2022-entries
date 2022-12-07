# Compiles reference into a `.wasm` module
cd reference
wasm-pack build --release
cd ..

# Compiles submission into a `.wasm` module
# Feel free to change
make -C submission submission.wasm
cp submission/submission.wasm www/submission.wasm

# Evaluation
cd ./www
npm install
npm run start
