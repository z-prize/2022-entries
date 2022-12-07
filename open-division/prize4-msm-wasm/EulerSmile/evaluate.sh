# Compiles reference into a `.wasm` module
cd reference
wasm-pack build --release
cd ..

# Compiles submission into a `.wasm` module
# Feel free to change
#cd submission
#cp submission.wasm ../www
#cd ..

cd submission
CC=emcc AR=llvm-ar wasm-pack build --release
rm -rf ../www/euler
cp -R ./pkg ../www/euler
cd ..

# Evaluation
cd ./www
npm install
npm run start
