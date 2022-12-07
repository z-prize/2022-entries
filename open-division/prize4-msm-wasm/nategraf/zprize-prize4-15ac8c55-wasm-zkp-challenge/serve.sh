#RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --release
set -e
wasm-pack build --release
cd ./www
npm install
npm run start
