# Compiles reference into a `.wasm` module
cd reference
wasm-pack build --release
cd ..

# This package contains a compiled binary version of the submission.wasm.
# If this has been deleted, attempt to rebuild it

if [ ! -f 'submission/submission.wasm' ]
then
  cd C
  make
  cd ..
  
  # check that the compiled sum matches, if not abort
  checksum=`sum submission/submission.wasm`
  if [ "$checksum" != "52216   107" ]
  then
    echo "Checksum does not match expected compiled checksum."
    echo "Please verify clang --version is Ubuntu clang version 14.0.0-1ubuntu1"
    exit
  fi
fi

# Compiles submission into a `.wasm` module
# Feel free to change
cd submission
cp submission.wasm ../www
cd ..

# Evaluation
cd ./www
npm install
npm run start
