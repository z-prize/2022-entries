../submission/submission.wasm: WMSM.c
	clang --target=wasm32 -O3 -flto -nostdlib -Wl,--no-entry -Wl,--export-all -Wl,--lto-O3 -o ../submission/submission.wasm WFastMSM.c -DWASM_BUILD 

