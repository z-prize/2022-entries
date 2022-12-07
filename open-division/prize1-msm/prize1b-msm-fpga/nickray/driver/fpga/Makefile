SIZE := 26

main:
	sudo -E cargo run --release -- $(SIZE)

stream:
	sudo -E cargo run --example stream --release -- $(SIZE)

reset:
	sudo fpga-load-local-image -S 0 -I agfi-04667e862f0e3b1fa

reset-old:
	sudo fpga-load-local-image -S 0 -I agfi-08559006491e9c9eb -a 187

