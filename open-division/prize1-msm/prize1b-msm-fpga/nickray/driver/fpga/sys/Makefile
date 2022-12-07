SIZE := 24
CARGO := /usr/bin/cargo

stream:
	sudo -E $(CARGO) run --release --example stream -- $(SIZE)

reset:
	sudo fpga-load-local-image -S 0 -I agfi-08559006491e9c9eb -a 187
