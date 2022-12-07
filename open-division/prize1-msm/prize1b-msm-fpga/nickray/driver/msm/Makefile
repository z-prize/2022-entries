SIZE := 26
NAME := /var/tmp/msm/size$(SIZE)

default: full

column:
	RUSTFLAGS='-C target-cpu=native' sudo -E cargo run --release --example column-harness -- $(SIZE) $(NAME)

full:
	RUSTFLAGS='-C target-cpu=native' sudo -E cargo run --release --example full-harness -- $(SIZE) $(NAME)

points:
	RUSTFLAGS='-C target-cpu=native' sudo -E cargo run --release --example point-gen -- $(SIZE) $(NAME)

reset: reset-8

reset-8:
	sudo fpga-load-local-image -S 0 -I agfi-0d25a1d127f1b497f

reset-7:
	sudo fpga-load-local-image -S 0 -I agfi-0b8f077575270002a

