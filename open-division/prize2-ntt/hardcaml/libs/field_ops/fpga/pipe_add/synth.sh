#!/bin/bash

dune build @default
vivado -mode batch -source ../../../scripts/run_vivado.tcl -tclargs pipe_add