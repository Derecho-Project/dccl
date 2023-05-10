#!/usr/bin/env bash
# This command needs to run on all members of the group.
~/.dccl/dccl-mpi-benchmarks/IMB-MT -count $1 -repeat $2 dcclallreduce
