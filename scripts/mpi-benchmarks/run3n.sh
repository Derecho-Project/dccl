#!/usr/bin/bash
if [ $# != 2 ]; then
    echo "USAGE: $0 <size in 4-bytes> <number of repeat>"
    exit 1
fi
let size=`expr $1 \* 4`
export OMP_NUM_THREADS=1
~/.dccl/opt/bin/mpirun -host compute31:1,compute32:1,compute29:1 -x OMP_NUM_THREADS  -n 3 IMB-MT -count $1 -repeat $2 -barrier on allreducemt 2>/dev/null | sed 's/^[[:space:]]\+//' | grep ^${size}
