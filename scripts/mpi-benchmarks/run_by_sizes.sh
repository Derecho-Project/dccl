#!/usr/bin/env bash

num_ints=( \
    1024        2048        4096        8192 \
    16384       32768       65536       131072 \
    262144      524288      1048576     2097152 \
    4194304     8388608     16777216    33554432 \
    67108864    134217728   268435456 \
)
num_reps=(
    100000      100000      100000      100000 \
    100000      50000       50000       25000 \
    25000       10000       10000       5000 \
    2000        1000        1000        1000 \
    1000        1000        1000 \
)

# num_ints=(134217728 268435456)
# num_reps=(500 500)

headline="#bytes #repetitions  t_min[usec]  t_max[usec]  t_avg[usec]"

echo $headline
for((i=0;i<${#num_ints[@]};i++))
do
    ./run3n.sh ${num_ints[$i]} ${num_reps[$i]}
done
