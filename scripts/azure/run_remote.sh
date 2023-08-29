#!/usr/bin/env bash

for alg in ring rabenseifner
do
    for n in `cat nodes.public.list`
    do
        ssh -oStrictHostKeyChecking=no ${n} "cd .dccl;./run_dccl.sh 268435456 40 200 ${alg}" &
    done
    wait
    root=`head -1 nodes.public.list`
    scp -oStrictHostKeyChecking=no ${root}:.dccl/${alg}-c268435456w40r200.tar.bz2 .
done
