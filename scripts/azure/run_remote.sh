#!/usr/bin/env bash

source common_env.sh

# run dccl
for alg in ring rabenseifner
do
    for n in `cat nodes.public.list`
    do
        ssh -oStrictHostKeyChecking=no ${n} "cd .dccl;./run_dccl.sh ${COUNT} ${WARMUP_ITER} ${RUN_ITER} ${alg}" &
    done
    wait
done

# run ompi
root=`head -1 nodes.public.list`
for alg in ring rabenseifner
do
    ssh -oStrictHostKeyChecking=no ${root} "cd .dccl;./run_ompi.sh ${COUNT} ${WARMUP_ITER} ${RUN_ITER} ${alg}"
done

# collect data
scp -oStrictHostKeyChecking=no ${root}:.dccl/${alg}-*-c${COUNT}w${WARMUP_ITER}r${RUN_ITER}.tar.bz2 .

