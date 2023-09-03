#!/usr/bin/env bash

source common_env.sh

for world_size in ${WORLD_SIZES[@]}
do
    # prepare nodes list
    list_files=("nodes.public.list" "nodes.private.list" "myhostfile" "myrankfile")
    for lf in ${list_files[@]}
    do
        head -${world_size} ${lf} > ${lf}.cur
    done

    # upload
    for cn in `cat nodes.public.list.cur`
    do
        for lf in ${list_files[@]}
        do
            scp ${lf}.cur ${cn}:${BENCHMARK_WORKSPACE}/${lf}
        done
    done

    # run dccl
    for alg in ring rabenseifner
    do
        for n in `cat nodes.public.list.cur`
        do
            ssh -oStrictHostKeyChecking=no ${n} "cd ${BENCHMARK_WORKSPACE};./run_dccl.sh ${COUNT} ${WARMUP_ITER} ${RUN_ITER} ${alg}" &
        done
        wait
    done
    
    # run ompi
    root=`head -1 nodes.public.list.cur`
    for alg in ring rabenseifner
    do
        ssh -oStrictHostKeyChecking=no ${root} "cd ${BENCHMARK_WORKSPACE};./run_ompi.sh ${COUNT} ${WARMUP_ITER} ${RUN_ITER} ${alg}"
    done
    
    # collect data
    rm -rf ws-${world_size}
    mkdir  ws-${world_size}
    for alg in ring rabenseifner
    do
        scp -oStrictHostKeyChecking=no ${root}:${BENCHMARK_WORKSPACE}/${alg}-*-c${COUNT}w${WARMUP_ITER}r${RUN_ITER}.tar.bz2 ws-${world_size}/
        ssh -oStrictHostKeyChecking=no ${root} "rm -f ${BENCHMARK_WORKSPACE}/${alg}-*-c${COUNT}w${WARMUP_ITER}r${RUN_ITER}.tar.bz2"
    done

done
