#!/usr/bin/bash

nodes=( \
    192.168.99.105 \
    192.168.99.106 \
    192.168.99.16 \
    192.168.99.21 \
    192.168.99.22 \
    192.168.99.23 \
    192.168.99.24 \
    192.168.99.25 \
    192.168.99.26 \
    192.168.99.27 \
)

for ((i=2;i<=10;i++))
do
    host_str=`echo ${nodes[@]:0:${i}} | sed 's/\s/:1,/g' | sed 's/$/:1/'`
    echo "Run within $host_str"
    for ((alg=1;alg<=6;alg++))
    do
        $HOME/.dccl/opt/bin/mpirun \
        -host ${host_str} -n $i \
        --mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 \
        -x OMP_NUM_THREADS=1 \
        --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allreduce_algorithm $alg \
        $HOME/.dccl/mpi-benchmarks/IMB-MT \
            -count 67108864 \
            -warmup 100 \
            -repeat 500 \
            -barrier off \
            -datatype int \
            allreducemt | grep 268435456 | sed "s/^/$i /" >> 256MB.mpich.ompi.a$alg.dat
        $HOME/.dccl/opt/bin/mpirun \
        -host ${host_str} -n $i \
        --mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 \
        -x OMP_NUM_THREADS=1 \
        --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allreduce_algorithm $alg \
        $HOME/.dccl/mpi-benchmarks/IMB-MT \
            -count 268435456 \
            -warmup 40 \
            -repeat 200 \
            -barrier off \
            -datatype int \
            allreducemt | grep 1073741824 | sed "s/^/$i /" >> 1GB.mpich.ompi.a$alg.dat
    done
done
