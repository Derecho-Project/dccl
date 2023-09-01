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
        $HOME/.dccl/dccl/build/src/application/ompi_cli -a all_reduce \
            -w 100 -r 500 -t int32 -c 67108864

        for((n=0;n<$i;n++))
        do
            scp ${nodes[$n]}:.dccl/ompi_cli.$n.tt 256MB.dccl.ompi.a$alg.$i-$n.dat
        done

        $HOME/.dccl/opt/bin/mpirun \
        -host ${host_str} -n $i \
        --mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 \
        -x OMP_NUM_THREADS=1 \
        --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allreduce_algorithm $alg \
        $HOME/.dccl/dccl/build/src/application/ompi_cli -a all_reduce \
            -w 40 -r 200 -t int32 -c 268435456

        for((n=0;n<$i;n++))
        do
            scp ${nodes[$n]}:.dccl/ompi_cli.$n.tt 1GB.dccl.ompi.a$alg.$i-$n.dat
        done
    done
done
