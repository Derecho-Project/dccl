#!/usr/bin/bash
source common_env.sh

if [ $# -lt 3 ]; then
    echo "USAGE: $0 <count> <#wi> <#ri> [alg]"
    echo "alg:=ring|rabenseifner"
    exit
fi

count=$1
warmup_iter=$2
run_iter=$3

ar_alg="ring"
if [ $# -ge 4 ]; then
    ar_alg=$4
fi

if [ ${ar_alg} == "ring" ]; then
    alg=4
elif [ ${ar_alg} == "rabenseifner" ]; then
    alg=6
else
    echo "Unknown algorithm: ${ar_alg}. Only ring or rabenseifner is allowed."
    exit
fi

world_size=`cat myhostfile | wc -l`

# run ob1
pkill -9 ompi_cli
sleep 3
$HOME/${BENCHMARK_WORKSPACE}/opt/bin/mpirun \
--hostfile myhostfile -n ${world_size} \
-rf myrankfile --report-bindings \
--mca btl_openib_device_type ib \
--mca btl_openib_allow_ib true \
--mca btl_openib_if_include ${IB_DEVICE}:1 \
--mca pml ob1 \
-x OMP_NUM_THREADS=1 \
--mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allreduce_algorithm $alg \
$HOME/${BENCHMARK_WORKSPACE}/dccl/build/src/application/ompi_cli -a all_reduce \
    -w ${warmup_iter} -r ${run_iter} -t int32 -c ${count}

dat=${ar_alg}-ob1-c${count}w${warmup_iter}r${run_iter}
mkdir ${dat}

let r=0
for h in `cat myhostfile`
do
    scp ${h}:${BENCHMARK_WORKSPACE}/ompi_cli.${r}.tt ${dat}/${h}.tt
    let r=$r+1
done
tar -jcf ${dat}.tar.bz2 ${dat}
rm -rf ${dat}

# run ucx
pkill -9 ompi_cli
sleep 3
$HOME/${BENCHMARK_WORKSPACE}/opt/bin/mpirun \
--hostfile myhostfile -n ${world_size} \
-rf myrankfile --report-bindings \
--mca pml ucx -x UCX_NET_DEVICES=${IB_DEVICE}:1 \
-x OMP_NUM_THREADS=1 \
--mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allreduce_algorithm $alg \
$HOME/${BENCHMARK_WORKSPACE}/dccl/build/src/application/ompi_cli -a all_reduce \
    -w ${warmup_iter} -r ${run_iter} -t ${DATA_TYPE} -c ${count}

ar_alg="ring"
if [ ${alg} -eq 6 ]; then
    ar_alg="rabenseifner"
fi

dat=${ar_alg}-ucx-c${count}w${warmup_iter}r${run_iter}
mkdir ${dat}

let r=0
for h in `cat myhostfile`
do
    scp ${h}:${BENCHMARK_WORKSPACE}/ompi_cli.${r}.tt ${dat}/${h}.tt
    let r=$r+1
done
tar -jcf ${dat}.tar.bz2 ${dat}
rm -rf ${dat}
