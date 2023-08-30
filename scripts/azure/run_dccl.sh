#!/usr/bin/env bash
source common_env.sh

if [ $# -lt 3 ]; then
    echo "USAGE: $0 <count> <#wi> <#ri> [alg]"
    echo "alg:=ring|rabenseifner"
    exit
fi

my_ip=`ip -4 address show type ipoib | grep -o -E '172\.16\.[[:digit:]]+\.[[:digit:]]+'`
world_size=`cat nodes.private.list | wc -l`
leader_ip=`head -1 nodes.private.list`

let my_id=0
for ip in `cat nodes.private.list`
do
    if [ $ip == "$my_ip" ]; then
        break
    else
        let my_id=${my_id}+1
    fi
done

count=$1
warmup_iter=$2
run_iter=$3
numa_node=0
ar_alg="ring"
if [ $# -ge 4 ]; then
    ar_alg=$4
fi
_count=${count}
if [ $ar_alg == "ring" ]; then
    _count=`expr ${count} - ${count} % \( 16 \* ${world_size} \)`
fi

# STEP 1 - generate layout.json
cat layout.json.template \
    | sed "s/@world_size@/${world_size}/g" \
    > layout.json

# STEP 2 - generate derecho.cfg
cat derecho.cfg.template \
    | sed "s/@leader_ip@/${leader_ip}/g" \
    | sed "s/@my_ip@/${my_ip}/g" \
    | sed "s/@my_id@/${my_id}/g" \
    | sed "s/@allreduce_alg@/${ar_alg}/g" \
    > derecho.cfg

# STEP 3 - run experiment
LD_LIBRARY_PATH=$HOME/${BENCHMARK_WORKSPACE}/opt/lib \
numactl -m ${numa_node} -N ${numa_node} \
dccl/build/src/application/dccl_cli \
    -a all_reduce \
    -t ${DATA_TYPE} \
    -c ${_count} \
    -w ${warmup_iter} \
    -r ${run_iter}

# STEP 4 - collect data
if [ ${my_id} == "0" ]; then
    dat=${ar_alg}-dccl-c${count}w${warmup_iter}r${run_iter}
    mkdir ${dat}

    for ip in `cat nodes.private.list`
    do
        scp -oStrictHostKeyChecking=no ${ip}:${BENCHMARK_WORKSPACE}/dccl_cli.tt ${dat}/${ip}.tt
    done
    tar -jcf ${dat}.tar.bz2 ${dat}
    rm -rf ${dat}
fi

