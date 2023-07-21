#!/usr/bin/env bash
if [ $# != 1 ]; then
    echo "Usage $0 <all|dccl|ompi>"
    exit 1
fi

TARGET=$1

source common_env.sh

for node in `cat nodes.list | grep ^[^#]`
do
    echo "installing to ${node}:${BENCHMARK_WORKSPACE}"
    ssh ${node} "rm -rf ${BENCHMARK_WORKSPACE};mkdir ${BENCHMARK_WORKSPACE}"
    scp deploy_${TARGET}.sh ${node}:${BENCHMARK_WORKSPACE}/
    ssh ${node} "chmod +x ${BENCHMARK_WORKSPACE}/deploy_${TARGET}.sh"
done
