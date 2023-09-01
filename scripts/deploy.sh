#!/usr/bin/env bash
if [ $# != 1 ]; then
    echo "Usage $0 <all|dccl|ompi|debug>"
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
    scp myrankfile ${node}:${BENCHMARK_WORKSPACE}/
    scp run_debug.sh ${node}:${BENCHMARK_WORKSPACE}/
    scp run_ompi_cli_4.1.5.ob1.sh ${node}:${BENCHMARK_WORKSPACE}/
    scp run_ompi_cli_4.1.5.ucx.sh ${node}:${BENCHMARK_WORKSPACE}/
done
