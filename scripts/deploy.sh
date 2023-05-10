#!/usr/bin/env bash
source common_env.sh

for node in `cat nodes.list`
do
    echo "installing to ${node}:${BENCHMARK_WORKSPACE}"
    ssh ${node} "rm -rf ${BENCHMARK_WORKSPACE};mkdir ${BENCHMARK_WORKSPACE}"
    scp deploy_local.sh ${node}:${BENCHMARK_WORKSPACE}/
    ssh ${node} "chmod +x ${BENCHMARK_WORKSPACE}/deploy_local.sh"
done
