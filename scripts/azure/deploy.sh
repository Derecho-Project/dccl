#!/usr/bin/env bash

source common_env.sh

for node in `cat nodes.public.list | grep ^[^#]`
do
    scp -oStrictHostKeyChecking=no $HOME/.ssh/id_rsa ${node}:.ssh/
    echo "installing to ${node}:${BENCHMARK_WORKSPACE}"
    ssh -oStrictHostKeyChecking=no ${node} "rm -rf ${BENCHMARK_WORKSPACE};mkdir ${BENCHMARK_WORKSPACE}"
    scp -oStrictHostKeyChecking=no deploy_all.sh ${node}:${BENCHMARK_WORKSPACE}/
    ssh -oStrictHostKeyChecking=no ${node} "chmod +x ${BENCHMARK_WORKSPACE}/deploy_all.sh"
    # scp myrankfile ${node}:${BENCHMARK_WORKSPACE}/
    scp -oStrictHostKeyChecking=no *.template ${node}:${BENCHMARK_WORKSPACE}/
    scp -oStrictHostKeyChecking=no *.list ${node}:${BENCHMARK_WORKSPACE}/
    scp -oStrictHostKeyChecking=no run_dccl.sh ${node}:${BENCHMARK_WORKSPACE}/
    ssh -oStrictHostKeyChecking=no ${node} "cd .dccl;time ./deploy_all.sh" &
done
wait
