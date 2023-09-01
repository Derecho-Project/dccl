#!/usr/bin/env bash
source common_env.sh
# az login --use-device-code
# public addresses
az vmss list-instance-public-ips -n ${VMSS_NAME} -g ${AZ_RESOURCEGROUP} --query "[].ipAddress" | sed -E 's/"|,|^\s+//g' | sed -E '/\[|\]/d' > nodes.public.list
# infiniband addresses
for h in `cat nodes.public.list`
do
    ssh -oStrictHostKeyChecking=no $h "ifconfig ib0|grep inet |grep -o -E '172\.16\.[[:digit:]]+\.[[:digit:]]+'"
done > nodes.private.list

rm -rf myrankfile myhostfile
let r=0
for h in `cat nodes.public.list`
do
    nodename=`ssh -oStrictHostKeyChecking=no $h "hostname"`
    echo "rank ${r}=${nodename} slot=${NUMA_NODE}:0-7" >> myrankfile
    echo "${nodename}" >> myhostfile
    let r=$r+1
done > myrankfile
