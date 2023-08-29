#!/usr/bin/env bash

# az login --use-device-code
# public addresses
az vmss list-instance-public-ips -n HPC -g WeijiaExperiment --query "[].ipAddress" | sed -E 's/"|,|^\s+//g' | sed -E '/\[|\]/d' > nodes.public.list
# infiniband addresses
for h in `cat nodes.public.list`
do
    ssh -oStrictHostKeyChecking=no $h "ifconfig ib0|grep inet |grep -o -E '172\.16\.[[:digit:]]+\.[[:digit:]]+'"
done > nodes.private.list
