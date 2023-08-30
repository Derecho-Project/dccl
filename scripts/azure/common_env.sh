# environment
AZ_RESOURCEGROUP=WeijiaExperiment
VMSS_NAME=HPC
BENCHMARK_WORKSPACE=.dccl
IB_DEVICE=mlx5_ib0
NUMA_NODE=0

# experiment parameters
COUNT=268435456
WARMUP_ITER=40
RUN_ITER=200
DATA_TYPE=int32
WORLD_SIZES=(2)