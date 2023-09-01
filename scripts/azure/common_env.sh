# environment
AZ_RESOURCEGROUP=WeijiaExperiment
VMSS_NAME=HPC
VMSS_SIZE=32
BENCHMARK_WORKSPACE=.dccl
IB_DEVICE=mlx5_ib0
NUMA_NODE=0

# experiment parameters
COUNT=1073741824
WARMUP_ITER=20
RUN_ITER=200
DATA_TYPE=float32
WORLD_SIZES=( 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 )
