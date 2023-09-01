Derecho Collective Communications Library {#mainpage}
=========================================

DCCL implements NCCL's API using with carefully zero-copy optimizations especially for RDMA.

# Installation

Currently, you need to build and install DCCL from source code.

## Prerequisites
- Linux (other operating systems don't currently support the RDMA features we use. We recommend Ubuntu 22.04. Any Ubuntu version newer than 18.04 will work. Other distributions should also work.)
- GNU G++ compiler with C++17 support.
- Derecho newer than 2.4.0. Please follow this [document](http://github.com/Derecho-Project/derecho) to install it.
- Doxygen >= 1.9.7, if you want to build DCCL documentation.

## Building DCCL
1) Download DCCL source code
```
# git clone https://github.com/Derecho-Project/dccl
```

2) Build DCCL source and install DCCL
```
# mkdir build
# cd build
# cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
# make -j
# make install
```
If you want to install DCCL to a location other than `/usr/local`, please pass it to `CMAKE_INSTALL_PREFIX` replacing `/usr/local`.

# Using DCCL

DCCL is a library very similar to NCCL or OpenMPI. Please refer to `include/dccl/dccl.hpp` for its API functions. Or, if you build DCCL documentation, open `build/html/index.html` and choose "Modules" tab and click "The DCCL API" link to see the HTML version.

## The Helloworld Application

We prepared a "hello world" in [examples/helloworld](examples/helloworld) folder to show how to create a simple DCCL application. The code is self-explanatory:
~~~~~~~~~~~~~~~~{.cpp}
/* hello world's main.cpp */
#include <dccl/dccl.hpp>
#include <iostream>

using namespace dccl;

int main(int argc, char** argv) {
    ncclComm_t comm;
    // STEP 1: initialize the communicator.
    auto ret = ncclCommInit(&comm);
    if (ret != ncclSuccess) {
        std::cerr << "DCCL Communicator initialization failed." << std::endl;
    }
    // STEP 2: print world size and rank information.
    std::cout << "Hello! world size is " << dcclGetWorldSize(comm)
              << ", my rank is " << dcclGetMyRank(comm)
              << std::endl;
    // STEP 3: finalize the communicator.
    ncclCommFinalize(comm);
    return 0;
}
~~~~~~~~~~~~~~~~

Besides getting the DCCL comunicator information, you can call other DCCL API functions like `ncclAllReduce`, `ncclBroadcast`, `ncclAllGather`, and etc..

We prepared the cmake file for your convenience. Follow the commands below to build it, assuming the current directory is `examples/helloworld`.
```
# mkdir build
# cd build
# cmake ..
# make
```
A binary file "helloworld" will be generated.

To run this helloworld application, we need the configuration for each process. We provides the configuration for four processes, from `n0` to `n3` in "cfg" folder. Please open four consoles and issue the following command in each console to start the application. We assume the current directory is `example/helloworld`.
```
# cd build/cfg/n0
# ../../helloworld
Hello! world size is 4, my rank is 0
```
Plesae change 'n0' in the first command to 'n1', 'n2', or 'n3' for the corresponding process. Once all processes start, they will join each other to create the distributed communicator, and then print the communicator information.

## The Full-fledged Application

Besides the helloworld application, we also provide a more complicated application called dccl_cli in "src/application/cli.cpp" for users to learn how to write their DCCL application. The 'dccl_cli' command line tool tests all APIs of DCCL. You run it the same way as the helloworld application. It will generate a `dccl_cli.tt` file on each process for performance analysis.
```
# ../../dccl_cli -a all_reduce
dccl api evaluation with the following configuration:
        api:all_reduce
        warmup:0
        repeat:1000
        type:3
        op:0
        count:1024
warm up...
done.
run test...
done.
flush timestamp...
...done
# ls -l dccl_cli.tt
-rw-r--r-- 1 dccluser sudo 150167 Aug 27 13:46 dccl_cli.tt
```

# FAQs

## How does DCCL decide on which network device/IP address to use?
DCCL relies on Derecho for network device/IP configuration. Each DCCL process has a derecho configuration file (generally, 'derecho.cfg' in the working directory) to decide which ip/ports, network device to use as well as other information like the message sizes used internally, persistent, logging level, and etc... Among those configurations, the following options are commonly used to tune the DCCL's layout.

```
[DERECHO]
# leader ip - the leader's ip address
leader_ip = 192.168.99.105
# leader gms port - the leader's gms port
leader_gms_port = 23580
# my local id - each node should have a different id
local_id = 1
# my local ip address
local_ip = 192.168.99.106
# my derecho gms port
gms_port = 23580
...
[RDMA]
# 1. provider = bgq|gni|efa|hook|netdir|psm|psm2|psm3|rxd|rxm|shm|udp|usnic|verbs
# possible options(only 'sockets' and 'verbs' providers are tested so far):
# bgq     - The Blue Gene/Q Fabric Provider
# efa     - The Amazon Elastic Fabric Adapter
# gni     - The GNI Fabric Provider (Cray XC (TM) systems)
# hook    - The Hook Fabric Provider Utility
# netdir  - The Network Direct Fabric Provider (Microsoft Network Direct SPI)
# psm     - The PSM Fabric Provider
# psm2    - The PSM2 Fabric Provider
# psm3    - The PSM3 Fabric Provider
# rxd     - The RxD (RDM over DGRAM) Utility Provider
# rxm     - The RxM (RDM over MSG) Utility Provider
# shm     - The SHM Fabric Provider
# tcp     - The TCP Fabric Provider
# udp     - The UDP Fabric Provider
# usnic   - The usNIC Fabric Provider (Cisco VIC)
# verbs   - The Verbs Fabric Provider
# Please note that only "tcp" and "verbs" are tested this moment.
provider = verbs
# 2. domain
# For sockets provider, domain is the NIC name (ifconfig | grep -v -e "^ ")
# For verbs provider, domain is the device name (ibv_devices)
domain = mlx5_0
...

```
Options `local_ip` and `gms_port` control which IP/PORT the process uses for communication. Options `leader_ip` and `leader_gms_port` tell the process who is the leader. Option `provider` controls which type of low-level communication mechanism to use. And option `domain` decides on which specific network device to use. Please refere to [Derecho documentation](https://github.com/Derecho-Project/derecho) for more.

## How are world size and rank decided ?
The world size is decided by Derecho's layout. The knob is in the layout file ('layout.json') side-by-side to the configuration file ('derecho.cfg').

```
[
    {
        "type_alias":   "DCCLSubgroupType",
        "layout":       [
                            {
                                "min_nodes_by_shard": ["4"],
                                "max_nodes_by_shard": ["4"],
                                "delivery_modes_by_shard": ["Ordered"],
                                "profiles_by_shard": ["DEFAULT"]
                            }
                        ]
    }
]

```
The option `min_nodes_by_shard` and `max_nodes_by_shard` specifies the range of the world size. Blocking call to `ncclCommInit` returns as soom as `min_nodes_by_shard` processes join. But more processes can join later dynamically.

Instead of controlling the rank directly, DCCL process specifies its node id, controlled by option `local_id` in the configuration above. Derecho enforces the uniqueness of its ID. Once enough processes join the system, the leader will assign the rank to each of the process. A user application has NO control of which rank it will be assigned.

## How does DCCL decide on which AllReduce algorithm to use?
In the configuration file, you can use option `DCCL/allreduce_algorithm` to control it.
```
[DCCL]
# allreduce_algorithm ring | rabenseifner
allreduce_algorithm = ring
```
Currently, we support only Ring and Rabenseifner algorithm.
