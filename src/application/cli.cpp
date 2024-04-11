#include <dccl/dccl.hpp>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include <derecho/utils/time.h>
#include <cassert>
#include "utils.hpp"

#ifdef __BUILD_FOR_OMPI__
#include <mpi.h>
#include <stdlib.h>
#endif//__BUILD_FOR_OMPI__

#if defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI)
//TODO: find a better way to determine CUDA L1 cache line size
#define ASSERTRT(stmt) \
    do { \
        cudaError_t err = (stmt); \
        if (err != cudaSuccess) { \
            const char *_err_desc = cudaGetErrorString(err); \
            std::cerr << "CUDA Runtime Error: " \
                      << "(" << err << ")" \
                      << _err_desc << std::endl; \
        } \
        assert(err == cudaSuccess); \
    } while(0)

__attribute__((visibility("hidden")))
void save_cuda_mem (const void* ptr, size_t size, const std::string& fname) {
    void* dat = malloc(size);

    if (dat == nullptr) {
        std::cerr << "Failed to allocate " << size << " bytes of memory." << std::endl;
        return;
    }

    if(cudaSuccess != cudaMemcpy(dat,ptr,size,cudaMemcpyDeviceToHost)) {
        std::cerr << "Cannot copy " << size << " bytes from device to host. giving up saving...@" 
                  << __FILE__ << ":" << __LINE__ << std::endl;
        return;
    }
    save_mem(dat,size,fname);
    free(dat);
}

#endif

using namespace dccl;

const char* help_string = 
    "\t--api,-a     name of the DCCL api to be tested. This option is mandatory. Full api list:\n"
    "\t             broadcast,send,recv,reduce,all_reduce,reduce_scatter,all_gather\n"
    "\t             Please note that\n"
    "\t             - 'send' and 'recv' only work between rank 0 and 1. \n"
    "\t             - only rank 0 will 'broadcast', all other node will receive. \n"
    "\t             - 'reduce' will reduce to rank 0.\n"
#if defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI__)
    "\t--gpu,-g     gpu device to use. If not specified, CPU is used.\n"
#endif
    "\t--warmup,-w  number of operations for warmup, defaulted to 0.\n"
    "\t--repeat,-r  number of operations for evaluation, defaulted to 1000.\n"
    "\t--type,-t    type of the data, defaulted to uint32. Full type list:\n"
    "\t             int8,uint8,int32,uint32,int64,uint64,float16,float32,float64\n"
    "\t--op,-o      the operation, defaulted to SUM. Full op list:\n"
    "\t             sum,prod,max,min,avg\n"
    "\t--count,-c   number of data entries in the array, defaulted to 1024\n"
    "\t--save,-s    save data in before.dat and after.dat for validation.\n"
    "\t--help,-h    print this message.\n";

#ifdef __BUILD_FOR_OMPI__

inline MPI_Datatype parse_data_type(const char* dt_str) {
    if (std::strcmp("int8",dt_str)==0) {
        return MPI_INT8_T;
    } else if (std::strcmp("uint8",dt_str)==0) {
        return MPI_UINT8_T;
    } else if (std::strcmp("int32",dt_str)==0) {
        return MPI_INT32_T;
    } else if (std::strcmp("uint32",dt_str)==0) {
        return MPI_UINT32_T;
    } else if (std::strcmp("int64",dt_str)==0) {
        return MPI_INT64_T;
    } else if (std::strcmp("uint64",dt_str)==0) {
        return MPI_UINT64_T;
    } else if (std::strcmp("float16",dt_str)==0) {
        std::cerr << "OpenMPI does not support float16, raising an exception..." << std::endl;
        throw std::runtime_error("OpenMPI does not support float16.");
    } else if (std::strcmp("float32",dt_str)==0) {
        return MPI_FLOAT;
    } else if (std::strcmp("float64",dt_str)==0) {
        return MPI_DOUBLE;
    }
    // default
    std::cerr << "Unknown data type:" << dt_str << " falling back to 'UINT8_T'" << std::endl;
    return MPI_UINT8_T;
}

/* Use MPI_Type_size() */

inline MPI_Op parse_reduce_operation(const char* ro_str) {
    if (std::strcmp("sum",ro_str)==0) {
        return MPI_SUM;
    } else if (std::strcmp("prod",ro_str)) {
        return MPI_PROD;
    } else if (std::strcmp("max",ro_str)) {
        return MPI_MAX;
    } else if (std::strcmp("min",ro_str)) {
        return MPI_MIN;
    } else if (std::strcmp("avg",ro_str)) {
        throw std::runtime_error("Open MPI does not support average Operation.");
    }
    // default
    std::cerr << "Unknown operation:" << ro_str << " falling back to 'sum'" << std::endl;
    return MPI_SUM;
}

#else
inline ncclDataType_t parse_data_type(const char* dt_str) {
    if (std::strcmp("int8",dt_str)==0) {
        return ncclDataType_t::ncclInt8;
    } else if (std::strcmp("uint8",dt_str)==0) {
        return ncclDataType_t::ncclUint8;
    } else if (std::strcmp("int32",dt_str)==0) {
        return ncclDataType_t::ncclInt32;
    } else if (std::strcmp("uint32",dt_str)==0) {
        return ncclDataType_t::ncclUint32;
    } else if (std::strcmp("int64",dt_str)==0) {
        return ncclDataType_t::ncclInt64;
    } else if (std::strcmp("uint64",dt_str)==0) {
        return ncclDataType_t::ncclUint64;
    } else if (std::strcmp("float16",dt_str)==0) {
        return ncclDataType_t::ncclFloat16;
    } else if (std::strcmp("float32",dt_str)==0) {
        return ncclDataType_t::ncclFloat32;
    } else if (std::strcmp("float64",dt_str)==0) {
        return ncclDataType_t::ncclFloat64;
    }
    // default
    std::cerr << "Unknown data type:" << dt_str << " falling back to 'uint8'" << std::endl;
    return ncclDataType_t::ncclUint8;
}

inline size_t size_of_type(ncclDataType_t datatype) {
    switch(datatype) {
    case ncclInt8:
    case ncclUint8:
        return 1;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
        return 8;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
        return 2;
    default:
        return 0;
    }
}

inline ncclRedOp_t parse_reduce_operation(const char* ro_str) {
    if (std::strcmp("sum",ro_str)==0) {
        return ncclSum;
    } else if (std::strcmp("prod",ro_str)) {
        return ncclProd;
    } else if (std::strcmp("max",ro_str)) {
        return ncclMax;
    } else if (std::strcmp("min",ro_str)) {
        return ncclMin;
    } else if (std::strcmp("avg",ro_str)) {
        return ncclAvg;
    }
    // default
    std::cerr << "Unknown operation:" << ro_str << " falling back to 'sum'" << std::endl;
    return ncclSum;
}
#endif//__BUILD_FOR_OMPI__

static void print_help(const char* command_name) {
    std::cout << "Usage: " << command_name << " [options]" << std::endl;
    std::cout << help_string << std::endl;
}

int main(int argc, char** argv) {
    // step 0 - parameters
    static struct option long_options[] = {
        {"api",     required_argument,  0,  'a'},
#if defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI__)
        {"gpu",     required_argument,  0,  'g'},
#endif
        {"warmup",  required_argument,  0,  'w'},
        {"repeat",  required_argument,  0,  'r'},
        {"type",    required_argument,  0,  't'},
        {"op",      required_argument,  0,  'o'},
        {"count",   required_argument,  0,  'c'},
        {"save",    no_argument,        0,  's'},
        {"help",    no_argument,        0,  'h'},
        {0}
    };

    int c;

    std::string api;
#if !defined(__BUILD_FOR_OMPI__) && defined(CUDA_FOUND)
    int32_t         gpu = -1;
#endif//!__BUILD_FOR_OMPI__ && CUDA_FOUND
    size_t          warmup_count = 0;
    size_t          repeat_count = 1000;
#ifdef __BUILD_FOR_OMPI__
    MPI_Datatype    data_type = MPI_UINT32_T;
    MPI_Op          operation = MPI_SUM;
#else
    ncclDataType_t  data_type = ncclUint32;
    ncclRedOp_t     operation = ncclSum;
#endif
    size_t          data_count = 1024;
    bool            save = false;

    while (true) {
        int option_index = 0;
        c = getopt_long(argc,argv, "a:g:w:r:t:o:c:sh", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
        case 'a':
            api = optarg;
            break;
#if defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI__)
        case 'g':
            gpu = std::stoi(optarg);
            break;
#endif//defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI__)
        case 'w':
            warmup_count = std::stoul(optarg);
            break;
        case 'r':
            repeat_count = std::stoul(optarg);
            break;
        case 't':
            data_type = parse_data_type(optarg);
            break;
        case 'o':
            operation = parse_reduce_operation(optarg);
            break;
        case 'c':
            data_count = std::stoul(optarg);
            break;
        case 's':
            save = true;
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        default:
            break;
        }
    }

    if (api.empty()) {
        print_help(argv[0]);
        return -1;
    }
#ifdef __BUILD_FOR_OMPI__
    std::cout << "ompi api evaluation with the following configuration:" << std::endl;
#else
    std::cout << "dccl api evaluation with the following configuration:" << std::endl;
#endif
    std::cout << "\tapi:" << api << std::endl;
#if defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI__)
    std::cout << "\tgpu:" << gpu << "\t(CPU == -1)" << std::endl;
#endif//defined(CUDA_FOUND) && !defined(__BUILD_FOR_OMPI__)
    std::cout << "\twarmup:" << warmup_count << std::endl;
    std::cout << "\trepeat:" << repeat_count << std::endl;
    std::cout << "\ttype:" << data_type << std::endl;
    std::cout << "\top:" << operation << std::endl;
    std::cout << "\tcount:" << data_count << std::endl;
#ifdef __BUILD_FOR_OMPI__
    int32_t my_rank;
    int ompi_err;
#else
    ncclResult_t ret;
    uint32_t my_rank;
    uint32_t world_size;
    ncclComm_t comm;
#endif//__BUILD_FOR_OMPI__

    // step 1 - initialize comm
#ifdef __BUILD_FOR_OMPI__
    ompi_err = MPI_Init(NULL,NULL);
    if (ompi_err != MPI_SUCCESS) {
        std::cerr << "failed to initialize mpi communicator." << std::endl;
        return ompi_err;
    }
    ompi_err = MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank);
    if (ompi_err != MPI_SUCCESS) {
        std::cerr << "failed to get rank from mpi communicator." << std::endl;
        return ompi_err;
    }
#else
    ret = ncclCommInit(&comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to initialize dccl communicator." << std::endl;
        return ret;
    }
    my_rank = dcclGetMyRank(comm);
    world_size = dcclGetWorldSize(comm);
#endif//__BUILD_FOR_OMPI__

    // step 2 - allocating data
    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
#define __ADDRESS_ALIGN__(ptr,align,ofst) \
    reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(ptr)&(~((align)-1)))+(ofst))
#define ENFORCE_BUFFER_OFFSET (0)
#ifdef __BUILD_FOR_OMPI__
    int data_size;
    void* ompi_sendbuf = nullptr;
    void* ompi_recvbuf = nullptr;
    MPI_Type_size(data_type,&data_size);
    if ((MPI_Alloc_mem(data_count*data_size + (CACHELINE_SIZE<<1),MPI_INFO_NULL,&ompi_sendbuf) != MPI_SUCCESS) ||
        (MPI_Alloc_mem(data_count*data_size + (CACHELINE_SIZE<<1),MPI_INFO_NULL,&ompi_recvbuf) != MPI_SUCCESS)) {
        std::cerr << "Failed to allocate " << (data_count*data_size + (CACHELINE_SIZE<<1)) << " bytes" << std::endl;
        std::cerr << "Error: " << std::strerror(errno) << std::endl;
        MPI_Finalize();
        return 1;
    }
    sendbuf = __ADDRESS_ALIGN__(reinterpret_cast<uintptr_t>(ompi_sendbuf)+CACHELINE_SIZE,
                                CACHELINE_SIZE,ENFORCE_BUFFER_OFFSET);
    recvbuf = __ADDRESS_ALIGN__(reinterpret_cast<uintptr_t>(ompi_recvbuf)+CACHELINE_SIZE,
                                CACHELINE_SIZE,ENFORCE_BUFFER_OFFSET);
#ifdef __USE_OMPI_WIN__
    MPI_Win s_win,r_win;
    if (MPI_Win_create(sendbuf,data_count*data_size,data_size,MPI_INFO_NULL,MPI_COMM_WORLD,&s_win)) {
        std::cerr << "Failed to create window for sendbuf@" << sendbuf << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (MPI_Win_create(recvbuf,data_count*data_size,data_size,MPI_INFO_NULL,MPI_COMM_WORLD,&r_win)) {
        std::cerr << "Failed to create window for recvbuf@" << recvbuf << std::endl;
        MPI_Finalize();
        return 1;
    }
#endif//__USE_OMPI_WIN__
    // initialize sendbuf and recvbuf
    memset(sendbuf,static_cast<int>(my_rank),data_count*data_size);
    memset(recvbuf,static_cast<int>(my_rank+128),data_count*data_size);
    if (save) {
        save_mem(sendbuf,data_count*data_size,"sendbuf.ompi.before.txt");
    }
#else//__BUILD_FOR_OMPI__
    size_t data_size = size_of_type(data_type);
    void* dccl_sendbuf = nullptr;
    void* dccl_recvbuf = nullptr;
#if defined(CUDA_FOUND)
    cudaStream_t stream = static_cast<cudaStream_t>(nullptr);

    if (gpu < 0) { // HOST Memory
#else
    cudaStream_t stream = nullptr;
#endif//CUDA_FOUND
        if (posix_memalign(&dccl_sendbuf,CACHELINE_SIZE,data_count*data_size + CACHELINE_SIZE) ||
            posix_memalign(&dccl_recvbuf,CACHELINE_SIZE,data_count*data_size + CACHELINE_SIZE)) {
            std::cerr << "Failed to allocate " << (data_count*data_size + CACHELINE_SIZE) << " bytes" << std::endl;
            std::cerr << "Error:" << std::strerror(errno) << std::endl;
            ncclCommFinalize(comm);
            return 1;
        }
        sendbuf = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dccl_sendbuf) + ENFORCE_BUFFER_OFFSET);
        recvbuf = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dccl_recvbuf) + ENFORCE_BUFFER_OFFSET);
        // initialize sendbuf and recvbuf
        memset(sendbuf,static_cast<int>(my_rank),data_count*data_size);
        memset(recvbuf,static_cast<int>(my_rank+128),data_count*data_size);
        if (save) {
            save_mem(sendbuf,data_count*data_size,"sendbuf.host.before.txt");
        }
#if defined(CUDA_FOUND)
    } else { // GPU Memory
        ASSERTRT(cudaSetDevice(gpu));
        ASSERTRT(cudaMalloc(&dccl_sendbuf, data_count * data_size + CUDA_L1_CACHELINE_SIZE));
        ASSERTRT(cudaMalloc(&dccl_recvbuf, data_count * data_size + CUDA_L1_CACHELINE_SIZE));
        ASSERTRT(cudaStreamCreate(&stream));
        sendbuf = __ADDRESS_ALIGN__(reinterpret_cast<uintptr_t>(dccl_sendbuf) + CUDA_L1_CACHELINE_SIZE,
                                    CUDA_L1_CACHELINE_SIZE,ENFORCE_BUFFER_OFFSET);
        recvbuf = __ADDRESS_ALIGN__(reinterpret_cast<uintptr_t>(dccl_recvbuf) + CUDA_L1_CACHELINE_SIZE,
                                    CUDA_L1_CACHELINE_SIZE,ENFORCE_BUFFER_OFFSET);
        ASSERTRT(cudaMemset(sendbuf,static_cast<int>(my_rank),data_count*data_size));
        ASSERTRT(cudaMemset(recvbuf,static_cast<int>(my_rank),data_count*data_size));
        if (save) {
            save_cuda_mem(sendbuf,data_count*data_size,"sendbuf.cuda.before.txt");
        }
    }
#endif//CUDA_FOUND
#endif//__BUILD_FOR_OMPI__

#ifdef __BUILD_FOR_OMPI__
#define RUN_WITH_COUNTER(cnt) \
    while (cnt--) { \
        if (api == "all_reduce") { \
            TIMESTAMP(TT_ALLREDUCE_ENTER,my_rank,0); \
            ompi_err = MPI_Allreduce(MPI_IN_PLACE,sendbuf,data_count,data_type,operation,MPI_COMM_WORLD); \
            TIMESTAMP(TT_ALLREDUCE_DONE,my_rank,0); \
        } else { \
            ompi_err = ~MPI_SUCCESS; \
        } \
        if (ompi_err != MPI_SUCCESS) { \
            std::cerr << "API:" << api << " failed with error:" << ompi_err << std::endl; \
            MPI_Finalize(); \
            return 2; \
        } \
    }
#else // !__BUILD_FOR_OMPI__
#define RUN_WITH_COUNTER(cnt) \
    while (cnt--) { \
        if (api == "all_reduce") { \
            ret = ncclAllReduce(sendbuf,sendbuf,data_count,data_type,operation,comm,stream); \
        } else if (api == "reduce_scatter") { \
            ret = ncclReduceScatter(sendbuf, \
                                    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(sendbuf) + my_rank*data_count*size_of_type(data_type)/world_size), \
                                    data_count/dcclGetWorldSize(comm),data_type,operation,comm,stream); \
        } else if (api == "all_gather") { \
            ret = ncclAllGather(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(sendbuf) + my_rank*data_count*size_of_type(data_type)/world_size), \
                                sendbuf, \
                                data_count/dcclGetWorldSize(comm),data_type,comm,stream); \
        } else if (api == "reduce") { \
            ret = ncclReduce(sendbuf,sendbuf,data_count,data_type,operation,0,comm,stream); \
        } else if (api == "broadcast") { \
            ret = ncclBroadcast(sendbuf,recvbuf,data_count,data_type,0,comm,stream); \
        } else if (api == "send") { \
            if (my_rank < 2) { \
                ret = ncclSend(sendbuf,data_count,data_type,1 - my_rank,comm,stream); \
            } \
        } else if (api == "recv") { \
            if (my_rank < 2) { \
                ret = ncclRecv(sendbuf,data_count,data_type,1 - my_rank,comm,stream); \
            } \
        } else { \
            ret = ncclInvalidArgument; \
        } \
        if (ret != ncclSuccess) { \
            std::cerr << "API:" << api << " failed with error:" << ret << std::endl; \
            ncclCommFinalize(comm); \
            return 2; \
        } \
    }

    // The memory should be registered with dccl before passing to dccl APIs.
    if (dcclRegisterCacheMemory(comm,sendbuf,data_count*size_of_type(data_type)) != ncclSuccess) {
        std::cerr << "Failed to register sendbuf@" << sendbuf << "to dccl." << std::endl;
        return 1;
    }
    if (dcclRegisterCacheMemory(comm,recvbuf,data_count*size_of_type(data_type)) != ncclSuccess) {
        std::cerr << "Failed to register recvbuf@" << recvbuf << "to dccl." << std::endl;
        return 1;
    }

#endif//__BUILD_FOR_OMPI__

    // step 3 - warmup
    std::cout << "warm up..." << std::endl;
    TIMESTAMP(TT_WARMUP_START,my_rank,0);
    RUN_WITH_COUNTER(warmup_count);
    TIMESTAMP(TT_WARMUP_END,my_rank,0);
    std::cout << "done." << std::endl;


    // step 4 - run test
    uint64_t cnt = repeat_count;
    std::cout << "run test..." << std::endl;
    TIMESTAMP(TT_TEST_START,my_rank,0);
    RUN_WITH_COUNTER(cnt);
    TIMESTAMP(TT_TEST_END,my_rank,0);
    std::cout << "done." << std::endl;
#ifdef __BUILD_FOR_OMPI__
#ifdef __USE_OMPI_WIN__
    MPI_Win_fence(0,s_win);
    MPI_Win_fence(0,r_win);
    MPI_Win_free(&s_win);
    MPI_Win_free(&r_win);
#endif//__USE_OMPI_WIN__
    // save data
    if (save) {
        save_mem(ompi_recvbuf,data_count*data_size,"recvbuf.ompi.txt");
        save_mem(ompi_sendbuf,data_count*data_size,"sendbuf.ompi.after.txt");
    }
    // free data
    MPI_Free_mem(ompi_sendbuf);
    MPI_Free_mem(ompi_recvbuf);
#else
    // deregister memory data
    if (dcclDeregisterCacheMemory(comm,sendbuf) != ncclSuccess) {
        std::cerr << "Failed to deregister sendbuf@" << sendbuf << "from dccl." << std::endl;
    }
    if (dcclDeregisterCacheMemory(comm,recvbuf) != ncclSuccess) {
        std::cerr << "Failed to deregister recvbuf@" << recvbuf << "from dccl." << std::endl;
    }
#if defined(CUDA_FOUND)
    // free data
    if (gpu < 0) {
#endif//CUDA_FOUND
        if (save) {
            save_mem(recvbuf,data_count*data_size,"recvbuf.host.txt");
            save_mem(sendbuf,data_count*data_size,"sendbuf.host.after.txt");
        }
        free(dccl_sendbuf);
        free(dccl_recvbuf);
#if defined(CUDA_FOUND)
    } else {
        if (save) {
            save_cuda_mem(recvbuf,data_count*data_size,"recvbuf.cuda.txt");
            save_cuda_mem(sendbuf,data_count*data_size,"sendbuf.cuda.after.txt");
        }
        ASSERTRT(cudaFree(dccl_sendbuf));
        ASSERTRT(cudaFree(dccl_recvbuf));
    }
#endif//CUDA_FOUND
#endif//__BUILD_FOR_OMPI__

    // step 5 -flush timestmap
    std::cout << "flush timestamp..." << std::endl;
#ifdef __BUILD_FOR_OMPI__
    std::string timestamp_fn = "ompi_cli." + std::to_string(my_rank) + ".tt";
    FLUSH_AND_CLEAR_TIMESTAMP(timestamp_fn);
#else
    FLUSH_AND_CLEAR_TIMESTAMP("dccl_cli.tt");
#endif
    std::cout << "...done" << std::endl;

    // step 6 - finalize comm
#ifdef __BUILD_FOR_OMPI__
    ompi_err = MPI_Finalize();
    if (ompi_err != MPI_SUCCESS) {
        std::cerr << "failed to finalize the ompi communicator." << std::endl;
    }
#else
    ret = ncclCommFinalize(comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to finalize the dccl communicator." << std::endl;
    }
#endif

    return 0;
}
