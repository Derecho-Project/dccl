#include <dccl/dccl.hpp>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include <derecho/utils/time.h>

using namespace dccl;

const char* help_string = 
    "\t--api,-a     name of the DCCL api to be tested. Full api list:\n"
    "\t             scatter,gather,broadcast,send,recv,reduce,all_reduce,reduce_scatter,all_gather\n"
    "\t--warmup,-w  number of operations for warmup, defaulted to 0.\n"
    "\t--repeat,-r  number of operations for evaluation, defaulted to 1000.\n"
    "\t--type,-t    type of the data, defaulted to uint32. Full type list:\n"
    "\t             int8,uint8,int32,uint32,int64,uint64,float16,float32,float64\n"
    "\t--op,-o      the operation, defaulted to SUM. Full op list:\n"
    "\t             sum,prod,max,min,avg\n"
    "\t--count,-c   number of data entries in the array, defaulted to 1024\n"
    "\t--help,-h    print this message.\n";

static ncclDataType_t parse_data_type(const char* dt_str) {
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

static ncclRedOp_t parse_reduce_operation(const char* ro_str) {
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

static void print_help(const char* command_name) {
    std::cout << "Usage: " << command_name << " [options]" << std::endl;
    std::cout << help_string << std::endl;
}

int main(int argc, char** argv) {
    // step 0 - parameters
    static struct option long_options[] = {
        {"api",     required_argument,  0,  'a'},
        {"warmup",  required_argument,  0,  'w'},
        {"repeat",  required_argument,  0,  'r'},
        {"type",    required_argument,  0,  't'},
        {"op",      required_argument,  0,  'o'},
        {"count",   required_argument,  0,  'c'},
        {"help",    no_argument,        0,  'h'},
        {0}
    };

    int c;

    std::string api;
    size_t warmup_count = 0;
    size_t repeat_count = 1000;
    ncclDataType_t data_type = ncclUint32;
    ncclRedOp_t operation = ncclSum;
    size_t data_count = 1024;

    while (true) {
        int option_index = 0;
        c = getopt_long(argc,argv, "a:w:r:t:o:c:h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
        case 'a':
            api = optarg;
            break;
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
        case 'h':
            print_help(argv[0]);
            return 0;
        default:
            break;
        }
    }

    std::cout << "dccl api evaluation with the following configuration:" << std::endl;
    std::cout << "\tapi:" << api << std::endl;
    std::cout << "\twarmup:" << warmup_count << std::endl;
    std::cout << "\trepeat:" << repeat_count << std::endl;
    std::cout << "\ttype:" << data_type << std::endl;
    std::cout << "\top:" << operation << std::endl;
    std::cout << "\tcount:" << data_count << std::endl;
    ncclComm_t comm;
    ncclResult_t ret;

    // step 1 - initialize comm
    ret = ncclCommInit(&comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to initialize dccl communication." << std::endl;
        return ret;
    }

    // step 2 - allocating data
    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
    if (posix_memalign(&sendbuf,CACHELINE_SIZE,data_count*size_of_type(data_type)) ||
        posix_memalign(&recvbuf,CACHELINE_SIZE,data_count*size_of_type(data_type))) {
        std::cerr << "Failed to allocate " << data_count*size_of_type(data_type) << " bytes" << std::endl;
        std::cerr << "Error:" << std::strerror(errno) << std::endl;
        ncclCommFinalize(comm);
        return 1;
    }
    // initialize each byte of sendbuf to 1
    memset(sendbuf,1,data_count*size_of_type(data_type));
    // zero recvbuf
    bzero(recvbuf,data_count*size_of_type(data_type));

#define RUN_WITH_COUNTER(cnt) \
    while (cnt--) { \
        if (api == "all_reduce") { \
            ret = ncclAllReduce(sendbuf,recvbuf,data_count,data_type,operation,comm); \
        } else if (api == "reduce_scatter") { \
            ret = ncclReduceScatter(sendbuf,recvbuf,data_count/dcclGetWorldSize(comm),data_type,operation,comm); \
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

    // step 3 - warmup
    std::cout << "warm up..." << std::endl;
    RUN_WITH_COUNTER(warmup_count);
    std::cout << "done." << std::endl;


    // step 4 - run test
    uint64_t cnt = repeat_count;
    std::cout << "run test..." << std::endl;
    uint64_t start_ts = get_time();
    RUN_WITH_COUNTER(cnt);
    uint64_t end_ts = get_time();
    std::cout << "done." << std::endl;

    // deregister memory data
    if (dcclDeregisterCacheMemory(comm,sendbuf) != ncclSuccess) {
        std::cerr << "Failed to deregister sendbuf@" << sendbuf << "from dccl." << std::endl;
    }
    if (dcclDeregisterCacheMemory(comm,recvbuf) != ncclSuccess) {
        std::cerr << "Failed to deregister recvbuf@" << recvbuf << "from dccl." << std::endl;
    }

    // free data
    free(sendbuf);
    free(recvbuf);

    // step 5 - finalize comm
    ret = ncclCommFinalize(comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to finalize the dccl communication." << std::endl;
    }

    // step 6 - get average
    std::cout << "Average: " << ((end_ts-start_ts)/1000/repeat_count) << " us" << std::endl;
    return 0;
}
