#include <dccl.hpp>
#include <iostream>
#include <cstring>
#include <unistd.h>

const char* help_string = 
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

int main(int argc, char** argv) {
    // step 0 - parameters
    static struct option long_options[] = {
        {"warmup",  required_argument,  0,  'w'},
        {"repeat",  required_argument,  0,  'r'},
        {"type",    required_argument,  0,  't'},
        {"op",      required_argument,  0,  'o'},
        {"count",   required_argument,  0,  'c'},
        {"help",    no_argument,        0,  'h'},
        {0}
    };

    int c;
    int digit_optind = 0;

    size_t warmup_count = 0;
    size_t repeat_count = 1000;
    size_t data_type = ncclUint32;
    size_t operation = ncclSum;
    size_t data_count = 1024;

    while (true) {
        int option_index = 0;
        c = getopt_long(argc,argv, "w:r:t:o:c:h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
        case 'w':
            warmup_count = std::stoul(optarg);
            break;
        case 'r':
            repeat_count = std::stoul(optarg);
            break;
        case 't':
            data_type = std::stoul(optarg);
            break;
        case 'c':
            data_count = std::stoul(optarg);
            break;
        default:
            break;
        }
    }

    std::cout << "dccl allreduce evaluation with the following configuration:" << std::endl;
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
    void* sendbuf,recvbuf;
    uint64_t cacheline_sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (posix_memalign(sendbuf,cacheline_sz,data_count*size_of_type(data_type)) ||
        posix_memalign(recvbuf,cacheline_sz,data_count*size_of_type(data_type))) {
        std::cerr << "Failed to allocate " << data_count*size_of_type(data_type) << " bytes" << std::endl;
        std::cerr << "Error:" << std::strerror(errno) << std::endl;
        ncclCommFinalize(comm);
        return 1;
    }

#define RUN_WITH_COUNTER(cnt) \
    while (cnt--) { \
        ret = ncclAllReduce(sendbuf,recvbuf,data_count,data_type,operation,comm); \
        if (ret != ncclSuccess) { \
            std::cerr << "all reduce failed with error:" << ret << std::endl; \
            ncclCommFinalize(comm); \
            return 2; \
        } \
    }

    // step 3 - warmup
    RUN_WITH_COUNTER(warmup_count);


    // step 4 - run test
    start_ts = get_time();
    RUN_WITH_COUNTER(repeat);
    end_ts = get_time();

    // step 5 - finalize comm
    ret = ncclCommFinalize(comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to finalize the dccl communication." << std::endl;
    }

    // step 6 - get average
    std::cout << "Average: " << ((end_ts-start_ts)/1000.0f) << " us" << std::endl;
    return 0;
}
