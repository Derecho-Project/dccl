#include <derecho/core/derecho.hpp>
#include <derecho/mutils-serialization/SerializationSupport.hpp>
#include <derecho/persistent/detail/PersistLog.hpp>
#include <derecho/utils/time.h>
#include <dccl/dccl.hpp>

#ifdef CUDA_FOUND
#include <cuda.h>
#endif

using namespace derecho;
using namespace dccl;

#define TT_OOB(x)               (2000000 + (x))

#define TT_OOB_WARMUP_START     TT_OOB(0001)
#define TT_OOB_WARMUP_END       TT_OOB(0002)
#define TT_OOB_TEST_START       TT_OOB(0003)
#define TT_OOB_TEST_END         TT_OOB(0004)

#define TT_OOB_SEND             TT_OOB(1001)
#define TT_OOB_ACKD             TT_OOB(1002)

/**
 * @cond    DoxygenSuppressed
 */
class P2PPerfSubgroupType : public mutils::ByteRepresentable,
                            public GroupReference {
private:
    int state;
public:
    int get() {return state;}
    void put(const int& s) {state=s;}
    P2PPerfSubgroupType():state(0) {}
    P2PPerfSubgroupType(int& s):state(s) {}

    DEFAULT_SERIALIZATION_SUPPORT(P2PPerfSubgroupType,state);
    REGISTER_RPC_FUNCTIONS(P2PPerfSubgroupType,ORDERED_TARGETS(get,put));

    void ensure_registerd(mutils::DeserializationManager&) {}
};

#ifdef CUDA_FOUND
#define ASSERTDRV(stmt) \
    do { \
        CUresult result = (stmt); \
        if (result != CUDA_SUCCESS) { \
            const char *_err_name; \
            cuGetErrorName(result, &_err_name); \
            std::cout << "CUDA error: (" << result << ")" << _err_name << std::endl; \
        } \
        assert(CUDA_SUCCESS == result); \
    } while(0)
#endif

static int oob_perf(
#ifdef CUDA_FOUND
                    int32_t 	cuda_dev,
#endif
		            size_t      size_byte,
                    size_t      depth,
                    uint32_t    warmup_sec,
                    uint32_t    duration_sec) {
    // STEP 1: prepare group, find my role
    SubgroupInfo si{make_subgroup_allocator<P2PPerfSubgroupType>()};
    Group<P2PPerfSubgroupType> g({},si,{},{},
            [](persistent::PersistentRegistry*,subgroup_id_t){return std::make_unique<P2PPerfSubgroupType>();});
    uint32_t    my_rank = g.get_my_rank();
    node_id_t   my_id   = g.get_my_id();

    std::cout << "Joined group with my_rank:" << my_rank
              << ", my_id:" << my_id
              << std::endl;

    // STEP 2: prepare memory pool
    size_t pool_size    = (size_byte*depth + 4095)/4096*4096;
    void*  pool_ptr;
#ifdef	CUDA_FOUND
    CUdevice    cuda_device;
    CUcontext   cuda_context;
    if (cuda_dev >= 0) {
        ASSERTDRV(cuInit(0));
        int n_devices = 0;
        ASSERTDRV(cuDeviceGetCount(&n_devices));

        if (cuda_dev >= n_devices) {
            std::cerr << "We found " << n_devices << " GPUs. dev id:" 
                      << cuda_dev << " is invalid." << std::endl;
            return -1;
        }

        ASSERTDRV(cuDeviceGet(&cuda_device,cuda_dev));
        ASSERTDRV(cuDevicePrimaryCtxRetain(&cuda_context, cuda_device));
        ASSERTDRV(cuCtxSetCurrent(cuda_context));

	    int rc = cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&pool_ptr),pool_size);
        if ( rc != CUDA_SUCCESS ) {
            std::cerr << "Failed to allocate cuda memory. cuMemAlloc() returns " << rc << std::endl;
            return -1;
        }
    } else
#endif
    if (posix_memalign(&pool_ptr,4096,pool_size)) {
        std::cerr << "Failed to allocate memory:" << strerror(errno) << std::endl;
        return -1;
    }
#ifdef CUDA_FOUND
    if (cuda_dev >= 0) {
        void* tptr = malloc(pool_size);
        if (tptr == nullptr) {
            std::cerr << "Failed to allocated temp memory" << strerror(errno) << std::endl;
            return -1;
        }
        bzero(tptr,pool_size);
        ASSERTDRV(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(pool_ptr),tptr,pool_size));
        free(tptr);
    } else {
#endif
        bzero(pool_ptr,pool_size);
#ifdef CUDA_FOUND
    }
#endif
    g.register_oob_memory(pool_ptr,pool_size);
    std::cout << pool_size << " bytes are registered as OOB cache." << std::endl;

    // STEP 3.0: get peer id.
    auto members = g.get_members();
    node_id_t peer_id;
    for(auto nid : members) {
        if (nid != my_id) {
            peer_id = nid;
            break;
        }
    }
    std::cout << "Peer id is " << peer_id << std::endl;

/**
 * b - base
 * s - buffer size
 * d - depth
 * c - count
 */
#define __BUF_PTR__(b,s,d,c) \
    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(b) + ((c)%(d))*(s))

#define OOB_SEND(id,iov,count) \
    g.get_subgroup<P2PPerfSubgroupType>().oob_send(id,iov,count)
#define OOB_RECV(id,iov,count) \
    g.get_subgroup<P2PPerfSubgroupType>().oob_recv(id,iov,count)

#define PERF_OOB_TIMEOUT_US (5000000)

#define OOB_WAIT_SEND(id,to_us) \
    g.get_subgroup<P2PPerfSubgroupType>().wait_for_oob_op(id,OOB_OP_SEND,to_us)
#define OOB_WAIT_RECV(id,to_us) \
    g.get_subgroup<P2PPerfSubgroupType>().wait_for_oob_op(id,OOB_OP_RECV,to_us)

#define __OOB_SEND \
    { \
        struct iovec siov; \
        siov.iov_base = __BUF_PTR__(pool_ptr,size_byte,depth,count); \
        siov.iov_len  = size_byte; \
        OOB_SEND(peer_id,&siov,1); \
        pending ++; \
        count ++; \
    }

#define RUN_WITH_DURATION(sec) \
    std::cout << "duration=" << sec << std::endl; \
    till    = get_time() + (sec)*1000000000ll; \
    do { \
        cur = get_time(); \
        while (cur < till && pending < depth) { \
            TIMESTAMP(TT_OOB_SEND,my_rank,count); \
            __OOB_SEND; \
            cur = get_time(); \
        } \
        while (pending > 0) { \
            try { \
                OOB_WAIT_SEND(peer_id,0); \
                TIMESTAMP(TT_OOB_ACKD,my_rank,acked++); \
                pending --; \
            } catch (derecho::derecho_exception& ex) { \
                break; \
            } \
        } \
    } while( (cur < till) || (pending > 0) );

    if (my_rank == 0) { // sender
        std::cout << "Start as a sender..." << std::endl;
        uint64_t cur;
        uint64_t till;
        size_t count = 0;
        size_t acked = 0;
        size_t pending = 0;

        // STEP 3.1: warmup
        std::cout << "Warming up" << std::endl;
        TIMESTAMP(TT_OOB_WARMUP_START,my_rank,0);
        RUN_WITH_DURATION(warmup_sec);
        TIMESTAMP(TT_OOB_WARMUP_END,my_rank,0);

        // STEP 3.2: run
        std::cout << "Running test" << std::endl;
        TIMESTAMP(TT_OOB_TEST_START,my_rank,0);
        RUN_WITH_DURATION(duration_sec);
        TIMESTAMP(TT_OOB_TEST_END,my_rank,0);

        // STEP 3.3: done
        std::cout << "Test done. sent " << count << " messages." << std::endl;
#ifdef CUDA_FOUND
        if (cuda_dev >= 0) {
            void* tptr = malloc(pool_size);
            if (tptr == nullptr) {
                std::cerr << "Failed to allocated temp memory" << strerror(errno) << std::endl;
                return -1;
            }
            memset(tptr,0xff,pool_size);
            ASSERTDRV(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(pool_ptr),tptr,pool_size));
            free(tptr);
        } else {
#endif
            memset(pool_ptr,0xff,pool_size);
#ifdef CUDA_FOUND
        }
#endif
        while(pending<depth) {
            __OOB_SEND;
        }
        while(pending > 0) {
            OOB_WAIT_SEND(peer_id,PERF_OOB_TIMEOUT_US);
            pending --;
        }
        std::cout << "Sender finished with total " << count << " messages." << std::endl;
        FLUSH_AND_CLEAR_TIMESTAMP("oob.dat");
    } else { // receiver
        std::cout << "Start as a receiver..." << std::endl;
        size_t nrecv = 0;
        size_t npost = 0;
        struct iovec riov;
        riov.iov_len = size_byte;

        while( npost < depth) {
            riov.iov_base = __BUF_PTR__(pool_ptr,size_byte,depth,npost);
            OOB_RECV(peer_id,&riov,1);
            npost ++;
        }
        while(npost > nrecv) {
            OOB_WAIT_RECV(peer_id,PERF_OOB_TIMEOUT_US);
#ifdef CUDA_FOUND
            if (cuda_dev >= 0) {
                uint8_t test_byte;
                ASSERTDRV(cuMemcpyDtoH(static_cast<void*>(&test_byte),
                                       reinterpret_cast<CUdeviceptr>(__BUF_PTR__(pool_ptr,size_byte,depth,nrecv)),
                                       1));
                if (test_byte != 0xFF) {
                    riov.iov_base = __BUF_PTR__(pool_ptr,size_byte,depth,npost);
                    OOB_RECV(peer_id,&riov,1);
                    npost ++;
                }
            } else {
#endif
                if (*static_cast<uint8_t*>(__BUF_PTR__(pool_ptr,size_byte,depth,nrecv)) != 0xFF) {
                    riov.iov_base = __BUF_PTR__(pool_ptr,size_byte,depth,npost);
                    OOB_RECV(peer_id,&riov,1);
                    npost ++;
                }
#ifdef CUDA_FOUND
            }
#endif
            nrecv ++;
        }
        // done.
        std::cout << "Receiver finished." << std::endl;
    }

    // STEP 4: finish.
    g.deregister_oob_memory(pool_ptr);
    g.barrier_sync();

#ifdef	CUDA_FOUND
    if (cuda_dev >= 0) {
        ASSERTDRV(cuMemFree(reinterpret_cast<CUdeviceptr>(pool_ptr)));
        ASSERTDRV(cuDevicePrimaryCtxRelease(cuda_device));
    } else {
#endif
        free(pool_ptr);
#ifdef  CUDA_FOUND
    }
#endif
    return 0;
}

const char* help_string = 
    "\t--transport,-t       name of the p2p transport driver. This option is mandatory.\n"
    "\t                     Transport choices: ucx,oob\n"
#ifdef	CUDA_FOUND
    "\t--cuda,-c            using gpu memory on specified cuda device.\n"
#endif
    "\t--size,-s            message size in bytes, default to 1024 bytes.\n"
    "\t--depth,-d           window deption, default to 16.\n"
    "\t--warmup,-w          duration of the warming up in seconds, default to one second.\n"
    "\t--duration,-D        duration for each of the evaluation in seconds, default to five seconds.\n"
    "\t--help,-h            print this message.\n";

static void print_help(const char* cmd) {
    std::cout << "Usage: " << cmd << " [options]" << std::endl;
    std::cout << help_string << std::endl;
}

int main(int argc, char** argv) {
    // step 0 - parameters
    static struct option long_options[] = {
        {"transport",   required_argument,  0,  't'},
#ifdef	CUDA_FOUND
	{"cuda",        required_argument,  0,  'c'},
#endif
        {"size",        required_argument,  0,  's'},
        {"depth",       required_argument,  0,  'd'},
        {"warmup",      required_argument,  0,  'w'},
        {"duration",    required_argument,  0,  'D'},
        {0,0,0,0}
    };

    int c;

    std::string transport;
#ifdef	CUDA_FOUND
    int32_t	cuda_dev = -1;
#endif
    size_t      size_byte = 1024;
    size_t      depth = 16;
    uint32_t    warmup_sec = 1;
    uint32_t    duration_sec = 5;
    int         ret = 0;
    
    while (true) {
        int option_index = 0;
        c = getopt_long(argc,argv, "t:c:s:d:w:D:h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch(c) {
        case 't':
            transport = optarg;
            break;
#ifdef	CUDA_FOUND
	    case 'c':
	        cuda_dev = std::stol(optarg);
	        break;
#endif
        case 's':
            size_byte = std::stol(optarg);
            break;
        case 'd':
            depth = std::stol(optarg);
            break;
        case 'w':
            warmup_sec = std::stoul(optarg);
            break;
        case 'D':
            duration_sec = std::stoul(optarg);
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        default:
            break;
        }
    }

    if (transport.empty()) {
        std::cerr << "Error:--transport,-t argument is required." << std::endl; 
        print_help(argv[0]);
        return 1;
    }

    if (transport != "oob" && transport != "ucx") {
        std::cerr << "Error: transport must be one of 'oob' and 'ucx'." << std::endl;
        print_help(argv[0]);
        return 2;
    }

    std::cout << "Evaluating transport performance with the following configuration:" << std::endl;
    std::cout << "\ttransport:  " << transport << std::endl;
#ifdef CUDA_FOUND
    std::cout << "\tcuda dev:   " << cuda_dev << std::endl;
#endif
    std::cout << "\tsize:       " << size_byte << std::endl;
    std::cout << "\tdepth:      " << depth << std::endl;
    std::cout << "\twarmup:     " << warmup_sec << std::endl;
    std::cout << "\tduration:   " << duration_sec << std::endl;

    if (transport == "oob") {
        ret = oob_perf(
#ifdef CUDA_FOUND
                       cuda_dev,
#endif
		       size_byte,depth,warmup_sec,duration_sec);
    } else {
        std::cerr << "'" << transport << "' support is under construction." << std::endl;
        return 3;
    }

    return ret;
}
/**
 * @endcond
 */
