#include <derecho/core/derecho.hpp>
#include <derecho/mutils-serialization/SerializationSupport.hpp>
#include <derecho/persistent/detail/PersistLog.hpp>
#include <derecho/utils/time.h>
#include <dccl/dccl.hpp>

using namespace derecho;
using namespace dccl;

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

static int oob_perf(size_t      size_byte,
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
    if (posix_memalign(&pool_ptr,4096,pool_size)) {
        std::cerr << "Failed to allocate memory:" << strerror(errno) << std::endl;
        return -1;
    }
    bzero(pool_ptr,pool_size);
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
    till    = get_time() + (sec)*1000000000; \
    do { \
        cur = get_time(); \
        while (cur < till) { \
            if (pending < depth) { \
                __OOB_SEND; \
            } \
            cur = get_time(); \
        } \
        while (pending > 0) { \
            try { \
                OOB_WAIT_SEND(peer_id,1); \
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
        size_t pending = 0;

        // STEP 3.1: warmup
        std::cout << "Warming up" << std::endl;
        RUN_WITH_DURATION(warmup_sec);

        // STEP 3.2: run
        std::cout << "Running test" << std::endl;
        RUN_WITH_DURATION(duration_sec);

        // STEP 3.3: done
        std::cout << "Test done." << std::endl;
        memset(pool_ptr,0xff,pool_size);
        while(pending<depth) {
            __OOB_SEND;
        }
        while(pending > 0) {
            OOB_WAIT_SEND(peer_id,PERF_OOB_TIMEOUT_US);
            pending --;
        }
        std::cout << "Sender finished." << std::endl;
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
            if (*static_cast<uint8_t*>(__BUF_PTR__(pool_ptr,size_byte,depth,nrecv)) != 0xFF) {
                riov.iov_base = __BUF_PTR__(pool_ptr,size_byte,depth,npost);
                OOB_RECV(peer_id,&riov,1);
                npost ++;
            }
            nrecv ++;
        }
        // done.
        std::cout << "Receiver finished." << std::endl;
    }

    // STEP 4: finish.
    g.deregister_oob_memory(pool_ptr);
    g.barrier_sync();
    return 0;
}

const char* help_string = 
    "\t--transport,-t       name of the p2p transport driver. This option is mandatory.\n"
    "\t                     Transport choices: ucx,oob\n"
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
        {"size",        required_argument,  0,  's'},
        {"depth",       required_argument,  0,  'd'},
        {"warmup",      required_argument,  0,  'w'},
        {"duration",    required_argument,  0,  'D'},
        {0,0,0,0}
    };

    int c;

    std::string transport;
    size_t      size_byte = 1024;;
    size_t      depth = 16;
    uint32_t    warmup_sec = 1;
    uint32_t    duration_sec = 5;
    int         ret = 0;
    
    while (true) {
        int option_index = 0;
        c = getopt_long(argc,argv, "t:s:d:w:D:h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch(c) {
        case 't':
            transport = optarg;
            break;
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
    std::cout << "\tsize:       " << size_byte << std::endl;
    std::cout << "\tdepth:      " << depth << std::endl;
    std::cout << "\twarmup:     " << warmup_sec << std::endl;
    std::cout << "\tduration:   " << duration_sec << std::endl;

    if (transport == "oob") {
        ret = oob_perf(size_byte,depth,warmup_sec,duration_sec);
    } else {
        std::cerr << "'" << transport << "' support is under construction." << std::endl;
        return 3;
    }

    return ret;
}
