#pragma once
#include <cstdint>
#include <cstddef>
#include <dccl/config.h>
#include <pthread.h>

#ifdef ENABLE_EVALUATION
#include <string>
#endif//ENABLE_EVALUATION

/**
 * @file    dccl.hpp
 * @brief   Derecho Collective Communications Library (DCCL) API
 * @see     dccl
 *
 * This file contains the declaration of the Derecho Collective Communication Library (DCCL) API. Similar to 
 * [RCCL](https://rccl.readthedocs.io/en/develop/api.html), DCCL is compatible to 
 * [NCCL](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_296/user-guide/docs/api.html).
 * We put all DCCL API to the `dccl` namespace. 
 *
 * @mainpage Derecho Collective Communications Library (DCCL)
 *
 * DCCL implements NCCL's API but built on Derecho. The source code arrangement follows Derecho and Cascade
 * conventions.
 *
 * @section concept_sec Design Concepts
 *
 * @section apiov_sec API Overview
 *
 * @section rm_sec Roadmap
 */

/**
 * @brief option key in derecho.cfg
 */
#define DCCL_ALLREDUCE_ALGORITHM_CONFSTR    "DCCL/allreduce_algorithm"
/**
 * @brief option value in derecho.cfg
 */
#define DCCL_ALLREDUCE_RING                 "ring"
/**
 * @brief option value in derecho.cfg
 */
#define DCCL_ALLREDUCE_RABINSEIFNER         "rabinseifner"

/**
 * @brief DCCL namespace contains all API declaration and definition.
 *
 * We put the declarations of all DCCL APIs, as well as their definitions in to this namespace. Using 
 * `using namespace dccl;` is recommended for application code.
 */
namespace dccl {

/**
 * @brief   Error types
 */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8 } ncclResult_t;

/**
 * @brief   Data types 
 */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
               ncclBfloat16   = 9,
               ncclNumTypes   = 10
#else
               ncclNumTypes   = 9
#endif
} ncclDataType_t;

/**
 * @brief   The dummy Operator.
 */
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;

/**
 * @brief   Reduction operation selector
 */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclAvg        = 4,
               /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
                * serves as the least possible value for dynamic ncclRedOp_t's
                * as constructed by ncclRedOpCreate*** functions. */
               ncclNumOps     = 5,
               /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
                * maintain ABI compatibility. */
               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t))
             } ncclRedOp_t;

/**
 * @brief   The opaque DCCL communicator struct
 */
struct dcclComm;
/**
 * @brief   DCCL communicator type
 */
typedef struct dcclComm* ncclComm_t ;

/* dccl init rank is slightly different, the rank and world size information
 * in in the configuration file. So we dont need them here. */
/*
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
*/

/**
 * @defgroup api The DCCL APIs
 * @{
 */

/**
 * @brief   Initialize the communication infrastructure.
 * @param[out]  comm    Pointer to a `ncclComm_t` object to be initialized
 *
 * @return      Error code
 */
ncclResult_t  ncclCommInit(ncclComm_t* comm);

/**
 * @brief   Finalize the communication infrastructure
 *
 * @param[in]   comm    An initialzied `ncclComm_t` object to be finalized.
 *
 * @return      Error code
 */
ncclResult_t  ncclCommFinalize(ncclComm_t comm);

/**
 * @brief   Register user-allocated memory as cache to DCCL.
 * Using registered memory in DCCL can avoid on-demand registeration on the critical path. We recommend doing so to
 * improve the performance. Also, the address and size of memory  registered has to be cacheline aligned.
 *
 * @param[in]   comm    An initialzied `ncclComm_t` object to be finalized.
 * @param[in]   buffer  The address of the memory buffer to be registered.
 * @param[in]   size    The size of the buffer.
 *
 * @throws      Runtime error might be raised on error.
 * @return      Error code.
 */
ncclResult_t  dcclRegisterCacheMemory(ncclComm_t comm, void* buffer, size_t size);

/**
 * @brief   Deregister a pre-registered user-allocated memory region from DCCL.
 *
 * @param[in]   comm    An initialzied `ncclComm_t` object to be finalized.
 * @param[in]   buffer  The address of the registered memory buffer.
 * @param[in]   size    The size of the buffer. Additional check on size would be performed if given.
 *
 * @throws      Runtime error might be raised on error.
 * @return      Error code.
 */
ncclResult_t  dcclDeregisterCacheMemory(ncclComm_t comm, void* buffer, size_t size = 0UL);

/**
 * @brief All-Reduce API
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 *
 * @param[in]   sendbuff    The buffer containing local data to be reduced.
 * @param[out]  recvbuff    The buffer to receive the reduced result.
 * @param[in]   count       The number of entries in the buffer.
 * @param[in]   datatype    The type of the data.
 * @param[in]   op          The reduced operation to be performed.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm);

/**
 * @brief Reduce-Scatter API
 *
 * This API is compatible to NVIDIA's NCCL:
 *
 * " Reduces data in sendbuff using op operation and leaves reduced result
 *   scattered over the devices so that recvbuff on rank i will contain the i-th
 *   block of the result.
 *   Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 *   should have a size of at least nranks*recvcount elements.
 *
 *   In-place operations will happen if recvbuff == sendbuff + rank * recvcount. "
 *
 * Please see NCCL's API in
 * [nccl.h](https://github.com/NVIDIA/nccl/blob/6e24ef4e1f1eac9f104d115ef65429f179924ee7/src/nccl.h.in#L311-L321).
 *
 * @param[in]   sendbuff    The buffer containing local data to be reduced.
 * @param[out]  recvbuff    The buffer to receive the reduced result.
 * @param[in]   recvcount   The number of entries in the receive buffer.
 * @param[in]   datatype    The type of the data.
 * @param[in]   op          The reduced operation to be performed.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm);

/**
 * @brief Broadcast API
 *
 * This API is compatible to NVIDIA's NCCL:
 * " Copies count values from root to all other devices.
 *   root is the rank (not the CUDA device) where data resides before the
 *   operation is started.
 *
 *   In-place operation will happen if sendbuff == recvbuff. "
 *
 * IMPORTANT: We assume that calls to broadcast arrive in the same order on all nodes.
 * IMPORTANT: Although our inner implementation can be non-blocking: a call to broadcast returns a uint64_t bcast_id,
 *            which can be used to query the state of broadcast later, we didn't do it yet. Why? let's say we have 
 *            two nodes in the system A and B. Both A and B calls the following broadcast in the same order:
 *            -# bid1 = nb_broadcast(...,root=A,...);
 *            -# bid2 = nb_broadcast(...,root=B,...);
 *            -# wait_for(bid1);
 *            -# wait_for(bid2);
 *            There is no guarantee that node A will broadcast earlier than node B without a causality between A and B.
 *            However, a series of broadcasts sent by a same thread can leverage the non-blocking mechanism because the
 *            sending thread introduce causality. Due to the above complexity, we decide to only expose blocking API so
 *            far.
 * IMPORTANT: Due to the current Derecho design, we DO need copy data on send and receive. This is going to be solved
 *            in our real-zerocopy design.
 *
 * @param[in]   sendbuff    The buffer containing local data to be sent.
 * @param[out]  recvbuff    The buffer to receive the data.
 * @param[in]   count       The number of entries in the receive buffer.
 * @param[in]   datatype    The type of the data.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm);

/**
 * @brief Bcast API
 *
 * This API is compatible to NVIDIA's NCCL
 *
 * @param[in/out]   buff    The buffer to receive the data.
 * @param[in]   count       The number of entries in the receive buffer.
 * @param[in]   datatype    The type of the data.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclBcast(void* buff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm);

/**
 * @brief Reduce API
 *
 * This API is compatible to NVIDIA's NCCL:
 * " Reduces data arrays of length count in sendbuff into recvbuff using op
 *   operation.
 *   recvbuff may be NULL on all calls except for root device.
 *   root is the rank (not the CUDA device) where data will reside after the
 *   operation is complete.
 *
 *   In-place operation will happen if sendbuff == recvbuff."
 *
 * @param[in]   sendbuff    The buffer containing local data to reduce.
 * @param[out]  recvbuff    The buffer receiving reduced data.
 * @param[in]   count       The number of entries in the receive buffer.
 * @param[in]   datatype    The type of the data.
 * @param[in]   op          The reduce operation to be performed.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm);

/**
 * @brief AllGather API
 *
 * This API is compatible to NVIDIA's NCCL:
 * " Each device gathers sendcount values from other GPUs into recvbuff,
 *   receiving data from rank i at offset i*sendcount.
 *   Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 *   should have a size of at least nranks*sendcount elements.
 *
 *   In-place operations will happen if sendbuff == recvbuff + rank * sendcount."
 *
 * @param[in]   sendbuff        The buffer containing the local data to gather.
 * @param[out]  recvbuff        The buffer receiving gathered data.
 * @param[in]   sendcount       The number of data entries in the send buffer.
 * @param[in]   datatype        The type of the data.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm);

/**
 * @brief Point-to-Point send
 *
 * @param[in]   sendbuff        The buffer containing the local data to send.
 * @param[in]   count           The number of data entries in the send buffer.
 * @param[in]   datatype        The type of the data.
 * @param[in]   peer            The rank of the receiver.
 * @param[in]   comm            The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm);

/**
 * @brief Point-to-Point recv
 *
 * @param[in]   recvbuff        The buffer receiving the data from peer.
 * @param[in]   count           The number of data entries in the buffer.
 * @param[in]   datatype        The type of the data.
 * @param[in]   peer            The rank of the receiver.
 * @param[in]   comm            The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm);

/**
 * @}
 */

/**
 * @defgroup helpers DCCL helper functions and macros
 * @{
 */

/**
 * @brief Test the offset of an address in a cacheline.
 *
 * @param[in]   addr        Some address of any pointer type
 *
 * @return      The offset of `addr` in a cacheline.
 */
#define CACHELINE_OFFSET(addr)  ( ((uint64_t)addr)%CACHELINE_SIZE)

/**
 * @brief Get the world size
 *
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      The number of members in the shard.
 */
uint32_t dcclGetWorldSize(ncclComm_t comm);

/**
 * @brief Get my rank
 *
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      My rank in the shard, starting from 0.
 */
uint32_t dcclGetMyRank(ncclComm_t comm);

/**
 * @brief Evaluation utilities
 */
#ifdef ENABLE_EVALUATION

/**
 * @brief The timestamper implementation.
 * Modified from [`derecho::cascade::TimestampLogger](https://github.com/Derecho-Project/cascade/blob/041639aab47a51b1ad12fabe13056a91d9ed4bda/include/cascade/utils.hpp#L438-L494)
 */
class Timestamp {
private:
    /**
     * @brief Timestamp storage
     * -# my rank
     * -# tag id
     * -# timestamp in nanosecond
     * -# extra info
     */
    uint64_t* _log;

    /**
     * @brief capacity (in entry number) of the log
     */
    size_t capacity;

    /**
     * @brief the current entry position
     */
    size_t position;

    /**
     * @brief   Timestamp spinlock
     */
    pthread_spinlock_t lck;

    /**
     * @brief   Constructor
     * @param   num_entries     The number of entries in the timestamp, defaulted to 2^24
     */
    Timestamp(size_t num_entries = 0);

    /**
     * @brief   Log the timestamp
     *
     * @param   tag         Event tag, a.k.a event identifier
     * @param   rank        My rank
     * @param   extra       Optional extra information
     */
    void instance_log(uint64_t tag, uint64_t rank, uint64_t extra = 0ull);

    /**
     * @brief   Flush the timestamps into a file
     *
     * @param   filename    The nameof the file
     * @param   clear       Clear the log after flush if `clear == true`.
     */
    void instance_flush(const std::string& filename, bool clear=true);

    /**
     * @brief   Clear the timestamps
     */
    void instance_clear();

    /**
     * @brief The timestamp singleton.
     */
    static Timestamp _t;

public:
    /**
     * @brief   The destructor
     */
    virtual ~Timestamp();

    /**
     * @brief log timestamp
     *
     * @param[in]   tag         Event tag, a.k.a event identifier
     * @param[in]   rank        My rank
     * @param[in]   extra       Optional extra information
     */
    static inline void log(uint64_t tag, uint64_t rank, uint64_t extra = 0ull) {
        _t.instance_log(tag,rank,extra);
    }
    
    /**
     * @brief Flush the timestamps into a file
     *
     * @param[in]   filename    The nameof the file
     * @param[in]   clear       Clear the log after flush if `clear == true`.
     */
    static inline void flush(const std::string& filename, bool clear=true) {
        _t.instance_flush(filename,clear);
    }

    /**
     * @brief clear the timestamps
     */
    static inline void clear() {
        _t.instance_clear();
    }
};

/**
 * @cond    DoxygenSuppressed
 * The DCCL timestamp tag is prefixed by 1000000
 */
#define     TT_DCCL(x)                      (1000000 + (x))

// OVERALL
#define     TT_WARMUP_START                 TT_DCCL(0001)
#define     TT_WARMUP_END                   TT_DCCL(0002)
#define     TT_TEST_START                   TT_DCCL(0003)
#define     TT_TEST_END                     TT_DCCL(0004)

// ALLREDUCE
#define     TT_ALLREDUCE_ENTER              TT_DCCL(1001)
#define     TT_ALLREDUCE_MEMCPY             TT_DCCL(1020)
#define     TT_ALLREDUCE_RDH_PREPROCESS     TT_DCCL(1021)
#define     TT_ALLREDUCE_REDUCESCATTER      TT_DCCL(1030)
#define     TT_ALLREDUCE_ALLGATHER          TT_DCCL(1040)
#define     TT_ALLREDUCE_RDH_POSTPROCESS    TT_DCCL(1041)
#define     TT_ALLREDUCE_DONE               TT_DCCL(1100)
/**
 * @endcond
 */

/**
 * @brief Timestamp::log() syntax sugar
 * Using this macro to avoid `#ifdef ENABLE_EVALUATION` in the code.
 * @param[in]   tag     Event tag, a.k.a event identifier
 * @param[in]   rank    My rank
 * @param[in]   extra   Optional extra information
 */
#define TIMESTAMP(tag,rank,extra)           Timestamp::log(tag,rank,extra)

/**
 * @brief Timestamp::flush() syntax sugar
 * Using this macro to avoid `#ifdef ENABLE_EVALUATION` in the code.
 * @param[in]   filename    The nameof the file
 * @param[in]   clear       Clear the log after flush if `clear == true`.
 */
#define FLUSH_AND_CLEAR_TIMESTAMP(filename) Timestamp::flush(filename,true)

/**
 * @brief Timestamp::clear() syntax sugar
 * Using this macro to avoid `#ifdef ENABLE_EVALUATION` in the code.
 */
#define CLEAR_TIMESTAMP()                   Timestamp::clear()

#else
/**
 * @cond DoxygenSuppressed
 */
#define TIMESTAMP(tag,rank,extra)
#define FLUSH_AND_CLEAR_TIMESTAMP(filename)
#define CLEAR_TIMESTAMP()

#define TIMESTAMP(tag,rank,extra)
/**
 * @endcond
 */
#endif//ENABLE_EVALUATION
/**
 * @}
 */

} // namespace dccl
