#pragma once

/**
 * @file    internal_common.hpp
 * @brief   Internal utilities (types, macros, and functions) which be hidden from DCCL API users.
 *
 * This file contains the internal types, macros, and functions used by multiple C++ files. Those utilities are for
 * internal use only and should be hidden from the users.
 */
#include <atomic>
#include <derecho/core/derecho.hpp>
#include <dccl/dccl.hpp>
#include "blob.hpp"
#include <mutex>
#include <condition_variable>

using namespace derecho;

namespace dccl {

/**
 * @brief Get the DCCL `spdlog` logger singleton
 * @return      A shared pointer to the logger, that can be used with Derecho's logger macros like the following:
 *              dbg_trace, dbg_debug, dbg_warn, dbg_error, ...
 */
std::shared_ptr<spdlog::logger>& getDcclLogger();
/**
 * @cond Doxygen_Suppressed
 */
#define dccl_trace(...) dbg_trace(getDcclLogger(), __VA_ARGS__)
#define dccl_debug(...) dbg_debug(getDcclLogger(), __VA_ARGS__)
#define dccl_info(...)  dbg_info(getDcclLogger(), __VA_ARGS__)
#define dccl_warn(...)  dbg_warn(getDcclLogger(), __VA_ARGS__)
#define dccl_error(...) dbg_error(getDcclLogger(), __VA_ARGS__)
#define dccl_crit(...)  dbg_crit(getDcclLogger(), __VA_ARGS__)
#define dccl_flush()    dbg_flush(getDcclLogger())
/**
 * @endcond
 */

/**
 * @brief   The timeout for oob transfer.
 * Default to 5 seconds, it should be enough for transfer 50GB data using 100Gb Link.
 */
#define DCCL_OOB_TIMEOUT_US     5000000

/**
 * @brief The DCCL Subgroup Class type
 * It defines the Derecho Subgroup type that supports the DCCL APIs.
 */
class DCCLSubgroupType : public mutils::ByteRepresentable,
                         public GroupReference {
public:
    /**
     * @brief   The receive buffer
     * It has to be atomic for concurrent access.
     */
    std::atomic<void*>  recvbuf;

    /**
     * @brief       Reduce operation
     *              It calculate the reduce value and put it into the buffer `recvbuf` pointed to. It is the caller's
     *              responsibility to make sure `recvbuf` is valid and registered with and OOB
     *
     * @todo        This implementation was used for AllReduce implementation, whick was an unsophisticated and
     *              inefficient API. Change it to a reduce API that accept one more param specifying which node
     *              will do the operation.
     *
     */
    virtual ncclResult_t reduce(const Blob& sendbuf, const size_t, ncclDataType_t datatype, ncclRedOp_t op, bool inplace);

    /**
     * @brief Register the RPC functions.
     */
    REGISTER_RPC_FUNCTIONS(DCCLSubgroupType,ORDERED_TARGETS(reduce));

    /**
     * serialization support
     * @cond    Doxygen_Suppressed
     */
    virtual std::size_t to_bytes(uint8_t*) const override {return 0;}

    virtual void post_object(const std::function<void(uint8_t const* const, std::size_t)>&) const override {}

    virtual std::size_t bytes_size() const {return 0;}

    static std::unique_ptr<DCCLSubgroupType> from_bytes(
            mutils::DeserializationManager*, const uint8_t*) {
        return std::make_unique<DCCLSubgroupType>();
    }

    static mutils::context_ptr<DCCLSubgroupType> from_bytes_noalloc(
            mutils::DeserializationManager*, const uint8_t*) {
        return mutils::context_ptr<DCCLSubgroupType>{new DCCLSubgroupType()};
    }
    static mutils::context_ptr<const DCCLSubgroupType> from_bytes_noalloc_const(
            mutils::DeserializationManager*, const uint8_t*) {
        return mutils::context_ptr<const DCCLSubgroupType>{new DCCLSubgroupType()};
    }
    void ensure_registered(mutils::DeserializationManager&) {}
    /**
     * @endcond
     */
    
    /**
     * @brief   The default constructor
     */
    DCCLSubgroupType():recvbuf(nullptr) {}
};

/**
 * @brief the DCCL communicator type definition
 *
 * We put it in source to hide it from DCCL applications.
 */
class dcclComm {
public:
    /**
     * @brief   Pointer to the derecho group
     */
    void* derecho_group_handle;
    /**
     * @brief   Pointer to the `DCCLSubgroupType` object living in Derecho.
     */
    void* derecho_group_object;

    /**
     * @brief   broadcast delivery states.
     */
    typedef enum {
        nonexist,
        undelivered,
        delivered,
        failed,
    } bcast_delivery_state_t;
private:
    /**
     * @brief   broadcast queue mutex
     *
     * Broadcast queue is used for coordination between the broadcast sender and receiver. The caller of 
     * `ncclBroadcast` or `ncclBcast` register the receiving buffer to the broadcast buffer map so that, once
     * a message is recieved (by derecho predicate thread), the data will be moved to the receiving buffer.
     *
     * If the data arrives earlier, the predicate thread will wait on the broadcast buffer map before 
     * buffer is received.
     *
     * TODO: this mechanism should be improved in the following ways
     * 1) We should use lockless design to avoid blocking the predicate thread.
     * 2) Update the SST/RDMA design for real zero-copy. Currently, we HAVE TO COPY because the internal derecho
     *    buffers are not usable for application. And this introduces overhead!!!
     */
    std::mutex                          broadcast_queue_mutex;
    /**
     * @brief   delivery state mutex
     */
    std::mutex                          delivery_state_mutex;
    /**
     * @brief   broadcast queue condition variable
     */
    std::condition_variable             broadcast_queue_cv;
    /**
     * @brief   delivery state condition variable
     */
    std::condition_variable             delivery_state_cv;
    /**
     * @brief   broadcast queue
     * a queue of 3-tuple (broadcast_id,buffer ptr,buffer size).
     */
    std::queue<std::tuple<uint64_t,void*,size_t>>   broadcast_queue;
    /**
     * @brief   broadcast id generator
     */
    uint64_t    bcast_id_seed;

    /**
     * @brief   broadcast delivery state table
     * The delivery state table is a map from broadcast_id to its delivery state.
     */
    std::unordered_map<uint64_t,bcast_delivery_state_t>               delivery_state;

public:
    /**
     * Constructor
     */
    dcclComm();
    /**
     * Destructor
     */
    virtual ~dcclComm();

    /**
     * @brief post broadcast receive buffer
     * This function is only called by ncclBroadcast/ncclBcast to pose a receiving buffer.
     *
     * @param[in]   recvbuff        pointer to the receive buffer
     * @param[in]   len             length of the receive buffer
     * 
     * @return      broadcast id    a token for retrieving the received buffer later.
     */
    uint64_t post_bcast_recv_buff(void* recvbuff,size_t len);
    /**
     * @brief handle a broadcast receiving
     *
     * @param[in]   data_generator  An r-value reference to a lambda to fill the buffer for the broadcast.
     *                              The parameters to the lambda are pointer/size. It returns false on error.
     */
    void on_bcast(const std::function<bool(void*,const size_t&)>& data_generator);
    /**
     * @brief query the state of a bcast.
     *
     * This query will return immediately reporting the broadcast's current state.
     * 
     * @param[in]   bcast_id        A value representing the broadcast state
     *
     * @return      state of the corresponding broadcast
     */
    bcast_delivery_state_t query_bcast(const uint64_t& bcast_id);
    /**
     * @brief blockingly query the state of a broadcast.
     *
     * The query will wait until the corresponding broadcast is not in `undelivered` state.
     *
     * @param[in]   bcast_id        A value representing the broadcast state
     *
     * @return      state of the corresponding broadcast
     */
    bcast_delivery_state_t wait_bcast(const uint64_t& bcast_id);
    /**
     * @brief clear the broadcast id from delivery table
     * IMPORTANT: If bcast_id in undelivered state, it will not be removed.
     *
     * @param[in]   bcast_id        A value representing the broadcast state
     *
     * @return      removed state of the corresponding broadcast id. 
     */
    bcast_delivery_state_t clear_bcast(const uint64_t& bcast_id);
};

/**
 * @brief Get the group handle from the DCCL communication object.
 *
 * @param[in]   comm        The DCCL communication object.
 *
 * @return      A pointer to `Group<DCCLSubgroupType>` object.
 */
#define GROUP_HANDLE(comm)      (reinterpret_cast<Group<DCCLSubgroupType>*>((comm)->derecho_group_handle))

/**
 * @brief Get the subgroup handle from the communication object.
 *
 * @param[in]   comm        The DCCL communication object.
 *
 * @return      A reference to `Replicated<DCCLSubgroupType>` object.
 */
#define SUBGROUP_HANDLE(comm)   (GROUP_HANDLE(comm)->get_subgroup<DCCLSubgroupType>())

/**
 * @brief Get the wrapped subgroup object.
 *
 * @param[in]   comm        The DCCL communication object.
 *
 * @return      A pointer to the wrapped object of type DCCLSubgroupType.
 */
#define SUBGROUP_OBJECT(comm)   (reinterpret_cast<DCCLSubgroupType*>((comm)->derecho_group_object))

/**
 * @brief Get the shard members
 *
 * @tparam      SubgroupType    The type of the derecho subgroup. For now, it must be `DCCLSubgroupType`
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      An std::vector object contains all node ids in my shard.
 */
template <typename SubgroupType>
inline auto _get_shard_members(ncclComm_t comm) {
    auto* group = GROUP_HANDLE(comm);
    auto subgroup_indexes = group->get_my_subgroup_indexes<SubgroupType>();

    assert(subgroup_indexes.size() == 1);

    uint32_t subgroup_index = subgroup_indexes.at(0);
    int32_t my_shard = group->get_my_shard<SubgroupType>(subgroup_index);
    return group->get_subgroup_members<SubgroupType>(subgroup_index)[my_shard];
}

/**
 * @brief A wrapper around _get_shard_members as syntax sugar
 *
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      An std::vector object contains all node ids in my shard.
 */
inline auto get_dccl_shard_members(ncclComm_t comm) {
    return _get_shard_members<DCCLSubgroupType>(comm);
}

/**
 * @brief Get the size (in bytes) of DCCL data types.
 * 
 * @param[in]   datatype    The DCCL data type
 *
 * @return      Size of the given data type in bytes. Returns `0` if the data type is invalid.
 */
inline size_t size_of_type(ncclDataType_t datatype) {
    switch(datatype) {
    case ncclInt8: // ncclChar
    case ncclUint8:
        return 1;
    case ncclInt32: // ncclInt
    case ncclUint32:
    case ncclFloat32: // ncclFloat
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64: // ncclDouble
        return 8;
    case ncclFloat16: // ncclHalf:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
        return 2;
    default:
        return 0;
    }
}

/**
 * @brief Macro expanding for specific DCCL data type.
 * TODO: we should support Float16 here using something like:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
 *  case ncclFloat16:
 * #if __cplusplus >= 202302L
 *      expr<float16_t>(__VA_ARGS__);
 * #endif
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~
 * @param[in]   datatype        The data type
 * @param[in]   expr            The expression suffixed with `<data type>`, like `<int32_t>`
 * @param[in]   ...             the arguments passed to `expr<data type>()`
 */
#define ON_DCCL_DATATYPE(datatype, expr, ... ) \
    switch (datatype) { \
    case ncclInt8: \
        expr<int8_t>(__VA_ARGS__); \
        break; \
    case ncclUint8: \
        expr<uint8_t>(__VA_ARGS__); \
        break; \
    case ncclInt32: \
        expr<int32_t>(__VA_ARGS__); \
        break; \
    case ncclUint32: \
        expr<uint32_t>(__VA_ARGS__); \
        break; \
    case ncclInt64: \
        expr<int64_t>(__VA_ARGS__); \
        break; \
    case ncclUint64: \
        expr<uint64_t>(__VA_ARGS__); \
        break; \
    case ncclFloat32: \
        expr<float>(__VA_ARGS__); \
        break; \
    case ncclFloat64: \
        expr<double>(__VA_ARGS__); \
        break; \
    case ncclFloat16: \
    default: \
        break; \
    }

/**
 * @brief Reverse the least significants bits of an integer.
 * This function reverse the order of least significant `nbits` bits in an integer.
 * For example,
 * - `0B*0000101` --[reversed with `nbits=7`]--> `0B*1010000`,
 * - `0B****1110` --[reversed with `nbits=4`]--> `0B****0111`,
 * - `0B*****011` --[reversed with `nbits=3`]--> `0B*****110`,
 *
 * The asterisk signs(*) represent irrelavent/unchanged bits.
 *
 * @tparam      T       Type of the integer, could be any primitive integer type.
 * @param[in]   x       the value to be reversed.
 * @param[in]   nbits   optional `nbits` specifies the range by the number of least significant bits, which must be
 *                      less or equal to sizeof(T)*8. This argument is defaulted to sizeof(T)*8.
 *
 * @return      The reversed value.
 */
template<typename T>
inline T reverse_bits(const T x, size_t nbits = sizeof(T)*8) {
    assert(nbits <= sizeof(T)*8);
    T rx = ((x)>>nbits);
    T tx = x;

    for (size_t p=0;p<nbits;p++) {
        rx = ((rx << 1) | (tx&1));
        tx = (tx >> 1);
    }

    return rx;
}

/**
 * @brief Test if an integer is power of two.
 *
 * @param[in]   n   The integer
 *
 * @return      true if n is power of two.
 */
#define IS_POWER_OF_TWO(n) ((n > 0) && (((n)&((n)-1)) == 0))

/**
 * @brief Calculate the exponent of a power-of-two integer
 *
 * @tparam      IntegerType The type of the integer
 * @param[in]   n           The integer, which must be power of two
 *
 * @return      log2(n)
 */
template<typename IntegerType>
inline IntegerType log_two(IntegerType n) {
    IntegerType shift = 1;
    assert(n>0);
    const IntegerType nbits = sizeof(IntegerType)*8 - (std::is_signed<IntegerType>::value?1:0);
    while (shift <= nbits) {
        if (n >> shift) {
            shift++;
        } else {
            break;
        }
    }
    assert((static_cast<IntegerType>(1)<<(shift-1))<=n);
    return shift-1;
}

/**
 * @brief Perform a local reduce
 * This is an optimized reduce operation on two local buffers.
 * It performs the following operation:
 *
 * `recvbuf[i] = op(recvbuf[i],senddat[i])`
 *
 * , for `i` in `[0, count)`.
 *
 * @tparam          DT          The type of the data.
 * @param[in]       sendbuf     List of operand 1.
 * @param[in,out]   recvbuf     List of operand 2, also used to receive the reduced results.
 * @param[in]       count       The number of data entries in the algorithm.
 * @param[in]       op          The reduce operation.
 *
 * @return          Error Code
 */
template<typename DT>
ncclResult_t do_reduce(const void*  sendbuf,
                       void*        recvbuf,
                       size_t       count,
                       ncclRedOp_t  op) {
    const DT*   psend = static_cast<const DT*>(sendbuf);
    DT*         precv = static_cast<DT*>(recvbuf);

    if (reinterpret_cast<uint64_t>(sendbuf)%sizeof(DT)) {
        dccl_warn("sendbuf@{:p} is not aligned with data type:{},size={}, performance might be degraded.",
                sendbuf,typeid(DT).name(),sizeof(DT));
    }

    if (reinterpret_cast<uint64_t>(recvbuf)%sizeof(DT)) {
        dccl_warn("recvbuf@{:p} is not aligned with data type:{},size={}, performance might be degraded.",
                recvbuf,typeid(DT).name(),sizeof(DT));
    }

    /*
     * The data is arranged in the following layout:
     *
     * HHH[DDDDDDDD][DDDDDDDD]...[DDDDDDDD]TTT
     *     <--CL-->  <--CL-->     <--CL-->
     *  ^     ^                             ^
     *  |     |                             +- tail count
     *  |     +- pack count
     *  +- head_count
     *
     * Special case:
     * [...DDDDDD...]
     *     Head = count
     *     tail = 0
     * 
     * The head and tail are handled separately.
     */
    std::size_t             head_count = (CACHELINE_SIZE - reinterpret_cast<uint64_t>(recvbuf)%CACHELINE_SIZE)%CACHELINE_SIZE/sizeof(DT);
    constexpr std::size_t   pack_count = CACHELINE_SIZE/sizeof(DT); // we assume CACHELINE_SIZE%sizeof(DT) == 0
    std::size_t             num_pack = count/pack_count;
    std::size_t             tail_count = (pack_count + (count%pack_count) - head_count)%pack_count;
    // for the special case.
    if ((tail_count + head_count) > count) {
        head_count = count;
        tail_count = 0;
    }

    dccl_trace("{}:head_count={},pack_count={},num_pack={},tail_count={}",
                      __func__,head_count,pack_count,num_pack,tail_count);
/**
 * @cond Doxygen_Suppressed
 */
#define OP_SUM(r,s) (r)+=(s)
#define OP_MIN(r,s) if((r)>(s))(r)=(s)
#define OP_MAX(r,s) if((r)<(s))(r)=(s)
#define OP_PROD(r,s) (r)*=(s)
#define REDUCE(OP) \
        for(size_t i=0;i<head_count;i++) { \
            OP(precv[i],psend[i]); \
        } \
        for(size_t j=0;j<num_pack;j++) \
        for(size_t i=0;i<pack_count;i++) { \
            OP(precv[head_count+j*pack_count+i],psend[head_count+j*pack_count+i]); \
        } \
        for(size_t i=0;i<tail_count;i++) { \
            OP(precv[count-1-i],psend[count-1-i]); \
        }
/**
 * @endcond
 */
    switch(op) {
    case ncclSum:
        REDUCE(OP_SUM);
        break;
    case ncclProd:
        REDUCE(OP_PROD);
        break;
    case ncclMax:
        REDUCE(OP_MAX);
        break;
    case ncclMin:
        REDUCE(OP_MIN);
        break;
    case ncclAvg:
        // TODO: do this later.
        return ncclInvalidUsage;
        break;
    default:
        return ncclInvalidArgument;
        break;
    }
    return ncclSuccess;
}

/**
 * @brief max message size
 * libfabric has a limitation of 1GB. My test with fractus show 256MB is close to optimal for large messages.
 */
#define DCCL_OOB_MESSAGE_SIZE    (1ul<<28)

/**
 * @brief oob send/recv segmentation
 * send/recv in `DCCL_OOB_MESSAGE_SIZE` chunks. User application should not call this directly, 
 * use `dccl_oob_send` and `dccl_oob_recv` instead.
 *
 * @param[in]   comm        the DCCL communicator
 * @param[in]   _id         peer id
 * @param[in]   buf         pointer to send/recv buffer
 * @param[in]   size        the total size for send/recv
 * @param[in]   is_send     true for send, false for receive
 *
 * @return number of chunks.
 */
inline uint32_t __dccl_oob_op(ncclComm_t& comm,const node_id_t& _id,void* buf,const size_t& size,const bool is_send) {
    size_t      leftover    = size;
    uint32_t    count       = 0;

    struct iovec iov;
    iov.iov_base = buf;
    while (leftover > 0) {
        iov.iov_len     = (leftover>DCCL_OOB_MESSAGE_SIZE)?DCCL_OOB_MESSAGE_SIZE:leftover;
        if (is_send) {
            SUBGROUP_HANDLE(comm).oob_send(_id,&iov,1);
        } else {
            SUBGROUP_HANDLE(comm).oob_recv(_id,&iov,1);
        }
        iov.iov_base    = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(iov.iov_base) + iov.iov_len);
        leftover        = leftover - iov.iov_len;
        count ++;
    }

    return count;
}

/**
 * @brief oob recv
 * send/recv in `DCCL_OOB_MESSAGE_SIZE` chunks
 *
 * @param[in]   comm        the DCCL communicator
 * @param[in]   fid         peer id
 * @param[in]   buf         pointer to recv buffer
 * @param[in]   size        the total size for recv
 *
 * @return number of chunks.
 */
inline uint32_t dccl_oob_recv(ncclComm_t& comm,const node_id_t& fid,void* buf,const size_t& size) {
    return __dccl_oob_op(comm,fid,buf,size,false);
}

/**
 * @brief oob send
 * send/recv in `DCCL_OOB_MESSAGE_SIZE` chunks
 *
 * @param[in]   comm        the DCCL communicator
 * @param[in]   tid         peer id
 * @param[in]   buf         pointer to send buffer
 * @param[in]   size        the total size for send
 *
 * @return number of chunks.
 */
inline uint32_t dccl_oob_send(ncclComm_t& comm,const node_id_t& tid,void* buf,const size_t& size) {
    return __dccl_oob_op(comm,tid,buf,size,true);
}

/**
 * @brief wait for oob send
 *
 * @param[in]   comm        the DCCL communicator
 * @param[in]   tid         peer id
 * @param[in]   num_chunks  number of chunks
 */
inline void dccl_oob_wait_for_send(ncclComm_t& comm, node_id_t& tid, uint32_t num_chunks) {
    uint32_t leftover = num_chunks;
    while(leftover) {
        SUBGROUP_HANDLE(comm).wait_for_oob_op(tid,OOB_OP_SEND,DCCL_OOB_TIMEOUT_US);
        leftover --;
    }
}

/**
 * @brief wait for oob recv
 *
 * @param[in]   comm        the DCCL communicator
 * @param[in]   fid         peer id
 * @param[in]   num_chunks  number of chunks
 */
inline void dccl_oob_wait_for_recv(ncclComm_t& comm, node_id_t& fid, uint32_t num_chunks) {
    uint32_t leftover = num_chunks;
    while(leftover) {
        SUBGROUP_HANDLE(comm).wait_for_oob_op(fid,OOB_OP_RECV,DCCL_OOB_TIMEOUT_US);
        leftover --;
    }
}

#ifdef CUDA_FOUND
/**
 * @brief test if a pointer is a host pointer or a device pointer.
 * 
 * @param[in]   ptr         The pointer to be tested.
 *
 * @return      True for device pointer, false for host pointer.
 */
inline bool is_device_ptr(const void* ptr) {
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs,ptr) == cudaSuccess) {
        switch(attrs.type) {
        case cudaMemoryTypeUnregistered:
        case cudaMemoryTypeHost:
            return false;
        case cudaMemoryTypeDevice:
        case cudaMemoryTypeManaged:
            return true;
        default:
            break;
        }
    }
    return false;
}

/**
 * @brief Synchronize with a stream
 *
 * @param[in]   stream      The cuda stream to synchronize.
 *
 * @return      error code
 */
inline cudaError_t sync_stream(cudaStream_t stream) {
    cudaEvent_t evt;
    cudaError_t err = cudaEventCreate(&evt);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaEventRecord(evt, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaEventSynchronize(evt);
}
#endif

} // namespace dccl
