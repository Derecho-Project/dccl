#include <atomic>
#include <cstddef>
#include <memory>
#include <limits>
#include <unistd.h>
#include <derecho/core/derecho.hpp>
#include <derecho/utils/logger.hpp>
#include <dccl.hpp>
#include "internal_common.hpp"
#include "blob.hpp"

using namespace derecho;

namespace dccl {
//----------------The derecho group---------------------
/*
class DCCLSubgroupType : public mutils::ByteRepresentable,
                         public GroupReference {
public:
    std::atomic<void*>  recvbuf;

    virtual ncclResult_t reduce(const Blob& sendbuf, const size_t, ncclDataType_t datatype, ncclRedOp_t op, bool inplace);

    REGISTER_RPC_FUNCTIONS(DCCLSubgroupType,ORDERED_TARGETS(reduce));

    // serialization support
    //
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
    
    // constructors
    DCCLSubgroupType():recvbuf(nullptr) {}
};
*/

template<typename DT>
ncclResult_t init_receive_buf(void* recvbuf,
                              size_t count,
                              ncclRedOp_t op) {
    DT e;
    switch(op) {
    case ncclSum:
    case ncclAvg:
        e = 0;
        break;
    case ncclProd:
        e = 1;
        break;
    case ncclMax:
        e = std::numeric_limits<DT>::lowest();
        break;
    case ncclMin:
        e = std::numeric_limits<DT>::max();
        break;
    default:
        e = 0;
    }

    std::fill(static_cast<DT*>(recvbuf),static_cast<DT*>(recvbuf) + count,e);

    return ncclSuccess;
}

template<typename DT>
ncclResult_t do_reduce(const void*  sendbuf,
                       void*        recvbuf,
                       size_t       count,
                       ncclRedOp_t  op) {
    const DT*   psend = static_cast<const DT*>(sendbuf);
    DT*         precv = static_cast<DT*>(recvbuf);
    // std::size_t clsz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    // we have to use constexp to enable SIMD optimization
    // Use CACHELINE_SZ macro passed during compilation

    if (reinterpret_cast<uint64_t>(sendbuf)%sizeof(DT)) {
        dbg_default_warn("sendbuf@{:p} is not aligned with data type:{},size={}, performance might be degraded.",sendbuf,typeid(DT).name(),sizeof(DT));
    }

    if (reinterpret_cast<uint64_t>(recvbuf)%sizeof(DT)) {
        dbg_default_warn("recvbuf@{:p} is not aligned with data type:{},size={}, performance might be degraded.",recvbuf,typeid(DT).name(),sizeof(DT));
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
    std::size_t             head_count = (CLSZ - reinterpret_cast<uint64_t>(recvbuf)%CLSZ)%CLSZ/sizeof(DT);
    constexpr std::size_t   pack_count = CLSZ/sizeof(DT); // we assume CLSZ%sizeof(DT) == 0
    std::size_t             num_pack = count/pack_count;
    std::size_t             tail_count = (pack_count + (count%pack_count) - head_count)%pack_count;
    // for the special case.
    if ((tail_count + head_count) > count) {
        head_count = count;
        tail_count = 0;
    }

    dbg_default_trace("{}:head_count={},pack_count={},num_pack={},tail_count={}",
                      __func__,head_count,pack_count,num_pack,tail_count);

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
        // we do not do average, but do sum and divide
        return ncclInvalidUsage;
        break;
    default:
        return ncclInvalidArgument;
        break;
    }
    return ncclSuccess;
}

ncclResult_t DCCLSubgroupType::reduce(const Blob& sendbuf, const size_t count, ncclDataType_t datatype, ncclRedOp_t op, bool inplace) {
    ncclResult_t ret = ncclSuccess;

    if (inplace  && (group->get_my_id() == group->get_rpc_caller_id())) {
        return ret;
    }

    void* rbuf = this->recvbuf.load();
    switch(datatype) {
    case ncclInt8: // ncclChar
        ret = do_reduce<int8_t>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclUint8:
        ret = do_reduce<uint8_t>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclInt32: // ncclInt
        ret = do_reduce<int32_t>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclUint32:
        ret = do_reduce<uint32_t>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclInt64:
        ret = do_reduce<uint64_t>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclUint64:
        ret = do_reduce<uint32_t>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclFloat16: // ncclHalf:
        // These types are only supported in C++23
        ret = ncclInvalidArgument;
        break;
    case ncclFloat32: // ncclFloat
        ret = do_reduce<float>(sendbuf.bytes,rbuf,count,op);
        break;
    case ncclFloat64: // ncclDouble
        ret = do_reduce<double>(sendbuf.bytes,rbuf,count,op);
        break;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
        // To be supported.
        ret = ncclInvalidUsage;
        break;
#endif
    default:
        // unknown type.
        ret = ncclInvalidArgument;
    }
    return ncclSuccess;
}

//------------------------------------------------------

ncclResult_t ncclCommInit(ncclComm_t* comm) {
    ncclComm_t      comm_handle = static_cast<ncclComm_t>(calloc(1,sizeof(*comm_handle)));

    if (!comm_handle) {
        // ENOMEM: no memory
        return ncclSystemError;
    }

    // create a subgroup
    SubgroupInfo si{make_subgroup_allocator<DCCLSubgroupType>()};
    Group<DCCLSubgroupType>* group = 
        new Group<DCCLSubgroupType>(
            {},
            si,{},{},
            [&comm_handle](
                persistent::PersistentRegistry*,
                subgroup_id_t) {
                DCCLSubgroupType* subgroup_object = new DCCLSubgroupType();
                comm_handle->derecho_group_object = reinterpret_cast<void*>(subgroup_object);
                return std::unique_ptr<DCCLSubgroupType>(subgroup_object);
            });

    comm_handle->derecho_group_handle = static_cast<void*>(group);
    *comm = comm_handle;

    return ncclSuccess;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    if (comm != nullptr) {
        Group<DCCLSubgroupType>* group = static_cast<Group<DCCLSubgroupType>*>(comm->derecho_group_handle);
        group->leave();
        delete group;
    }
    free(comm);
    return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void*      sendbuff,
                           void*            recvbuff,
                           size_t           count,
                           ncclDataType_t   datatype,
                           ncclRedOp_t      op,
                           ncclComm_t       comm) {
    if (!comm || !comm->derecho_group_handle) {
        return ncclInvalidArgument;
    }
    std::unique_ptr<Blob> blob_to_send;
    bool inplace = (sendbuff == recvbuff);
    // if inplace, we have to copy the send data first, so set emplaced = !inplace
    // otherwise, we skip the copy.
    blob_to_send = std::make_unique<Blob>(
                        reinterpret_cast<const uint8_t*>(sendbuff),
                        count*size_of_type(datatype),!inplace);
    ncclResult_t ret;
    if (!inplace) {
        // we have to initialize the receive buffer if it's not inplace
        switch(datatype) {
        case ncclInt8: // ncclChar
            ret = init_receive_buf<int8_t>(recvbuff,count,op);
            break;
        case ncclUint8:
            ret = init_receive_buf<uint8_t>(recvbuff,count,op);
            break;
        case ncclInt32: // ncclInt
            ret = init_receive_buf<int32_t>(recvbuff,count,op);
            break;
        case ncclUint32:
            ret = init_receive_buf<uint32_t>(recvbuff,count,op);
            break;
        case ncclInt64:
            ret = init_receive_buf<int64_t>(recvbuff,count,op);
            break;
        case ncclUint64:
            ret = init_receive_buf<uint64_t>(recvbuff,count,op);
            break;
        case ncclFloat16: // ncclHalf:
            // These types are only supported in C++23
            ret = ncclInvalidArgument;
            break;
        case ncclFloat32: // ncclFloat
            ret = init_receive_buf<float>(recvbuff,count,op);
            break;
        case ncclFloat64: // ncclDouble
            ret = init_receive_buf<double>(recvbuff,count,op);
            break;
#if defined(__CUDA_BF16_TYPES_EXIST__)
        case ncclBfloat16:
            // To be supported.
            ret = ncclInvalidUsage;
            break;
#endif
        default:
            // unknown type.
            ret = ncclInvalidArgument;
        }
    }


    if (ret != ncclSuccess) {
        return ret;
    }

    Group<DCCLSubgroupType>*            group = 
                                        reinterpret_cast<Group<DCCLSubgroupType>*>(comm->derecho_group_handle);
    DCCLSubgroupType*                   group_object = 
                                        reinterpret_cast<DCCLSubgroupType*>(comm->derecho_group_object);
    auto&                               dccl_subgroup_handle =
                                        group->get_subgroup<DCCLSubgroupType>();

    group_object->recvbuf.store(recvbuff);
    group->barrier_sync();
    auto results = dccl_subgroup_handle.ordered_send<RPC_NAME(reduce)>(*blob_to_send,count,datatype,op,inplace);
    for(auto& reply_pair : results.get()) {
        ret = reply_pair.second.get();
        if (ret != ncclSuccess) {
            break;
        }
    }
    group->barrier_sync();
    return ret;
}
}/*namespace dccl*/
