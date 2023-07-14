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

ncclResult_t DCCLSubgroupType::reduce(const Blob& sendbuf, const size_t count, ncclDataType_t datatype, ncclRedOp_t op, bool inplace) {
    ncclResult_t ret = ncclSuccess;

    if (inplace && (group->get_my_id() == group->get_rpc_caller_id())) {
        return ret;
    }

    void* rbuf = this->recvbuf.load();
    /*
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
    */
    ret = ncclInvalidArgument;
    ON_DCCL_DATATYPE(datatype,ret=do_reduce,sendbuf.bytes,rbuf,count,op);
    return ret;
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

ncclResult_t ncclRegisterCacheMemory(ncclComm_t comm, void* buffer, size_t size) {
    if (!comm || !comm->derecho_group_handle) {
        dccl_error("{}: invalid comm handle.", __func__);
        return ncclInvalidArgument;
    }

    if (CACHELINE_OFFSET(buffer) != 0) {
        dccl_error("{}: buffer@{:p} is not cacheline aligned.", __func__, buffer);
        return ncclInvalidArgument;
    }

    if (CACHELINE_OFFSET(size) != 0) {
        dccl_error("{}: buffer size {} is not cacheline aligned.", __func__, size);
        return ncclInvalidArgument;
    }

    GROUP_HANDLE(comm)->register_oob_memory(buffer,size);

    return ncclSuccess;
}

ncclResult_t ncclDeregisterCacheMemory(ncclComm_t comm, void* buffer, size_t size) {

    // TODO: verify size...
    GROUP_HANDLE(comm)->deregister_oob_memory(buffer);

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
