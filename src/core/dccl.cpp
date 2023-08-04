/**
 * @file    dccl.cpp
 * @brief   The DCCP API implementations
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <limits>
#include <unistd.h>
#include <derecho/core/derecho.hpp>
#include <derecho/utils/logger.hpp>
#include <derecho/conf/conf.hpp>
#include <derecho/utils/time.h>
#include <dccl/dccl.hpp>
#include "internal_common.hpp"
#include "algorithms.hpp"
#include "blob.hpp"

using namespace derecho;

namespace dccl {

/**
 * @brief   Validate DCCL communicator
 * If communicator is not initialized, throw an exception
 *
 * @param[in]   comm        The communicator object.
 *
 * @throw   std::runtime_error  Runtime error will be raised on null communicator.
 */
#define VALIDATE_COMM(comm) \
if (!comm || !comm->derecho_group_handle) { \
    dccl_error("{}: invalid comm handle.", __func__); \
    throw std::runtime_error (std::string(__func__) + " is unable to handle invalid comm handle."); \
}

uint32_t dcclGetWorldSize(ncclComm_t comm) {
    return _get_shard_members<DCCLSubgroupType>(comm).size();
}

uint32_t dcclGetMyRank(ncclComm_t comm) {
    auto my_id = GROUP_HANDLE(comm)->get_my_id();
    auto shard_members = _get_shard_members<DCCLSubgroupType>(comm);
    uint32_t my_rank = 0;
    while(my_rank < shard_members.size() && shard_members.at(my_rank) != my_id) {
        my_rank ++;
    }
    assert (my_rank < shard_members.size());
    return my_rank;
}


/**
 * @brief The initial size of the scratchpad memory is 64 MB per thread.
 */
#define SCRATCHPAD_INI_SIZE     (1L<<26)
/**
 * @brief The maximum size of the scratchpad memory is 4 GBytes per thread.
 */
#define SCRATCHPAD_MAX_SIZE     (1L<<32)
/**
 * @brief the thread local scratch pad memory
 * The scratch pad memory is pre-registered derecho OOB memory for zero-copy
 * operations.
 */
thread_local void*  scratchpad = nullptr;
/**
 * @brief the current size of scratchpad memory.
 */
thread_local size_t scratchpad_size = 0L;

/**
 * @brief Make sure the scratch pad is big enough.
 * This function will go through the following process:
 *
 * - If the requested scratchpad size is bigger than `SCRATCHPAD_MAX_SIZE`, throw an exception.
 *
 * - If the scratchpad is uninitialized or its size is smaller than the requested size, enlarge
 *   the scratchpad to the smallest page aligned size no less than `max(size,SCRATCHPAD_INI_SIZE)`
 *
 * @param[in]   size        The requested scratchpad size.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error  In case of failure, raise an runtime error.
 *
 * @return      error code
 */
static ncclResult_t verify_scratchpad(size_t size, ncclComm_t comm) {
    ncclResult_t ret = ncclSuccess;

    if (size > SCRATCHPAD_MAX_SIZE) {
        dccl_error("{}: Unable to allocate a scratchpad of {} Bytes, which is bigger than {} Bytes. "
                   "See {}:{}",
                   __func__, size, SCRATCHPAD_MAX_SIZE, __FILE__, __LINE__);
        return ncclInvalidArgument;
    }

    size_t new_size = (scratchpad_size < size) ?
        ((size<=SCRATCHPAD_INI_SIZE)? SCRATCHPAD_INI_SIZE : ((size + getpagesize() - 1) & ~(getpagesize()-1))) : 0;

    if (new_size > 0) {
        if (scratchpad_size > 0) {
            ret = dcclDeregisterCacheMemory(comm, scratchpad, scratchpad_size);
            if (ret != ncclSuccess) {
                dccl_error("{} is unable to deregister existing scratchpad. "
                           "See {}:{}",
                           __func__, __FILE__, __LINE__);
                return ret;
            }

            free(scratchpad);
            scratchpad_size = 0;
        }

        if(posix_memalign(&scratchpad,CACHELINE_SIZE,new_size) != 0) {
            dccl_error("{} is unable to allocate memory of size {}, error = {}, See {}:{}",
                        __func__, new_size, strerror(errno), __FILE__, __LINE__);
            return ncclSystemError;
        }
        scratchpad_size = new_size;

        ret = dcclRegisterCacheMemory(comm, scratchpad, new_size);
        if (ret != ncclSuccess) {
            free(scratchpad);
            scratchpad_size = 0;
            dccl_error("{} is unable to register scratchpad with size {}. "
                       "See {}:{}",
                       __func__, new_size, __FILE__, __LINE__);
            return ret;
        }

        scratchpad_size = new_size;
    }

    return ret;
}

/**
 * @brief   Initialize receive buffer(Deprecated API)
 * @deprecated  Stop using this function.
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

ncclResult_t ncclAllReduce(const void*      sendbuff,
                           void*            recvbuff,
                           size_t           count,
                           ncclDataType_t   datatype,
                           ncclRedOp_t      op,
                           ncclComm_t       comm) {
    uint32_t        my_rank =           dcclGetMyRank(comm);
    ncclResult_t    ret =               ncclSuccess;
    size_t          total_data_size =   count * size_of_type(datatype);
    TIMESTAMP(TT_ALLREDUCE_ENTER,my_rank,op);

    // STEP 1: test constraints.
    if (CACHELINE_OFFSET(sendbuff) || CACHELINE_OFFSET(recvbuff)) {
        dccl_warn("Either sendbuff@{:p} or recvbuff@{:p} is not cacheline ({} bytes) aligned. "
                  "Possible performance degradation might occur.",
                  sendbuff, recvbuff, CACHELINE_SIZE);
    }

    // STEP 2: check buffer
    if (sendbuff != recvbuff) {
        memcpy(recvbuff,sendbuff,total_data_size);
    }
    TIMESTAMP(TT_ALLREDUCE_MEMCPY,my_rank,op);

    if (hasCustomizedConfKey(DCCL_ALLREDUCE_ALGORITHM_CONFSTR) == false ||
        getConfString(DCCL_ALLREDUCE_ALGORITHM_CONFSTR) == DCCL_ALLREDUCE_RING) {

        // STEP 3: verify_scratchpad
        ret = verify_scratchpad(total_data_size/dcclGetWorldSize(comm),comm);
        if (ret != ncclSuccess) {
            dccl_error("{} failed to verify scratchpad memory with size {}",
                       __func__, total_data_size/dcclGetWorldSize(comm));
            return ret;
        }
    
        ret = algorithm::all_reduce_ring(recvbuff,scratchpad,count,datatype,op,comm);
        if (ret != ncclSuccess) {
            dccl_error("{}: all_reduce_ring() failed.",
                       __func__);
            return ret;
        }

    } else if (getConfString(DCCL_ALLREDUCE_ALGORITHM_CONFSTR) == DCCL_ALLREDUCE_RABINSEIFNER) {
    
        // STEP 3: verify_scratchpad
        ret = verify_scratchpad(total_data_size>>1,comm);
        if (ret != ncclSuccess) {
            dccl_error("{} failed to verify scratchpad memory with size {}",
                       __func__, total_data_size>>1);
            return ret;
        }
    
        ret = algorithm::all_reduce_recursive_halving_and_doubling(recvbuff,scratchpad,count,datatype,op,comm);
        if (ret != ncclSuccess) {
            dccl_error("{}: all_reduce_recursive_halving_and_doubling() failed.",
                       __func__);
            return ret;
        }
    }

    TIMESTAMP(TT_ALLREDUCE_DONE,my_rank,op);

    return ret;
}

/* deprecated
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
*/

ncclResult_t dcclRegisterCacheMemory(ncclComm_t comm, void* buffer, size_t size) {
    VALIDATE_COMM(comm);

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

ncclResult_t dcclDeregisterCacheMemory(ncclComm_t comm, void* buffer, size_t size) {
    VALIDATE_COMM(comm);
    //TODO: check size
    GROUP_HANDLE(comm)->deregister_oob_memory(buffer);
    return ncclSuccess;
}

ncclResult_t ncclReduceScatter(const void*      sendbuffer,
                               void*            recvbuffer,
                               size_t           recvcount,
                               ncclDataType_t   datatype,
                               ncclRedOp_t      op,
                               ncclComm_t       comm) {
 /*
    VALIDATE_COMM(comm);

    ncclResult_t ret = ncclSuccess;

    //TODO: This is a brutal force wrapper for test. Do the following afterward:
    // - prepare a writable send buffer
    // - copy the corresponding block to destination.
    ret = verify_scratchpad(recvcount*dcclGetWorldSize(comm),comm);
    if (ret != ncclSuccess) {
        return ret;
    }
    return   algorithm::reduce_scatter_recursive_halving(const_cast<void*>(sendbuffer),
                                                        scratchpad,
                                                        recvcount*dcclGetWorldSize(comm),
                                                        datatype,op,comm);
*/
    return ncclInvalidUsage;
}

#ifdef ENABLE_EVALUATION
Timestamp::Timestamp(size_t num_entries):
    _log(nullptr),capacity(0),position(0) {
    // lock it
    pthread_spin_init(&lck,PTHREAD_PROCESS_PRIVATE);
    pthread_spin_lock(&lck);

    capacity = ((num_entries == 0) ? (1ul<<16) : num_entries);
    size_t capacity_in_bytes = (capacity)*4*sizeof(uint64_t);

    if ( posix_memalign(reinterpret_cast<void**>(&_log), CACHELINE_SIZE,capacity_in_bytes) ) {
        dccl_error("{} failed to allocate {} bytes for log space: {}. See {}:{}",
                __func__, capacity, strerror(errno), __FILE__, __LINE__);
        throw std::runtime_error("Failed to allocate memory for log space.");
    }

    // warm it up
    for (int i=0; i<6; i++) {
        bzero(_log,capacity_in_bytes);
    }

    // unlock
    pthread_spin_unlock(&lck);
}

void Timestamp::instance_log(uint64_t tag, uint64_t rank, uint64_t extra) {
    uint64_t ts = get_time();
    pthread_spin_lock(&lck);

    _log[(position<<2)] = tag;
    _log[(position<<2)+1] = rank;
    _log[(position<<2)+2] = extra;
    _log[(position<<2)+3] = ts;
    position ++;

    pthread_spin_unlock(&lck);
}

void Timestamp::instance_flush(const std::string& filename, bool clear) {
    pthread_spin_lock(&lck);
    std::ofstream outfile(filename);

    outfile << "# number of entries:" << position << std::endl;
    outfile << "# tag rank extra tsns" << std::endl;
    for (size_t i=0;i<position;i++) {
        outfile << _log[i<<2] << " " 
                << _log[(i<<2)+1] << " " 
                << _log[(i<<2)+2] << " " 
                << _log[(i<<2)+3] << std::endl;
    }
    outfile.close();
    pthread_spin_unlock(&lck);

    if (clear) {
        _t.clear();
    }
}

void Timestamp::instance_clear() {
    pthread_spin_lock(&lck);
    position=0;
    pthread_spin_unlock(&lck);
}

Timestamp::~Timestamp() {
    if (_log != nullptr) {
        free(_log);
    }
}

Timestamp Timestamp::_t{};

#endif//ENABLE_EVALUATION
}/*namespace dccl*/
