#include "algorithms.hpp"
#include "internal_common.hpp"


namespace dccl {
namespace algorithm {

ncclResult_t reduce_scatter_ring(
        void*           buffer,
        void*           scratchpad,
        size_t          count,
        ncclDataType_t  datatype,
        ncclRedOp_t     op,
        ncclComm_t      comm,
        cudaStream_t    stream,
        const rank_converter_t& to_new_rank,
        const rank_converter_t& to_old_rank) {
    uint32_t        my_rank = to_new_rank(dcclGetMyRank(comm));
    ncclResult_t    ret = ncclSuccess;
    size_t          total_data_size = count * size_of_type(datatype);
    uint32_t        world_size =    dcclGetWorldSize(comm);
    size_t          data_slot_size = total_data_size/world_size;
    auto            shard_members  = get_dccl_shard_members(comm);

#ifdef  CUDA_FOUND
    bool            is_in_device = is_device_ptr(buffer);
#endif

    // STEP 1 check contraints
#ifdef  CUDA_FOUND
    if (is_in_device) {
        if (CUDA_L1_CACHELINE_OFFSET(buffer)) {
            dccl_warn("The buffer @{:p} is not cacheline ({} bytes) aligned. "
                      "Performance degradation might occur.",
                      buffer, CUDA_L1_CACHELINE_SIZE);
        }
        if (CUDA_L2_CACHELINE_OFFSET(buffer)) {
            dccl_warn("The buffer @{:p} is not cacheline ({} bytes) aligned. "
                      "Performance degradation might occur.",
                      buffer, CUDA_L2_CACHELINE_SIZE);
        }
    } else {
#endif // CUDA_FOUND
        if (CACHELINE_OFFSET(buffer)) {
            dccl_warn("The buffer @{:p} is not cacheline ({} bytes) aligned. "
                      "Possible performance degradation might occur.",
                      buffer, CACHELINE_SIZE);
        }
#ifdef  CUDA_FOUND
    }
#endif // CUDA_FOUND
    // TODO: the latter constrain can be lifted.
    if (count < world_size || count % world_size) {
        dccl_error("Entry count {} cannot be distributed evenly among {} nodes.",
                   count, world_size);
        return ncclInvalidArgument;
    }


    // STEP 2 ring reduce scatter
/**
 * @cond Doxygen_Suppressed.
 */
#define __DATA__(i) \
    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + (total_data_size/world_size)*((world_size+(i))%world_size))
#define __NEXT__(r) ((world_size + (r) + 1) % world_size)
#define __PREV__(r) ((world_size + (r) - 1) % world_size)
/**
 * @endcond
 */
    node_id_t to_id     = shard_members.at(to_old_rank(__NEXT__(my_rank)));
    node_id_t from_id   = shard_members.at(to_old_rank(__PREV__(my_rank)));
    for (uint32_t s=0;s<world_size-1;s++) { // s - step
        // 2.1 - send dat[r-s] to rank r+1
        uint32_t s_chunks = dccl_oob_send(comm,to_id,__DATA__(my_rank - s),data_slot_size);
        // 2.2 - recv dat[r-s-1] from rank r-1
        uint32_t r_chunks = dccl_oob_recv(comm,from_id,scratchpad,data_slot_size);
        // 2.3 - wait
        dccl_oob_wait_for_send(comm,to_id,s_chunks);
        dccl_oob_wait_for_recv(comm,from_id,r_chunks);
        // 2.4 - do reduce
#ifdef CUDA_FOUND
        if (is_in_device) {
            ON_DCCL_DATATYPE(datatype,
                             ret=do_device_reduce,
                             scratchpad,__DATA__(my_rank - s - 1),
                             count/world_size,op,stream);
            ret = (sync_stream(stream) == cudaSuccess)? ncclSuccess:ncclUnhandledCudaError;
        } else {
#endif
            ON_DCCL_DATATYPE(datatype,
                             ret=do_host_reduce,
                             scratchpad,__DATA__(my_rank - s - 1),
                             count/world_size,op);
#ifdef CUDA_FOUND
        }
#endif
        if (ret != ncclSuccess) {
            return ret;
        }
    }

    dccl_trace("{}: ring reduce_scatter done.", __func__);

    return ret;
}

}
}
