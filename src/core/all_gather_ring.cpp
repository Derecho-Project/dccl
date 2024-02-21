#include "algorithms.hpp"
#include "internal_common.hpp"


namespace dccl {
namespace algorithm {

ncclResult_t all_gather_ring(
        void*                   buffer,
        size_t                  count,
        ncclDataType_t          datatype,
        ncclComm_t              comm,
        cudaStream_t            stream,
        const rank_converter_t& to_new_rank,
        const rank_converter_t& to_old_rank) {
    uint32_t        my_rank = to_new_rank(dcclGetMyRank(comm));
    ncclResult_t    ret = ncclSuccess;
    size_t          data_slot_size = count * size_of_type(datatype);
    uint32_t        world_size =    dcclGetWorldSize(comm);
    auto            shard_members  = get_dccl_shard_members(comm);

    // STEP 1 check contraints
    if (CACHELINE_OFFSET(buffer)) {
        dccl_warn("The buffer @{:p} is not cacheline ({} bytes) aligned. "
                  "Possible performance degradation might occur.",
                  buffer, CACHELINE_SIZE);

    }

/**
 * @cond Doxygen_Suppressed.
 */
#define __DATA__(i) \
    reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + (data_slot_size)*((world_size+(i))%world_size))
#define __NEXT__(r) ((world_size + (r) + 1) % world_size)
#define __PREV__(r) ((world_size + (r) - 1) % world_size)
/**
 * @endcond
 */
    node_id_t to_id     = shard_members.at(to_old_rank(__NEXT__(my_rank)));
    node_id_t from_id   = shard_members.at(to_old_rank(__PREV__(my_rank)));

    // STEP 3 ring all gather
    for (uint32_t s=0;s<world_size-1;s++) {
        // 3.1 recv dat[r-s] from rank r-1
        uint32_t r_chunks = dccl_oob_recv(comm,from_id,__DATA__(my_rank - s - 1),data_slot_size);
        // 3.2 send dat[r-s+1] to rank r+1
        uint32_t s_chunks = dccl_oob_send(comm,to_id,__DATA__(my_rank - s),data_slot_size);
        // 3.3 blocking wait
        dccl_oob_wait_for_recv(comm,from_id,r_chunks);
        dccl_oob_wait_for_send(comm,to_id,s_chunks);

        /*****
        struct iovec siov,riov;
        siov.iov_base   = __DATA__(my_rank - s);
        siov.iov_len    = data_slot_size;
        riov.iov_base   = __DATA__(my_rank - s -1);
        riov.iov_len    = data_slot_size;
        SUBGROUP_HANDLE(comm).oob_recv(from_id,&riov,1);
        SUBGROUP_HANDLE(comm).oob_send(to_id,&siov,1);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(from_id,OOB_OP_RECV,DCCL_OOB_TIMEOUT_US);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(to_id,OOB_OP_SEND,DCCL_OOB_TIMEOUT_US);
        *****/
    }

    dccl_trace("{}: ring allgather done.", __func__);

    return ret;
}

}
}
