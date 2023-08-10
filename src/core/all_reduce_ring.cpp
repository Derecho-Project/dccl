#include "algorithms.hpp"
#include "internal_common.hpp"


namespace dccl {
namespace algorithm {

ncclResult_t all_reduce_ring(
        void*           buffer,
        void*           scratchpad,
        size_t          count,
        ncclDataType_t  datatype,
        ncclRedOp_t     op,
        ncclComm_t      comm) {
    uint32_t        my_rank = dcclGetMyRank(comm);
    ncclResult_t    ret = ncclSuccess;
    size_t          total_data_size = count * size_of_type(datatype);
    uint32_t        world_size =    dcclGetWorldSize(comm);
    size_t          data_slot_size = total_data_size/world_size;
    auto            shard_members  = get_dccl_shard_members(comm);

    // STEP 1 check contraints
    if (CACHELINE_OFFSET(buffer)) {
        dccl_warn("The buffer @{:p} is not cacheline ({} bytes) aligned. "
                  "Possible performance degradation might occur.",
                  buffer, CACHELINE_SIZE);

    }

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
    node_id_t to_id     = shard_members.at(__NEXT__(my_rank));
    node_id_t from_id   = shard_members.at(__PREV__(my_rank));
    for (uint32_t s=0;s<world_size-1;s++) { // s - step
        // 2.1 - send dat[r-s] to rank r+1
        struct iovec siov,riov;
        siov.iov_base   = __DATA__(my_rank - s);
        siov.iov_len    = data_slot_size;
        // 2.2 - recv dat[r-s-1] from rank r-1
        riov.iov_base   = scratchpad;
        riov.iov_len    = data_slot_size;
        SUBGROUP_HANDLE(comm).oob_send(to_id,&siov,1);
        SUBGROUP_HANDLE(comm).oob_recv(from_id,&riov,1);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(to_id,OOB_OP_SEND,DCCL_OOB_TIMEOUT_US);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(from_id,OOB_OP_RECV,DCCL_OOB_TIMEOUT_US);
        // 2.3 - do reduce...
        ON_DCCL_DATATYPE(datatype,
                         ret=do_reduce,
                         scratchpad,__DATA__(my_rank - s - 1),
                         count/world_size,op);
        if (ret != ncclSuccess) {
            return ret;
        }
    }

    TIMESTAMP(TT_ALLREDUCE_REDUCESCATTER,my_rank,op);
    dccl_trace("{}: ring reduce_scatter done.", __func__);

    // STEP 3 ring all gather
    for (uint32_t s=0;s<world_size-1;s++) {
        // 3.1 send dat[r-s+1] to rank r+1
        struct iovec siov,riov;
        siov.iov_base   = __DATA__(my_rank - s + 1);
        siov.iov_len    = data_slot_size;
        // 3.2 recv dat[r-s] from rank r-1
        riov.iov_base   = __DATA__(my_rank - s);
        riov.iov_len    = data_slot_size;
        SUBGROUP_HANDLE(comm).oob_send(to_id,&siov,1);
        SUBGROUP_HANDLE(comm).oob_recv(from_id,&riov,1);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(to_id,OOB_OP_SEND,DCCL_OOB_TIMEOUT_US);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(from_id,OOB_OP_RECV,DCCL_OOB_TIMEOUT_US);
    }

    TIMESTAMP(TT_ALLREDUCE_ALLGATHER,my_rank,op);
    dccl_trace("{}: ring allgather done.", __func__);

    return ret;
}

}
}