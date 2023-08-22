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
    ret = reduce_scatter_ring(buffer,scratchpad,count,datatype,op,comm,
                              [](uint32_t r){return r;},
                              [](uint32_t r){return r;});
    if (ret != ncclSuccess) {
        dccl_error("{}: failed to run reduce_scatter_ring.", __func__);
        return ret;
    }

    TIMESTAMP(TT_ALLREDUCE_REDUCESCATTER,my_rank,op);

    // STEP 3 ring all gather
    ret = all_gather_ring(buffer,count/world_size,datatype,comm,
                          [world_size](uint32_t r){return (r + 1)%world_size;},
                          [world_size](uint32_t r){return (r - 1 + world_size)%world_size;});
    

    TIMESTAMP(TT_ALLREDUCE_ALLGATHER,my_rank,op);
    dccl_trace("{}: ring allgather done.", __func__);

    return ret;
}

}
}
