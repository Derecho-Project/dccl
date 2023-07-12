#include "algorithms.hpp"
#include "internal_common.hpp"

/**
 * @file    reduce_scatter_recursive_halving.cpp
 * @brief   The recursive halving reduce scatter algorithm implementation
 */

namespace dccl {
namespace algorithm {

ncclResult_t reduce_scatter_recursive_halving(
        void*           buffer,
        void*           scratchpad,
        size_t          count,
        ncclDataType_t  datatype,
        ncclRedOp_t     op,
        ncclComm_t      comm) {
    ncclResult_t ret = ncclSuccess;
    // STEP 1: test constraints
    if (CACHELINE_OFFSET(buffer) != 0 || CACHELINE_OFFSET(scratchpad)) {
        dccl_warn("Function {} got CL-unaligned buffer@{:p} or scratchpad@{:p}. Performance might be compromised."
                  " See {}:{}.",
                  __func__,buffer,scratchpad,__FILE__,__LINE__);
    }
    const uint32_t world_size = get_world_size(comm);
    assert(IS_POWER_OF_TWO(world_size)); // TODO: to be lifted.
    assert(count%world_size == 0); // TODO: to be lifted.
    const size_t data_entry_size = size_of_type(datatype);
    const uint32_t slice_size = count/world_size*data_entry_size;
    if (CACHELINE_OFFSET(slice_size) != 0) {
        dccl_warn("Function {} got CL-unaligned slice size ({} bytes). Performance might be compromised. See {}:{}.",
                  __func__,slice_size,__FILE__,__LINE__);
    }
    const uint32_t my_rank  = get_my_rank(comm);
    const uint32_t exponent = log_two(world_size);
    const uint32_t my_slice = reverse_bits(my_rank,exponent);
    auto shard_members      = get_dccl_shard_members(comm);

    void* send_buffer = buffer; // pointer to the data to send in a step
    void* recv_buffer = buffer; // pointer to the data to reduce (with the received data) in a step
    size_t step_bsize = count*data_entry_size;  // size of the buffer in bytes to be processed in a step
#define __UPPER_HALF_PTR__(ptr,bsize)   (ptr)
#define __LOWER_HALF_PTR__(ptr,bsize)   reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + (bsize>>1))
    // STEP 2: prepare buffer and exchange data.
    for (uint32_t step = 0; step < exponent; step ++) {
        uint32_t peer_rank = (my_rank&(~((1<<step)-1))) + (my_rank+(1<<step))%(1<<(step+1));
        auto peer_id = shard_members.at(peer_rank);
        if (((my_rank>>step)&1) == 0) {
            send_buffer = __LOWER_HALF_PTR__(send_buffer,step_bsize);
            recv_buffer = __UPPER_HALF_PTR__(recv_buffer,step_bsize);
        } else {
            send_buffer = __UPPER_HALF_PTR__(send_buffer,step_bsize);
            recv_buffer = __LOWER_HALF_PTR__(recv_buffer,step_bsize);
        }
        step_bsize = (step_bsize>>1);
        dccl_trace("STEP {}, me{}:id-{} <--> peer{}:id-{}, data size = {} Bytes", 
                   my_rank, shard_members.at(my_rank), 
                   peer_rank, peer_id, step_bsize);
        struct iovec siov,riov;
        siov.iov_base   = send_buffer;
        siov.iov_len    = step_bsize;
        riov.iov_base   = scratchpad;
        riov.iov_len    = step_bsize;
        SUBGROUP_HANDLE(comm).oob_send(shard_members.at(peer_rank),&siov,1);
        SUBGROUP_HANDLE(comm).oob_recv(shard_members.at(peer_rank),&riov,1);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(peer_id,OOB_OP_SEND,100); // TODO: change this according to message size. 
                                                                        // This might affect p2p heart beat
        SUBGROUP_HANDLE(comm).wait_for_oob_op(peer_id,OOB_OP_RECV,100); // TODO: change this according to message size. 
                                                                        // This might affect p2p heart beat
        // TODO: optimization opportunities here: we can wait OOB_OP_RECV first and then do reduce. But currently, 
        // the wait_for_oob_op needs improved to distinguish OOB_OP_RECV and OOB_OP_SEND.

        // do reduce...
        ON_DCCL_DATATYPE(datatype,
                         ret=do_reduce,
                         scratchpad,recv_buffer,
                         step_bsize/size_of_type(datatype),op);
        if (ret != ncclSuccess) {
            break;
        }
    }

    return ret;
}

} // namespace algorithm
} // namespace dccl
