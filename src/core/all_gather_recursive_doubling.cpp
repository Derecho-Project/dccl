#include "algorithms.hpp"
#include "internal_common.hpp"

/**
 * @file    all_gather_recursive_doubling.cpp
 * @brief   The recursive doubling all gather algorithm implementation
 */

namespace dccl {
namespace algorithm {

ncclResult_t all_gather_recursive_doubling (
        void*           buffer,
        size_t          count,
        ncclDataType_t  datatype,
        ncclComm_t      comm) {
    ncclResult_t ret = ncclSuccess;

    dccl_trace("{}: STEP 1: testing constraints.", __func__);

    if (CACHELINE_OFFSET(buffer) != 0) {
        dccl_warn("Function {} got CL-unaligned buffer@{:p}. Performance might be compromised."
                  " See {}:{}.",
                  __func__,buffer,__FILE__,__LINE__);
    }
    const uint32_t world_size = dcclGetWorldSize(comm);
    assert(IS_POWER_OF_TWO(world_size)); 
    assert(count%world_size == 0);
    const size_t data_entry_size = size_of_type(datatype);
    const uint32_t slice_size = count/world_size*data_entry_size;
    if (CACHELINE_OFFSET(slice_size) != 0) {
        dccl_warn("Function {} got CL-unaligned slice size ({} bytes). Performance might be compromised. See {}:{}.",
                  __func__,slice_size,__FILE__,__LINE__);
    }
    const uint32_t my_rank  = dcclGetMyRank(comm);
    const uint32_t exponent = log_two(world_size);
    // const uint32_t my_slice = reverse_bits(my_rank,exponent);
    auto shard_members      = get_dccl_shard_members(comm);

    void* send_buffer = nullptr; // pointer to the data to send in a step
    void* recv_buffer = nullptr; // pointer to the buffer to recv in a step
    size_t step_bsize = slice_size;  // size of the buffer in bytes to be processed in a step
/**
 * @cond Doxygen_Suppressed.
 */
#define __UPPER_HALF_PTR__(ptr,bsize)   (ptr)
#define __LOWER_HALF_PTR__(ptr,bsize)   reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + (bsize>>1))
/**
 * @endcond
 */
    dccl_trace("{}: STEP 2: prepare buffer and exchange data. Buffer = {} bytes @ {:p} ",
               __func__, count*data_entry_size, buffer);

    for (uint32_t step = 0; step < exponent; step ++) {

        dccl_trace("{}: all_gather step-{}", __func__, step);

        uint32_t peer_rank = (my_rank&(~((1<<(exponent-step))-1))) + my_rank+(1<<(exponent-step-1))%(1<<(exponent-step));
        auto peer_id = shard_members.at(peer_rank);
        dccl_trace("{}: current peer is rank:{}(node_id:{}).", __func__, peer_rank, peer_id);

        if (step > 0) {
            if ((my_rank>>step)&1) {
                // doubling low
                send_buffer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(send_buffer) - step_bsize/2);
            }
        } else {
            send_buffer = 
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer)+reverse_bits(my_rank)*slice_size);
        }
        recv_buffer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer)^(1<<(step))); 

        dccl_trace("step-{}, me{}:id-{} <--> peer{}:id-{}, data size = {} Bytes, recv_buffer = {:p}, send_buffer = {:p}",
                   step, my_rank, shard_members.at(my_rank), 
                   peer_rank, peer_id, step_bsize, recv_buffer, send_buffer);

        struct iovec siov,riov;
        siov.iov_base   = send_buffer;
        siov.iov_len    = step_bsize;
        riov.iov_base   = recv_buffer;
        riov.iov_len    = step_bsize;
        SUBGROUP_HANDLE(comm).oob_send(shard_members.at(peer_rank),&siov,1);
        SUBGROUP_HANDLE(comm).oob_recv(shard_members.at(peer_rank),&riov,1);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(peer_id,OOB_OP_SEND,100); // TODO: change this according to message size. 
                                                                        // This might affect p2p heart beat
        SUBGROUP_HANDLE(comm).wait_for_oob_op(peer_id,OOB_OP_RECV,100); // TODO: change this according to message size. 
                                                                        // This might affect p2p heart beat
        step_bsize = (step_bsize>>1);
    }

    dccl_trace("{}: Finished.", __func__);

    return ret;
}

} // namespace algorithm
} // namespace dccl
