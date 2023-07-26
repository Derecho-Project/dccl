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
        ncclComm_t      comm,
        uint32_t        subworld_size,
        const rank_converter_t& to_new_rank,
        const rank_converter_t& to_old_rank) {
    ncclResult_t ret = ncclSuccess;

    dccl_trace("{}: STEP 1: testing constraints.", __func__);

    if (CACHELINE_OFFSET(buffer) != 0) {
        dccl_warn("Function {} got CL-unaligned buffer@{:p}. Performance might be compromised."
                  " See {}:{}.",
                  __func__,buffer,__FILE__,__LINE__);
    }
    assert(subworld_size<dcclGetWorldSize(comm));
    assert(IS_POWER_OF_TWO(subworld_size));
    const uint32_t  exponent = log_two(subworld_size);
    assert(count%subworld_size == 0);
    const size_t    data_entry_size = size_of_type(datatype);
    const uint32_t  slice_size = count/subworld_size*data_entry_size;
    if (CACHELINE_OFFSET(slice_size) != 0) {
        dccl_warn("Function {} got CL-unaligned slice size ({} bytes). Performance might be compromised. See {}:{}.",
                  __func__,slice_size,__FILE__,__LINE__);
    }
    const uint32_t my_rank  = to_new_rank(dcclGetMyRank(comm));
    auto shard_members      = get_dccl_shard_members(comm);

    size_t step_bsize = slice_size;  // size of the buffer in bytes to be processed in a step

    dccl_trace("{}: STEP 2: prepare buffer and exchange data. Buffer = {} bytes @ {:p} ",
               __func__, count*data_entry_size, buffer);

    uint32_t send_block = reverse_bits(my_rank,exponent);

    for (uint32_t step = 0; step < exponent; step ++) {

        uint32_t peer_rank = (my_rank&(~((1<<(exponent-step))-1))) + (my_rank+(1<<(exponent-step-1)))%(1<<(exponent-step));
        auto peer_id = shard_members.at(to_old_rank(peer_rank));

        if (step > 0) {
            send_block = send_block & (~(1<<(step-1)));
        }
        uint32_t recv_block = send_block^(1<<step);
        dccl_trace("step-{}, me{}:id-{} <--> peer{}:id-{}, data size = {} Bytes, recv_block = {}, send_block = {}",
                   step, my_rank, shard_members.at(to_old_rank(my_rank)), 
                   peer_rank, peer_id, step_bsize, recv_block, send_block);

        /**
         * @cond Doxygen_Suppressed
         */
#define BLOCK_ID_TO_BUF_ADDR(id) \
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + (id * slice_size))
        /**
         * @endcond
         */
        struct iovec siov,riov;
        siov.iov_base   = BLOCK_ID_TO_BUF_ADDR(send_block);
        siov.iov_len    = step_bsize;
        riov.iov_base   = BLOCK_ID_TO_BUF_ADDR(recv_block);
        riov.iov_len    = step_bsize;
        SUBGROUP_HANDLE(comm).oob_send(shard_members.at(to_old_rank(peer_rank)),&siov,1);
        SUBGROUP_HANDLE(comm).oob_recv(shard_members.at(to_old_rank(peer_rank)),&riov,1);
        SUBGROUP_HANDLE(comm).wait_for_oob_op(peer_id,OOB_OP_SEND,30000); // TODO: change this according to message size. 
                                                                        // This might affect p2p heart beat
        SUBGROUP_HANDLE(comm).wait_for_oob_op(peer_id,OOB_OP_RECV,30000); // TODO: change this according to message size. 
                                                                        // This might affect p2p heart beat
        step_bsize = (step_bsize<<1);
    }

    dccl_trace("{}: Finished.", __func__);

    return ret;
}

} // namespace algorithm
} // namespace dccl
