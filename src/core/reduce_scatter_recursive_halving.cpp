#include "algorithms.hpp"
#include "internal_common.hpp"

/**
 * @file    reduce_scatter_recursive_halving.cpp
 * @brief   The recursive halving reduce scatter algorithm implementation
 */

namespace dccl {
namespace algorithm {

ncclResult_t reduce_scatter_recursive_halving(
        void*                   buffer,
        void*                   scratchpad,
        size_t                  count,
        ncclDataType_t          datatype,
        ncclRedOp_t             op,
        ncclComm_t              comm,
        cudaStream_t            stream,
        uint32_t                subworld_size,
        const rank_converter_t& to_new_rank,
        const rank_converter_t& to_old_rank) {
    ncclResult_t ret = ncclSuccess;

    dccl_trace("{}: STEP 1: testing constraints.", __func__);
#ifdef CUDA_FOUND
    bool  is_in_device = is_device_ptr(buffer);
    if (!is_in_device) {
#endif
        if (CACHELINE_OFFSET(buffer) != 0 || CACHELINE_OFFSET(scratchpad)) {
            dccl_warn("Function {} got CL-unaligned buffer@{:p} or scratchpad@{:p}. Performance might be compromised."
                      " See {}:{}.",
                      __func__,buffer,scratchpad,__FILE__,__LINE__);
        }
#ifdef CUDA_FOUND
    }
#endif
    assert(subworld_size <= dcclGetWorldSize(comm));
    assert(IS_POWER_OF_TWO(subworld_size));
    const uint32_t exponent = log_two(subworld_size);
    assert(count%subworld_size == 0);
    const size_t data_entry_size = size_of_type(datatype);
    const uint32_t slice_size = count/subworld_size*data_entry_size;
    if (CACHELINE_OFFSET(slice_size) != 0) {
        dccl_warn("Function {} got CL-unaligned slice size ({} bytes). Performance might be compromised. See {}:{}.",
                  __func__,slice_size,__FILE__,__LINE__);
    }
    const uint32_t my_rank  = to_new_rank(dcclGetMyRank(comm));
    // const uint32_t my_slice = reverse_bits(my_rank,exponent);
    auto shard_members      = get_dccl_shard_members(comm);

    void* send_buffer = buffer; // pointer to the data to send in a step
    void* recv_buffer = buffer; // pointer to the data to reduce (with the received data) in a step
    size_t step_bsize = count*data_entry_size;  // size of the buffer in bytes to be processed in a step
/**
 * @cond Doxygen_Suppressed.
 */
#define __UPPER_HALF_PTR__(ptr,bsize)   (ptr)
#define __LOWER_HALF_PTR__(ptr,bsize)   reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + (bsize>>1))
/**
 * @endcond
 */
    dccl_trace("{}: STEP 2: prepare buffer and exchange data. Buffer = {} bytes @ {:p} ",
               __func__, step_bsize, buffer);

    for (uint32_t step = 0; step < exponent; step ++) {

        dccl_trace("{}: reduce_scatter step-{}", __func__, step);

        uint32_t peer_rank = (my_rank&(~((1<<(step+1))-1))) + (my_rank+(1<<step))%(1<<(step+1));
        auto peer_id = shard_members.at(to_old_rank(peer_rank));
        dccl_trace("{}: current peer is rank:{}(node_id:{}).", __func__, peer_rank, peer_id);
        if (((my_rank>>step)&1) == 0) {
            send_buffer = __LOWER_HALF_PTR__(recv_buffer,step_bsize);
            recv_buffer = __UPPER_HALF_PTR__(recv_buffer,step_bsize);
        } else {
            send_buffer = __UPPER_HALF_PTR__(recv_buffer,step_bsize);
            recv_buffer = __LOWER_HALF_PTR__(recv_buffer,step_bsize);
        }
        step_bsize = (step_bsize>>1);
        dccl_trace("step-{}, me{}:id-{} <--> peer{}:id-{}, data size = {} Bytes, recv_buffer = {:p}, send_buffer = {:p}",
                   step, my_rank, shard_members.at(to_old_rank(my_rank)), 
                   peer_rank, peer_id, step_bsize, recv_buffer, send_buffer);
        uint32_t s_chunks = dccl_oob_send(comm,peer_id,send_buffer,step_bsize);
        uint32_t r_chunks = dccl_oob_recv(comm,peer_id,scratchpad,step_bsize);
        dccl_oob_wait_for_send(comm,peer_id,s_chunks);
        dccl_oob_wait_for_recv(comm,peer_id,r_chunks);

        // do reduce...
#ifdef CUDA_FOUND
        if (is_in_device) {
            ON_DCCL_DATATYPE(datatype,
                             ret=do_device_reduce,
                             scratchpad,recv_buffer,
                             step_bsize/size_of_type(datatype),op,stream);
            if (ret == ncclSuccess) {
                ret = (sync_stream(stream) == cudaSuccess)?ncclSuccess:ncclUnhandledCudaError;
            }
        } else {
#endif // CUDA_FOUND
            ON_DCCL_DATATYPE(datatype,
                             ret=do_host_reduce,
                             scratchpad,recv_buffer,
                             step_bsize/size_of_type(datatype),op);
#ifdef CUDA_FOUND
        }
#endif // CUDA_FOUND
        if (ret != ncclSuccess) {
            break;
        }
    }

    dccl_trace("{}: Finished.", __func__);

    return ret;
}

} // namespace algorithm
} // namespace dccl
