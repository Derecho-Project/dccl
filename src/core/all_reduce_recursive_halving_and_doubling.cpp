#include "algorithms.hpp"
#include "internal_common.hpp"


namespace dccl {
namespace algorithm {

ncclResult_t all_reduce_recursive_halving_and_doubling(
        void*           buffer,
        void*           scratchpad,
        size_t          count,
        ncclDataType_t  datatype,
        ncclRedOp_t     op,
        ncclComm_t      comm,
        cudaStream_t    stream) {
    uint32_t        my_rank = dcclGetMyRank(comm);
    ncclResult_t    ret = ncclSuccess;
    size_t          total_data_size = count * size_of_type(datatype);
    uint32_t        world_size =    dcclGetWorldSize(comm);
    auto            shard_members  = get_dccl_shard_members(comm);

    // STEP 1 check contraints
#ifdef CUDA_FOUND
    bool            is_in_device = is_device_ptr(buffer);
    if (!is_in_device) {
#endif
        if (CACHELINE_OFFSET(buffer)) {
            dccl_warn("The buffer @{:p} is not cacheline ({} bytes) aligned. "
                      "Possible performance degradation might occur.",
                      buffer, CACHELINE_SIZE);
    
        }
#ifdef CUDA_FOUND
    }
#endif

    // STEP 2 preprocessing -- > power of two
    // # new rank is calculated as follows:
    // '''
    // world_size = 2^n + r
    // if old_rank < 2*r and old_rank is even:
    //      new_rank = old_rank/2
    //      perform as leader
    // if old_rank < 2*r and old_rank is odd:
    //      new_rank = old_rank/2
    //      perform as follower
    // if old_rank >= 2*r:
    //      new_rank = old_rank - r
    //      perform as independent
    uint32_t subworld_size  = (1<<log_two(world_size));
    if (count % subworld_size) {
        dccl_error("Support for uneven data per node to be added yet.");
        return ncclInvalidArgument;
    }
    uint32_t r              = world_size - subworld_size;
    const rank_converter_t to_new_rank  = [r] (uint32_t _or) {
        return (_or < 2*r)? ( _or / 2) : (_or - r);
    };
    const rank_converter_t to_old_rank  = [r] (uint32_t _nr) {
        return (_nr < r)? (_nr * 2) : (_nr + r);
    };
    uint32_t my_new_rank = to_new_rank(my_rank);
    enum ar_rhd_role {
        Leader,
        Follower,
        Independent
    } my_role = (my_rank < 2*r)? ((my_rank%2)?Follower:Leader) : Independent;

    dccl_trace("{}: p={},r={}, old_rank:{} --> new_rank:{}, perform as {}.",
               __func__, world_size, r, my_rank, my_new_rank, my_role);

    if (my_role == Leader)
    {
        uint32_t peer_rank = my_rank+1;
        uint32_t peer_id = shard_members.at(peer_rank);
        // send the bottom half to the follower
        uint32_t s_chunks = dccl_oob_send(comm,peer_id,
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + (total_data_size>>1)),
                total_data_size>>1);
        // receive the top half from the follower
        uint32_t r_chunks = dccl_oob_recv(comm,peer_id,scratchpad,total_data_size>>1);
        dccl_oob_wait_for_send(comm,peer_id,s_chunks);
        dccl_oob_wait_for_recv(comm,peer_id,r_chunks);
        // do reduce for the top half
#ifdef CUDA_FOUND
        if (is_in_device) {
            ON_DCCL_DATATYPE(datatype,
                             ret=do_device_reduce,
                             scratchpad,buffer,
                             count>>1,op,stream);
        } else {
#endif
            ON_DCCL_DATATYPE(datatype,
                             ret=do_host_reduce,
                             scratchpad,buffer,
                             count>>1,op);
#ifdef CUDA_FOUND
        }
#endif
        if (ret != ncclSuccess) {
            dccl_error("{} failed to do reduce, error= {}.",
                       __func__, ret);
            return ret;
        }
        // receive the bottom half result from the follower
        r_chunks = dccl_oob_recv(comm,peer_id,
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + (total_data_size>>1)),
                total_data_size>>1);
        dccl_oob_wait_for_recv(comm,peer_id,r_chunks);
    } else if (my_role == Follower) {
        uint32_t    peer_rank   = my_rank - 1;
        auto        peer_id     = shard_members.at(peer_rank);
        // send the top half to the leader
        uint32_t s_chunks = dccl_oob_send(comm,peer_id,buffer,total_data_size>>1);
        // receive the bottom half from the leader
        uint32_t r_chunks = dccl_oob_recv(comm,peer_id,scratchpad,total_data_size>>1);
        dccl_oob_wait_for_send(comm,peer_id,s_chunks);
        dccl_oob_wait_for_recv(comm,peer_id,r_chunks);
        // do reduce for the bottom half
#ifdef CUDA_FOUND
        if (is_in_device) {
            ON_DCCL_DATATYPE(datatype,
                             ret=do_device_reduce,
                             scratchpad,reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer)+(total_data_size>>1)),
                             count>>1,op,stream);
        } else {
#endif
            ON_DCCL_DATATYPE(datatype,
                             ret=do_host_reduce,
                             scratchpad,reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer)+(total_data_size>>1)),
                             count>>1,op);
#ifdef CUDA_FOUND
        }
#endif
        if (ret != ncclSuccess) {
            dccl_error("{} failed to do reduce, error= {}.",
                       __func__, ret);
            return ret;
        }
        // send the bottom half result to the leader
        s_chunks = dccl_oob_send(comm,peer_id,
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + (total_data_size>>1)),
                total_data_size>>1);
        dccl_oob_wait_for_send(comm,peer_id,s_chunks);
    }

    TIMESTAMP(TT_ALLREDUCE_RDH_PREPROCESS,my_rank,op);

    // STEP 3 reduce scatter and all gather on subworld.
    if (my_role == Leader || my_role == Independent) {
        ret = reduce_scatter_recursive_halving(buffer,scratchpad,count,datatype,op,comm,stream,
                                               subworld_size,to_new_rank,to_old_rank);
        if (ret != ncclSuccess) {
            dccl_error("{}: reduce scatter failed with error = {}.",
                       __func__, ret);
            return ret;
        }

        TIMESTAMP(TT_ALLREDUCE_REDUCESCATTER,my_rank,op);

        ret = all_gather_recursive_doubling(buffer,count,datatype,comm,stream,
                                            subworld_size,to_new_rank,to_old_rank);

        if (ret != ncclSuccess) {
            dccl_error("{}: all gather failed with error = {}.",
                       __func__, ret);
            return ret;
        }
        TIMESTAMP(TT_ALLREDUCE_ALLGATHER,my_rank,op);
    } else {
        TIMESTAMP(TT_ALLREDUCE_REDUCESCATTER,my_rank,op);
        TIMESTAMP(TT_ALLREDUCE_ALLGATHER,my_rank,op);
    }

    // STEP 4 postprocessing.
    if (my_role == Leader) {
        // send total data to Follower
        uint32_t    peer_rank   = my_rank + 1;
        auto        peer_id     = shard_members.at(peer_rank);
       
        uint32_t s_chunks = dccl_oob_send(comm,peer_id,buffer,total_data_size);
        dccl_oob_wait_for_send(comm,peer_id,s_chunks);
    } else if (my_role == Follower) {
        // receive total data from Leader
        uint32_t    peer_rank   = my_rank - 1;
        auto        peer_id     = shard_members.at(peer_rank);
       
        uint32_t r_chunks = dccl_oob_recv(comm,peer_id,buffer,total_data_size);
        dccl_oob_wait_for_recv(comm,peer_id,r_chunks);
    }

    TIMESTAMP(TT_ALLREDUCE_RDH_POSTPROCESS,my_rank,op);

    return ret;
}

}
}
