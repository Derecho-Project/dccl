#pragma once
#include <dccl.hpp>

/**
 * @file algorithms.hpp
 * @brief Internal implementations for the collective communication algorithms
 * 
 * This is the internal interface between DCCL API implementations and the internal collective communication
 * algorithms.
 */

namespace dccl {

/**
 * @brief   The `algorithm` namespace contains all internal collective communication implementations.
 *
 * The building blocks for DCCL API.
 */
namespace algorithm {
    /**
     * @brief Recursive-halving reduce scatter algorithm
     * The recursive-halving reduce scatter algorithm is from Thakur, Rabinseifner, and Gropp's paper
     * called [_Optimization of Collective Communication Operations in MPICH_]
     * (https://journals.sagepub.com/doi/pdf/10.1177/1094342005051521). Please see section 4.4.
     *
     * @param[in,out]   buffer      The `buffer` containing local data to be reduced, it also receives the reduced value
     *                              after the operation. Its size must be equal or greater than 
     *                              `count*size_of_type(datatype)`. `buffer` must be pre-registered for Derecho OOB
     *                              operations.
     * @param[in]       scratchpad  The `scratchpad` is a memory cache of half the size of `buffer`. It must be
     *                              pre-registered for Derecho OOB operations.
     * @param[in]       count       The number of entries in the buffer.
     * @param[in]       datatype    The type of the data.
     * @param[in]       op          The reduced operation to be performed.
     * @param[in]       comm        The DCCL communication object.
     *
     * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
     *
     * @return      Error code
     */
    ncclResult_t reduce_scatter_recursive_halving(
            void*           buffer,
            void*           scratchpad,
            size_t          count,
            ncclDataType_t  datatype,
            ncclRedOp_t     op,
            ncclComm_t      comm);

}
}
