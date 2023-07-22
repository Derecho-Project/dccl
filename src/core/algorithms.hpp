#pragma once
#include <dccl/dccl.hpp>

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
 * @brief Recursive-halving ReduceScatter algorithm
 * This algorithm is the first part of the AllReduce algorithm introduced by Rabenseifner in his paper called
 * [_Optimization of Collective Reduction Operations_]
 * (https://link.springer.com/chapter/10.1007/978-3-540-24685-5_1). Please see Figure 3.1.
 *
 * @param[in,out]   buffer      The `buffer` contains local data to be reduced, it also receives the reduced value
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

/**
 * @brief Recursive-doubling AllGather algorithm
 * This algorithm is the second part of the AllReduce algorithm introduced by Rabenseifner in his paper called
 * [_Optimization of Collective Reduction Operations_]
 * (https://link.springer.com/chapter/10.1007/978-3-540-24685-5_1). Please see Figure 3.1.
 *
 * Important: do NOT use this for a normal AllGather operation because it assumes the results are from the above
 * `reduce_scatter_recursive_halving()` algorithm, where the data block is NOT in rank order (but in an order I
 * call it 'bits-reverse' order.)
 *
 * For example, eight nodes with rank zero to seven will have the eight blocks of reduced data in the following order:
 * ~~~~~~~~~~~~~~~~~~~~~
 * Rank     normal_block_order      bits_reverse_order
 * 0        0                       0 (000b)
 * 1        1                       4 (100b)
 * 2        2                       2 (010b)
 * 3        3                       6 (110b)
 * 4        4                       1 (001b)
 * 5        5                       5 (101b)
 * 6        6                       3 (011b)
 * 7        7                       7 (111b)
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * @param[in,out]   buffer      The `buffer` contains the results from ReduceScatter. It also receives the allgather
 *                              value after the operation. Its size must be equal or greater than 
 *                              `count*size_of_type(datatype)`. `buffer` must be pre-registered for Derecho OOB
 *                              operations.
 * @param[in]       count       The number of entries in the buffer
 * @param[in]       datatype    The type of the data.
 * @param[in]       comm        The DCCL communication object.
 *
 * @throws      std::runtime_erro   A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t all_gather_recursive_doubling(
        void*           buffer,
        size_t          count,
        ncclDataType_t  datatype,
        ncclComm_t      comm);

}
}
