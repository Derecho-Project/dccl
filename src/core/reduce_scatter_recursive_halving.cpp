#include "algorithms.hpp"

namespace dccl {
namespace algorithm {

ncclResult_t reduce_scatter_recursive_halving(
        const void*     sendbuff,
        void*           recvbuff,
        size_t          count,
        ncclDataType_t  datatype,
        ncclRedOp_t     op,
        ncclComm_t comm) {
    // STEP 1: test the buffer's offset

    return ncclSuccess;
}

} // namespace algorithm
} // namespace dccl
