#pragma once
#include <sys/types.h>

/**
 * @file    dccl.hpp
 * @brief   Derecho Collective Communication Library (DCCL) API
 * @see     dccl
 *
 * This file contains the declaration of the Derecho Collective Communication Library (DCCL) API. Similar to 
 * [RCCL](https://rccl.readthedocs.io/en/develop/api.html), DCCL is compatible to 
 * [NCCL](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_296/user-guide/docs/api.html).
 * We put all DCCL API to the `dccl` namespace. 
 */

/**
 * @brief DCCL namespace contains all API declaration and definition.
 *
 * We put the declarations of all DCCL APIs, as well as their definitions in to this namespace. Using 
 * `using namespace dccl;` is recommended for application code.
 */
namespace dccl {


/**
 * @brief   Error types
 */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8 } ncclResult_t;

/**
 * @brief   Data types 
 */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
               ncclBfloat16   = 9,
               ncclNumTypes   = 10
#else
               ncclNumTypes   = 9
#endif
} ncclDataType_t;

typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;

/**
 * @brief   Reduction operation selector
 */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclAvg        = 4,
               /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
                * serves as the least possible value for dynamic ncclRedOp_t's
                * as constructed by ncclRedOpCreate*** functions. */
               ncclNumOps     = 5,
               /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
                * maintain ABI compatibility. */
               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t))
             } ncclRedOp_t;

/**
 * @brief   DCCL communication type
 */
struct dcclComm {
    void* derecho_group_handle;
    void* derecho_group_object;
};
typedef struct dcclComm* ncclComm_t ;

/* dccl init rank is slightly different, the rank and world size information
 * in in the configuration file. So we dont need them here. */
/*
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
*/

/**
 * @defgroup DCCL APIs
 * @{
 */

/**
 * @brief   Initialize the communication infrastructure.
 * @param[out]  comm    Pointer to a `ncclComm_t` object to be initialized
 *
 * @return      Error code
 */
ncclResult_t  ncclCommInit(ncclComm_t* comm);

/**
 * @brief   Finalize the communication infrastructure
 *
 * @param[in]   comm    An initialzied `ncclComm_t` object to be finalized.
 *
 * @return      Error code
 */
ncclResult_t  ncclCommFinalize(ncclComm_t comm);

/**
 * @brief   Register user-allocated memory as cache to DCCL.
 * Using registered memory in DCCL can avoid on-demand registeration on the critical path. We recommend doing so to
 * improve the performance. Also, the address and size of memory  registered has to be cacheline aligned.
 *
 * @param[in]   comm    An initialzied `ncclComm_t` object to be finalized.
 * @param[in]   buffer  The address of the memory buffer to be registered.
 * @param[in]   size    The size of the buffer.
 *
 * @throws      Runtime error might be raised on error.
 * @return      Error code.
 */
ncclResult_t  dcclRegisterCacheMemory(ncclComm_t comm, void* buffer, size_t size);

/**
 * @brief   Deregister a pre-registered user-allocated memory region from DCCL.
 *
 * @param[in]   comm    An initialzied `ncclComm_t` object to be finalized.
 * @param[in]   buffer  The address of the registered memory buffer.
 * @param[in]   size    The size of the buffer. Additional check on size would be performed if given.
 *
 * @throws      Runtime error might be raised on error.
 * @return      Error code.
 */
ncclResult_t  dcclDeregisterCacheMemory(ncclComm_t comm, void* buffer, size_t size = 0UL);

/**
 * @brief All-Reduce API
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 *
 * @param[in]   sendbuff    The buffer containing local data to be reduced.
 * @param[out]  recvbuff    The buffer to receive the reduced result.
 * @param[in]   count       The number of entries in the buffer.
 * @param[in]   datatype    The type of the data.
 * @param[in]   op          The reduced operation to be performed.
 * @param[in]   comm        The DCCL communication object.
 *
 * @throws      std::runtime_error A runtime error might be raised in case of exceptions.
 *
 * @return      Error code
 */
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm);

/**
 * @}
 */

/**
 * @defgroup DCCL helper functions and macros
 * @{
 */

/**
 * @brief Test the offset of an address in a cacheline.
 *
 * @param[in]   addr        Some address of any pointer type
 *
 * @return      The offset of `addr` in a cacheline.
 */
#define CACHELINE_OFFSET(addr)  ( ((uint64_t)addr)%CLSZ )

/**
 * @}
 */

} // namespace dccl
