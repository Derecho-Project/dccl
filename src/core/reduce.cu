#define NVCC_VISIBLE
#include <dccl/dccl.hpp>
#include "internal_common.hpp"
#include <cuda_fp16.h>

namespace dccl{

template<typename DT>
__global__ void reduce_kernel(
    const void*     sendbuf,
    void*           recvbuf,
    size_t          count,
    ncclRedOp_t     op) {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    const DT*   sbuf = static_cast<const DT*>(sendbuf);
    DT*         rbuf = static_cast<DT*>(recvbuf);
    while(idx < count) {
        switch (op) {
        case ncclSum:
        case ncclAvg:
            rbuf[idx] += sbuf[idx];
            break;
        case ncclProd:
            rbuf[idx] *= rbuf[idx];
            break;
        case ncclMax:
            rbuf[idx] = (rbuf[idx]>=sbuf[idx])?rbuf[idx]:sbuf[idx];
            break;
        case ncclMin:
            rbuf[idx] = (rbuf[idx]<=sbuf[idx])?rbuf[idx]:sbuf[idx];
            break;
        default:
            break;
        }
        idx += gridDim.x*blockDim.x;
    }
}

ncclResult_t do_device_reduce(
    const void*     sendbuf,
    void*           recvbuf,
    ncclDataType_t  dtype,
    size_t          count,
    ncclRedOp_t     op,
    cudaStream_t    stream) {
    const size_t nblock = std::min(count/256,static_cast<size_t>(40)); // Tesla T4 has 40 SM Processors.
    switch(dtype) {
    case ncclInt8:
        reduce_kernel<int8_t><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclUint8:
        reduce_kernel<uint8_t><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclInt32:
        reduce_kernel<int32_t><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclUint32:
        reduce_kernel<uint32_t><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclInt64:
        reduce_kernel<int64_t><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclUint64:
        reduce_kernel<uint64_t><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclFloat32:
        reduce_kernel<float><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclFloat64:
        reduce_kernel<double><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    case ncclFloat16:
        reduce_kernel<half><<<nblock,256,0,stream>>>(sendbuf,recvbuf,count,op);
        break;
    default:
        return ncclInvalidArgument;
    }

    return ncclSuccess;
}

}
