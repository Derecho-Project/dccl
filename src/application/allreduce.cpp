#include <dccl.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "dccl allreduce tester." << std::endl;
    ncclComm_t comm;
    ncclResult_t ret;

    // step 1 - initialize comm
    ret = ncclCommInit(&comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to initialize dccl communication." << std::endl;
        return ret;
    }

    // step 2 - simple all reduce test
    uint32_t sendbuf[] = {1,2,3,4,5};
    uint32_t recvbuf[5];

    ret = ncclAllReduce(reinterpret_cast<const void*>(sendbuf),
                        reinterpret_cast<void*>(recvbuf),
                        5,ncclUint32,ncclProd/*ncclSum*/,comm);

    if (ret != ncclSuccess) {
        std::cerr << "all reduce failed with error:" << ret << std::endl;
        return ret;
    } else {
        std::cout << "all reduce result: [";
        for (int i = 0; i < 5; i++) {
            std::cout << recvbuf[i] << ",";
        }
        std::cout << "]" << std::endl;
    }

    // step 3 - finalize comm
    ret = ncclCommFinalize(comm);
    if (ret != ncclSuccess) {
        std::cerr << "failed to finalize the dccl communication." << std::endl;
    }
    return 0;
}
