#include <dccl/dccl.hpp>
#include <iostream>

using namespace dccl;

int main(int argc, char** argv) {
    ncclComm_t comm;
    auto ret = ncclCommInit(&comm);
    if (ret != ncclSuccess) {
        std::cerr << "DCCL Communicator initialization failed." << std::endl;
    }

    std::cout << "Hello! world size is " << dcclGetWorldSize(comm)
              << ", my rank is " << dcclGetMyRank(comm)
              << std::endl;

    ncclCommFinalize(comm);
    return 0;
}
