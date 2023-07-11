#pragma once

/**
 * @file    internal_common.hpp
 * @brief   Internal utilities (types, macros, and functions) which be hidden from DCCL API users.
 *
 * This file contains the internal types, macros, and functions used by multiple C++ files. Those utilities are for
 * internal use only and should be hidden from the users.
 */
#include <atomic>
#include <derecho/core/derecho.hpp>
#include "dccl.hpp"
#include "blob.hpp"

using namespace derecho;

namespace dccl {

/**
 * @brief The DCCL Subgroup Class type
 * It defines the Derecho Subgroup type that supports the DCCL APIs.
 */
class DCCLSubgroupType : public mutils::ByteRepresentable,
                         public GroupReference {
public:
    std::atomic<void*>  recvbuf;

    /**
     * @brief       Reduce operation
     *              It calculate the reduce value and put it into the buffer `recvbuf` pointed to. It is the caller's
     *              responsibility to make sure `recvbuf` is valid and registered with and OOB
     *
     * @todo        This implementation was used for AllReduce implementation, whick was an unsophisticated and
     *              inefficient API. Change it to a reduce API that accept one more param specifying which node
     *              will do the operation.
     *
     */
    virtual ncclResult_t reduce(const Blob& sendbuf, const size_t, ncclDataType_t datatype, ncclRedOp_t op, bool inplace);

    REGISTER_RPC_FUNCTIONS(DCCLSubgroupType,ORDERED_TARGETS(reduce));

    // serialization support
    virtual std::size_t to_bytes(uint8_t*) const override {return 0;}

    virtual void post_object(const std::function<void(uint8_t const* const, std::size_t)>&) const override {}

    virtual std::size_t bytes_size() const {return 0;}

    static std::unique_ptr<DCCLSubgroupType> from_bytes(
            mutils::DeserializationManager*, const uint8_t*) {
        return std::make_unique<DCCLSubgroupType>();
    }

    static mutils::context_ptr<DCCLSubgroupType> from_bytes_noalloc(
            mutils::DeserializationManager*, const uint8_t*) {
        return mutils::context_ptr<DCCLSubgroupType>{new DCCLSubgroupType()};
    }
    static mutils::context_ptr<const DCCLSubgroupType> from_bytes_noalloc_const(
            mutils::DeserializationManager*, const uint8_t*) {
        return mutils::context_ptr<const DCCLSubgroupType>{new DCCLSubgroupType()};
    }
    void ensure_registered(mutils::DeserializationManager&) {}
    
    // constructors
    DCCLSubgroupType():recvbuf(nullptr) {}
};

/**
 * @brief Get the group handle from the DCCL communication object.
 *
 * @param[in]   comm        The DCCL communication object.
 *
 * @return      A pointer to `Group<DCCLSubgroupType>` object.
 */
#define GROUP_HANDLE(comm)      (reinterpret_cast<Group<DCCLSubgroupType>*>((comm)->derecho_group_handle))

/**
 * @brief Get the subgroup handle from the communication object.
 *
 * @param[in]   comm        The DCCL communication object.
 *
 * @return      A pointer to `Replicated<DCCLSubgroupType>` object.
 */
#define SUBGROUP_HANDLE(comm)   (GROUP_HANDLE(comm)->get_subgroup<DCCLSubgroupType>())

/**
 * @brief Get the wrapped subgroup object.
 *
 * @param[in]   comm        The DCCL communication object.
 *
 * @return      A pointer to the wrapped object of type DCCLSubgroupType.
 */
#define SUBGROUP_OBJECT(comm)   (reinterpret_cast<DCCLSubgroupType*>((comm)->derecho_group_object))

} // namespace dccl
