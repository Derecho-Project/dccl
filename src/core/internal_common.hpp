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
 * @return      A reference to `Replicated<DCCLSubgroupType>` object.
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

/**
 * @brief Get the shard members
 *
 * @tparam      SubgroupType    The type of the derecho subgroup. For now, it must be `DCCLSubgroupType`
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      An std::vector object contains all node ids in my shard.
 */
template <typename SubgroupType>
inline auto _get_shard_members(ncclComm_t comm) {
    auto* group = GROUP_HANDLE(comm);
    auto subgroup_indexes = group->get_my_subgroup_indexes<SubgroupType>();

    assert(subgroup_indexes.size() == 1);

    uint32_t subgroup_index = subgroup_indexes.at(0);
    int32_t my_shard = group->get_my_shard<SubgroupType>(subgroup_index);
    return group->get_subgroup_members<SubgroupType>(subgroup_index)[my_shard];
}

/**
 * @brief A wrapper around _get_shard_members as syntax sugar
 *
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      An std::vector object contains all node ids in my shard.
 */
inline auto get_dccl_shard_members(ncclComm_t comm) {
    return _get_shard_members<DCCLSubgroupType>(comm);
}

/**
 * @brief Get the world size
 *
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      The number of members in the shard.
 */
inline uint32_t get_world_size(ncclComm_t comm) {
    return _get_shard_members<DCCLSubgroupType>(comm).size();
}

/**
 * @brief Get my rank
 *
 * @param[in]   comm            The DCCL communication object.
 *
 * @return      My rank in the shard, starting from 0.
 */
inline uint32_t get_my_rank(ncclComm_t comm) {
    auto my_id = GROUP_HANDLE(comm)->get_my_id();
    auto shard_members = _get_shard_members<DCCLSubgroupType>(comm);
    uint32_t my_rank = 0;
    while(my_rank < shard_members.size() && shard_members.at(my_rank) != my_id) {
        my_rank ++;
    }
    assert (my_rank < shard_members.size());
    return my_rank;
}

/**
 * @brief Get the size (in bytes) of DCCL data types.
 * 
 * @param[in]   datatype    The DCCL data type
 *
 * @return      Size of the given data type in bytes. Returns `0` if the data type is invalid.
 */
inline size_t size_of_type(ncclDataType_t datatype) {
    switch(datatype) {
    case ncclInt8: // ncclChar
    case ncclUint8:
        return 1;
    case ncclInt32: // ncclInt
    case ncclUint32:
    case ncclFloat32: // ncclFloat
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64: // ncclDouble
        return 8;
    case ncclFloat16: // ncclHalf:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
        return 2;
    default:
        return 0;
    }
}


/**
 * @brief Reverse the least significants bits of an integer.
 * This function reverse the order of least significant `nbits` bits in an integer.
 * For example,
 * - `0B*0000101` --[reversed with `nbits=7`]--> `0B*1010000`,
 * - `0B****1110` --[reversed with `nbits=4`]--> `0B****0111`,
 * - `0B*****011` --[reversed with `nbits=3`]--> `0B*****110`,
 *
 * The asterisk signs(*) represent irrelavent/unchanged bits.
 *
 * @tparam      T       Type of the integer, could be any primitive integer type.
 * @param[in]   x       the value to be reversed.
 * @param[in]   nbits   optional `nbits` specifies the range by the number of least significant bits, which must be
 *                      less or equal to sizeof(T)*8. This argument is defaulted to sizeof(T)*8.
 *
 * @return      The reversed value.
 */
template<typename T>
inline T reverse_bits(const T x, size_t nbits = sizeof(T)*8) {
    assert(nbits <= sizeof(T)*8);
    T rx = ((~static_cast<T>(0))<<nbits);
    T tx = x;

    for (size_t p=0;p<nbits;p++) {
        rx = ((rx << 1) | (tx&1));
        tx = (tx >> 1);
    }
    return rx;
}

/**
 * @brief Test if an integer is power of two.
 *
 * @param[in]   n   The integer
 *
 * @return      true if n is power of two.
 */
#define IS_POWER_OF_TWO(n) ((n > 0) && (((n)&((n)-1)) == 0))

/**
 * @brief Calculate the exponent of a power-of-two integer
 *
 * @tparam      IntegerType The type of the integer
 * @param[in]   n           The integer, which must be power of two
 *
 * @return      log2(n)
 */
template<typename IntegerType>
inline IntegerType log_two(IntegerType n) {
    assert(IS_POWER_OF_TWO(n));
    IntegerType e = 0;
    const IntegerType nbits = sizeof(IntegerType)*8 - (std::is_signed<IntegerType>::value?1:0);
    while (e < nbits) {
        if (static_cast<IntegerType>(1<<e) >= n) {
            break;
        }
        e ++;
    }
    assert(static_cast<IntegerType>(1<<e) == n);
    return e;
}

/**
 * @brief Get the DCCL `spdlog` logger singleton
 * @return      A shared pointer to the logger, that can be used with Derecho's logger macros like the following:
 *              dbg_trace, dbg_debug, dbg_warn, dbg_error, ...
 */
std::shared_ptr<spdlog::logger>& getDcclLogger();

#define dccl_trace(...) dbg_trace(getDcclLogger(), __VA_ARGS__)
#define dccl_debug(...) dbg_debug(getDcclLogger(), __VA_ARGS__)
#define dccl_info(...)  dbg_info(getDcclLogger(), __VA_ARGS__)
#define dccl_warn(...)  dbg_warn(getDcclLogger(), __VA_ARGS__)
#define dccl_error(...) dbg_error(getDcclLogger(), __VA_ARGS__)
#define dccl_crit(...)  dbg_crit(getDcclLogger(), __VA_ARGS__)
#define dccl_flush()    dbg_flush(getDcclLogger())

} // namespace dccl
