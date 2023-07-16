#pragma once
#include <iostream>
#include <memory>

#include <derecho/mutils-serialization/SerializationSupport.hpp>

/**
 * @file blob.hpp
 * @brief The Blob object for derecho RPCs.
 */

namespace dccl{

/**
 * @brief   blob memory modes
 * - By `DEFAULT`, a blob object owns the memory: it allocates the memory in constructor and frees it afterward.
 * - `EMPLACED` means the blob does not own the memory passed to the constructor. The blob object neither allocates nor
 * free it in constructor and destructor respectively.
 * - `BLOB_GENERATOR` dictates using of a data generator callback to avoid copy.
 */
enum object_memory_mode_t {
    DEFAULT,
    EMPLACED,
    BLOB_GENERATOR,
};

/**
 * @brief   The blob data generator callback type
 *
 * Using this function to avoid message copy.
 */
using blob_generator_func_t = std::function<std::size_t(uint8_t*,const std::size_t)>;

/**
 * @brief The blob type go through Derecho's RPC
 * 
 * Optimized for passing big message through Derecho's RPC.
 */
class Blob : public mutils::ByteRepresentable {
public:
    /**
     * @brief   Pointer to the data buffer.
     */
    const uint8_t* bytes;
    /**
     * @brief   The size of the data.
     */
    std::size_t size;
    /**
     * @brief   The capacity of the data buffer.
     */
    std::size_t capacity;

    /**
     * @brief   For BLOB_GENERATOR mode only
     */
    blob_generator_func_t blob_generator;

    /**
     * @brief   the memory mode.
     */
    object_memory_mode_t memory_mode;

    /**
     * @brief   constructor - copy to own the data.
     *
     * @param[in]   b       Pointer to a buffer containing the data.
     * @param[in]   s       The size of the data.
     */
    Blob(const uint8_t* const b, const decltype(size) s);

    /**
     * @brief   constructor - with emplaced option.
     *
     * @param[in]   b           Pointer to a buffer containing the data.
     * @param[in]   s           The size of the data.
     * @param[in]   emplaced    If true, the object will not allocate memory buffer. The `bytes` member just
     *                          points to the data buffer.
     */
    Blob(const uint8_t* b, const decltype(size) s, bool emplaced);

    /**
     * @brief   constructor - with data generator callback.
     *
     * @param[in]   generator   The user generator
     * @param[in]   s           The size of the data.
     */
    Blob(const blob_generator_func_t& generator, const decltype(size) s);

    /**
     * @brief   copy constructor
     * 
     * @param[in]   other       The other `Blob` object.
     */
    Blob(const Blob& other);

    /**
     * @brief   move constructor
     *
     * @param[in]   other       The other `Blob` object.
     */
    Blob(Blob&& other);

    /**
     * @brief   The default constructor
     */
    Blob();

    /**
     * @brief   The destructor
     */
    virtual ~Blob();

    /**
     * @brief   The move evaluator
     */
    Blob& operator=(Blob&& other);

    /**
     * @brief   The copy evaluator
     */
    Blob& operator=(const Blob& other);

    /**
     * serialization/deserialization supports
     * @cond    Doxygen_Suppressed
     */
    std::size_t to_bytes(uint8_t* v) const;

    std::size_t bytes_size() const;

    void post_object(const std::function<void(uint8_t const* const, std::size_t)>& f) const;

    void ensure_registered(mutils::DeserializationManager&) {}

    static std::unique_ptr<Blob> from_bytes(mutils::DeserializationManager*, const uint8_t* const v);

    static mutils::context_ptr<Blob> from_bytes_noalloc(
        mutils::DeserializationManager* ctx,
        const uint8_t* const v);

    static mutils::context_ptr<const Blob> from_bytes_noalloc_const(
        mutils::DeserializationManager* ctx,
        const uint8_t* const v);
    /**
     * @endcond
     */
};

} /* namespace dccl*/
