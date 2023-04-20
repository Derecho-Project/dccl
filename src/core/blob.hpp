#pragma once
#include <iostream>
#include <memory>

#include <derecho/mutils-serialization/SerializationSupport.hpp>

namespace derecho{

enum object_memory_mode_t {
    DEFAULT,
    EMPLACED,
    BLOB_GENERATOR,
};

using blob_generator_func_t = std::function<std::size_t(uint8_t*,const std::size_t)>;

class Blob : public mutils::ByteRepresentable {
public:
    const uint8_t* bytes;
    std::size_t size;
    std::size_t capacity;

    // for BLOB_GENERATOR mode only
    blob_generator_func_t blob_generator;

    object_memory_mode_t memory_mode;

    // constructor - copy to own the data
    Blob(const uint8_t* const b, const decltype(size) s);

    Blob(const uint8_t* b, const decltype(size) s, bool emplaced);

    // generator constructor - data to be generated on serialization
    Blob(const blob_generator_func_t& generator, const decltype(size) s);

    // copy constructor - copy to own the data
    Blob(const Blob& other);

    // move constructor - accept the memory from another object
    Blob(Blob&& other);

    // default constructor - no data at all
    Blob();

    // destructor
    virtual ~Blob();

    // move evaluator:
    Blob& operator=(Blob&& other);

    // copy evaluator:
    Blob& operator=(const Blob& other);

    // serialization/deserialization supports
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
};

}
