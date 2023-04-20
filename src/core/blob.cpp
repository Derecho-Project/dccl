#include <unistd.h>
#include <stdlib.h>
#include "blob.hpp"

namespace derecho {

/*
 *  IMPORTANT NOTICE of Blob Implementation
 *
 *  Blob is an inner type of ObjectWithXXXKeys class, which is repsonsible for store the objects with various length.
 *  Copy constructor is heavy for large objects. We found that sometimes, the Linux malloc/memcpy behave wiredly if the
 *  memory is not aligned to page boundary. It incurs a lot of page walks (The phenomenon can be reproduced by
 *  'wired_reconnecting_external_client' branch with a setup of 1 VCSS node, 1MB message, and 1 external client). To
 *  solve this issue, we only use full pages for Blob data buffer. To further improve the performance, we should use
 *  hugepages.
 *
 *  Update on Sept 1st, 2021
 *  The reason for the "wired" issue is found. It was because the malloc system adatps the size of idle pages to use
 *  memory smartly. The glibc tunable 'glibc.malloc.trim_threshold' controls that behaviour. If
 *  glibc.malloc.trim_threshold is not set, the default value of the threshold is 128KB and dynamically changing by the
 *  workload. So for the first time, when the workload keeps malloc-ing and free-ing 1MB memory chunks, the free
 *  operation will return the pages to OS. Therefore, the malloc-ed new pages does not exist in the page table.
 *  Accessing the new allocated memory cause a page fault and page walk for each new page (256 4K pages per 1MB),
 *  causing 4~5x overhead in memcpy(). But later, when the threshold adapts to the new workload, The performance is back
 *  to normal.
 *
 *  The right way to avoid this is to use an optimal trim_threshold value instead of setting page alignment.
 */
// static const std::size_t page_size = sysconf(_SC_PAGESIZE);
// #define PAGE_ALIGNED_NEW(x) (new uint8_t[((x)+page_size-1)/page_size*page_size])

Blob::Blob(const uint8_t* const b, const decltype(size) s) :
    bytes(nullptr), size(0), capacity(0), memory_mode(object_memory_mode_t::DEFAULT) {
    if(s > 0) {
        // uint8_t* t_bytes = PAGE_ALIGNED_NEW(s);
        uint8_t* t_bytes = static_cast<uint8_t*>(malloc(s));
        if (b != nullptr) {
            memcpy(t_bytes, b, s);
        } else {
            bzero(t_bytes, s);
        }
        bytes = t_bytes;
        size = s;
        capacity = size;
    }
}

Blob::Blob(const uint8_t* b, const decltype(size) s, bool emplaced) :
    bytes(b), size(s), capacity(s), memory_mode((emplaced)?object_memory_mode_t::EMPLACED:object_memory_mode_t::DEFAULT) {
    if ( (size>0) && (emplaced==false)) {
        // uint8_t* t_bytes = PAGE_ALIGNED_NEW(s);
        uint8_t* t_bytes = static_cast<uint8_t*>(malloc(s));
        if (b != nullptr) {
            memcpy(t_bytes, b, s);
        } else {
            bzero(t_bytes, s);
        }
        bytes = t_bytes;
    }
    // exclude illegal argument combinations like (0x982374,0,false)
    if (size == 0) {
        bytes = nullptr;
    }
}

Blob::Blob(const blob_generator_func_t& generator, const decltype(size) s):
    bytes(nullptr), size(s), capacity(0), blob_generator(generator), memory_mode(object_memory_mode_t::BLOB_GENERATOR) {
    // no data is generated here.
}

Blob::Blob(const Blob& other) :
    bytes(nullptr), size(0), capacity(0), memory_mode(object_memory_mode_t::DEFAULT) {
    if(other.size > 0) {
        uint8_t* t_bytes = static_cast<uint8_t*>(malloc(other.size));
        if (memory_mode == object_memory_mode_t::BLOB_GENERATOR) {
            // instantiate data.
            auto number_bytes_generated = other.blob_generator(t_bytes,other.size);
            if (number_bytes_generated != other.size) {
                std::string exception_message("Expecting");
                throw std::runtime_error(std::string("Expecting ") + std::to_string(other.size) 
                        + " bytes, but blob generator writes "
                        + std::to_string(number_bytes_generated) + " bytes.");
            }
        } else {
            // uint8_t* t_bytes = PAGE_ALIGNED_NEW(other.size);
            memcpy(t_bytes, other.bytes, other.size);
        }
        bytes = t_bytes;
        size = other.size;
        capacity = other.size;
    }
}

Blob::Blob(Blob&& other) : 
    bytes(other.bytes), size(other.size), capacity(other.size),
    blob_generator(other.blob_generator), memory_mode(other.memory_mode) {
    other.bytes = nullptr;
    other.size = 0;
    other.capacity = 0;
}

Blob::Blob() : bytes(nullptr), size(0), capacity(0), memory_mode(object_memory_mode_t::DEFAULT) {}

Blob::~Blob() {
    if(bytes && (memory_mode == object_memory_mode_t::DEFAULT)) {
        free(const_cast<void*>(reinterpret_cast<const void*>(bytes)));
    }
}

Blob& Blob::operator=(Blob&& other) {
    auto swp_bytes = other.bytes;
    auto swp_size = other.size;
    auto swp_cap  = other.capacity;
    auto swp_blob_generator = other.blob_generator;
    auto swp_memory_mode = other.memory_mode;
    other.bytes = bytes;
    other.size = size;
    other.capacity = capacity;
    other.blob_generator = blob_generator;
    other.memory_mode = memory_mode;
    bytes = swp_bytes;
    size = swp_size;
    capacity = swp_cap;
    blob_generator = swp_blob_generator;
    memory_mode = swp_memory_mode;
    return *this;
}

Blob& Blob::operator=(const Blob& other) {
    // 1) this->is_emplaced has to be false;
    if (memory_mode != object_memory_mode_t::DEFAULT) {
        throw std::runtime_error("Copy to a Blob that does not own the data (object_memory_mode_T::DEFAULT) is prohibited.");
    }

    // 2) verify that this->capacity has enough memory;
    if (this->capacity < other.size) {
        bytes = static_cast<uint8_t*>(realloc(const_cast<void*>(static_cast<const void*>(bytes)),other.size));
        this->capacity = other.size;
    } 

    // 3) update this->size; copy data, if there is any.
    this->size = other.size;
    if(this->size > 0) {
        if (other.memory_mode == object_memory_mode_t::BLOB_GENERATOR) {
            auto number_bytes_generated = other.blob_generator(const_cast<uint8_t*>(this->bytes),other.size);
            if (number_bytes_generated != other.size) {
                std::string exception_message("Expecting");
                throw std::runtime_error(std::string("Expecting ") + std::to_string(other.size) 
                        + " bytes, but blob generator writes "
                        + std::to_string(number_bytes_generated) + " bytes.");
            }
        } else {
            memcpy(const_cast<void*>(static_cast<const void*>(this->bytes)), other.bytes, size);
        }
    }

    return *this;
}

std::size_t Blob::to_bytes(uint8_t* v) const {
    ((std::size_t*)(v))[0] = size;
    if(size > 0) {
        if (memory_mode == object_memory_mode_t::BLOB_GENERATOR) {
            auto number_bytes_generated = blob_generator(v+sizeof(size), size);
            if (number_bytes_generated != size) {
                std::string exception_message("Expecting");
                throw std::runtime_error(std::string("Expecting ") + std::to_string(size) 
                        + " bytes, but blob generator writes "
                        + std::to_string(number_bytes_generated) + " bytes.");
            }
        } else {
            memcpy(v + sizeof(size), bytes, size);
        }
    }
    return size + sizeof(size);
}

std::size_t Blob::bytes_size() const {
    return size + sizeof(size);
}

void Blob::post_object(const std::function<void(uint8_t const* const, std::size_t)>& f) const {
    if (size > 0 && (memory_mode == object_memory_mode_t::BLOB_GENERATOR)) {
        // we have to instatiate the data. CAUTIOUS: this is inefficient. Please use BLOB_GENERATOR mode carefully.
        uint8_t* local_bytes = static_cast<uint8_t*>(malloc(size));
        auto number_bytes_generated = blob_generator(local_bytes,size);
        if (number_bytes_generated != size) {
            free(local_bytes);
            std::string exception_message("Expecting");
            throw std::runtime_error(std::string("Expecting ") + std::to_string(size) 
                    + " bytes, but blob generator writes "
                    + std::to_string(number_bytes_generated) + " bytes.");
        }
        f((uint8_t*)&size, sizeof(size));
        f(local_bytes, size);
        free(local_bytes);
    } else {
        f((uint8_t*)&size, sizeof(size));
        f(bytes, size);
    }
}

mutils::context_ptr<Blob> Blob::from_bytes_noalloc(mutils::DeserializationManager* ctx, const uint8_t* const v) {
    return mutils::context_ptr<Blob>{new Blob(const_cast<uint8_t*>(v) + sizeof(std::size_t), ((std::size_t*)(v))[0], true)};
}

mutils::context_ptr<const Blob> Blob::from_bytes_noalloc_const(mutils::DeserializationManager* ctx, const uint8_t* const v) {
    return mutils::context_ptr<const Blob>{new Blob(const_cast<uint8_t*>(v) + sizeof(std::size_t), ((std::size_t*)(v))[0], true)};
}

std::unique_ptr<Blob> Blob::from_bytes(mutils::DeserializationManager*, const uint8_t* const v) {
    return std::make_unique<Blob>(v + sizeof(std::size_t), ((std::size_t*)(v))[0]);
}

}
