#include "internal_common.hpp"
#include <derecho/utils/logger.hpp>

/**
 * @file internal_common.cpp
 * @brief internal utilities shared by implementation components which should be hidden from DCCL users.
 */

namespace dccl {

std::shared_ptr<spdlog::logger>& getDcclLogger() {
    static std::shared_ptr<spdlog::logger> _logger;
#define DCCL_LOGGER_UNINITIALIZED   0
#define DCCL_LOGGER_INITIALIZING    1
#define DCCL_LOGGER_INITIALIZED     2
    static std::atomic<uint32_t> _logger_state(DCCL_LOGGER_UNINITIALIZED);
    uint32_t expected = DCCL_LOGGER_UNINITIALIZED;
    if (_logger_state.compare_exchange_strong(
                expected,
                DCCL_LOGGER_INITIALIZING,
                std::memory_order_acq_rel)) {
        _logger = LoggerFactory::createLogger("DCCL");
        _logger_state.store(DCCL_LOGGER_INITIALIZED,std::memory_order_acq_rel);
    }
    // make sure initialization finished by concurrent callers
    while(_logger_state.load(std::memory_order_acquire)!=DCCL_LOGGER_INITIALIZED){
    }
    return _logger;
}

} // namespace dccl
