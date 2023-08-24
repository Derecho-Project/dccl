#include "internal_common.hpp"
#include <derecho/utils/logger.hpp>
#include <derecho/utils/time.h>

/**
 * @file internal_common.cpp
 * @brief internal utilities shared by implementation components which should be hidden from DCCL users.
 */

namespace dccl {

dcclComm::dcclComm():
    derecho_group_handle(nullptr),
    derecho_group_object(nullptr),
    bcast_id_seed(get_time()) {
}

dcclComm::~dcclComm() {
}

uint64_t dcclComm::post_bcast_recv_buff(void* recvbuff, size_t len) {
    std::unique_lock<std::mutex> state_lock(this->delivery_state_mutex);
    // update seed
    // The following parameters are from MIMIX(Donald Knuth)
    // a = 6364136223846793005
    // c = 1442695040888963407
    // m = 2^64
    do {
        bcast_id_seed = bcast_id_seed * 6364136223846793005 + 1442695040888963407;
    } while (delivery_state.find(bcast_id_seed)!=delivery_state.cend());
    delivery_state.emplace(bcast_id_seed,bcast_delivery_state_t::undelivered);
    state_lock.unlock();

    std::unique_lock<std::mutex> queue_lock(this->broadcast_queue_mutex);
    broadcast_queue.emplace(bcast_id_seed,recvbuff,len);
    queue_lock.unlock();
    broadcast_queue_cv.notify_all();

    return bcast_id_seed;
}

void dcclComm::on_bcast(const std::function<bool(void*,const size_t&)>& data_generator) {
    std::unique_lock<std::mutex> queue_lock(this->broadcast_queue_mutex);
    // pick bcast_id
    broadcast_queue_cv.wait(queue_lock, [this]{return !broadcast_queue.empty();});
    uint64_t bcast_id = std::get<0>(broadcast_queue.front());
    // assert(delivery_state.find(bcast_id) != delivery_state.cend());
    void*    recvbuff = std::get<1>(broadcast_queue.front());
    size_t   buffsize = std::get<2>(broadcast_queue.front());
    broadcast_queue.pop();
    queue_lock.unlock();
    // process data
    bcast_delivery_state_t target_state = (data_generator(recvbuff,buffsize)) ? bcast_delivery_state_t::delivered : bcast_delivery_state_t::failed;
    // update state
    std::unique_lock<std::mutex> state_lock(this->delivery_state_mutex);
    delivery_state[bcast_id] = target_state;
    state_lock.unlock();
    delivery_state_cv.notify_all();
}

dcclComm::bcast_delivery_state_t dcclComm::query_bcast(const uint64_t& bcast_id) {
    std::lock_guard<std::mutex> state_lock(this->delivery_state_mutex);
    if (delivery_state.find(bcast_id) == delivery_state.cend()) {
        return bcast_delivery_state_t::nonexist;
    } else {
        return delivery_state.at(bcast_id);
    }
}

dcclComm::bcast_delivery_state_t dcclComm::wait_bcast(const uint64_t& bcast_id) {
    std::unique_lock<std::mutex> state_lock(this->delivery_state_mutex);
    delivery_state_cv.wait(state_lock,[this,bcast_id]{
            return  delivery_state.find(bcast_id)==delivery_state.cend() || 
                    delivery_state.at(bcast_id) != undelivered;});
    if (delivery_state.find(bcast_id) == delivery_state.cend()) {
        return bcast_delivery_state_t::nonexist;
    } else {
        return delivery_state.at(bcast_id);
    }
}

dcclComm::bcast_delivery_state_t dcclComm::clear_bcast(const uint64_t& bcast_id) {
    std::unique_lock<std::mutex> state_lock(this->delivery_state_mutex);
    bcast_delivery_state_t ret = nonexist;
    if (delivery_state.find(bcast_id) != delivery_state.cend()) {
        bcast_delivery_state_t ret = delivery_state.at(bcast_id);
        if (ret != undelivered) {
            delivery_state.erase(bcast_id);
        }
    }
    return ret;
}

std::shared_ptr<spdlog::logger>& getDcclLogger() {
    static std::shared_ptr<spdlog::logger> _logger;
/**
 * @cond Doxygen_Suppressed
 */
#define DCCL_LOGGER_UNINITIALIZED   0
#define DCCL_LOGGER_INITIALIZING    1
#define DCCL_LOGGER_INITIALIZED     2
/**
 * @endcond
 */
    static std::atomic<uint32_t> _logger_state(DCCL_LOGGER_UNINITIALIZED);
    uint32_t expected = DCCL_LOGGER_UNINITIALIZED;
    if (_logger_state.compare_exchange_strong(
                expected,
                DCCL_LOGGER_INITIALIZING,
                std::memory_order_acq_rel)) {
        _logger = LoggerFactory::createLogger("DCCL",derecho::getConfString(CONF_LOGGER_DEFAULT_LOG_LEVEL));
        _logger_state.store(DCCL_LOGGER_INITIALIZED,std::memory_order_acq_rel);
    }
    // make sure initialization finished by concurrent callers
    while(_logger_state.load(std::memory_order_acquire)!=DCCL_LOGGER_INITIALIZED){
    }
    return _logger;
}

} // namespace dccl
