#pragma once
#include <dccl/config.h>
#include <string>
/**
 * @file    utils.hpp
 * @brief   Common utilities for applications.
 */

/**
 * @brief Save host memory to file.
 *
 * @param[in]   ptr     The pointer data.
 * @param[in]   size    The data size.
 * @param[in]   fname   The file name.
 */
void save_mem(const void* ptr, size_t size, const std::string& fname);
