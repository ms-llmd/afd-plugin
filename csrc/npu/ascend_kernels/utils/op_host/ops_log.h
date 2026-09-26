/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: dfx log interface (lightweight printf-stub; CANN 8.5 has no alog/dfx headers)
 * Create: 2026-08-28
 * Note:
 * History: 2026-08-28 create log interface file
 */
#ifndef OPS_BUILT_IN_OP_TILING_OPS_LOG_H_
#define OPS_BUILT_IN_OP_TILING_OPS_LOG_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>  // getpid()

namespace optiling {

// ----- log macros -----
// OPS_LOG_D is a no-op by default (debug logging off in production). Define
// OPS_DEBUG_LOG_ON to enable printf output for debugging.
#ifdef OPS_DEBUG_LOG_ON
#define OPS_LOG_D(OPS_DESC, ...)                              \
    do {                                                      \
        printf("[DEBUG] " __VA_ARGS__);                       \
        printf("\n");                                         \
    } while (0)
#define OPS_LOG_I(OPS_DESC, ...)                              \
    do {                                                      \
        printf("[INFO] " __VA_ARGS__);                        \
        printf("\n");                                         \
    } while (0)
#define OPS_LOG_W(OPS_DESC, ...)                              \
    do {                                                      \
        printf("[WARN] " __VA_ARGS__);                        \
        printf("\n");                                         \
    } while (0)
#else
#define OPS_LOG_D(OPS_DESC, ...) ((void)0)
#define OPS_LOG_I(OPS_DESC, ...) ((void)0)
#define OPS_LOG_W(OPS_DESC, ...) ((void)0)
#endif

// OPS_LOG_E always reports to stderr (real error, not debug).
#define OPS_LOG_E(OPS_DESC, ...)                              \
    do {                                                      \
        fprintf(stderr, "[ERROR] " __VA_ARGS__);              \
        fprintf(stderr, "\n");                                \
    } while (0)
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_EVENT(OPS_DESC, ...) ((void)0)

// ----- helpers (migrated from error_log.h) -----
constexpr char LCCL_BUFFER_SIZE[] = "LCCL_BUFFER_SIZE";
constexpr char BATCH_SIZE_FACTOR[] = "BATCH_SIZE_FACTOR";
constexpr int DEFAULT_BUFFER_SIZE = 2 * (200 + 4);  // 408MB
constexpr int MAX_BUFFER_SIZE = 32 * 1024;          // 32GB
constexpr float DEFAULT_BATCH_SIZE_FACTOR = 1.0;

static inline uint64_t GetMaxWindowSize()
{
    int size = DEFAULT_BUFFER_SIZE;
    auto env = std::getenv(LCCL_BUFFER_SIZE);
    if (env != nullptr) {
        try {
            std::string envStr(env);
            size = std::stoi(envStr);
            if (size > MAX_BUFFER_SIZE) {
                fprintf(stderr, "LCCL_BUFFER_SIZE %d larger than MAX %d, clamped\n", size, MAX_BUFFER_SIZE);
                size = MAX_BUFFER_SIZE;
            }
        } catch (...) {
            fprintf(stderr, "Unknown exception parsing LCCL_BUFFER_SIZE\n");
        }
    }
    return static_cast<uint64_t>(size) * 1024UL * 1024UL;
}

static inline float GetBatchSizeFactor()
{
    float factor = DEFAULT_BATCH_SIZE_FACTOR;
    auto env = std::getenv(BATCH_SIZE_FACTOR);
    if (env == nullptr) {
        return factor;
    }
    try {
        factor = std::stof(std::string(env));
    } catch (...) {
        fprintf(stderr, "Unknown exception parsing BATCH_SIZE_FACTOR\n");
    }
    return factor;
}

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_OPS_LOG_H_
