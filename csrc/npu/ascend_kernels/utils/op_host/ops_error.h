/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: dfx check outer interface (lightweight; CANN 8.5 has no err_msg/dfx headers)
 * Create: 2026-08-28
 * Note:
 * History: 2026-08-28 create check interface file
 */
#ifndef OPS_BUILT_IN_OP_TILING_OPS_ERROR_H_
#define OPS_BUILT_IN_OP_TILING_OPS_ERROR_H_

#include "ops_log.h"

// Conditional check: if (COND) { LOG_FUNC; EXPR; }
// LOG_FUNC is caller-supplied (e.g. OPS_LOG_E); EXPR is the failure action
// (e.g. return ge::GRAPH_FAILED). No external dependency.
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) \
    do {                                  \
        if (COND) {                       \
            LOG_FUNC;                     \
            EXPR;                         \
        }                                 \
    } while (0)

#define OPS_CHECK(COND, LOG_FUNC, EXPR) OPS_ERR_IF(COND, LOG_FUNC, EXPR)

#endif  // OPS_BUILT_IN_OP_TILING_OPS_ERROR_H_
