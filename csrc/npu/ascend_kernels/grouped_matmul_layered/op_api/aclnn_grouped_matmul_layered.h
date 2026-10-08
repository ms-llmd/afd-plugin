/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software and is distributed under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for details.
 */

#ifndef ACLNN_GROUPED_MATMUL_LAYERED_API_H
#define ACLNN_GROUPED_MATMUL_LAYERED_API_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Activation type for the act_type attribute (moved here from the dropped V4 header). */
typedef enum {
    GMM_ACT_TYPE_NONE = 0L,
    GMM_ACT_TYPE_RELU = 1L,
    GMM_ACT_TYPE_GELU_TANH = 2L,
    GMM_ACT_TYPE_GELU_ERR_FUNC = 3L,
    GMM_ACT_TYPE_FAST_GELU = 4L,
    GMM_ACT_TYPE_SILU = 5L,
} GMMActType;

/*
 * cam_async layered migration of GroupedMatmul (ascend910_93, A8W4/A8W8 symmetric quant).
 * Trimmed interface: offset / antiquantScale / antiquantOffset and the V2-V5/WeightNz
 * variants are dropped; only the main entry is exposed.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulLayeredGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight,
    const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
    const aclTensorList* perTokenScaleOptional, const aclTensor* layerIndex,
    const aclTensor* groupListOptional, int64_t splitItem, int64_t groupListType,
    const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulLayered(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GROUPED_MATMUL_LAYERED_API_H
