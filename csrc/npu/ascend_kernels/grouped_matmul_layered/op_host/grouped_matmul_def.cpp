/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */

#include "register/op_def_registry.h"
namespace ops {

// cam_async layered migration: trimmed from the upstream 26 dtype columns to the 13 columns
// covering the A8W4 (x INT8 + weight INT4, cols for y BF16/FP16, weight ND/NZ) and A8W8
// (x INT8 + weight INT8, y INT8/BF16/FP16/INT32, weight ND/NZ) cases on ascend910_93.
// offset / antiquant_scale / antiquant_offset inputs are dropped: only used by asymmetric
// quant and A16 antiquant modes which are out of scope (symmetric quant only).
class GroupedMatmulLayered : public OpDef {
public:
    explicit GroupedMatmulLayered(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("all_weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                    ge::DT_INT4, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                    ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("all_bias")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                    ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("all_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                    ge::DT_UINT64, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // layered: current layer index resolved on device (shape [1]); the all_* tensor lists
        // hold one element per layer, each element covering all experts of that layer.
        this->Input("layer_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("per_token_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                    ge::DT_INT32, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("split_item").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(0); // Reserved parameter, indicate output data type, not enabled.
        this->Attr("transpose_weight")
            .AttrType(OPTIONAL)
            .Bool(false); // Reserved parameter, indicate whether input weight is transposed, not enabled.
        this->Attr("transpose_x")
            .AttrType(OPTIONAL)
            .Bool(false); // Reserved parameter, indicate whether input x is transposed, not enabled.
        this->Attr("group_type").AttrType(OPTIONAL).Int(-1); // Indicates the split dimension.
        this->Attr("group_list_type")
            .AttrType(OPTIONAL)
            .Int(0); // Indicates whether the value in group_dist is cumsum or count.
        this->Attr("act_type").AttrType(OPTIONAL).Int(0);            // Indicate activation function type.
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0}); // Indicate activation function type.

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(GroupedMatmulLayered);
} // namespace ops
