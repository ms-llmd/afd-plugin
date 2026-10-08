/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncDispatchSend kernel function definition file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncDispatchSend operator definition file
 */
#include "kernel_operator.h"
#include "afd_async_dispatch_send.h"
#include "afd_async_dispatch_send_tiling.h"

using namespace AscendC;
using namespace MoeDistributeDispatchImpl;
using namespace Cam;

extern "C" __global__ __aicore__ void afd_async_dispatch_send(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR commArgs, GM_ADDR expandXOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AfdAsyncDispatchSendTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000000000", AfdAsyncDispatchSendTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncDispatchSendTilingData, tilingData, tilingGM);

    int dynamicQuant = tilingData.moeDistributeDispatchInfo.dynamicQuant;

    if (TILING_KEY_IS(100)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchSend<bfloat16_t, bfloat16_t, false> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchSend<bfloat16_t, int8_t, true> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    } else if (TILING_KEY_IS(101)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchSend<float16_t, float16_t, false> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchSend<float16_t, int8_t, true> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    }
}
