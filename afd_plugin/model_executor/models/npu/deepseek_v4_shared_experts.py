# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local shared-expert computation for DSV4 Async CAM."""

import torch
import torch_npu
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm_ascend.models.deepseek_v4 import DeepseekV2MLP
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod,
)


class AFDDeepseekV4SharedExperts(DeepseekV2MLP):
    """Preserve Ascend's fused W8A8 path outside its routed-expert runner.

    Async CAM owns communication and invokes this replicated, sequence-parallel
    MLP directly, bypassing FusedMoE._forward_shared_experts. Keep the native
    constructor and weight loading, and mirror that runner's INT8 arithmetic
    without its routed-expert events or TP collectives. This adapter can be
    removed when upstream exposes standalone fused shared-expert execution.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for projection in (self.gate_up_proj, self.down_proj):
            method = projection.quant_method
            if not isinstance(method, AscendLinearMethod):
                return super().forward(x)
            scheme = method.quant_method
            if (
                not isinstance(scheme, AscendW8A8DynamicLinearMethod)
                or scheme.act_quant_type != torch.int8
            ):
                return super().forward(x)

        # Match the actual activation, including nondefault sigmoid/up terms.
        clamp_limit, glu_alpha, glu_bias = 0.0, 1.0, 0.0
        if isinstance(self.act_fn, SiluAndMulWithClamp):
            clamp_limit = self.act_fn.swiglu_limit
            glu_alpha = self.act_fn.alpha
            glu_bias = self.act_fn.beta

        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        gate_up = torch_npu.npu_quant_matmul(
            quantized_x,
            self.gate_up_proj.weight,
            self.gate_up_proj.weight_scale,
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int32,
        )
        quantized_activation, activation_scale = (
            torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=gate_up,
                weight_scale=self.gate_up_proj.weight_scale_fp32,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=None,
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=clamp_limit,
                glu_alpha=glu_alpha,
                glu_bias=glu_bias,
            )
        )
        return torch_npu.npu_quant_matmul(
            quantized_activation,
            self.down_proj.weight,
            self.down_proj.weight_scale,
            pertoken_scale=activation_scale,
            bias=None,
            output_dtype=x.dtype,
        )
