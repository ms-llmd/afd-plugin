// SPDX-License-Identifier: MIT
// Copyright contributors to the AFD plugin project
//
// Adapted from cam_async/src/comm_operator/pybind/gmm_layered.cpp.
// AFD adapter: shared allocation and validation between NPU and Meta, template
// argument selects execution only. The source's autograd Function subclass is
// replaced by registering the same adapter for AutogradPrivateUse1, matching the
// other AFD Ascend bindings; these are inference-only operators.

#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#include "pytorch_extension/op_api_common.h"
#include "grouped_matmul_layered/op_api/aclnn_grouped_matmul_layered.h"

namespace afd_plugin::grouped_matmul_layered {
namespace {

constexpr int64_t SPLIT_ITEM_MIN = 0;
constexpr int64_t SPLIT_ITEM_MAX = 3;
constexpr int64_t B4_PER_B32 = 8;  // eight int4 nibbles per int32 word

// Source: gmm_layered.cpp::DeriveOutputDtype.
// AFD adaptation: none; kept verbatim so the layered operator derives y exactly
// as the CAM build does. An explicit output_dtype wins; otherwise A8W4
// antiquant (int8 x with int32-packed int4 weight) yields bfloat16 while the
// quantized A8W8 family keeps the activation dtype. Never inherit from the
// packed weight tensor, whose dtype describes the storage, not the result.
at::ScalarType derive_output_dtype(
    const at::TensorList &x, const at::TensorList &all_weight,
    const c10::optional<at::ScalarType> &output_dtype) {
  if (output_dtype.has_value()) {
    return *output_dtype;
  }
  if (x[0].scalar_type() == at::kChar && all_weight[0].scalar_type() == at::kInt) {
    return at::kBFloat16;
  }
  return x[0].scalar_type();
}

// Source: gmm_layered.cpp::IsWeightTransposed.
// AFD adaptation: none. Detects the native is_weight_trans layout by the
// last-two-dimension stride swap; a transposed int32-packed int4 weight carries
// its nibbles along a different axis, which this adapter does not model, so it
// is rejected rather than used to compute a wrong N.
bool is_weight_transposed(const at::Tensor &tensor) {
  if (tensor.dim() < 2) {
    return false;
  }
  const int64_t dim1 = tensor.dim() - 1;
  const int64_t dim2 = tensor.dim() - 2;
  return tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2);
}

// Source: gmm_layered.cpp::AllocGmmOutputs.
// AFD adaptation: TORCH_CHECK wording only. The layered output model gives one
// [rows, n] tensor whose rows are the per-group rows concatenated; the kernel
// writes each group at its own offset inside that single buffer. A fresh
// at::empty storage is format-neutral, so the weight's FRACTAL_NZ tag cannot
// leak into y.
tensor_list alloc_outputs(const at::TensorList &x, const at::TensorList &all_weight,
                          const std::vector<int64_t> &group_rows,
                          const c10::optional<at::ScalarType> &output_dtype) {
  // torch_npu expresses int4 weight as int32, eight nibbles per word along the
  // last axis, so the logical N is the unpacked int4 extent.
  int64_t n = all_weight[0].size(-1);
  if (all_weight[0].scalar_type() == at::kInt) {
    TORCH_CHECK(!is_weight_transposed(all_weight[0]),
                "all_weight: transposed int32-packed int4 weight is not supported");
    n *= B4_PER_B32;
  }
  const auto out_options =
      all_weight[0].options().dtype(derive_output_dtype(x, all_weight, output_dtype));
  int64_t rows = 0;
  if (x.size() == 1) {
    rows = x[0].size(0);
  } else {
    // Per-group x: the row count is the sum of the group list.
    TORCH_CHECK(!group_rows.empty(),
                "group_list must be non-empty when x is a per-group list");
    for (const auto r : group_rows) {
      rows += r;
    }
  }
  return {at::empty({rows, n}, out_options)};
}

// Source: gmm_layered.cpp::PrepareGroupList.
// AFD adaptation: TORCH_CHECK wording only. group_list follows the V5 contract
// (a device int64 tensor). Every compiled A8W4 variant is count-style, so a
// cumsum-style list is normalized to counts on the host; the same counts drive
// the y row allocation. The H2D upload rides the current stream and nothing
// synchronizes.
void prepare_group_list(const c10::optional<std::vector<int64_t>> &group_list_optional,
                        int64_t &group_list_type, const at::Device &device,
                        at::Tensor &group_list_tensor, std::vector<int64_t> &group_rows) {
  if (!group_list_optional.has_value() || group_list_optional->empty()) {
    return;
  }
  const auto &group_list = *group_list_optional;
  if (group_list_type == 0) {  // cumsum -> count
    std::vector<int64_t> counts;
    counts.reserve(group_list.size());
    int64_t previous = 0;
    for (const auto value : group_list) {
      counts.push_back(value - previous);
      previous = value;
    }
    group_list_tensor = at::from_blob(const_cast<int64_t *>(counts.data()),
                                      {static_cast<int64_t>(counts.size())},
                                      at::TensorOptions().dtype(at::kLong))
                            .clone()
                            .to(device);
    group_rows = std::move(counts);
  } else {
    group_list_tensor = at::from_blob(const_cast<int64_t *>(group_list.data()),
                                      {static_cast<int64_t>(group_list.size())},
                                      at::TensorOptions().dtype(at::kLong))
                            .clone()
                            .to(device);
    group_rows = group_list;
  }
  group_list_type = 1;  // the normalized form sent downstream
}

// Source: gmm_layered.cpp::cam_gmm_layered_impl_npu.
// AFD adaptation: the device guard is kept (it is what makes the kernel launch
// on the device the tensors live on rather than the process default), the
// symbols carry the AFD prefix, and argument validation is stated explicitly.
void check_lists(const at::TensorList &x, const at::TensorList &all_weight,
                 const at::TensorList &all_bias, const at::TensorList &all_scale,
                 const at::Tensor &layer_index) {
  TORCH_CHECK(!x.empty() && !all_weight.empty() && !all_bias.empty() && !all_scale.empty(),
              "x, all_weight, all_bias and all_scale must be non-empty lists");
  // The all_* lists each hold one element per layer, so their lengths must agree
  // and every element of a list must describe the same layer geometry.
  const size_t layer_count = all_weight.size();
  TORCH_CHECK(all_bias.size() == layer_count && all_scale.size() == layer_count,
              "all_weight, all_bias and all_scale must have the same length; got ",
              all_weight.size(), ", ", all_bias.size(), ", ", all_scale.size());
  TORCH_CHECK(layer_index.defined() && layer_index.scalar_type() == at::kLong &&
                  layer_index.numel() == 1,
              "layer_index must be a one-element int64 tensor");
  for (size_t i = 0; i < layer_count; ++i) {
    TORCH_CHECK(all_weight[i].sizes() == all_weight[0].sizes(),
                "all_weight[", i, "] must match all_weight[0]");
  }
}

template <bool EXECUTE_NPU>
tensor_list grouped_matmul_layered(
    const at::TensorList &x, const at::TensorList &all_weight,
    const at::TensorList &all_bias, const at::TensorList &all_scale,
    const at::Tensor &layer_index,
    const c10::optional<at::Tensor> &per_token_scale_optional,
    const c10::optional<std::vector<int64_t>> &group_list_optional,
    const int64_t group_list_type, const int64_t split_item,
    const c10::optional<at::ScalarType> &output_dtype) {
  TORCH_CHECK(split_item >= SPLIT_ITEM_MIN && split_item <= SPLIT_ITEM_MAX,
              "split_item must be one of 0/1/2/3, got ", split_item);
  TORCH_CHECK(group_list_type == 0 || group_list_type == 1,
              "group_list_type must be 0 (cumsum) or 1 (count), got ", group_list_type);
  check_lists(x, all_weight, all_bias, all_scale, layer_index);

  // Pin the NPU device to the one the tensors live on, before the first device
  // operation below (the group list upload and the output allocation both touch
  // the device). Without this the current device stays at the process default
  // even when the inputs are elsewhere, and EXEC_NPU_CMD derives its stream from
  // the current device - so the kernel would launch in one device's context
  // while holding another's addresses, faulting as an MTE DDR address error.
  const c10::OptionalDeviceGuard device_guard(at::device_of(x[0]));

  at::Tensor group_list_tensor;
  std::vector<int64_t> group_rows;
  int64_t group_list_type_norm = group_list_type;
  prepare_group_list(group_list_optional, group_list_type_norm, x[0].device(),
                     group_list_tensor, group_rows);
  auto outputs = alloc_outputs(x, all_weight, group_rows, output_dtype);

  if constexpr (EXECUTE_NPU) {
    // per_token_scale follows the upstream TensorList contract; when omitted,
    // synthesize a shape-[0] tensor as the op_api empty normalization would.
    at::Tensor per_token_scale_tensor =
        per_token_scale_optional.has_value()
            ? *per_token_scale_optional
            : at::empty({0}, x[0].options().dtype(at::kFloat));
    at::TensorList per_token_scale_list(&per_token_scale_tensor, 1);
    at::TensorList output_list(outputs);

    // EXEC_NPU_CMD's parameter packing needs lvalues.
    EXEC_NPU_CMD(aclnnGroupedMatmulLayered,
                 x, all_weight, all_bias, all_scale,
                 per_token_scale_list, layer_index, group_list_tensor,
                 split_item, group_list_type_norm,
                 output_list);
  }
  return outputs;
}

}  // namespace
}  // namespace afd_plugin::grouped_matmul_layered

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
  ops.def(
      "grouped_matmul_layered(Tensor[] x, Tensor[] all_weight, "
      "Tensor[] all_bias, Tensor[] all_scale, Tensor layer_index, "
      "Tensor? per_token_scale=None, int[]? group_list=None, "
      "int group_list_type=0, int split_item=0, ScalarType? output_dtype=None) "
      "-> Tensor[]");
}

TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, ops) {
  ops.impl("grouped_matmul_layered",
           &afd_plugin::grouped_matmul_layered::grouped_matmul_layered<true>);
}

// Explicitly enforce the inference-only contract, including when grad mode is on.
TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, ops) {
  ops.impl("grouped_matmul_layered",
           &afd_plugin::grouped_matmul_layered::grouped_matmul_layered<true>);
}

TORCH_LIBRARY_IMPL(afd_ascend, Meta, ops) {
  ops.impl("grouped_matmul_layered",
           &afd_plugin::grouped_matmul_layered::grouped_matmul_layered<false>);
}
