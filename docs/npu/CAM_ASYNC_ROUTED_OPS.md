# Routed-only CAM asynchronous operators

The AFD native package contains four experimental asynchronous communication
operators for Ascend 910C (`ascend910_93`). They use the
`cam_async_routed_only_compact_v2` protocol: only routed experts participate
in dispatch and combine; shared-expert payloads and reserved expert slots are
absent.

`CAMAsyncAFDConnector` uses these four source-built operators and their compact
metadata. The package includes the native sources, build registration,
PyTorch bindings and standalone numerical tests.

## Operator identities and ownership

| Phase | PyTorch operator under `torch.ops.afd_ascend` | CANN operator |
| --- | --- | --- |
| Dispatch send (Attention) | `afd_async_dispatch_send` | `AfdAsyncDispatchSend` |
| Dispatch receive (FFN) | `afd_async_dispatch_recv` | `AfdAsyncDispatchRecv` |
| Combine send (FFN) | `afd_async_combine_send` | `AfdAsyncCombineSend` |
| Combine receive (Attention) | `afd_async_combine_recv` | `AfdAsyncCombineRecv` |

Sources live under `csrc/npu/ascend_kernels/afd_async_*/`.
Each operator carries its host, kernel, and ACLNN wrapper. The existing
`msopgen` / `npu_op_*` build collects them through
`operator_registry.json`; no separate CAM wheel or build framework is added.

The extension remains `afd_plugin._C_ascend`, and the CANN vendor remains
`afd-plugin`. CANN registrations, ACLNN public/inner symbols, and kernel
entry points also carry the AFD prefix. PyTorch namespace isolation alone
would not isolate CANN operator types from an older external CAM package.
The AFD prefix prevents registration-name collisions; it does not make the
old and new communication protocols interoperable.

The operators are adapted from the MIT-licensed `cam_async-routed-only-candidate`
snapshot, under `src/comm_operator/ascend_kernels/`. The integration renames
operator symbols and shared headers for AFD isolation while preserving the
compact routed-only communication layout.
The complete upstream MIT copyright and permission notice is retained in
[the repository license](../../LICENSE) and included in package license data.
The candidate's separate build system,
generated binaries, placeholder backward functions, and unrelated components
are not imported.

## Build and loading

Use a matching Ascend development environment and the normal package build:

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 \
  pip install -e . -v --no-build-isolation
```

For 910C, the registry selects A2E/E2A and all four routed-only operators.
For 950, only A2E/E2A are selected, and the new PyTorch registrations are not
compiled. `setup.py` passes the same `SOC_VERSION` to the ACLNN build and
the extension CMake configuration (`AFD_SOC_VERSION`).

`AFD_SKIP_ACLNN_BUILD=1` still requires an existing, matching AFD vendor
package. An older package containing only A2E/E2A does not provide the new
ACLNN symbols. Rebuild the complete operator package after changing this
source or the SOC. The candidate lists CANN 8.5 / 9.0 and matching
PyTorch/torch-npu 2.8 as its build environment; the integrated package still
requires qualification on the actual deployment stack.

Load the package using the existing lazy loader before accessing the new
operators:

```python
import torch
import torch_npu

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
dispatch_send = torch.ops.afd_ascend.afd_async_dispatch_send
dispatch_recv = torch.ops.afd_ascend.afd_async_dispatch_recv
combine_send = torch.ops.afd_ascend.afd_async_combine_send
combine_recv = torch.ops.afd_ascend.afd_async_combine_recv
```

The loader's existing A2E/E2A check is unchanged; loading it is not proof that
an old installed extension contains the new operators. The new binding tests
check the four registrations explicitly. Importing the Python plugin remains
safe without the native extension.

## Compact protocol contract

Let `B` be the Attention batch size, `H` the hidden dimension, `K` top-k,
`M` the MoE rank count, `A` the Attention rank count, `R` the routed
expert count per MoE rank, and `T` the Attention TP size.

- Global routed expert IDs are in `[0, M * R)`; each MoE rank has exactly
  `R` zero-based expert slots. There is no leading shared-expert slot.
- Attention ranks occupy `[0, A)`, and MoE ranks occupy `[A, A + M)`.
  `world_size = A + M`, and `T` divides `A`.
- `B >= 1`, `1 <= K <= M * R`, and
  `B <= max_seq_len / T`. Inputs must be contiguous, on the same device,
  and not require gradients. The binding rejects invalid shapes and scalar
  configuration before executing an NPU command.
- `expert_ids` is `int32[B, K]`; combine weights are
  `float32[B, K]`. Routing values themselves must satisfy the protocol;
  shape checks do not validate every device-side expert ID.
- `comm_id` is a retained reserved argument and is not used to select
  communication resources. `comm_args` remains a float16 tensor placeholder;
  the kernels obtain communication windows through the HCCL context selected
  by `group_name`.
- All participating ranks must use the same protocol, topology, quantization,
  dimensions, window configuration, and lifetime rules.

| Operator | Principal inputs | Return contract |
| --- | --- | --- |
| Dispatch send | `x[B,H]` in FP16/BF16, expert IDs, scalar topology/configuration | `int8[1]` placeholder |
| Dispatch receive | FP16/BF16 one-element anchor, topology/configuration | Four tensors: routed payload, dynamic scales, batch metadata, routed expert counts |
| Combine send | Routed expert results in FP16/BF16, compact batch metadata | `int8[1]` placeholder; no shared-expert result argument |
| Combine receive | FP16/BF16 one-element anchor, expert IDs, routing weights | Weighted routed output `[B,H]` |

The send placeholders are not received payloads and should not be passed as
the floating-point anchors of receive operations. They also do not prove that
another rank has consumed the transfer.

Dispatch receive allocates `N = floor(262144 * BATCH_SIZE_FACTOR)` rows.
The environment variable defaults to `1.0`; its parsed value must be in
`(0,1]` and produce at least one row. Its float parsing and malformed-value
fallback follow the source implementation. Keep it fixed across the entire
run and consistent with host tiling.

| Dispatch receive output | Shape and dtype |
| --- | --- |
| Routed payload | `[N,H]`, FP16/BF16 when `dynamic_quant=0`, int8 when `dynamic_quant=1` |
| Dynamic scales | float32 `[1]` without quantization, float32 `[N]` with quantization |
| Batch metadata | int64 `[5 + T + R*T]`: five header fields, `T` prefixes, then routed counts |
| Routed expert counts | int64 `[R]` |

Only the valid rows described by the returned metadata are meaningful.
Expert interval indices are zero-based. Empty expert intervals and empty MoE
participants must retain the completion-notification behavior; callers cannot
drop the corresponding combine-send step.

Some source counters and token addresses use uint16. Values must remain
representable, and each expert's TP-aggregated token count must fit in one
receive chunk. The kernels do not provide automatic recovery when an expert
cannot fit. Dispatch statistics must remain intact until the matching
combine-receive completes; a later dispatch must not overwrite them early.

The binding is inference-only. It provides PrivateUse1 and Meta implementations
with the same allocation logic, and rejects gradient-bearing inputs instead
of importing the candidate's placeholder autograd backward. Meta execution
checks registration, shapes, and allocation contracts; it performs no
communication and proves no device correctness or graph support.

## Validation

CPU checks cover SOC selection and rejection, source packaging, staging
collisions, and propagation of the SOC into the extension build:

```bash
python -m pytest -q \
  tests/unit/package/test_ascend_build_files.py \
  tests/unit/package/test_cam_async_build.py \
  tests/unit/compat/test_ascend_ops.py
```

Run the dedicated Meta binding tests with a built 910C extension and its
matching torch-npu/CANN libraries:

```bash
SOC_VERSION=910c python -m pytest -q -ra \
  tests/unit/compat/npu/test_cam_async_ops.py
```

Missing dependencies or an unspecified target SOC are reported as skips, not
NPU validation. An installed 910C extension missing a required operator fails
the tests.

Before claiming device support, record the exact AFD revision, CANN/HCCL,
torch-npu and PyTorch versions, device IDs, topology, commands, logs, and
cleanup results. Validate all four phases together, including:

- FP16/BF16, quantization off/on, and an independent routed-only weighted-sum
  reference.
- TP greater than one, multiple receive chunks, sparse routing, empty experts
  and empty MoE ranks, plus repeated use of the same communication windows.
- Inference rejection paths, invalid metadata, timeout/abort behavior, and
  completion flags under the intended caller lifecycle.
- Rebuilding and loading with existing A2E/E2A operators, with the 950 build
  continuing to exclude these 910C-only operators.

The standalone device check is
`tests/e2e/operators/async_cam_roundtrip.py`. Run it with `torchrun` using
TP1/2/4 plus two FFN ranks, both FP16/BF16, and quantization off/on. Each run
compares against an independent CPU weighted-sum/quantization reference and
covers sparse/empty ranks, multiple chunks and repeated window reuse. Model
E2E and per-layer reference checks remain separate acceptance gates.

## Routed-only connector and model contract

`CAMAsyncAFDConnector` requires these four source-built operators on every
rank. Its receive path preserves the full compact metadata for combine-send
and computes chunk length from expert counts. Shared activations and outputs
do not cross the A/F boundary.

DeepSeek-V2 and DeepSeek-V4 Attention roles construct the pinned native shared
MLP using replicated weights and local model tokens. They load shared weights,
scales and biases on Attention, then add the shared result after restoring the
routed output layout. FFN ranks construct routed experts only. Two-stage
execution keeps shared results separate by stage; interrupted model forwards
release pending routing references and require process-group teardown.

The synchronous CAMP2P and GPU paths retain their existing ownership contracts.
