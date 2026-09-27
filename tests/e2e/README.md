# End-to-End Tests

These tests validate DeepSeek-V2-Lite on real GPU or Ascend NPU hardware,
Qwen3 MoE on real GPU hardware, and Qwen3.6 MoE through the Qwen3.5/3.6
adapter family on real CUDA hardware.
Each default gate runs four scenarios:

- `baseline-graph`
- `afd-eager-2a2f` (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use `afd-eager-2a1f`)
- `afd-graph-2a2f` (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use `afd-graph-2a1f`)
- `afd-graph-dbo-2a2f` (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use `afd-graph-dbo-2a1f`)

Each scenario evaluates the first 7 GSM8K samples; the DBO scenarios run 24
samples with 12 concurrent requests so that live requests actually execute
as two ubatches (see the accuracy gate in
`docs/design/module/e2e_testing.md`). If `AFD_E2E_DEVICES` is set,
that value is used as-is; otherwise the defaults are:

- `0,1,2,3` for the gate scenarios. The 2A2F AFD cases (DeepSeek-V2-Lite gate
  only) use the first two for Attention DP2/TP1 and the last two for FFN
  DP2/TP1/EP2; `baseline-graph` uses all four for DP4/TP1/EP4.
- The 2A1F cases use the first two for Attention and the third for FFN,
  leaving the fourth device idle. For DeepSeek-V2-Lite these are local-only
  cases; for Qwen3 MoE and Qwen3.6 MoE they are the suite's gate scenarios.

Tests run sequentially and must not skip. Every GSM8K evaluation uses 8
few-shot examples and a 4096-token maximum model length.

See the [E2E testing design](../../docs/design/module/e2e_testing.md) before
adding a model or case.

## Run

Run from the repository root. The environment needs `vllm`, `pytest`,
`afd_plugin`, `lm_eval`, `datasets`, and `huggingface_hub` (install
by running `uv sync --group dev --group e2e-tests`). NPU also needs
`torch_npu`.

The selected test downloads/caches `openai/gsm8k` and its Hugging Face model
when the backend model env var is unset. Point `HF_HOME` at a persistent cache
if you want to reuse downloads across runs.

GPU:

```bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=gpu
# Optional: export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_GPU_E2E_MODEL=/path/to/model
```

NPU:

```bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=npu
# Optional: export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
```

Then run the selected model suite:

```bash
# DeepSeek-V2-Lite gate scenarios
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a2f]"

# Qwen3 MoE
python -m pytest -q -s \
  tests/e2e/models/qwen3_moe/test_qwen3_moe.py

# Qwen3.6 MoE (text-only CUDA lane)
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_GPU_E2E_MODEL=/path/to/Qwen3.6-35B-A3B
python -m pytest -q -s \
  tests/e2e/models/qwen3_6/test_qwen3_6.py
```

Success means 4 passed and 0 skipped.

The repository CUDA E2E evidence for the Qwen3.5/3.6 adapter family uses
`Qwen/Qwen3.6-35B-A3B`, vLLM V1, and the
`P2pNcclAFDConnector` on CUDA. Every scenario passes `--language-model-only`;
multimodal execution is not covered. `baseline-graph` uses native DP4/TP1/EP4
on four devices. The three AFD scenarios use synchronous 2A1F: Attention on
the first two devices and FFN on the third. The fourth device remains unused
by AFD scenarios. The suite uses the same GSM8K-7, eight-shot, 4096-token,
0.27 minimum exact-match gate as the other default suites. NPU, multimodal,
`compute_gate_on_attention=true`, pipeline-parallel, asynchronous, and
multi-node execution are not covered; quantization is unverified.

### DeepSeek-V2-Lite local 2A1F cases

The DeepSeek-V2-Lite 2A1F scenarios are local-only; its CI gate selects the
2A2F scenarios. They use the first two devices for Attention DP2/TP1 and the
third for FFN DP1/TP1/EP1, and run GSM8K-7.

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a1f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a1f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a1f]"
```

### GPU ModelRunnerV2 evidence matrix

The GPU-only ModelRunnerV2 regression matrix contains six representative
scenarios:

- `afd-v2-eager-1a1f` and `afd-v2-graph-1a1f`
- `afd-v2-eager-dp2` and `afd-v2-graph-dp2`
- `afd-v2-eager-tp2` and `afd-v2-graph-tp2`

The 1A1F scenarios use two devices and are local-only. DP2 and TP2 use four
devices, split evenly between Attention and FFN, and run in the CI gate on
`l4_4`. These rows record hardware-tested coverage; they are not a production
topology allowlist. Other valid DP/TP topologies use the same AFD and native
vLLM topology contracts.

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py" \
  -k 'afd-v2'
```

### GLM-5.2 (`glm_moe_dsa`) — multi-pod only

The GLM-5.2 suite lives at
`tests/e2e/models/glm_moe_dsa/test_glm_moe_dsa.py` and follows the
DeepSeek-V2-Lite multi-pod shape: AFD eager, graph, and graph+DBO scenarios on
GSM8K-7, each run as one pod's slice of an already-provisioned deployment. It
cannot run on a single node.

GLM-5.2 is a ~753B checkpoint. With `compute_gate_on_attention=false` the FFN
role owns 730.3B of the parameters — the routed experts alone are 724.8B, or
98% of the model — while the Attention role owns 15.5B. The FFN rank count is
therefore set by expert memory, not by preference. On H200 (~140 GiB, of
which vLLM uses 0.9):

| dtype | FFN weights per rank | minimum FFN ranks | minimum topology | H200s |
| --- | --- | --- | --- | --- |
| FP8 (`zai-org/GLM-5.2-FP8`) | 730.3 GB / y | 8 (~85 GiB) | `8A8F` | 16 |
| BF16 | 1460.6 GB / y | 16 (~85 GiB) | `16A16F` | 32 |

Four FFN ranks need ~170 GiB each in FP8 and do not fit.
`n_routed_experts=256` also requires the FFN rank count to divide 256, and
`P2pNcclAFDConnector` requires `num_attention_ranks >= num_ffn_ranks`
(`afd_plugin/distributed/topology.py`), which pins the Attention side at 8
even though it only holds 15.5 GB. The suite targets FP8 `8A8F`.

With at most four ranks per pod, `8A8F` spans four pods. Two layouts are
exercised:

| Case | Pod layout |
| --- | --- |
| `afd-graph-8a8f-4pod-role-split` | `4A0F,4A0F,0A4F,0A4F` |
| `afd-graph-8a8f-4pod-interleaved` | `2A2F,2A2F,2A2F,2A2F` |
| `afd-eager-8a8f-4pod-role-split` | `4A0F,4A0F,0A4F,0A4F` |
| `afd-graph-dbo-8a8f-4pod-role-split` | `4A0F,4A0F,0A4F,0A4F` |

There is no native baseline: serving GLM-5.2 without AFD needs at least
eight devices in one pod, and the multi-pod runner refuses to split a
baseline. Pass/fail is the absolute GSM8K threshold, as for every suite.

Bring the pods up per `.agents/skills/run-e2e/resources/k8-multi-pod.md`
with four GPUs each and a ReadWriteMany PVC holding the FP8 snapshot, then run
the same pytest node inside every pod:

```bash
export AFD_E2E_RUN_ID=... AFD_E2E_STORE_HOST=<job>-0.<job>
export AFD_GPU_E2E_MODEL=/models/.hf_home/hub/models--zai-org--GLM-5.2-FP8/snapshots/<rev>
export AFD_E2E_GSM8K_OUTPUT=/work/gsm8k
python -m pytest -s \
  'tests/e2e/models/glm_moe_dsa/test_glm_moe_dsa.py::test_glm_moe_dsa[afd-graph-8a8f-4pod-role-split]'
```

Every rank scans the whole ~761 GB checkpoint, which took about 65 minutes
from a shared VAST PVC, so the suite raises `--serving-timeout` to 7200 s.
The roles also finish loading minutes apart, so it passes the same value as
`--afd-process-group-timeout-s`; at the connector's 120 s default the first
role to load gives up on the AFD world join before the other arrives.
The command contract and the per-pod placement of both layouts are covered
by unit tests in `tests/unit/test_e2e_runner.py`, which run everywhere.

### Weekly GSM8K

The weekly pipeline runs the Qwen3 MoE and Qwen3.6 MoE suites (baseline,
eager, and graph scenarios; the DBO scenario is excluded pending the FFN
CUDA fault investigation from build 68) plus the DeepSeek-V2-Lite
`afd-graph-dbo-2a1f` scenario, all with the default GSM8K sample limit. To
reproduce locally:

```bash
# Qwen3 MoE (all four scenarios)
python -m pytest -q -s tests/e2e/models/qwen3_moe/test_qwen3_moe.py

# Qwen3.6 MoE (all four scenarios)
python -m pytest -q -s tests/e2e/models/qwen3_6/test_qwen3_6.py
```

For a full 1319-sample run, export `AFD_GSM8K_LIMIT=all` before invoking
pytest. Without `AFD_GSM8K_LIMIT`, each scenario evaluates the first 7
samples.

## DSV4 Flash async CAM concurrent requests (local, 16 NPUs)

`afd-dsv4-flash-async-cam-dp2tp4-ep8` runs Attention DP2/TP4 on the first
eight devices and FFN DP8/TP1/EP8 on the last eight. This is a standalone
Ascend 910C case, outside the four-device PR gate. Use DSV4 Flash W8A8
weights and a DSV4-capable vLLM/vLLM-Ascend runtime with CAM operators.

The fixed deployment uses eager execution, MBT=8192, max-model-len=1048576,
max-num-seqs=16, block-size=128, memory utilization=0.7, and seed=1024.
Both roles explicitly disable `enable_dsv4_shared_compressor_workspace`.
CAM uses `dynamicQuant=1`, Attention-side gating, and two token-split async
MoE ubatches. FlashComm1 is enabled only on Attention. CPU binding and
128-thread weight loading follow the reference prefill scripts. Prefix
caching, native DBO, and KV transfer are not enabled.

After startup, an async HTTP client schedules ten independent chat requests together.
They ask for `12 + 7` through `21 + 7`, with temperature=0, thinking=false,
and max_tokens=256. Every response must contain one nonempty answer and
finish with `stop`; HTTP errors, truncated answers, missing responses, or
non-overlapping request timings fail the case. The test prints all outputs
and saves complete responses and monotonic timestamps in the pytest temporary
directory, including per-request errors and partial results on cancellation.
SIGTERM/SIGINT cancels pending HTTP operations before service cleanup, without
waiting for the request timeout. It does **not** run GSM8K or establish general
model accuracy.
Service liveness and owned-process cleanup are also required to pass.
This 16-NPU case allows 60 seconds for service shutdown before escalation;
Attention SIGKILL escalation remains a failure. FFN uses the existing scoped
async CAM cleanup exception.

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V4-Flash-w8a8-mtp
export HCCL_IF_IP=<local-communication-ip>
export HCCL_SOCKET_IFNAME=eth0
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[afd-dsv4-flash-async-cam-dp2tp4-ep8]'
```

Defaults: API ports 19280/19281, AFD rendezvous port 6455, startup timeout
1800 seconds. Override these using `AFD_NPU_DSV4_E2E_API_PORT`,
`AFD_NPU_DSV4_E2E_AFD_PORT`, and `AFD_NPU_E2E_STARTUP_TIMEOUT`.
`AFD_NPU_E2E_VLLM_BIN` selects the executable. Build the plugin-owned 910C
operators before running. Model and all sixteen device IDs must be supplied explicitly; missing setup fails rather than skips.

## Run with the Codex skill

The repository includes the [`run-e2e`](../../.agents/skills/run-e2e/SKILL.md)
skill. Open the repository in Codex and ask, for example:

```text
Use run-e2e to run the Qwen3 MoE GPU E2E tests with HF_HOME
/data/huggingface.
```

Provide `HF_HOME` and `AFD_E2E_BACKEND`. `AFD_E2E_DEVICES` is optional; when
unset, the test module picks the defaults above. The model path is optional
when Hugging Face download is available. The skill checks prerequisites, runs
the same four tests, and reports failures and process cleanup.

## NPU async CAM smoke test

This separate test still reads `AFD_E2E_DEVICES`. It uses four NPUs: the first
two for Attention TP=2, and the last two for FFN DP=2/TP=1. It sends one
prompt and requests 32 tokens. It does not run GSM8K.

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_cam
```

The CANN runtime and source-built AFD custom operators must be installed. Missing
model configuration or a device list other than four unique IDs fails the
test.

## NPU async CAM ubatching test

This case runs the `afd-async-ubatch` scenario (`CAMAsyncAFDConnector` with
AFD-managed two-stage MoE token-split ubatching). It uses three NPUs: the first
two for Attention DP=1/TP=2, and the last one for FFN DP=1/TP=1. Unlike the
smoke test above, it reuses the shared runner's GSM8K path with batch size 2
(first 7 samples, 8-shot, sample-count and accuracy gates).

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_ubatch
```

The device count is derived from the scenario's Attention/FFN rank constants,
not hard-coded; `AFD_E2E_DEVICES` may supply any three unique NPU IDs. The
topology is fixed at 2A1F, with TP=2 on Attention as required by token split.

Prerequisites match the async CAM smoke test (CAM/CANN runtime and custom
operators), plus a reachable GSM8K dataset source — offline pods need a local
HF mirror (`HF_ENDPOINT`). Both async CAM tests configure a 4096 MB buffer only
for connector-owned HCCL groups and remove `HCCL_BUFFSIZE` from child process
environments so unrelated groups retain their normal defaults. See
[`docs/npu/TROUBLESHOOTING.md`](../../docs/npu/TROUBLESHOOTING.md).
