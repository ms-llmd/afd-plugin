# P2pNcclAFDConnector Recipes

Launch scripts for running Attention-FFN Disaggregation (AFD) with
`P2pNcclAFDConnector`, a GPU point-to-point connector built on vLLM's
`PyNcclCommunicator`. See
[`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`](../../../docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md)
for the connector's rank mapping and configuration contract.

## Directory layout

```text
.
├── deepseek_v2_lite/
│   ├── README.md
│   ├── prefill_decode_colocation/       # 2A2F, DP/TP variants, eager/graph
│   └── prefill_decode_disaggregation/   # 2P1A1F
└── qwen3_5_122b_a10b_fp8/
    └── prefill_decode_colocation/       # 4A2F, graph
```

Each model directory holds its own recipe scripts; open its `README.md`
(where present) before running anything in it -- it documents prerequisites,
topology, ports, and known limitations for that model.

Recipes under `prefill_decode_colocation/` background an attention worker and
an FFN worker (no prefill split). Recipes under `prefill_decode_disaggregation/`
add a separate prefill stage and a proxy in front of it -- see the caveat in
"Deploying on Kubernetes" below.

## Running on a local host

Run any script directly from the repository root, e.g.:

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh
```

Each script backgrounds its workers and writes per-worker logs (`attn.log`,
`ffn.log`, and for disaggregation `afd_prefill*.log`) into the current
directory. Wait for `Application startup complete` in each log before
sending traffic. See the model README for exact prerequisites (GPU count,
weights, ports) and the benchmark command for that model.

## Deploying on Kubernetes

Use the **`deploy-afd-k8s`** skill (`.agents/skills/deploy-afd-k8s/SKILL.md`)
to stand up a recipe from this folder on a Kubernetes/OpenShift cluster. It
provisions the model PVC, mounts the chosen recipe script into a serve pod
via a ConfigMap (no image rebuild needed to try an edited or new recipe),
and waits until an OpenAI-compatible endpoint answers.

The skill requires the recipe path (e.g.
`recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh`)
plus the image, model id, and PVC name it needs. It derives `GPU_COUNT`
itself for a plain single-node recipe; for a multi-node-capable recipe it
instead confirms the recipe's attention/FFN rank totals and asks you for a
placement plan (see below).

**Scope.** The skill only covers `prefill_decode_colocation` recipes (and
their `baseline*` counterparts) under `recipe/gpu/P2pNcclAFDConnector/**`.
`prefill_decode_disaggregation` recipes are out of scope for it -- they need
a NIXL-enabled image and a proxy, and rely on a `SCRIPT_DIR`-relative path
that breaks once the script is mounted flat into the pod. Run those locally
instead, or ask for a different workflow.

**Single-node.** Any recipe deploys as one pod holding every rank -- this is
the simple, non-experimental path. The skill computes `GPU_COUNT` itself from
the script's `CUDA_VISIBLE_DEVICES` blocks and deploys a single `vllm-pod` /
`vllm-service` pair.

**Multi-node.** Only recipes that gate their `vllm serve` blocks on
`ATTENTION_DP_RANKS`/`FFN_DP_RANKS` and read the AFD `host` field from an
`AFD_CONNECTOR_HOST` env var support spreading attention/FFN ranks across
more than one pod (e.g. `deepseek_v2_lite/prefill_decode_colocation/*.sh`).
For these, the skill reads the recipe's fixed attention/FFN rank totals and
asks you for an explicit **placement plan** -- how many of those ranks go in
each pod, and how many pods -- rather than assuming a shape. It then creates
one pod per plan entry, an internal FFN rendezvous Service when the plan
spans more than one pod, and DP-coordination Services for any role that is
itself split across pods.

Pods and Services are left running after deployment so weights stay warm for
follow-up runs; see the skill's teardown step to tear them down explicitly.
