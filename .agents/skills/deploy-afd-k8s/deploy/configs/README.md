# AFD deployment configs

A JSON config here generalizes one `recipe/gpu/P2pNcclAFDConnector/**` shell
recipe: model, topology (`num_attention_ranks` / `num_ffn_ranks`, DP/TP per
role), single-node vs multi-node placement, execution mode, DBO, and the
vLLM serve knobs that vary between recipes. [`../generate_recipe.py`](../generate_recipe.py)
turns a config into a recipe script (or a pair of them, for multi-node) plus
a `<name>.manifest.json`; the `deploy-afd-k8s` skill consumes either the
generated script(s) or a hand-written one.

Schema: [`schema/afd-deploy-config.schema.json`](schema/afd-deploy-config.schema.json).
Worked examples: [`examples/`](examples/).

## Fields

| Field | Meaning |
| --- | --- |
| `name` | Used to name generated files, ConfigMaps, and k8s labels. |
| `model.family` | Recipe directory name this generalizes (e.g. `deepseek_v2_lite`). Labeling only. |
| `model.model_id` | HF Hub repo id (e.g. `deepseek-ai/DeepSeek-V2-Lite`) or a literal in-container path (e.g. `/models/Qwen3.5-122B-A10B-FP8`) staged on the PVC. Becomes the script's `MODEL_PATH` default. |
| `model.trust_remote_code` | Emits `--trust-remote-code` when true. |
| `model.extra_serve_args` | Model-specific fixed flags that don't fit the generic AFD shape, e.g. Qwen3.5's `{"dtype": "bfloat16", "language-model-only": true, "mamba-cache-mode": "align", "all2all-backend": "allgather_reducescatter", "seed": 0}`. `true` emits a bare flag; any other value emits `--flag value`. |
| `topology.strategy` | Must be `"colocation"` -- `prefill_decode_disaggregation` recipes are out of scope for `deploy-afd-k8s` (see its SKILL.md Scope section) and this config format doesn't represent them. |
| `topology.afd_enabled` | `false` generates a baseline (non-AFD) recipe: one DP/TP group from `topology.attention`, no `--additional-config`, no DBO. |
| `topology.num_attention_ranks` / `num_ffn_ranks` | Total AFD worker counts (DP x TP for that role), required when `afd_enabled`. Must satisfy `num_attention_ranks >= num_ffn_ranks` and `num_attention_ranks % num_ffn_ranks == 0` (every FFN rank maps to the same number of consecutive Attention ranks -- see `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`). |
| `topology.attention` / `topology.ffn` | `{data_parallel_size, tensor_parallel_size}` per role. Their product must equal the matching `num_*_ranks`. |
| `placement.node_mode` | `"single"`: attention + FFN run as two backgrounded processes in one recipe script / one Pod (today's behavior). `"multi"`: each role gets its own recipe script and Pod, scheduled on different nodes. Requires `afd_enabled: true`. **Experimental** -- see the caveat in `deploy-afd-k8s`'s SKILL.md; cross-node AFD is not validated upstream. |
| `execution.mode` | `"eager"` (`--enforce-eager`) or `"graph"` (`--max-cudagraph-capture-size` + `--compilation-config` with `FULL_DECODE_ONLY`). |
| `execution.cudagraph_capture_size` | Required when `mode` is `"graph"`. Should generally match `serving.max_num_seqs`. |
| `dbo.enabled` | Dual Batch Overlap. Must be `false` when `afd_enabled` is `false` -- baseline recipes don't use DBO. |
| `dbo.decode_token_threshold` / `prefill_token_threshold` | Example defaults `2` / `12` in every checked-in recipe; tune per workload. |
| `serving.max_num_seqs` / `max_num_batched_tokens` / `max_model_len` | Standard vLLM serving limits. `max_num_batched_tokens` is usually the first thing a load profile saturates -- report it alongside the endpoint. |

There is no `connector` section -- every recipe uses the same connector
type (`P2pNcclAFDConnector`), so it's hardcoded in the generator. The base
AFD rendezvous port isn't part of the config either; it defaults to `6269`
and can be overridden by exporting `AFD_CONNECTOR_PORT` before running
`generate_recipe.py`. The connector also needs `port + 1 .. port +
num_ffn_ranks` reachable (one derived port per FFN subgroup).

There is no `serving.client_port` either -- the OpenAI-compatible HTTP port
defaults to `18305` (every checked-in recipe uses it) and can be overridden
by exporting `AFD_CLIENT_PORT` before running `generate_recipe.py`.

There is no `deployment` section -- k8s specifics (namespace, image, PVC
name/size, storage class, HF token secret name) aren't part of this config.
`deploy-afd-k8s` confirms those with the user at deploy time, the same way
it does when starting from a raw `.sh` recipe.

## Generating a recipe

```bash
python3 .agents/skills/deploy-afd-k8s/deploy/generate_recipe.py .agents/skills/deploy-afd-k8s/deploy/configs/examples/deepseek_v2_lite-2a2f-graph-singlenode.json --out-dir /tmp/afd-recipe-gen
```

Writes `<out-dir>/<name>.recipe.sh` (single-node) or
`<name>.attention.recipe.sh` + `<name>.ffn.recipe.sh` (multi-node), plus
`<name>.manifest.json` describing GPU counts per role, the resolved
`MODEL_ID` form (`hf` vs `path`), and -- for AFD topologies -- the port
range the FFN side needs exposed (`ffn_port_range`). Set
`AFD_CONNECTOR_PORT=<port>` in the environment before running this to use a
base rendezvous port other than the default `6269`, or `AFD_CLIENT_PORT=<port>`
to use an HTTP port other than the default `18305`.

For `node_mode: "multi"`, both generated scripts contain the literal
placeholder `AFD_FFN_HOST_PLACEHOLDER` wherever the AFD `host` field would
otherwise be `127.0.0.1` (per `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`,
`host` must resolve to FFN's first rank). `deploy-afd-k8s` substitutes the
real FFN Service DNS name into that placeholder once it exists -- the
generator itself has no k8s knowledge and never needs to be re-run for that.

## Examples

| File | Topology | Placement |
| --- | --- | --- |
| `deepseek_v2_lite-2a2f-graph-singlenode.json` | `2A2F`, DP1/TP2 per role, graph mode, DBO | single-node |
| `deepseek_v2_lite-4a4f-graph-multinode.json` | `4A4F`, DP2/TP2 per role, graph mode, DBO | multi-node (experimental) |
| `deepseek_v2_lite-baseline-singlenode.json` | non-AFD, DP4/TP1 | single-node |
| `qwen3_5_122b_a10b_fp8-2a2f-graph-singlenode.json` | `2A2F`, DP2/TP1 per role, graph mode, no DBO | single-node |
