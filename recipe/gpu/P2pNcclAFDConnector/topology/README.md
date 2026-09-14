# AFD topology recipes: one table, every deployment shape

The scenario scripts under `deepseek_v2_lite/prefill_decode_colocation/` each
hard-code one shape. This directory replaces that with a table plus two scripts,
so single-host, multi-host, and split-role deployments all come from the same
source of truth.

| File | Role |
| --- | --- |
| `topologies.tsv` | **the source of truth** -- one row per vLLM process |
| `afd_launch.sh` | expands a scenario into per-process launches |
| `afd_serve.sh` | launches one process; validates the topology and prints the rank map |

```bash
./afd_launch.sh --list                       # what shapes exist
./afd_launch.sh 2a2f-dp2tp1 --plan           # what would run where
MODEL_PATH=/models/DeepSeek-V2-Lite ./afd_launch.sh 2a2f-dp2tp1
```

## The parameter model

An AFD deployment is described by three independent groups of settings. Getting
them to agree by hand is where most bring-ups fail, which is why the table
derives all but the first group.

### 1. AFD topology -- what the connector is told

Passed through `--additional-config` as the `afd` object, identically on
**every** process in the deployment:

| Key | Meaning |
| --- | --- |
| `role` | `attention` or `ffn`. The only key that differs per process. |
| `connector` | `P2pNcclAFDConnector` for GPU. |
| `host` | Address of **FFN role rank 0**. Every rank passes the same value, and it must be an address that FFN rank 0's process owns. |
| `port` | Base rendezvous port, default `6269`, on `host`. |
| `num_attention_ranks` | `A`, the total Attention ranks across all hosts. |
| `num_ffn_ranks` | `F`, the total FFN ranks across all hosts. |
| `compute_gate_on_attention` | Optional; Attention computes MoE gate outputs before sending. |

Constraints, enforced in `afd_plugin/distributed/topology.py`: `A >= F` and
`A % F == 0`. `ratio = A / F` is how many Attention ranks fan into each FFN rank.

### 2. Per-role parallelism -- how many ranks actually exist

A role's rank count is not declared separately; it **falls out of that role's
parallelism**:

```text
role_rank = (dp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
A = dp_size(attention) * tp_size(attention)      # pcp = 1 on GPU
F = dp_size(ffn)       * tp_size(ffn)            # pipeline parallel must be 1
```

If `A` does not equal `dp_size * tp_size` for the Attention role, some AFD world
ranks are never filled and the rendezvous (`world_size = A + F`) hangs for its
full two minutes before failing. `afd_serve.sh` checks this before loading the
model and fails immediately instead.

The two roles are independent here: `2a2f-dp2tp1` and `2a2f-dp1tp2` both give
`A = F = 2`, but the first spreads each role over two DP engines and the second
runs one engine with TP 2. The connector cannot tell the difference; the GPUs
and the memory profile can.

### 3. Placement -- which process runs which ranks

Each role's DP ranks are split across processes with the standard vLLM flags.
Per process:

| Flag | Value |
| --- | --- |
| `--data-parallel-size` | that role's **total** DP size, same on every process of the role |
| `--data-parallel-size-local` | DP engines **this** process runs |
| `--data-parallel-start-rank` | global DP rank of this process's first engine |
| `--data-parallel-address` / `--data-parallel-rpc-port` | address and port of **that role's** DP rank 0 |
| `--headless` | on every process whose start rank is not 0 |

`--data-parallel-address` is per role and is **not** the same as `afd.host`:
the Attention role rendezvouses its own DP group on its own rank 0. A fully
split deployment therefore has three addresses in play -- `afd.host` (FFN rank
0), the FFN DP address (also FFN rank 0), and the Attention DP address
(Attention rank 0) -- and two DP RPC ports, one per role.

### GPU assignment

`CUDA_VISIBLE_DEVICES` gives each **process** its slice of host-local devices.
The slice must hold exactly `dp_local * tp_size` devices; vLLM assigns them to
local DP engine `i`, TP rank `t` in order, so device index `i * tp_size + t`.
That mapping is what the `gpus` column in the table sets, and what
`afd_serve.sh` prints per rank:

```text
afd_serve:   gpu 2 -> dp_rank 0 tp_rank 0 = ffn rank 0, afd world rank 0, subgroup 0, binds subgroup store on 127.0.0.1:6270
afd_serve:   gpu 3 -> dp_rank 1 tp_rank 0 = ffn rank 1, afd world rank 1, subgroup 1, binds subgroup store on 127.0.0.1:6271
```

Attention and FFN are separate processes, so on a single host they take disjoint
device lists (`0,1` and `2,3`). Expect asymmetric memory use -- on
DeepSeek-V2-Lite/H100 roughly 77 GB per Attention GPU against 18 GB per FFN GPU.
That asymmetry is AFD working, not a misconfiguration.

## Single host vs multi host

Nothing in the AFD configuration changes between the two. What changes is
whether `afd.host` and the DP addresses are loopback or routable, and whether a
role's ranks are split across processes.

```bash
# Single host: slots resolve to 127.0.0.1, so no configuration is needed.
./afd_launch.sh 2a2f-dp2tp1

# Multi host: map the slots, then run the SAME command on each host with a
# different --host. Start the FFN rank 0 host first; the rendezvous window is
# two minutes.
export HOSTS="h0=10.0.0.1 h1=10.0.0.2"
./afd_launch.sh 2a2f-two-host --host h0 --dry-run   # on 10.0.0.1
./afd_launch.sh 2a2f-two-host --host h1 --dry-run   # on 10.0.0.2
```

Multi-host extras, none of them AFD-specific:

- `NIC_NAME=eth0` pins `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and
  `TP_SOCKET_IFNAME` when a host has more than one route.
- Reachable ports on the FFN rank 0 host: `AFD_PORT` (world plus control plane)
  and `AFD_PORT + subgroup + 1` for each subgroup, plus each role's DP RPC port
  on that role's rank 0 host.
- Weights must be reachable from every host -- on Kubernetes, an RWX PVC.

## Runtime knobs are environment variables, not rows

So that any shape can be run in any mode without duplicating the table:
`MODEL_PATH`, `SERVED_MODEL_NAME`, `MODE` (`eager` or `graph`), `DBO`,
`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `AFD_PORT`, `NIC_NAME`, `AFD_VLLM_LAUNCHER`,
`EXTRA_ARGS`, `LOG_DIR`.

```bash
MODE=graph MODEL_PATH=/models/DeepSeek-V2-Lite ./afd_launch.sh 4a4f-dp2tp2
```

`AFD_VLLM_LAUNCHER` exists for containers where `uv run` cannot work, for
example `AFD_VLLM_LAUNCHER='/work/venv/bin/python -m vllm.entrypoints.cli.main'`
under a read-only project directory or an unwritable uv cache.

## Using it on Kubernetes

`afd_serve.sh` takes plain environment variables and launches exactly one
process, which is the shape of a pod. Take the values from
`afd_launch.sh <scenario> --plan`, set them in the pod spec, and have the
container run `afd_serve.sh`. Point each role's `DP_ADDRESS`, and `AFD_HOST`, at
the relevant pod's IP (or a headless Service address) rather than a node IP.

Note that a pod has its own network namespace: two FFN pods on one node are as
separate, for binding purposes, as two pods on different nodes.

## The `afd.host` locality check

Before PR #328, the GPU P2P connector created each subgroup's store at
`afd.host:port + subgroup_index + 1`, and **every FFN rank is rank 0 of its own
subgroup** -- so every FFN rank tried to bind `afd.host`. Only the process
owning that address can. Any other FFN process died in `init_afd_connector`
with `OSError: [Errno 99] Cannot assign requested address`
([issue #327](https://github.com/vllm-project/afd-plugin/issues/327)).

`afd_serve.sh` probes `afd.host` on FFN processes and warns before the model
load rather than after it. The `2a2f-split-ffn` scenario exists to reproduce the
failure deliberately; `2a2f-split-attn` is the control that splits the other
role and succeeds either way. Set `SKIP_HOST_CHECK=1` to silence the probe when
the failure is the point.

After PR #328 a subgroup reuses the AFD world's store through a `PrefixStore`,
no second bind happens, and only `AFD_PORT` needs to be reachable.

## Scenarios in the table

| Scenario | GPUs | Hosts | Shape |
| --- | --- | --- | --- |
| `1a1f` | 2 | 1 | smallest shape that exercises the connector |
| `2a2f-dp2tp1` | 4 | 1 | DP 2, TP 1 per role |
| `2a2f-dp1tp2` | 4 | 1 | DP 1, TP 2 per role |
| `4a4f-dp2tp2` | 8 | 1 | DP 2, TP 2 per role |
| `4a2f-ratio2` | 6 | 1 | ratio 2: two Attention ranks per FFN rank |
| `2a2f-two-host` | 4 | 2 | one role per host |
| `2a2f-split-ffn` | 4 | 2 | FFN role split across hosts (issue #327) |
| `2a2f-split-attn` | 4 | 2 | Attention role split across hosts (control) |
| `2a2f-four-proc` | 4 | 4 | one rank per process |

Adding a shape means adding rows, not writing a script. Row order defines rank
order: a role's first row owns DP rank 0, and for the FFN role that row also
owns `afd.host`.
