# Phase 4b report: cluster-portable network setup

- Date: 2026-10-15
- Scope: items 1–4 of the cluster-portability review (after Phase 4):
  discover the topology, derive one network plan, turn constants into
  configuration, add fabric presets. Items 5–7 (payload format by GPU
  generation, build portability, deployment templates) are in `TODO.md`.
- Branch: `fastafd-connector`.
- Hardware: pokprod `ronenkat-test1`, pods `deepep-m2n-node0`
  (`pokprod-b93r39s2`) and `deepep-m2n-node1` (`pokprod-b93r44s2`), 4× H100
  each, RoCE (`k8s/pods-2node.yaml`).

## Result

**The connector now works out the network setup itself.** On pokprod, with
no NIC setting at all, every rank agrees on one shared rail and the two-node
suite passes (6 runs × 2 nodes), at the same speed as Phase 4's hand-picked
`mlx5_0`. Clusters with routed rails, InfiniBand or an NVL72 rack need only a
preset (or nothing): their logic is unit-tested on synthetic topologies, but
only pokprod (rail-isolated RoCE) has been run on hardware.

## Pins addressed

| Pin (review) | Before | Now |
| --- | --- | --- |
| A1 GIN on/off | `disable_gin: auto` = hostname rule (all ranks on one host → GIN off) | `network.gin: auto` = GIN only when ranks span several **NVLink domains** (NVML fabric clique on NVL72, else hostname) |
| A2 Layout check | Equal, contiguous ranks per hostname | Equal, contiguous ranks per NVLink domain (DeepEP's actual rule); an NVL72 rack may hold uneven ranks per tray |
| A3 NIC choice | `ib_hca_by_local_rank`, set by hand (pokprod: `["mlx5_0"]`) | `network.nic_policy: auto` picks from discovered NICs and rails (below) |
| A4 Hybrid mode | `allow_hybrid_mode: true` | `network.hybrid` (`auto` = on) |
| A5 RDMA traffic class | Constant `"3"` | `network.rdma_traffic_class` (default 3) |
| A6 Tuning | QPs, timeouts, 24 SMs fixed | `num_qps` (0 = DeepEP default), `cpu_timeout_s`/`gpu_timeout_s` (120), `num_sms` default = 24/132 of the device's SMs |
| Site NCCL settings | Deployment env only | `network.nccl_env` (`NCCL_*`/`EP_*` keys only), applied before the first communicator |

The removed keys (`disable_gin`, `allow_hybrid_mode`, `ib_hca_by_local_rank`)
are rejected as unknown options, not silently ignored.

## How it works

`afd_plugin/connectors/gpu/deepep_m2n/network.py`:

1. **Discovery, per rank** (`discover_local_topology`): host name, NVLink
   domain (NVML `GpuFabricInfo` cluster UUID + clique when the GPU is in a
   multi-node NVLink fabric, else `host:<hostname>`), local GPU index, and
   the active RDMA ports in `/sys/class/infiniband`: link layer, PCIe
   distance to the GPU, and a **rail** = the RoCE interface's IPv4 subnet
   (`roce:172.22.144.0/21`) or the IB GID subnet prefix.
2. **Handshake**: each rank's topology joins the existing init
   `all_gather_object`, which now runs on a gloo group created before any
   NCCL communicator. The `network` config must match on every rank.
3. **Plan, identical on every rank** (`plan_network`):
   - group ranks into NVLink domains; reject non-contiguous or unequal
     blocks;
   - GIN on iff there is more than one domain (or `gin: on|off`);
   - NIC policy under `auto`: `nccl` (keep NCCL's nearest NIC per GPU) when
     `rails_routed` is declared or all NICs share one rail; otherwise
     `shared`: the rail present on every rank with the lowest total PCIe
     distance (or the rail of `nics[0]`), each rank using its own host's NIC
     on that rail;
   - `explicit`: `nics` by local rank.
4. **Environment** (`apply_rank_environment`): `nccl_env`, `NCCL_IB_HCA` for
   a pinned NIC, `EP_DISABLE_GIN`. Then the control group (nccl or the gloo
   group) and the DeepEP buffers are created with the plan's hybrid mode and
   traffic class. Rank 0 logs the plan; the test harness reports it per rank.

Whether rails route to each other cannot be seen from inside a pod, so it is
declared (`rails_routed`); unknown means "not routed", which works
everywhere at the cost of using one NIC per node.

### Presets

`network: {preset: ...}`; explicit keys override the preset.

| Preset | Settings | For |
| --- | --- | --- |
| `auto` (default) | none | Anything; isolated rails assumed when NICs sit on several subnets |
| `single-node` | `gin: off`, `nic_policy: nccl` | One host |
| `roce-rail-isolated` | `nic_policy: shared`, `rails_routed: false` | pokprod: each NIC on its own unrouted /21 |
| `roce-routed` | `nic_policy: nccl`, `rails_routed: true` | Rail-optimized RoCE with routing between rails |
| `ib` | `nic_policy: nccl`, `rails_routed: true` | InfiniBand fat-tree |
| `nvl72` | `nic_policy: nccl`, `rails_routed: true` | GB200 NVL72; one rack is one domain → GIN off |

Examples: `{"network": {"preset": "ib", "rdma_traffic_class": 0}}`,
`{"network": {"preset": "roce-routed", "nccl_env": {"NCCL_IB_GID_INDEX": "3"}}}`.

### Caveat: NCCL reads its environment once per process

`NCCL_IB_HCA` and `nccl_env` take effect only if no NCCL communicator exists
in the process yet. The connector creates its own groups after applying them,
and the bootstrap now uses gloo for that reason. Under the runner, vLLM may
already have created NCCL groups (TP/EP) before the connector initializes.
In that case the plan is still computed and logged, but the deployment must
set the same variables in the pod environment (the plan log tells which).

## Topology seen on pokprod

`probe_topology` on node0 (H100, so no NVLink fabric; domain = host):

- 16 active RoCE ports, each on its own rail (separate /21 subnets).
- The nearest NICs differ per GPU (gpu0 `mlx5_8/9`, gpu1 `mlx5_10/11`, gpu2
  `mlx5_12/13`, gpu3 `mlx5_16/17`); `mlx5_0` is far from all of them.

With `auto`, the plan picks `shared` on rail `roce:172.22.144.0/21`
(`mlx5_8`, nearest to GPU 0) on both nodes, not Phase 4's `mlx5_0`. Logged by
every rank:

```text
domains=2x4 gin=True hybrid=True nic_policy=shared rail=roce:172.22.144.0/21
nics=['mlx5_8'] traffic_class=3 this_rank_nic=mlx5_8
```

## Tests

### Two nodes (`phase4b-results/`)

No NIC setting in any run (only the last column's options). All pass on both
nodes.

| Run | Config | Domain | Result |
| --- | --- | --- | --- |
| 2F\|2A FP8 correctness | `auto` | (2, 2) | pass; experts ≤ 2.8e-2, transport 9.6e-8 |
| 2F\|2A FP8 CUDA graphs | `auto` | (2, 2) | pass |
| 4F\|4A BF16 correctness | `auto` | (2, 4) | pass; experts ≤ 6.0e-3, transport 9.4e-10 |
| 4F\|4A FP8, two lanes | `preset: roce-rail-isolated` | (2, 4) | pass; plan above |
| 2F\|2A FP8 correctness | `auto`, `control_backend: gloo` | (2, 2) | pass; same plan |
| 2F\|2A FP8 graph benchmark | `auto` | (2, 2) | pass; below |

Graph round trip per MoE layer on Attention rank 0 (µs):

| T/attn | Phase 4, `mlx5_0` pinned | Phase 4b, auto (`mlx5_8`) |
| ---: | ---: | ---: |
| 1 | 163 | 171 |
| 8 | 268 | 249 |
| 32 | 311 | 316 |
| 128 | 438 | 422 |
| 512 | 851 | 872 |
| 2048 | 2664 | 2709 |

Same within run-to-run noise: both use one 100 Gb/s port per node. The
plan's choice of a NIC near GPU 0 makes no measurable difference here, since
the port is the bottleneck.

### Unit tests

- `tests/unit/connectors/test_deepep_m2n_network.py` (new, CPU): presets
  and overrides, invalid options, single host (no GIN, no NIC), pokprod-like
  isolated rails → one shared rail, each host's own NIC name on that rail,
  preferred NIC, routed RoCE and IB → `nccl`, IB fat-tree as one rail, NVL72
  with uneven trays → one domain, explicit NICs, no common rail → error,
  forced GIN, environment application, and discovery from a fake sysfs
  (rails from GIDs and interface subnets, PCIe distance, down ports
  skipped).
- `test_deepep_m2n_connector.py`: handshake carries topology and network
  config, rejects mismatches and bad domain layouts; new options validated.
- Single-node regression on node0 (`tests/unit/connectors` + `tests/unit/config`):
  172 passed, 3 skipped; `test_deepep_m2n_gpu.py` 16 passed (one host, so the
  plan turns GIN off).

## Not covered

- Real runs on routed RoCE, InfiniBand or NVL72 (no such cluster here). On
  NVL72 the fabric-ID lookup and `nvl72` preset are tested only with
  synthetic topologies.
- Using every NIC on pokprod: still blocked by the fabric (unrouted rails)
  and DeepEP's all-pairs GIN connections, as in Phase 4.
- Items 5–7 and the other open items: `TODO.md`.
