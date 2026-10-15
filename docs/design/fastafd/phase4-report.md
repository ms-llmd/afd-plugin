# Phase 4 report: two-node DeepEP M2N over RoCE

- Date: 2026-10-15
- Plan: `.scratchpad/fastafd/fastafd-deepep-m2n-connector-plan.md`, §9 Phase 4.
  Agreed scope: connector-level multi-node tests and benchmark (no runner E2E),
  FFN ranks on one node and Attention ranks on the other, equal rank counts
  per node, Attention TP > 1 deferred.
- Branch: `fastafd-connector`.
- Hardware: pokprod `ronenkat-test1`, two pods on different nodes
  (`pokprod-b93r43s0`, `pokprod-b93r44s2`), 4× H100 each, RoCE via
  `multi-nic-compute` + `rdma/roce_gdr` (`k8s/pods-2node.yaml`). Each RoCE
  port is 100 Gb/s; each NIC port is its own rail (a separate /21 subnet).

## Result

**The connector works across nodes**, with all Phase 1–3 checks passing on
2F|2A and 4F|4A. **On pokprod it needs every rank to use the same NIC**,
because the rails are unrouted subnets and DeepEP opens RDMA connections
between ranks of different local index. That caps cross-node traffic at one
100 Gb/s port per node, and large batches are network-bound at ~1.6M tokens
per layer per second.

## What changed

> Phase 4b replaced `ib_hca_by_local_rank` with the `network` section (an
> automatic NIC plan); see `phase4b-report.md`.

| Area | Change |
| --- | --- |
| `ib_hca_by_local_rank` | Sets `NCCL_IB_HCA` per rank (by local rank) before any NCCL communicator exists. On pokprod use a single entry (`["mlx5_0"]`). |
| `control_backend` | `nccl` (default) or `gloo` for the connector's own group (handshake, DeepEP bootstrap, control payload); `gloo` keeps that traffic off GPUs and NICs. |
| Handshake | Rejects host layouts DeepEP's hybrid mode cannot use: each host must hold the same number of contiguous ranks. |
| `connector.domain` | DeepEP's `(scale-out, scale-up)` sizes, reported by the tests: `(2, 2)` / `(2, 4)` confirm traffic crossed nodes. |
| Tests | `run_ranks(..., Placement)` runs one node's ranks with a shared master address; `tests/unit/connectors/deepep_m2n_multinode.py` runs any scenario per node. |

## Finding the working NIC setting

The two pods received different physical GPUs (node0 bus 18/2A/3A/5D, node1
18/3A/5D/9A), so the NIC nearest local GPU `j` is on a different rail on each
node.

| Experiment | NIC setting | Outcome |
| --- | --- | --- |
| E1 | NCCL default (nearest NIC) | DeepEP GIN setup: `ibv_modify_qp` INIT→RTR timeout from `mlx5_3` (rail 200) to a peer on another rail; NCCL hints "NICs are not cross-rail connected". |
| E2 | One NIC per local rank, rail-aligned across nodes (`mlx5_0`, `mlx5_5`) | The connector's NCCL group fails first (`IBV_WC_RETRY_EXC_ERR`): NCCL rings pair local rank 1 on one node with local rank 0 on the other. |
| E4 | As E2, connector group on `gloo` | DeepEP's own GIN setup (`ncclDevCommCreate`) times out from node1 `mlx5_0` (172.22.177.97) to node1's other rank on `mlx5_5` (172.22.139.97): GIN connects same-node peers too. |
| E3 / E5 | Every rank on `mlx5_0` (E5 also with `gloo`) | **Pass.** |

Using all NICs needs routing between the rail subnets (a cluster change) or
a DeepEP/NCCL GIN mode that connects only rail-aligned peers.

## Tests (all with every rank on `mlx5_0`)

15 runs, all passing on both nodes (`phase4-results/summary.txt`):

| Run | Domain | Worst combine vs BF16 model | Transport (rows) |
| --- | --- | ---: | ---: |
| 2F\|2A FP8, eager (10 cases × 2 layers) | (2, 2) | 6.6e-2 | 9.6e-8 |
| 2F\|2A FP8, two lanes | (2, 2) | 6.9e-2 | 9.5e-8 |
| 2F\|2A FP8, CUDA graphs, 1 and 2 lanes | (2, 2) | 6.7e-2 | 1.3e-7 |
| 4F\|4A BF16, eager | (2, 4) | 5.5e-3 | 1.3e-9 |
| 4F\|4A FP8, eager / two lanes / CUDA graphs | (2, 4) | 6.9e-2 | 1.6e-7 |

The experts check (FFN side) stays at ≤ 6.0e-3 (BF16) and ≤ 2.9e-2 (FP8), as
on one node.

## Benchmark

Per MoE layer, round trip on Attention rank 0; FP8 unless noted. Single-node
2F2A numbers are from Phase 3 (`phase3-report.md`).

| T/attn | 2F2A one node, graph | 2F\|2A two nodes, graph | 2F\|2A two nodes, eager (no sync) | 4F\|4A two nodes, graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 98 | 163 | 493 | 303 |
| 8 | 170 | 268 | 551 | 380 |
| 32 | 211 | 311 | 585 | 409 |
| 128 | 215 | 438 | 645 | 575 |
| 512 | 251 | 851 | 1097 | 1515 |
| 2048 | 462 | 2664 | 2869 | 5021 |
| 8192 | — | — | 10216 | — |

Eager breakdown at 8192 tokens (2F|2A): dispatch 3554 µs, experts 725 µs,
combine 540 µs. Two lanes under graphs save 25% at 128 tokens and 6% at 2048
(2F|2A). BF16 graph 2F|2A: 247 µs at 1 token, 4014 µs at 2048.

### Findings

1. **Cross-node latency adds ~65–220 µs at decode sizes** (2F|2A graph: 98 →
   163 µs at 1 token, 215 → 438 µs at 128), all of it in dispatch and combine.
2. **Large batches are bound by the one usable NIC.** At 8192 tokens per
   rank the 2F|2A dispatch moves ~33 MB of FP8 rows in ~3.5 ms (~75 Gb/s on a
   100 Gb/s port); 2F|2A and 4F|4A both level off at ~1.6M tokens per layer
   per second. From 512 tokens on, 4F|4A moves twice the tokens per layer in
   about twice the time, as expected when both share the same port.
3. **Spreading traffic over a node's NICs is the next win and is blocked by
   the fabric**, not the connector (see E2/E4). With per-rank rails it would
   scale with the number of NICs per node.
4. **Expert compute is unchanged** (eager 2F|2A experts 160–725 µs, as on one
   node): the FFN side does the same work, only the network differs.

## Open items

- Rail routing (cluster) or a rail-only GIN connection mode, to use every NIC.
- Attention TP > 1 (sequence-split dispatch), deferred.
- Runner integration and a two-node E2E with a real model.
- Teardown when a peer dies (DeepEP's barrier timeout, then an abort).
