# AFD connector regression suite

Compare **P2P → Masked DeepEP → afdEP v1** on the same GPUs and requests.
The primary regression control is Masked DeepEP; the P2P comparison measures
complete serving paths, including its FFN expert-parallel exchange.

```mermaid
flowchart LR
  subgraph P2P
    PA[Attention] --> PF[Paired FFN]
    PF <--> PE[FFN expert-parallel exchange]
    PF --> PA
  end
  subgraph Masked_DeepEP
    MA[Attention routing] --> MF[Owning FFN experts]
    MF --> MA
  end
  subgraph afdEP_v1
    AA[Attention routing] --> AF[Owning FFN experts]
    AF --> AA
  end
```

Masked DeepEP and v1 share scheduling, payload precision, expert compute and
transport. V1 enables explicit roles, removes unused Attention payload arenas
and empty-side epilogues, and bounds FFN receives by Attention source count.
Logical expert slots, collective participation and registered transport capacity
remain. P2P naturally uses BF16 payloads and FFN routing; the routed paths use
FP8 plus scales and Attention routing. Their difference is not transport alone.

## Matrix

| Architecture | GPU baseline | Topology | Three-way suite |
|---|---|---|---|
| Hopper SM90 | H100 80 GB | 4 GPUs, one NVLink domain | Supported profile |
| Hopper SM90 | H200 | 4 GPUs, one NVLink domain | Supported profile |
| Hopper SM90 | H100 / H200 | Multiple domains over RoCE | Blocked: v1 role specialization rejects this topology |

H100 and H200 baselines remain separate despite sharing SM90. The runner checks
actual GPU names, count, architecture and NVLink connectivity. It refuses occupied
GPUs. Each role uses DP2/TP1; FFN EP2 owns 64 of Qwen's 128 experts per rank.

## Measure

| Workload | Input tokens | Output | Concurrency | Requests/repeat |
|---|---|---|---|---|
| Decode | 32 | 64 | 1 | 32 |
| Prefill | 2,048 | 1 | 8 | 128 |
| Mixed | 32–2,048 | 64 | 32 | 256 |
| Sustained | 256 | 256 | 128 | 512 |

Metrics: output/input tokens per second, requests per second, request latency,
TTFT, TPOT and observed ITL. Report medians and P95. One-token prefill has no
TPOT. ITL uses adjacent single-token SSE events; coalesced events are reported,
not expanded into invented timings. GPU memory snapshots are total device usage
after a workload, not peak memory or registered communication capacity.

Startup, compilation, graph capture and request warmup are excluded. Every timed
graph replay, eager allocation and host-count wait is included. All variants use
`FULL_DECODE_ONLY`, identical capture sizes and scheduler limits; prefill is eager.
Prefix caching is disabled. The workload is closed-loop, not an offered-load/SLO
experiment. The common CUDA DBO import guard is applied to all three paths.

## Run and gate

Install the checkout with its pinned afdEP extra and vLLM 0.30.0. Select matching
NCCL 2.30.7 headers/runtime **before** Python starts. Configure writable DeepEP,
DeepGEMM, vLLM and afdEP caches. `MODEL_PATH` must point to the pinned Qwen
snapshot; see the model revision emitted by `plan`.

```bash
python -m tools.benchmarks.afdep_suite plan --profile h100-nvlink
python -m tools.benchmarks.afdep_suite run --profile h100-nvlink \
  --model-path "$MODEL_PATH" --output results/h100-v1 --repeats 6
python -m tools.benchmarks.afdep_suite compare \
  results/h100-v1/manifest.json --output results/h100-v1/comparison.json
```

Use `h200-nvlink` for H200. Six repeats cover all connector-order permutations;
paired variants receive the same seeded prompts. Each server pair restarts between
variants. Keep raw request artifacts and role logs with the manifest and hashes.
Future changes rerun the same matrix with fresh P2P/Masked controls; no cross-SKU,
cross-fabric, model, graph-policy or runtime comparisons are accepted as a speedup.

The manifest records initial budgets: **3%** for throughput/median latency and
**5%** for tail latency. The gate resamples paired *repeats*, not individual
requests, with a 95% bootstrap interval. Fewer than six repeats are provisional;
intervals crossing the budget are inconclusive. Neither passes the gate.
A regression exits 2; inconclusive/provisional exits 3. Profiling runs must be kept
separate and cannot create an accepted baseline.

For transport diagnosis, exclude model GEMMs and compare the actual two backends:

```bash
EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
  -m tools.benchmarks.afdep_exchange --variant afdep --payload fp8 \
  --mode graph --output exchange-afdep.json
```

Repeat with `--variant masked` and modes `eager-sync` / `eager-capacity`.
This includes a weighted identity reply; it is not pure network bandwidth.
It reports every rank, critical Attention round-trip time, registered transport
bytes and PyTorch allocation separately. P2P remains in the serving suite so its
second communication stage is preserved. Use separate Nsight Systems runs to
investigate regressions; never substitute trace timings for uninstrumented results.
