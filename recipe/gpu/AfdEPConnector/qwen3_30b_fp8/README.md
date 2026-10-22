# afdEP · DeepEP transport backend

![Collocated serving comparison](serving-h100.svg)

`AfdEPConnector` keeps the DeepEP M2N scheduler, Attention-side routing and
DeepGEMM runner, and loads communication through the standalone
[afdEP library](https://github.com/ms-llmd/afdEP).
This backend retains Masked DeepEP's union-rank protocol and dummy expert slots.
These results do not measure the earlier directional afdEP kernels.

## H100 experiment

| Setting | Measured configuration |
| --- | --- |
| Topology | One NVLink domain, four H100 80 GB; 2 Attention + 2 FFN |
| Model | Qwen3-30B-A3B-FP8, [pinned revision](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/tree/d206ba732169f29bb77fbf80fc2c4b81d4d30782); 64 real experts per FFN |
| Runtime | vLLM 0.30.0, PyTorch 2.13.0+cu130, NCCL 2.30.7 |
| Parallelism | DP2 / TP1 per role, FFN EP2, no DBO |
| Graph policy | Eager prefill; decode graphs at 1 / 8 / 32 / 128 tokens |
| Scheduling | Max 128 sequences, 2,048 batched tokens, model length 4,096 |
| Measurement | 3 repeats per path/workload, 8 warmup requests per repeat |

[Results and source hashes](serving-h100.json) cover 27 runs, 3,744 requests
and 678,240 observed token intervals, with no coalesced events. afdEP closely
matches Masked DeepEP: median ITL is 6.21 / 10.38 / 12.61 ms across the three
workloads. Both improve median ITL and throughput over P2P here, while TTFT
is worse, particularly under sustained load. Mixed-workload P95 ITL is also worse.

This compares natural serving paths: DeepEP/afdEP send FP8 plus scales and
route on Attention; P2P sends BF16 and routes on FFN. Expert integration differs.
Concurrent generated sequences are not all identical, including between
DeepEP and afdEP. This is a performance experiment, not an accuracy-equivalence
or transport-only claim. Startup, compilation and capture are excluded;
every timed graph replay is included. Whiskers show run ranges, not confidence intervals.

## Run

Install this checkout with `uv sync --group dev --extra vllm --extra afdep`.
Provide matching NCCL headers/runtime with device API support through
`AFDEP_NCCL_ROOT` and `LD_LIBRARY_PATH`, plus CUDA compilation tools.
Set `MODEL_PATH` to the local model snapshot. In two terminals on the same
four-GPU host, start FFN first, then Attention:

```bash
ROLE=ffn bash recipe/gpu/AfdEPConnector/qwen3_30b_fp8/serve.sh
ROLE=attention bash recipe/gpu/AfdEPConnector/qwen3_30b_fp8/serve.sh
```

Set `CONNECTOR=DeepEPM2NAFDConnector` or `CONNECTOR=P2pNcclAFDConnector` on
both roles for each baseline. Stop both servers before changing connector.
`AfdEPConnector` runs afdEP's AFD role path; `ROLE_AWARE=0` runs its Masked
DeepEP protocol on the same build. `DBO=1` enables two micro-batches, one
connector lane each; set it on both roles. `EXPERT_NUM_SMS` sizes the FFN's
DeepGEMM grid (even, on the FFN role).

### DBO: size the expert GEMM to the SMs transport leaves free

DeepGEMM splits its tiles statically over a persistent grid of every SM. Under
DBO the other lane's receive kernel waits on the GPU while a GEMM runs, so an
overlapping GEMM leaves one CTA without an SM and finishes a share late (1.4x
on afdEP, about 1.8x on DeepEP M2N, same graph node overlapped vs alone).
afdEP's role-path receive holds one SM while it waits, DeepEP M2N's holds 24.
4 x H200, this recipe with `DBO=1`, 3 runs per cell, ITL median in ms:

| Connector, `EXPERT_NUM_SMS` | Short | Sustained |
| --- | --- | --- |
| `DeepEPM2NAFDConnector`, all (132) | 6.09 | 12.13 |
| `DeepEPM2NAFDConnector`, 108 | 6.20 | 11.52 |
| `AfdEPConnector`, all (132) | 5.66 | 11.54 |
| `AfdEPConnector`, 130 | 5.65 | 10.90 |

Under DBO with this recipe, decode at 60 requests per Attention engine
(concurrency 120) falls back from CUDA graphs to eager steps at about 78 ms per
token on every connector; 64 to 68 per engine replay graphs. Runs in that mode
are excluded above.
The single-node network preset selects local transport; these runs do not
exercise NCCL GIN across nodes. `cpu_sync_min_tokens=0` enables exact receive
counts via CPU synchronization during eager prefill; captured decode uses GPU counts.

After both servers are ready, run the client for each row with
`--warmup 8 --model qwen3-30b-afdep --url http://127.0.0.1:18305`:

| Workload | `--input-tokens` | `--output-tokens` | `--concurrency` | `--requests` |
| --- | --- | --- | --- | --- |
| Short | `32` | `64` | `1` | `32` |
| Mixed | `32 128 512 1024 1536 2048` | `64` | `32` | `128` |
| Sustained | `256` | `256` | `128` | `256` |

Use `python tools/benchmarks/bench_afdep.py ... --output result.json`.
For repeats 0–2, set `--seed` to `9242 + 1000 × output_tokens + 100 × concurrency + repeat`.
The client records token IDs and receipt timestamps; it reports unresolved
coalesced intervals rather than inventing individual token timings.
