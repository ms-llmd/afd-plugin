#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Run inside the spike pod from /work/spike. Usage:
#   run_in_pod.sh env                      # toolchain / NCCL probe
#   run_in_pod.sh build                    # JIT-build the extension once
#   run_in_pod.sh test <num_ffn> <nproc> [extra test_m2n.py args]
set -euo pipefail
cd /work/spike
mkdir -p /work/results

case "${1:-}" in
  env)
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    nvidia-smi topo -m > /work/results/topo.txt || true
    ls /dev/infiniband 2>/dev/null || echo "no /dev/infiniband"
    which nvcc || echo "nvcc not on PATH"
    ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null || true
    python3 -c 'import torch,importlib.metadata as m; print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version()); print([ (d.metadata["Name"], d.version) for d in m.distributions() if "nccl" in d.metadata["Name"].lower() or "cuda-nvcc" in d.metadata["Name"].lower() or d.metadata["Name"]=="vllm"])'
    ;;
  build)
    time python3 -c 'from afd_deepep.build import build_extension; print(build_extension())'
    ;;
  test)
    num_ffn=$2; nproc=$3; shift 3
    tag="${num_ffn}f$((nproc - num_ffn))a"
    timeout 1200 torchrun --standalone --nproc-per-node "$nproc" test_m2n.py \
      --num-ffn "$num_ffn" --out "/work/results/${tag}.json" "$@" 2>&1 \
      | tee "/work/results/${tag}.log"
    ;;
  *)
    echo "usage: $0 env|build|test <num_ffn> <nproc> [args]" >&2
    exit 2
    ;;
esac
