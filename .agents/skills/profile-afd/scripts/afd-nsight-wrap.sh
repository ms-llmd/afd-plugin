#!/usr/bin/env bash
# Drop-in replacement for the `vllm` executable that runs selected AFD roles
# under Nsight Systems (nsys) or Nsight Compute (ncu).
#
# Pass it to the E2E runners as the vLLM binary:
#   single host : AFD_GPU_E2E_VLLM_BIN=/path/afd-nsight-wrap.sh pytest ...
#   multi pod   : python -m tests.e2e.multi_pod.runner --vllm-bin /path/afd-nsight-wrap.sh ...
# or call it instead of `vllm` in a recipe launcher.
#
# The role is taken from the `--served-model-name` suffix the E2E runner
# writes (`<prefix>-attention`, `<prefix>-ffn`, `<prefix>-baseline`).
# Roles not listed in AFD_NSIGHT_ROLES run unwrapped.
#
# Environment (all optional except AFD_NSIGHT_TOOL):
#   AFD_NSIGHT_TOOL        nsys | ncu | none          (default: none)
#   AFD_NSIGHT_ROLES       comma list of roles to wrap (default: attention,ffn)
#   AFD_NSIGHT_OUTPUT_DIR  report directory            (default: ./nsight_reports)
#   AFD_NSIGHT_VLLM_BIN    real vLLM executable        (default: vllm)
#   AFD_NSYS_ARGS          extra/override nsys profile args (word-split)
#   AFD_NCU_ARGS           extra/override ncu args (word-split)
#
# The wrapper is NOT exec'd into the tool: it stays the process-group leader
# the runner holds a Popen handle on, and waits until no live (non-zombie)
# process remains in its group. Otherwise an nsys that detaches after a
# `--duration` window would look to the runner like the role exiting.
set -uo pipefail

tool="${AFD_NSIGHT_TOOL:-none}"
roles=",${AFD_NSIGHT_ROLES:-attention,ffn},"
out_dir="${AFD_NSIGHT_OUTPUT_DIR:-./nsight_reports}"
vllm_bin="${AFD_NSIGHT_VLLM_BIN:-vllm}"

role="unknown"
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--served-model-name" ]]; then
        role="${arg##*-}"
    fi
    case "$arg" in
        --served-model-name=*) role="${arg##*-}" ;;
    esac
    prev="$arg"
done

if [[ "$tool" == "none" || "$roles" != *",${role},"* ]]; then
    exec "$vllm_bin" "$@"
fi

mkdir -p "$out_dir"
pod="${JOB_COMPLETION_INDEX:-${AFD_E2E_POD_INDEX:-0}}"
stem="${out_dir}/$(hostname)-pod${pod}-${role}-$$"

# nsys/ncu inject into every child; spawn gives each worker a clean exec.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

case "$tool" in
    nsys)
        # shellcheck disable=SC2206
        extra=(${AFD_NSYS_ARGS:-})
        # shellcheck disable=SC2054  # commas are nsys's own list syntax
        cmd=(nsys profile
            --trace=cuda,nvtx
            --cuda-graph-trace=node
            --trace-fork-before-exec=true
            --sample=none --cpuctxsw=none
            --force-overwrite=true
            --output="${stem}"
            "${extra[@]}"
            "$vllm_bin" "$@")
        ;;
    ncu)
        # shellcheck disable=SC2206
        extra=(${AFD_NCU_ARGS:-})
        cmd=(ncu
            --target-processes all
            --graph-profiling node
            --force-overwrite
            --export "${stem}"
            "${extra[@]}"
            "$vllm_bin" "$@")
        ;;
    *)
        echo "afd-nsight-wrap: unknown AFD_NSIGHT_TOOL=${tool}" >&2
        exit 2
        ;;
esac

echo "afd-nsight-wrap: role=${role} pid=$$ report=${stem} cmd=${cmd[*]}" >&2

"${cmd[@]}" &
tool_pid=$!
trap 'kill -TERM "$tool_pid" 2>/dev/null' TERM INT
wait "$tool_pid"
rc=$?

# The tool may have exited while the served app keeps running (nsys
# `--duration ... --kill none`). Hold the group leader until every other
# member of this process group is gone or a zombie.
group_has_live_member() {
    local stat pid rest state pgrp
    [[ -d /proc/self ]] || return 1
    for stat in /proc/[0-9]*/stat; do
        { read -r pid rest < "$stat"; } 2>/dev/null || continue
        [[ "$pid" == "$$" ]] && continue
        rest="${rest##*) }"
        read -r state _ pgrp _ <<< "$rest"
        if [[ "$pgrp" == "$$" && "$state" != "Z" ]]; then
            return 0
        fi
    done
    return 1
}
while group_has_live_member; do
    sleep 2
done
exit "$rc"
