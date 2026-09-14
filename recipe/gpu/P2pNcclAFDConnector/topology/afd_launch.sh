#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Expand one scenario from `topologies.tsv` into per-process launches of
# `afd_serve.sh`. The table is the only place a topology is written down; every
# derived value -- A, F, --data-parallel-size, --data-parallel-start-rank,
# --headless, AFD_HOST, the per-role DP address -- is computed here, so the
# pieces cannot drift out of agreement.
#
# Usage:
#   afd_launch.sh --list
#   afd_launch.sh <scenario> --plan
#   afd_launch.sh <scenario> [--host SLOT] [--dry-run]
#
#   --list       print the scenarios defined in the table
#   --plan       print the resolved deployment and exit without launching
#   --host SLOT  launch only the processes placed on that host slot. Required
#                for multi-host scenarios; optional for single-host ones.
#   --dry-run    print each vllm command instead of running it
#
# Environment:
#   HOSTS       slot-to-address map, e.g. HOSTS="h0=10.0.0.1 h1=10.0.0.2".
#               Unmapped slots resolve to 127.0.0.1.
#   TABLE       path to the topology table (default: topologies.tsv beside this
#               script)
#   LOG_DIR     directory for per-process logs (default: current directory)
#   Everything afd_serve.sh accepts -- MODEL_PATH, MODE, DBO, MAX_NUM_SEQS,
#   NIC_NAME, AFD_PORT, AFD_VLLM_LAUNCHER, ... -- is passed straight through.
#
# Multi-host scenarios: run the SAME command on every host, changing only
# --host. The AFD world rendezvous times out after 2 minutes, so start the hosts
# close together, and start the host holding FFN rank 0 first.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TABLE=${TABLE:-$SCRIPT_DIR/topologies.tsv}
LOG_DIR=${LOG_DIR:-.}
HOSTS=${HOSTS:-}

die() { echo "afd_launch: $*" >&2; exit 2; }

[ -f "$TABLE" ] || die "topology table not found: $TABLE"

list_scenarios() {
    awk '!/^[[:space:]]*#/ && NF >= 7 { if (!seen[$1]++) print $1 }' "$TABLE"
}

SCENARIO=""
PLAN_ONLY=0
HOST_FILTER=""
DRY_RUN_FLAG=${DRY_RUN:-0}

while [ $# -gt 0 ]; do
    case "$1" in
        --list)
            list_scenarios
            exit 0
            ;;
        --plan)
            PLAN_ONLY=1
            shift
            ;;
        --host)
            HOST_FILTER=${2:?--host needs a slot name}
            shift 2
            ;;
        --dry-run)
            DRY_RUN_FLAG=1
            shift
            ;;
        -h|--help)
            sed -n '5,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            die "unknown option $1"
            ;;
        *)
            [ -z "$SCENARIO" ] || die "only one scenario may be given, got $SCENARIO and $1"
            SCENARIO=$1
            shift
            ;;
    esac
done

[ -n "$SCENARIO" ] || die "no scenario given. Try --list"

# --- Read the scenario's rows ------------------------------------------------

PROCS=()
ROLES=()
SLOTS=()
DP_LOCALS=()
TPS=()
GPU_SETS=()

while read -r proc role slot dp_local tp gpus; do
    PROCS+=("$proc")
    ROLES+=("$role")
    SLOTS+=("$slot")
    DP_LOCALS+=("$dp_local")
    TPS+=("$tp")
    GPU_SETS+=("$gpus")
done < <(awk -v scenario="$SCENARIO" \
    '!/^[[:space:]]*#/ && NF >= 7 && $1 == scenario { print $2, $3, $4, $5, $6, $7 }' "$TABLE")

ROW_COUNT=${#PROCS[@]}
if (( ROW_COUNT == 0 )); then
    echo "afd_launch: no rows for scenario '$SCENARIO'. Known scenarios:" >&2
    list_scenarios >&2
    exit 2
fi

# --- Derive DP sizes and start ranks, in row order ---------------------------
# Row order defines rank order: the first row of a role owns DP rank 0.

DP_STARTS=()
attn_running=0
ffn_running=0
attn_tp=""
ffn_tp=""

for (( i = 0; i < ROW_COUNT; i++ )); do
    case "${ROLES[$i]}" in
        attention)
            DP_STARTS+=("$attn_running")
            attn_running=$(( attn_running + DP_LOCALS[i] ))
            if [ -z "$attn_tp" ]; then
                attn_tp=${TPS[$i]}
            elif [ "$attn_tp" != "${TPS[$i]}" ]; then
                die "scenario $SCENARIO: attention rows disagree on tp ($attn_tp vs ${TPS[$i]}); vLLM requires one tensor-parallel size per role"
            fi
            ;;
        ffn)
            DP_STARTS+=("$ffn_running")
            ffn_running=$(( ffn_running + DP_LOCALS[i] ))
            if [ -z "$ffn_tp" ]; then
                ffn_tp=${TPS[$i]}
            elif [ "$ffn_tp" != "${TPS[$i]}" ]; then
                die "scenario $SCENARIO: ffn rows disagree on tp ($ffn_tp vs ${TPS[$i]}); vLLM requires one tensor-parallel size per role"
            fi
            ;;
        *)
            die "scenario $SCENARIO: row ${PROCS[$i]} has unknown role '${ROLES[$i]}'"
            ;;
    esac
done

(( attn_running > 0 )) || die "scenario $SCENARIO has no attention rows"
(( ffn_running > 0 )) || die "scenario $SCENARIO has no ffn rows"

DP_SIZE_ATTN=$attn_running
DP_SIZE_FFN=$ffn_running
NUM_ATTENTION_RANKS=$(( DP_SIZE_ATTN * attn_tp ))
NUM_FFN_RANKS=$(( DP_SIZE_FFN * ffn_tp ))

(( NUM_ATTENTION_RANKS >= NUM_FFN_RANKS )) ||
    die "scenario $SCENARIO gives A=$NUM_ATTENTION_RANKS F=$NUM_FFN_RANKS; P2pNcclAFDConnector requires A >= F"
(( NUM_ATTENTION_RANKS % NUM_FFN_RANKS == 0 )) ||
    die "scenario $SCENARIO gives A=$NUM_ATTENTION_RANKS F=$NUM_FFN_RANKS; P2pNcclAFDConnector requires A % F == 0"

# --- Resolve host slots to addresses -----------------------------------------

resolve_slot() {
    local slot=$1 entry
    for entry in $HOSTS; do
        case "$entry" in
            "$slot"=*) printf '%s' "${entry#*=}"; return 0 ;;
        esac
    done
    printf '127.0.0.1'
}

# AFD_HOST is the address of FFN rank 0's process; each role's DP address is the
# address of that role's DP rank 0 process.
AFD_HOST_SLOT=""
DP_SLOT_ATTN=""
DP_SLOT_FFN=""
for (( i = 0; i < ROW_COUNT; i++ )); do
    if [ "${DP_STARTS[$i]}" = "0" ]; then
        if [ "${ROLES[$i]}" = "ffn" ]; then
            AFD_HOST_SLOT=${SLOTS[$i]}
            DP_SLOT_FFN=${SLOTS[$i]}
        else
            DP_SLOT_ATTN=${SLOTS[$i]}
        fi
    fi
done

AFD_HOST_RESOLVED=$(resolve_slot "$AFD_HOST_SLOT")
DP_ADDRESS_ATTN=$(resolve_slot "$DP_SLOT_ATTN")
DP_ADDRESS_FFN=$(resolve_slot "$DP_SLOT_FFN")

# --- Plan --------------------------------------------------------------------

SLOT_LIST=$(printf '%s\n' "${SLOTS[@]}" | sort -u | tr '\n' ' ')
SLOT_COUNT=$(printf '%s\n' "${SLOTS[@]}" | sort -u | wc -l | tr -d ' ')

echo "afd_launch: scenario $SCENARIO"
echo "afd_launch:   A=$NUM_ATTENTION_RANKS (dp $DP_SIZE_ATTN x tp $attn_tp), F=$NUM_FFN_RANKS (dp $DP_SIZE_FFN x tp $ffn_tp), ratio $(( NUM_ATTENTION_RANKS / NUM_FFN_RANKS ))"
echo "afd_launch:   host slots: $SLOT_LIST"
echo "afd_launch:   AFD_HOST = $AFD_HOST_RESOLVED (slot $AFD_HOST_SLOT, owns FFN rank 0)"
echo "afd_launch:   DP address: attention $DP_ADDRESS_ATTN (slot $DP_SLOT_ATTN), ffn $DP_ADDRESS_FFN (slot $DP_SLOT_FFN)"
if [ "$AFD_HOST_RESOLVED" = "127.0.0.1" ] && (( SLOT_COUNT > 1 )); then
    echo "afd_launch:   WARNING slot $AFD_HOST_SLOT is unmapped and resolved to 127.0.0.1, which no other host can reach." >&2
    echo "afd_launch:   WARNING Set HOSTS=\"$AFD_HOST_SLOT=<address> ...\" for a multi-host scenario." >&2
fi
printf 'afd_launch:   %-8s %-10s %-6s %-9s %-4s %s\n' proc role slot dp_ranks tp gpus
for (( i = 0; i < ROW_COUNT; i++ )); do
    first=${DP_STARTS[$i]}
    last=$(( first + DP_LOCALS[i] - 1 ))
    headless=""
    (( first > 0 )) && headless=" (headless)"
    printf 'afd_launch:   %-8s %-10s %-6s %-9s %-4s %s%s\n' \
        "${PROCS[$i]}" "${ROLES[$i]}" "${SLOTS[$i]}" "$first-$last" "${TPS[$i]}" "${GPU_SETS[$i]}" "$headless"
done

# --- Select the rows to launch on this host ----------------------------------

if [ -z "$HOST_FILTER" ] && (( SLOT_COUNT > 1 )) && (( PLAN_ONLY == 0 )); then
    die "scenario $SCENARIO spans $SLOT_COUNT host slots ($SLOT_LIST). Pass --host <slot> to say which one this is, or --plan to only print the layout."
fi

if (( PLAN_ONLY == 1 )); then
    exit 0
fi

SELECTED=()
for (( i = 0; i < ROW_COUNT; i++ )); do
    if [ -z "$HOST_FILTER" ] || [ "$HOST_FILTER" = "${SLOTS[$i]}" ]; then
        SELECTED+=("$i")
    fi
done

if (( ${#SELECTED[@]} == 0 )); then
    die "no processes placed on host slot '$HOST_FILTER' in scenario $SCENARIO (slots: $SLOT_LIST)"
fi

mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    local pid
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for i in "${SELECTED[@]}"; do
    if [ "${ROLES[$i]}" = "attention" ]; then
        dp_size=$DP_SIZE_ATTN
        dp_address=$DP_ADDRESS_ATTN
    else
        dp_size=$DP_SIZE_FFN
        dp_address=$DP_ADDRESS_FFN
    fi
    log="$LOG_DIR/${SCENARIO}.${PROCS[$i]}.log"
    echo "afd_launch: starting ${PROCS[$i]} (${ROLES[$i]}) -> $log"
    env \
        AFD_ROLE="${ROLES[$i]}" \
        AFD_HOST="$AFD_HOST_RESOLVED" \
        NUM_ATTENTION_RANKS="$NUM_ATTENTION_RANKS" \
        NUM_FFN_RANKS="$NUM_FFN_RANKS" \
        DP_SIZE="$dp_size" \
        TP_SIZE="${TPS[$i]}" \
        DP_LOCAL="${DP_LOCALS[$i]}" \
        DP_START_RANK="${DP_STARTS[$i]}" \
        DP_ADDRESS="$dp_address" \
        GPUS="${GPU_SETS[$i]}" \
        DRY_RUN="$DRY_RUN_FLAG" \
        bash "$SCRIPT_DIR/afd_serve.sh" > "$log" 2>&1 &
    PIDS+=("$!")
done

if [ "$DRY_RUN_FLAG" = "1" ]; then
    wait
    for i in "${SELECTED[@]}"; do
        echo "--- ${PROCS[$i]} ---"
        cat "$LOG_DIR/${SCENARIO}.${PROCS[$i]}.log"
    done
    exit 0
fi

echo "afd_launch: ${#PIDS[@]} process(es) started; tail the logs in $LOG_DIR"
wait
