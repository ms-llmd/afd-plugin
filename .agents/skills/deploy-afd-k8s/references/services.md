# Services

Create as step 4d of [deploy.md](deploy.md) -- **before** deploying any Pod
(step 4e), not after (idempotent -- safe to re-apply either way). The
headless Services below are what a role's head Pod *binds* its own
rendezvous/DP-RPC server to during its own startup, within seconds of the
container starting; if that Service doesn't exist yet, the bind/resolve
fails fast and, since every pod has `restartPolicy: Never`, it never gets a
second chance. Which of the four to create is derived from the plan parsed
in [resolve-recipe.md](resolve-recipe.md), never asked.

## `vllm-service` -- always

Client-facing. Selects whichever pod holds `afd-attn-node-role: head` --
true for the sole pod of a plain recipe, and for whichever pod is the
attention head of a placement plan.

```bash
CLIENT_PORT="$CLIENT_PORT" envsubst '${CLIENT_PORT}' \
  < templates/service-client.yaml | kubectl apply -f -
```

## `vllm-ffn-p2p-service` -- iff the plan has more than one pod

Internal only, carries AFD rendezvous traffic to whichever pod holds
`afd-ffn-node-role: head`. Must exist before any multi-pod deploy, since the
recipe's own `AFD_CONNECTOR_HOST` default now points at this name directly.
Exposes the control-plane port plus one derived port per FFN rank
(`[AFD_CONNECTOR_PORT, AFD_CONNECTOR_PORT + NUM_FFN_RANKS]` -- read
`AFD_CONNECTOR_PORT` off the recipe's own shell default and `NUM_FFN_RANKS`
off its `"num_ffn_ranks"` JSON config key, the *total* rank count for the
FFN role across all pods, not a per-pod count):

```bash
FFN_PORT_START="$(grep -oE 'AFD_CONNECTOR_PORT:-[0-9]+' "$RECIPE_SCRIPT_PATH" | grep -oE '[0-9]+')"
NUM_FFN_RANKS="$(grep -oE '"num_ffn_ranks":[[:space:]]*[0-9]+' "$RECIPE_SCRIPT_PATH" | grep -oE '[0-9]+$' | head -1)"
PORT_ENTRIES="$(for p in $(seq "$FFN_PORT_START" "$((FFN_PORT_START + NUM_FFN_RANKS))"); do
  printf '\n    - {name: p%s, port: %s, targetPort: %s}' "$p" "$p" "$p"
done)"
PORT_ENTRIES="$PORT_ENTRIES" envsubst '${PORT_ENTRIES}' \
  < templates/service-ffn-p2p.yaml | kubectl apply -f -
```

## `vllm-attn-dp-service` / `vllm-ffn-dp-service` -- iff that role is split

Only when **more than one** plan entry's `POD` carries the same role (i.e.
that role is split across pods, not just placed in its own single pod).
Carries that role's DP-RPC coordination traffic to its head pod. One per
split role, generic over role via `templates/service-dp-coord.yaml`:

```bash
TEMPLATE_ROLE=attn TEMPLATE_PORT=13345 \
  envsubst '${TEMPLATE_ROLE} ${TEMPLATE_PORT}' < templates/service-dp-coord.yaml | kubectl apply -f -
# symmetrically: TEMPLATE_ROLE=ffn TEMPLATE_PORT=13346, if FFN is ever split
```

All three internal Services (`vllm-ffn-p2p-service`, `vllm-attn-dp-service`,
`vllm-ffn-dp-service`) are headless (`clusterIP: None`) -- the role's head
pod *binds* its rendezvous/DP-RPC server to that name, and a normal
ClusterIP has no real interface to bind to, only to connect through. Their
templates also set `publishNotReadyAddresses: true`: headless-Service DNS
by default only lists Pods that have already passed readiness, but a head
pod resolving its own Service name to bind is inherently not-yet-ready at
that moment -- without this flag, any future readinessProbe added to
`templates/pod.yaml` would deadlock every head pod at startup.
