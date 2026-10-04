# 5. Report back

Tell the caller: the endpoint `http://vllm-service:${CLIENT_PORT}`,
`MODEL_ID` (and whether it's a HF repo id or an in-container path),
`PVC_NAME`, `GPU_COUNT` (per pod, if more than one), the recipe deployed,
its `--max-num-batched-tokens`, and the node(s) it landed on:

```bash
kubectl get pod <pod_name...> -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}'
```

Every node matters if a follow-up pod needs to attach the same
`ReadWriteOnce` PVC -- it can only attach from that node.

If the plan spans more than one pod, repeat the experimental caveat from
[resolve-recipe.md](resolve-recipe.md): correct AFD/DP flags and port
exposure, but cross-pod `P2pNcclAFDConnector` throughput is not
upstream-validated.

# 6. Teardown

Left running by design, so weights stay warm -- report what's up and how to
remove it; do not delete unless asked:

```bash
kubectl delete pod <pod_name...>
kubectl delete svc vllm-service vllm-ffn-p2p-service vllm-attn-dp-service vllm-ffn-dp-service --ignore-not-found
```

The model PVC is intentionally not listed -- deleting it discards the warm
cache and forces a full re-download.
