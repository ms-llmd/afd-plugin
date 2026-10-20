# 5. Report back

Tell the caller: the endpoint `http://${DEPLOY_NAME}-service:${CLIENT_PORT}`,
`MODEL_ID` (and whether it's a HF repo id or an in-container path),
`PVC_NAME`, `GPU_COUNT` (per role, if the plan spans pods), the recipe deployed,
its `--max-num-batched-tokens`, and the node(s) it landed on:

```bash
kubectl get pods -l app=afd-recipe -o wide
kubectl get sts -l app=afd-recipe
```

Every node matters if a follow-up pod needs to attach the same
`ReadWriteOnce` PVC -- it can only attach from that node.

# 6. Teardown

Left running by design, so weights stay warm -- report what's up and how to
remove it; do not delete unless asked:

```bash
kubectl delete sts <sts_name...>
kubectl delete svc "${DEPLOY_NAME}-recipe-sts" "${DEPLOY_NAME}-service" \
  "${DEPLOY_NAME}-ffn-p2p-service" "${DEPLOY_NAME}-attn-dp-service" \
  "${DEPLOY_NAME}-ffn-dp-service" --ignore-not-found
kubectl delete configmap "${DEPLOY_NAME}-recipe-script" --ignore-not-found
```

The model PVC is intentionally not listed -- deleting it discards the warm
cache and forces a full re-download.
