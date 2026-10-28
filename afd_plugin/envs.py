# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_ASYNC_CAM_LAYERED_GMM = "AFD_ASYNC_CAM_LAYERED_GMM"
# Default on; a false value restores vLLM's eager two-micro-batch step when the
# empty-micro-batch guard rejects a FULL DBO graph (gpu_mrv2_ubatching.py).
AFD_DBO_GUARD_FALLBACK = "AFD_DBO_GUARD_FALLBACK"
# Default on; a false value restores vLLM's eager step for a batch whose only
# prefills are final one-token prompt chunks (attention_model_runner_v2.py).
AFD_FINAL_PREFILL_GRAPH = "AFD_FINAL_PREFILL_GRAPH"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


def async_cam_layered_gmm_enabled() -> bool:
    return os.environ.get(AFD_ASYNC_CAM_LAYERED_GMM, "").lower() in ENV_TRUE_VALUES


def dbo_guard_fallback_enabled() -> bool:
    return os.environ.get(AFD_DBO_GUARD_FALLBACK, "").lower() not in ENV_FALSE_VALUES


def final_prefill_graph_enabled() -> bool:
    return os.environ.get(AFD_FINAL_PREFILL_GRAPH, "").lower() not in ENV_FALSE_VALUES


__all__ = [
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_ASYNC_CAM_LAYERED_GMM",
    "AFD_DBO_GUARD_FALLBACK",
    "AFD_FINAL_PREFILL_GRAPH",
    "async_cam_layered_gmm_enabled",
    "dbo_guard_fallback_enabled",
    "final_prefill_graph_enabled",
    "force_balanced_topk_ids_enabled",
]
