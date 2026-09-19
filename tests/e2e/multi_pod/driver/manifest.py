# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Render the Kubernetes objects a multi-pod AFD E2E run needs.

Pure functions. Rendering to a file and applying it by hand must produce an
identical, complete run -- that is the check that keeps the driver a driver and
not an orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

APP_LABEL = "afd-e2e"
DEFAULT_APP_DIR = "/opt/afd-plugin"
OVERLAY_MOUNT_PATH = "/overlay"
OVERLAY_ARCHIVE_NAME = "source.tgz"
WORK_SOURCE_DIR = "/work/src"
MODEL_MOUNT_PATH = "/models"
WORK_MOUNT_PATH = "/work"
RUNNER_MODULE = "tests.e2e.multi_pod.runner"
# Indexed Job pod hostnames are "<job>-<index>"; with a matching headless
# Service every pod is reachable at "<job>-<index>.<job>".
POD_ADDRESS_TEMPLATE_SUFFIX = "-{index}"

# The /work emptyDir shadows whatever the image created there, so every
# writable path the run needs is recreated at start.
WORK_DIRECTORIES = (
    f"{WORK_MOUNT_PATH}/home",
    f"{WORK_MOUNT_PATH}/tmp",
    f"{WORK_MOUNT_PATH}/hf_modules",
)
_PRELUDE = "set -euo pipefail\nmkdir -p " + " ".join(WORK_DIRECTORIES)
PLAIN_BOOTSTRAP = f'{_PRELUDE}\nexec "$@"'
OVERLAY_BOOTSTRAP = f"""{_PRELUDE}
cp -a {DEFAULT_APP_DIR} {WORK_SOURCE_DIR}
tar xzf {OVERLAY_MOUNT_PATH}/{OVERLAY_ARCHIVE_NAME} -C {WORK_SOURCE_DIR}
cd {WORK_SOURCE_DIR}
exec "$@"
"""

# Not cluster-specific: these follow from running as an arbitrary UID with an
# emptyDir at /work. Anything that varies per cluster goes through
# --container-env, which overrides these.
DEFAULT_CONTAINER_ENV = {
    "HOME": f"{WORK_MOUNT_PATH}/home",
    "TMPDIR": f"{WORK_MOUNT_PATH}/tmp",
    # torch._inductor calls getpass.getuser() at import time and an arbitrary
    # UID has no /etc/passwd entry.
    "USER": "afd",
    "LOGNAME": "afd",
    "XDG_CACHE_HOME": f"{WORK_MOUNT_PATH}/xdg",
    "TORCHINDUCTOR_CACHE_DIR": f"{WORK_MOUNT_PATH}/inductor",
    "TRITON_CACHE_DIR": f"{WORK_MOUNT_PATH}/triton",
    "VLLM_CACHE_ROOT": f"{WORK_MOUNT_PATH}/vllm",
    "UV_CACHE_DIR": f"{WORK_MOUNT_PATH}/uv",
    # With --trust-remote-code and a model cache it cannot write, transformers
    # dies with Errno 30 before any weight load.
    "HF_MODULES_CACHE": f"{WORK_MOUNT_PATH}/hf_modules",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}


@dataclass(frozen=True)
class JobSpec:
    """Everything the manifest needs; nothing about how the test behaves."""

    name: str
    namespace: str
    image: str
    scenario: str
    pod_layout: str
    num_pods: int
    run_id: str
    model: str
    gsm8k_output_path: str
    gpus_per_pod: int
    model_pvc: str
    cpu_request: str = "16"
    cpu_limit: str = "32"
    memory_request: str = "128Gi"
    memory_limit: str = "200Gi"
    shm_size: str = "16Gi"
    fs_group: int | None = None
    app_dir: str = DEFAULT_APP_DIR
    source_overlay_configmap: str | None = None
    pod_env: Mapping[str, str] = field(default_factory=dict)
    container_env: Mapping[str, str] = field(default_factory=dict)
    runner_args: Sequence[str] = ()
    spread_across_nodes: bool = False
    pack_onto_one_node: bool = False
    excluded_nodes: Sequence[str] = ()
    active_deadline_seconds: int | None = None
    image_pull_policy: str = "Always"

    @property
    def store_host(self) -> str:
        return f"{self.name}-0.{self.name}"

    @property
    def pod_address_template(self) -> str:
        return f"{self.name}{POD_ADDRESS_TEMPLATE_SUFFIX}.{self.name}"

    @property
    def labels(self) -> dict[str, str]:
        return {"app": APP_LABEL, "run": self.name}


def render(spec: JobSpec) -> list[dict]:
    """Render the headless Service and the Indexed Job, in apply order."""
    return [render_service(spec), render_job(spec)]


def render_service(spec: JobSpec) -> dict:
    """A headless Service gives every pod a stable, predictable DNS name."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": spec.name,
            "namespace": spec.namespace,
            "labels": spec.labels,
        },
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": spec.labels,
            "ports": [{"name": "rendezvous", "port": 29500}],
        },
    }


def render_job(spec: JobSpec) -> dict:
    job_spec = {
        "completionMode": "Indexed",
        "completions": spec.num_pods,
        # Partial parallelism would deadlock every barrier, so the two must
        # always be equal.
        "parallelism": spec.num_pods,
        # A test failure is a failure; a retried pod would rejoin a store whose
        # barriers have already advanced.
        "backoffLimit": 0,
        "template": render_pod_template(spec),
    }
    if spec.active_deadline_seconds is not None:
        job_spec["activeDeadlineSeconds"] = spec.active_deadline_seconds
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": spec.name,
            "namespace": spec.namespace,
            "labels": spec.labels,
        },
        "spec": job_spec,
    }


def render_pod_template(spec: JobSpec) -> dict:
    pod_spec: dict = {
        "restartPolicy": "Never",
        "subdomain": spec.name,
        "volumes": render_volumes(spec),
        "containers": [render_container(spec)],
    }
    if spec.fs_group is not None:
        pod_spec["securityContext"] = {"fsGroup": spec.fs_group}
    affinity = render_affinity(spec)
    if affinity:
        pod_spec["affinity"] = affinity
    return {"metadata": {"labels": spec.labels}, "spec": pod_spec}


def render_affinity(spec: JobSpec) -> dict:
    """Pin the physical axis. Nothing else in the design depends on it."""
    if spec.spread_across_nodes and spec.pack_onto_one_node:
        raise ValueError("pods cannot both spread across nodes and pack onto one")
    affinity: dict = {}
    if spec.pack_onto_one_node:
        affinity["podAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {"matchLabels": {"run": spec.name}},
                    "topologyKey": "kubernetes.io/hostname",
                },
            ],
        }
    if spec.spread_across_nodes:
        affinity["podAntiAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "labelSelector": {"matchLabels": {"run": spec.name}},
                    "topologyKey": "kubernetes.io/hostname",
                },
            ],
        }
    if spec.excluded_nodes:
        affinity["nodeAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "NotIn",
                                "values": list(spec.excluded_nodes),
                            },
                        ],
                    },
                ],
            },
        }
    return affinity


def render_volumes(spec: JobSpec) -> list[dict]:
    volumes: list[dict[str, Any]] = [
        {
            "name": "model-storage",
            "persistentVolumeClaim": {"claimName": spec.model_pvc},
        },
        # Per pod, deliberately: a private /dev/shm is part of what makes these
        # pods genuinely distinct hosts to vLLM.
        {
            "name": "dshm",
            "emptyDir": {"medium": "Memory", "sizeLimit": spec.shm_size},
        },
        {"name": "work", "emptyDir": {}},
    ]
    if spec.source_overlay_configmap is not None:
        volumes.append(
            {
                "name": "source-overlay",
                "configMap": {"name": spec.source_overlay_configmap},
            },
        )
    return volumes


def render_container(spec: JobSpec) -> dict:
    volume_mounts: list[dict[str, Any]] = [
        {"name": "model-storage", "mountPath": MODEL_MOUNT_PATH},
        {"name": "dshm", "mountPath": "/dev/shm"},
        {"name": "work", "mountPath": WORK_MOUNT_PATH},
    ]
    bootstrap = PLAIN_BOOTSTRAP
    working_dir = spec.app_dir
    if spec.source_overlay_configmap is not None:
        volume_mounts.append(
            {
                "name": "source-overlay",
                "mountPath": OVERLAY_MOUNT_PATH,
                "readOnly": True,
            },
        )
        bootstrap = OVERLAY_BOOTSTRAP
        working_dir = WORK_SOURCE_DIR
    return {
        "name": "e2e",
        "image": spec.image,
        "imagePullPolicy": spec.image_pull_policy,
        "workingDir": working_dir,
        "command": ["/bin/bash", "-c", bootstrap, APP_LABEL],
        "args": render_runner_args(spec),
        "env": render_env(spec),
        "resources": {
            "requests": {
                "nvidia.com/gpu": str(spec.gpus_per_pod),
                "cpu": spec.cpu_request,
                "memory": spec.memory_request,
            },
            "limits": {
                "nvidia.com/gpu": str(spec.gpus_per_pod),
                "cpu": spec.cpu_limit,
                "memory": spec.memory_limit,
            },
        },
        "volumeMounts": volume_mounts,
    }


def render_runner_args(spec: JobSpec) -> list[str]:
    """The identical argv every pod receives. Identity comes from the env."""
    args = [
        "python",
        "-m",
        RUNNER_MODULE,
        "--scenario",
        spec.scenario,
        "--pod-layout",
        spec.pod_layout,
        "--run-id",
        spec.run_id,
        "--model",
        spec.model,
        "--gsm8k-output-path",
        spec.gsm8k_output_path,
        "--store-host",
        spec.store_host,
        "--pod-address-template",
        spec.pod_address_template,
    ]
    for name, value in spec.pod_env.items():
        args.extend(["--pod-env", f"{name}={value}"])
    args.extend(spec.runner_args)
    return args


def render_env(spec: JobSpec) -> list[dict]:
    environment: list[dict[str, Any]] = [
        {
            "name": "POD_IP",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        },
    ]
    resolved = {**DEFAULT_CONTAINER_ENV, **spec.container_env}
    for name, value in resolved.items():
        environment.append({"name": name, "value": value})
    return environment
