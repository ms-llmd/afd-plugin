# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment parsing shared by local NPU E2E entrypoints."""

import os
from pathlib import Path


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def devices_from_env(name: str, expected_count: int) -> list[str]:
    devices = [item.strip() for item in required_env(name).split(",") if item.strip()]
    if len(devices) != expected_count:
        raise RuntimeError(f"{name} must contain exactly {expected_count} devices")
    if len(devices) != len(set(devices)):
        raise RuntimeError(f"{name} devices must be unique")
    return devices


def prepend_env_paths(env: dict[str, str], name: str, *paths: Path) -> None:
    existing = [path for path in env.get(name, "").split(os.pathsep) if path]
    env[name] = os.pathsep.join(
        dict.fromkeys([*(str(path) for path in paths), *existing]),
    )
