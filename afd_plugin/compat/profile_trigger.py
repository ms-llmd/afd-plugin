# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""File-driven start/stop trigger for profiling the AFD FFN role.

The FFN ``vllm serve`` process exposes no API server and its EngineCore never
drains utility requests, so vLLM's ``/start_profile`` cannot reach it. The FFN
EngineCore busy loop polls a file named by ``AFD_FFN_PROFILE_TRIGGER_FILE``
instead, and forwards each change to the workers' native ``profile`` RPC.

File content is ``start[:<token>]`` or ``stop[:<token>]``. Commands are
edge-triggered: only a change of content acts, and the content present when
the loop starts is the baseline, so a value left over from an earlier run
never fires. Use a new token to repeat a command. A Kubernetes ConfigMap
mounted as a volume (not via ``subPath``) is updated in place by the kubelet,
so editing the ConfigMap drives a running pod.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

AFD_FFN_PROFILE_TRIGGER_FILE_ENV: Final[str] = "AFD_FFN_PROFILE_TRIGGER_FILE"
START_COMMAND: Final[str] = "start"
STOP_COMMAND: Final[str] = "stop"
TOKEN_SEPARATOR: Final[str] = ":"


class ProfileTrigger:
    """Edge-triggered profile commands read from one watched file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._last_content = self._read()
        logger.info(
            "AFD FFN profile trigger watching %s (baseline %r)",
            path,
            self._last_content,
        )

    def poll(self) -> bool | None:
        """Return True to start, False to stop, or None when nothing changed."""

        content = self._read()
        if content == self._last_content:
            return None
        self._last_content = content
        if not content:
            return None

        command = content.split(TOKEN_SEPARATOR, 1)[0].strip().lower()
        if command not in (START_COMMAND, STOP_COMMAND):
            logger.warning(
                "Ignoring AFD FFN profile trigger %r in %s; expected "
                "'start[:token]' or 'stop[:token]'",
                content,
                self.path,
            )
            return None

        # A repeated start or stop is safe to forward: vLLM's WorkerProfiler
        # ignores a start while active and a stop while inactive.
        return command == START_COMMAND

    def _read(self) -> str:
        try:
            return self.path.read_text().strip()
        except FileNotFoundError:
            return ""
        except OSError:
            logger.debug(
                "Cannot read AFD FFN profile trigger %s",
                self.path,
                exc_info=True,
            )
            return ""


def create_ffn_profile_trigger() -> ProfileTrigger | None:
    """Create the FFN trigger when ``AFD_FFN_PROFILE_TRIGGER_FILE`` is set."""

    path = os.getenv(AFD_FFN_PROFILE_TRIGGER_FILE_ENV)
    if not path:
        return None
    return ProfileTrigger(Path(path))


__all__ = [
    "AFD_FFN_PROFILE_TRIGGER_FILE_ENV",
    "ProfileTrigger",
    "create_ffn_profile_trigger",
]
