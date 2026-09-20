# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared process-group cleanup for E2E subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

PROC_ROOT = Path("/proc")


@dataclass(frozen=True)
class ProcessIdentity:
    """A PID plus a handle that remains bound to the same Linux process."""

    pid: int
    pidfd: int


def close_process_identities(processes: Sequence[ProcessIdentity]) -> None:
    """Close pidfds returned by ``find_processes_matching_environment``."""
    for process in processes:
        # Closing an owned pidfd is best effort during process teardown.
        with suppress(OSError):
            os.close(process.pidfd)


def reap_orphan_descendants() -> int:
    """Reap every already-exited descendant, returning how many were reaped.

    Only safe once the caller has waited on its own ``Popen`` children: a bare
    ``waitpid(-1)`` would otherwise consume an exit status ``Popen.wait`` still
    needs.
    """
    reaped = 0
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        if pid == 0:
            return reaped
        reaped += 1


def terminate_process_groups(
    processes: Sequence[subprocess.Popen[str]],
    *,
    termination_timeout_s: float,
    poll_interval_s: float,
    reap_timeout_s: float,
    process_name: str = "",
    deferred_sigkill_pgids: Collection[int] = (),
    reap_orphans: Callable[[], int] | None = None,
    proc_root: Path = PROC_ROOT,
) -> list[str]:
    """Terminate process groups with one deadline and reap their leaders.

    ``deferred_sigkill_pgids`` identifies process groups whose successful
    SIGKILL escalation and liveness verification are owned by caller-specific
    cleanup. Signal-delivery and process-reaping failures are always reported.
    """
    failures: list[str] = []
    live_pgids: list[int] = []
    group_description = f"{process_name} process group".strip()
    process_description = f"{process_name} process".strip()

    for process in processes:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"SIGTERM failed for {group_description} {process.pid}: {exc}",
                )
        except OSError as exc:
            failures.append(
                f"SIGTERM failed for {process_description} {process.pid}: {exc}",
            )
        live_pgids.append(process.pid)

    deadline = time.monotonic() + termination_timeout_s
    leaders_to_poll = list(processes)
    while live_pgids:
        unreaped_leaders: list[subprocess.Popen[str]] = []
        for process in leaders_to_poll:
            try:
                returncode = process.poll()
            except Exception as exc:
                failures.append(
                    f"poll failed for {process_description} {process.pid}: {exc}",
                )
                continue
            if returncode is None:
                unreaped_leaders.append(process)
        leaders_to_poll = unreaped_leaders

        surviving_pgids: list[int] = []
        for pgid in live_pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"liveness check failed for {group_description} {pgid}: {exc}",
                )
            if group_is_spent(pgid, proc_root=proc_root):
                continue
            surviving_pgids.append(pgid)
        live_pgids = surviving_pgids
        if not live_pgids or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    forced_pgids: list[int] = []
    for pgid in live_pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError as exc:
            failures.append(
                f"SIGKILL failed for {group_description} {pgid}: {exc}",
            )
            forced_pgids.append(pgid)
        else:
            if pgid not in deferred_sigkill_pgids:
                failures.append(
                    f"forced SIGKILL for {group_description} {pgid} after "
                    f"{termination_timeout_s}s timeout",
                )
            forced_pgids.append(pgid)

    # A caller that is PID 1 must reap orphaned descendants before the liveness
    # check below: an unreaped zombie still answers ``killpg(pgid, 0)``, so it
    # would be reported as a survivor although it is already dead.
    reap_deadline = time.monotonic() + reap_timeout_s
    for process in processes:
        try:
            process.wait(timeout=max(reap_deadline - time.monotonic(), 0))
        except Exception as exc:
            failures.append(
                f"wait failed for {process_description} {process.pid}: {exc}",
            )

    if reap_orphans is not None:
        reap_orphans()

    surviving_pgids = [
        pgid for pgid in forced_pgids if pgid not in deferred_sigkill_pgids
    ]
    while surviving_pgids:
        still_alive: list[int] = []
        for pgid in surviving_pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"post-SIGKILL liveness check failed for "
                    f"{group_description} {pgid}: {exc}",
                )
            if group_is_spent(pgid, proc_root=proc_root):
                continue
            still_alive.append(pgid)
        surviving_pgids = still_alive
        if not surviving_pgids or time.monotonic() >= reap_deadline:
            break
        time.sleep(poll_interval_s)
        if reap_orphans is not None:
            reap_orphans()

    for pgid in surviving_pgids:
        failures.append(
            f"{group_description} {pgid} still alive after SIGKILL",
        )

    return failures


@dataclass(frozen=True)
class ProcessGroupMember:
    """One process seen in a process group, as /proc reports it."""

    pid: int
    command: str
    state: str
    parent_pid: int

    @property
    def is_zombie(self) -> bool:
        """Whether this process has exited and is only awaiting a reap."""
        return self.state == "Z"

    def __str__(self) -> str:
        return (
            f"pid {self.pid} ({self.command}) state={self.state} ppid={self.parent_pid}"
        )


def process_group_members(
    pgid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> list[ProcessGroupMember]:
    """Return the processes currently in ``pgid``, empty if none can be read."""
    members: list[ProcessGroupMember] = []
    try:
        entries = sorted(proc_root.iterdir(), key=lambda path: path.name)
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # "pid (comm) state ppid pgrp ..."; comm may itself contain spaces and
        # parentheses, so split after its final ')'.
        close = stat.rfind(")")
        if close == -1:
            continue
        fields = stat[close + 2 :].split()
        if len(fields) < 3 or fields[2] != str(pgid):
            continue
        members.append(
            ProcessGroupMember(
                pid=int(entry.name),
                command=stat[stat.find("(") + 1 : close],
                state=fields[0],
                parent_pid=int(fields[1]),
            ),
        )
    return members


def describe_process_group(
    pgid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> list[str]:
    """Describe the members of a process group, for teardown diagnostics.

    The state letter distinguishes the two ways a group can appear to survive
    SIGKILL: ``Z`` means already dead and merely unreaped, ``D`` means an
    uninterruptible kernel wait that SIGKILL genuinely cannot interrupt.
    """
    return [str(member) for member in process_group_members(pgid, proc_root=proc_root)]


def group_is_spent(pgid: int, *, proc_root: Path = PROC_ROOT) -> bool:
    """Whether every remaining member of ``pgid`` is an unreaped zombie.

    A zombie still answers ``killpg(pgid, 0)``, so without this a process group
    whose members have all exited is reported as surviving teardown. Returns
    False when the group cannot be inspected at all, so an uninspectable group
    is still reported rather than silently assumed dead.
    """
    members = process_group_members(pgid, proc_root=proc_root)
    if not members:
        return False
    return all(member.is_zombie for member in members)


def find_processes_matching_environment(
    required_environment: Mapping[str, str],
    *,
    proc_root: Path = PROC_ROOT,
) -> list[ProcessIdentity]:
    """Open stable handles for matching processes for the caller to close."""
    if not required_environment:
        raise ValueError("required_environment must not be empty")

    required_entries = {
        f"{name}={value}".encode() for name, value in required_environment.items()
    }
    try:
        process_entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(
            f"could not scan process directory {proc_root}: {exc}",
        ) from exc

    matching_processes: list[ProcessIdentity] = []
    for process_entry in process_entries:
        if not process_entry.name.isdecimal():
            continue
        pid = int(process_entry.name)
        try:
            # Open the pidfd before reading environ so a later PID reuse cannot
            # redirect the handle used for signaling to another process.
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        except OSError as exc:
            close_process_identities(matching_processes)
            raise RuntimeError(
                f"could not open pidfd for process {pid}: {exc}"
            ) from exc
        process = ProcessIdentity(pid=pid, pidfd=pidfd)
        try:
            environment = (process_entry / "environ").read_bytes()
        except OSError:
            # Processes may exit or be inaccessible while /proc is scanned.
            close_process_identities((process,))
            continue
        if required_entries.issubset(environment.split(b"\0")):
            matching_processes.append(process)
        else:
            close_process_identities((process,))
    return sorted(matching_processes, key=lambda process: process.pid)


def kill_processes_matching_environment(
    required_environment: Mapping[str, str],
    *,
    timeout_s: float,
    poll_interval_s: float,
    process_name: str,
    proc_root: Path = PROC_ROOT,
) -> list[str]:
    """SIGKILL matching processes until none remain or the deadline expires."""
    failures: list[str] = []
    reported_signal_failures: set[int] = set()
    deadline = time.monotonic() + timeout_s

    while matching_processes := find_processes_matching_environment(
        required_environment,
        proc_root=proc_root,
    ):
        try:
            for process in matching_processes:
                try:
                    signal.pidfd_send_signal(process.pidfd, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except OSError as exc:
                    if process.pid not in reported_signal_failures:
                        failures.append(
                            f"SIGKILL failed for {process_name} process "
                            f"{process.pid}: {exc}",
                        )
                        reported_signal_failures.add(process.pid)
        finally:
            close_process_identities(matching_processes)

        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    surviving_processes = find_processes_matching_environment(
        required_environment,
        proc_root=proc_root,
    )
    try:
        failures.extend(
            f"{process_name} process {process.pid} still alive after SIGKILL"
            for process in surviving_processes
        )
    finally:
        close_process_identities(surviving_processes)
    return failures
