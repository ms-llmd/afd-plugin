# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared process-group cleanup for E2E subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

PROC_ROOT = Path("/proc")
# Offsets into ``/proc/<pid>/stat`` counted from the ``state`` field, which is
# the first field after the parenthesised command name (see proc(5)).
STAT_STATE_OFFSET = 0
STAT_PARENT_PID_OFFSET = 1
STAT_PROCESS_GROUP_OFFSET = 2
STAT_THREAD_COUNT_OFFSET = 17
UNKNOWN_PROC_FIELD = "?"
# A process in uninterruptible sleep cannot be reaped by SIGKILL until it
# leaves the kernel call it is blocked in; a zombie is already dead but still
# occupies the process table. Both keep ``killpg(pgid, 0)`` succeeding.
UNINTERRUPTIBLE_STATE = "D"
ZOMBIE_STATE = "Z"


@dataclass(frozen=True)
class ProcessGroupMember:
    """One ``/proc`` entry still attached to a process group during teardown."""

    pid: int
    parent_pid: int
    process_group: int
    state: str
    command: str
    wait_channel: str
    thread_count: int


def read_process_group_member(
    pid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> ProcessGroupMember | None:
    """Read one process's teardown-relevant ``/proc`` fields, or None if gone."""
    process_root = proc_root / str(pid)
    try:
        stat_text = (process_root / "stat").read_text()
    except OSError:
        return None
    command_start = stat_text.find("(")
    command_end = stat_text.rfind(")")
    if command_start < 0 or command_end < command_start:
        return None
    fields = stat_text[command_end + 2 :].split()
    if len(fields) <= STAT_THREAD_COUNT_OFFSET:
        return None
    # wchan names the kernel function a blocked process is waiting in, which is
    # what distinguishes a driver hang from an ordinary slow exit.
    try:
        wait_channel = (process_root / "wchan").read_text().strip()
    except OSError:
        wait_channel = ""
    return ProcessGroupMember(
        pid=pid,
        parent_pid=int(fields[STAT_PARENT_PID_OFFSET]),
        process_group=int(fields[STAT_PROCESS_GROUP_OFFSET]),
        state=fields[STAT_STATE_OFFSET],
        command=stat_text[command_start + 1 : command_end],
        wait_channel=wait_channel or UNKNOWN_PROC_FIELD,
        thread_count=int(fields[STAT_THREAD_COUNT_OFFSET]),
    )


def describe_process(
    pid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> str:
    """Summarise one surviving process for a teardown failure message."""
    member = read_process_group_member(pid, proc_root=proc_root)
    if member is None:
        return "no longer present at description time"
    return (
        f"pgid={member.process_group} ppid={member.parent_pid} "
        f"state={member.state} threads={member.thread_count} "
        f"wchan={member.wait_channel} comm={member.command}"
    )


def describe_process_group(
    pgid: int,
    *,
    proc_root: Path = PROC_ROOT,
) -> str:
    """Summarise the surviving members of ``pgid`` for a teardown failure.

    Cleanup failures report only that a group is still alive, which cannot
    distinguish a real driver hang from a process that merely exited slowly.
    This renders each surviving member with the state letter and wait channel
    needed to tell those apart from the failure message alone.
    """
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        return f"could not scan {proc_root}: {exc}"

    members: list[ProcessGroupMember] = []
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        member = read_process_group_member(int(entry.name), proc_root=proc_root)
        if member is not None and member.process_group == pgid:
            members.append(member)

    if not members:
        # The group emptied between the liveness check and this description,
        # which is itself worth recording rather than reporting nothing.
        return "no surviving members found at description time"

    descriptions = [
        f"pid={member.pid} ppid={member.parent_pid} state={member.state} "
        f"threads={member.thread_count} wchan={member.wait_channel} "
        f"comm={member.command}"
        for member in sorted(members, key=lambda member: member.pid)
    ]
    stuck = sum(1 for member in members if member.state == UNINTERRUPTIBLE_STATE)
    dead = sum(1 for member in members if member.state == ZOMBIE_STATE)
    header = (
        f"{len(members)} surviving member(s), {stuck} uninterruptible, {dead} zombie"
    )
    return f"{header}: [{'; '.join(descriptions)}]"


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


def terminate_process_groups(
    processes: Sequence[subprocess.Popen[str]],
    *,
    termination_timeout_s: float,
    poll_interval_s: float,
    reap_timeout_s: float,
    process_name: str = "",
    deferred_sigkill_pgids: Collection[int] = (),
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
                    f"{termination_timeout_s}s timeout "
                    f"({describe_process_group(pgid, proc_root=proc_root)})",
                )
            forced_pgids.append(pgid)

    reap_deadline = time.monotonic() + reap_timeout_s
    for process in processes:
        try:
            process.wait(timeout=max(reap_deadline - time.monotonic(), 0))
        except Exception as exc:
            failures.append(
                f"wait failed for {process_description} {process.pid}: {exc}",
            )

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
            still_alive.append(pgid)
        surviving_pgids = still_alive
        if not surviving_pgids or time.monotonic() >= reap_deadline:
            break
        time.sleep(poll_interval_s)

    for pgid in surviving_pgids:
        failures.append(
            f"{group_description} {pgid} still alive after SIGKILL "
            f"({describe_process_group(pgid, proc_root=proc_root)})",
        )

    return failures


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
            f"{process_name} process {process.pid} still alive after SIGKILL "
            f"({describe_process(process.pid, proc_root=proc_root)})"
            for process in surviving_processes
        )
    finally:
        close_process_identities(surviving_processes)
    return failures
