# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import signal

from tests.e2e import process_utils


def test_terminate_process_groups_signals_the_leader_before_the_group(
    monkeypatch,
):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_alive = True
    leader_signal_calls = []
    group_signal_calls = []

    def fake_kill(pid, sig):
        nonlocal group_alive
        leader_signal_calls.append((pid, sig))
        group_alive = False

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == signal.SIGTERM:
            group_alive = False
        if sig == 0 and not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],  # type: ignore[list-item]
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == []
    assert leader_signal_calls == [(101, signal.SIGTERM)]
    assert group_signal_calls == [(101, 0)]


def test_terminate_process_groups_cleans_children_after_leader_exits(monkeypatch):
    class ExitedLeader:
        pid = 101

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    group_alive = True

    def leader_is_gone(*_args):
        raise ProcessLookupError

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == signal.SIGTERM:
            group_alive = False
        elif not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "kill", leader_is_gone, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [ExitedLeader()],  # type: ignore[list-item]
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == []
    assert group_alive is False


def test_terminate_process_groups_reports_force_kill_escalation(monkeypatch):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_alive = True

    def fake_kill(_pid, _sig):
        return None

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == 9:
            group_alive = False

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert not any("still alive after SIGKILL" in failure for failure in failures)


def test_terminate_process_groups_reports_a_group_that_survives_sigkill(
    monkeypatch,
):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_liveness_checks = 0

    def fake_kill(_pid, _sig):
        return None

    def fake_killpg(_pid, sig):
        nonlocal group_liveness_checks
        if sig == 0:
            group_liveness_checks += 1

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],  # type: ignore[list-item]
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert any("still alive after SIGKILL" in failure for failure in failures)
    assert group_liveness_checks >= 2


def test_find_processes_matching_environment_uses_exact_entries(
    monkeypatch,
    tmp_path,
):
    environments = {
        101: b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=ffn\0",
        102: b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=attention\0",
        103: b"AFD_E2E_RUN_ID=run-10\0AFD_E2E_PROCESS_ROLE=ffn\0",
    }
    for pid, environment in environments.items():
        process_dir = tmp_path / str(pid)
        process_dir.mkdir()
        (process_dir / "environ").write_bytes(environment)
    (tmp_path / "self").mkdir()
    closed_pidfds: list[int] = []
    monkeypatch.setattr(
        process_utils.os,
        "pidfd_open",
        lambda pid: pid + 1000,
    )
    monkeypatch.setattr(
        process_utils.os,
        "close",
        closed_pidfds.append,
    )

    matching_processes = process_utils.find_processes_matching_environment(
        {
            "AFD_E2E_RUN_ID": "run-1",
            "AFD_E2E_PROCESS_ROLE": "ffn",
        },
        proc_root=tmp_path,
    )

    assert [process.pid for process in matching_processes] == [101]
    assert [process.pidfd for process in matching_processes] == [1101]
    process_utils.close_process_identities(matching_processes)
    assert set(closed_pidfds[:2]) == {1102, 1103}
    assert closed_pidfds[-1] == 1101


def test_kill_matching_process_does_not_signal_recycled_pid(monkeypatch, tmp_path):
    process_dir = tmp_path / "101"
    process_dir.mkdir()
    environment_path = process_dir / "environ"
    environment_path.write_bytes(
        b"AFD_E2E_RUN_ID=run-1\0AFD_E2E_PROCESS_ROLE=ffn\0",
    )
    opened_pidfds = iter((501, 502))
    closed_pidfds: list[int] = []
    signal_calls: list[tuple[int, int]] = []

    def fake_pidfd_open(pid):
        assert pid == 101
        return next(opened_pidfds)

    def fake_close(pidfd):
        closed_pidfds.append(pidfd)

    def fake_pidfd_send_signal(pidfd, sig):
        signal_calls.append((pidfd, sig))
        # The matched process exits and PID 101 is immediately recycled for an
        # unrelated process. Signaling the old pidfd must not target the reuse.
        environment_path.write_bytes(b"UNRELATED_PROCESS=1\0")
        raise ProcessLookupError

    def fail_numeric_pid_signal(*_args):
        raise AssertionError("numeric PID signaled")

    monkeypatch.setattr(process_utils.os, "pidfd_open", fake_pidfd_open)
    monkeypatch.setattr(process_utils.os, "close", fake_close)
    monkeypatch.setattr(process_utils.os, "kill", fail_numeric_pid_signal)
    monkeypatch.setattr(
        process_utils.signal,
        "pidfd_send_signal",
        fake_pidfd_send_signal,
    )
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.kill_processes_matching_environment(
        {
            "AFD_E2E_RUN_ID": "run-1",
            "AFD_E2E_PROCESS_ROLE": "ffn",
        },
        timeout_s=0,
        poll_interval_s=0,
        process_name="FFN",
        proc_root=tmp_path,
    )

    assert failures == []
    assert signal_calls == [(501, signal.SIGKILL)]
    assert closed_pidfds == [501, 502]


def test_terminate_process_groups_uses_one_deadline_and_reaps_every_process(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None
            self.wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = 0

    first_process = FakeProcess(101)
    second_process = FakeProcess(102)
    leader_signal_calls = []
    group_signal_calls = []

    def fake_kill(pid, sig):
        leader_signal_calls.append((pid, sig))
        if pid == first_process.pid and sig == signal.SIGTERM:
            first_process.returncode = 0

    def fake_killpg(pid, sig):
        group_signal_calls.append((pid, sig))
        process = first_process if pid == first_process.pid else second_process
        if sig == 0 and process.returncode is not None:
            raise ProcessLookupError
        if pid == second_process.pid and sig == signal.SIGKILL:
            second_process.returncode = 0

    monotonic_values = iter([100.0, 121.0, 200.0, 201.0, 204.0])
    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        process_utils.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    failures = process_utils.terminate_process_groups(
        [first_process, second_process],  # type: ignore[list-item]
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert len(failures) == 1
    assert "forced SIGKILL" in failures[0]
    assert leader_signal_calls == [
        (101, signal.SIGTERM),
        (102, signal.SIGTERM),
    ]
    assert group_signal_calls == [
        (101, 0),
        (102, 0),
        (102, 9),
        (102, 0),
    ]
    assert first_process.wait_calls == [4]
    assert second_process.wait_calls == [1]


def _write_fake_proc_entry(
    proc_root,
    pid,
    *,
    ppid,
    pgid,
    state,
    command,
    wait_channel,
    thread_count=1,
):
    """Write the ``/proc/<pid>`` fields ``read_process_group_member`` reads."""
    entry = proc_root / str(pid)
    entry.mkdir()
    # proc(5) stat: pid (comm) state ppid pgrp session ... num_threads is the
    # 20th field, i.e. offset 17 counting from state.
    leading = f"{pid} ({command}) {state} {ppid} {pgid} 0"
    padding = " ".join("0" for _ in range(13))
    (entry / "stat").write_text(f"{leading} {padding} {thread_count} 0 0\n")
    (entry / "wchan").write_text(wait_channel)


def test_describe_process_group_reports_state_and_wait_channel(tmp_path):
    _write_fake_proc_entry(
        tmp_path,
        4875,
        ppid=1635,
        pgid=551,
        state="D",
        command="VLLM::Worker_DP",
        wait_channel="nv_kthread_q_flush",
        thread_count=9,
    )
    _write_fake_proc_entry(
        tmp_path,
        4876,
        ppid=1635,
        pgid=999,
        state="S",
        command="unrelated",
        wait_channel="do_wait",
    )

    description = process_utils.describe_process_group(551, proc_root=tmp_path)

    assert "1 surviving member(s), 1 uninterruptible, 0 zombie" in description
    assert "pid=4875" in description
    assert "ppid=1635" in description
    assert "state=D" in description
    assert "threads=9" in description
    assert "wchan=nv_kthread_q_flush" in description
    assert "comm=VLLM::Worker_DP" in description
    # A process in another group must not be attributed to this one.
    assert "unrelated" not in description


def test_describe_process_group_counts_zombies_separately(tmp_path):
    for pid in (700, 701):
        _write_fake_proc_entry(
            tmp_path,
            pid,
            ppid=1,
            pgid=550,
            state="Z",
            command="python3",
            wait_channel="",
        )

    description = process_utils.describe_process_group(550, proc_root=tmp_path)

    assert "2 surviving member(s), 0 uninterruptible, 2 zombie" in description
    # An empty wchan must render as the unknown marker, not as an empty field.
    assert f"wchan={process_utils.UNKNOWN_PROC_FIELD}" in description


def test_describe_process_group_reports_an_emptied_group(tmp_path):
    description = process_utils.describe_process_group(4242, proc_root=tmp_path)

    assert description == "no surviving members found at description time"


def test_describe_process_reports_a_single_process(tmp_path):
    _write_fake_proc_entry(
        tmp_path,
        321,
        ppid=1,
        pgid=320,
        state="D",
        command="VLLM::Worker_DP",
        wait_channel="nv_kthread_q_flush",
    )

    description = process_utils.describe_process(321, proc_root=tmp_path)

    assert "pgid=320" in description
    assert "state=D" in description
    assert "wchan=nv_kthread_q_flush" in description


def test_describe_process_reports_a_vanished_process(tmp_path):
    assert (
        process_utils.describe_process(321, proc_root=tmp_path)
        == "no longer present at description time"
    )


def test_read_process_group_member_tolerates_commands_with_parentheses(tmp_path):
    _write_fake_proc_entry(
        tmp_path,
        900,
        ppid=1,
        pgid=900,
        state="S",
        command="odd (name) here",
        wait_channel="do_wait",
    )

    member = process_utils.read_process_group_member(900, proc_root=tmp_path)

    assert member is not None
    assert member.command == "odd (name) here"
    assert member.state == "S"
    assert member.process_group == 900


def test_terminate_process_groups_appends_diagnostics_to_survivor_failures(
    monkeypatch,
    tmp_path,
):
    _write_fake_proc_entry(
        tmp_path,
        202,
        ppid=201,
        pgid=201,
        state="D",
        command="VLLM::Worker_DP",
        wait_channel="nv_kthread_q_flush",
    )

    class FakeProcess:
        pid = 201

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(process_utils.os, "kill", lambda pid, sig: None, raising=False)
    monkeypatch.setattr(
        process_utils.os, "killpg", lambda pid, sig: None, raising=False
    )

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
        proc_root=tmp_path,
    )

    forced = next(f for f in failures if "forced SIGKILL" in f)
    survived = next(f for f in failures if "still alive after SIGKILL" in f)
    for failure in (forced, survived):
        assert "state=D" in failure
        assert "wchan=nv_kthread_q_flush" in failure
        assert "comm=VLLM::Worker_DP" in failure
