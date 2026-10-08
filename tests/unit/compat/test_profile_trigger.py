# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from afd_plugin.compat.profile_trigger import (
    AFD_FFN_PROFILE_TRIGGER_FILE_ENV,
    ProfileTrigger,
    create_ffn_profile_trigger,
)


def test_trigger_is_disabled_without_env(monkeypatch):
    monkeypatch.delenv(AFD_FFN_PROFILE_TRIGGER_FILE_ENV, raising=False)

    assert create_ffn_profile_trigger() is None


def test_trigger_reads_path_from_env(monkeypatch, tmp_path):
    path = tmp_path / "ffn"
    monkeypatch.setenv(AFD_FFN_PROFILE_TRIGGER_FILE_ENV, str(path))

    trigger = create_ffn_profile_trigger()

    assert trigger is not None
    assert trigger.path == path


def test_trigger_ignores_content_present_at_startup(tmp_path):
    path = tmp_path / "ffn"
    path.write_text("start:1\n")

    trigger = ProfileTrigger(path)

    assert trigger.poll() is None


def test_trigger_fires_once_per_content_change(tmp_path):
    path = tmp_path / "ffn"
    trigger = ProfileTrigger(path)

    assert trigger.poll() is None
    path.write_text("start:1")
    assert trigger.poll() is True
    assert trigger.poll() is None
    path.write_text("stop:1")
    assert trigger.poll() is False
    path.write_text("start:2")
    assert trigger.poll() is True


@pytest.mark.parametrize("content", ["", "restart", "  "])
def test_trigger_ignores_empty_or_unknown_commands(tmp_path, content):
    path = tmp_path / "ffn"
    path.write_text("stop:0")
    trigger = ProfileTrigger(path)

    path.write_text(content)

    assert trigger.poll() is None


def test_trigger_accepts_command_without_token(tmp_path):
    path = tmp_path / "ffn"
    trigger = ProfileTrigger(path)

    path.write_text("START\n")

    assert trigger.poll() is True


def test_trigger_treats_removed_file_as_empty(tmp_path):
    path = tmp_path / "ffn"
    path.write_text("stop:0")
    trigger = ProfileTrigger(path)

    path.unlink()

    assert trigger.poll() is None
    path.write_text("stop:0")
    assert trigger.poll() is False
