# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The proposed connector must actually activate the merged v1 specializations."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("afdep")

from afd_plugin.connectors.gpu.afdep import AfdEPConnector  # noqa: E402
from afd_plugin.connectors.gpu.deepep_m2n.connector import (  # noqa: E402
    DeepEPM2NAFDConnector,
)


def test_v1_constructor_geometry_and_empty_attention_arena():
    connector = object.__new__(AfdEPConnector)
    connector.layout = SimpleNamespace(ffn_size=2)
    assert connector._buffer_options() == {"role_aware": True, "ffn_ranks": 2}
    assert connector.BUFFER_PROTOCOL != DeepEPM2NAFDConnector.BUFFER_PROTOCOL
    assert object.__new__(DeepEPM2NAFDConnector)._buffer_options() == {}
    connector.hidden_size = 2048
    connector.dtype = torch.bfloat16
    connector.device = torch.device("cpu")
    connector._allocate_combine_zeros()
    assert connector._combine_zeros.shape == (0, 2048)
    assert connector._combine_zero_rows((0, 2048)).numel() == 0
    with pytest.raises(RuntimeError, match="combine needs zero rows"):
        connector._combine_zero_rows((1, 2048))
