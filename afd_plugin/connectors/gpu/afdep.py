# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Use afdEP's DeepEP-derived transport with the existing routed-expert runner."""

from afdep.deepep import ElasticBuffer

from afd_plugin.connectors.gpu.deepep_m2n.connector import DeepEPM2NAFDConnector


class AfdEPConnector(DeepEPM2NAFDConnector):
    """Preserve DeepEP M2N scheduling and compute; load transport from afdEP."""

    BUFFER_TYPE = ElasticBuffer
