# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Union-world expert layout for DeepEP M2N.

Adapted from FastAFD ``python/minisgl/moe/deepep_m2n_union.py`` at ``3c71619``.

FastAFD orders the union world ``[AG.., EG..]`` and remaps real expert ids by a
``dummy_expert_offset``. afd-plugin orders process groups FFN first
(``[F0.., A0..]``, see ``afd_plugin/connectors/gpu/p2p.py``), so FFN rank ``g``
owns DeepEP experts ``[g * E_local, (g + 1) * E_local)`` which equal the real
expert ids: the remap is the identity. Attention ranks own dummy expert slots
after the real ones that are never routed to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepEPUnionM2NLayout:
    ffn_size: int
    attn_size: int
    real_num_experts: int

    def __post_init__(self) -> None:
        if self.ffn_size <= 0:
            raise ValueError(f"ffn_size must be positive, got {self.ffn_size}")
        if self.attn_size <= 0:
            raise ValueError(f"attn_size must be positive, got {self.attn_size}")
        if self.real_num_experts <= 0:
            raise ValueError(
                f"real_num_experts must be positive, got {self.real_num_experts}"
            )

    @property
    def union_size(self) -> int:
        return self.ffn_size + self.attn_size

    @property
    def experts_per_union_rank(self) -> int:
        return (self.real_num_experts + self.ffn_size - 1) // self.ffn_size

    @property
    def deepep_num_experts(self) -> int:
        return self.union_size * self.experts_per_union_rank

    def is_ffn_rank(self, union_rank: int) -> bool:
        _validate_rank("union_rank", union_rank, self.union_size)
        return union_rank < self.ffn_size

    def union_rank_for_attn_rank(self, attn_rank: int) -> int:
        _validate_rank("attn_rank", attn_rank, self.attn_size)
        return self.ffn_size + attn_rank

    def expert_range_for_union_rank(self, union_rank: int) -> range:
        _validate_rank("union_rank", union_rank, self.union_size)
        start = union_rank * self.experts_per_union_rank
        return range(start, start + self.experts_per_union_rank)

    def real_expert_range_for_ffn_rank(self, ffn_rank: int) -> range:
        _validate_rank("ffn_rank", ffn_rank, self.ffn_size)
        deepep_range = self.expert_range_for_union_rank(ffn_rank)
        return range(
            min(self.real_num_experts, deepep_range.start),
            min(self.real_num_experts, deepep_range.stop),
        )


def _validate_rank(name: str, rank: int, size: int) -> None:
    if not 0 <= rank < size:
        raise ValueError(f"{name} must be in [0, {size}), got {rank}")
