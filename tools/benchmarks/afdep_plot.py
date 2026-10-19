# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plot a completed suite; bars are repeat medians and whiskers are run ranges."""

import argparse
import json
import statistics
from pathlib import Path

from tools.benchmarks.afdep_suite import VARIANTS, WORKLOADS


def main() -> None:
    # Optional plotting dependency; the benchmark runner does not load it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["status"] != "complete":
        raise ValueError("Plot only a complete campaign")
    figures = (
        ("output_tokens_per_second", "Output tokens/s", True),
        ("ttft_p95_ms", "P95 TTFT (ms)", True),
        ("tpot_median_ms", "Median TPOT (ms)", False),
    )
    colors = ("#778295", "#2864a3", "#218457")
    labels = ("P2P", "Masked DeepEP", "afdEP v1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (metric, title, logarithmic) in zip(axes, figures, strict=True):
        for index, (variant, _) in enumerate(VARIANTS):
            for case, workload in enumerate(WORKLOADS):
                values = [
                    row["result"][metric]
                    for row in manifest["runs"]
                    if row["variant"] == variant
                    and row["workload"] == workload.name
                    and row["result"][metric] is not None
                ]
                if not values:
                    continue
                median = statistics.median(values)
                axis.bar(
                    case + (index - 1) * 0.24,
                    median,
                    width=0.22,
                    color=colors[index],
                    label=labels[index] if case == 0 else None,
                    yerr=[[median - min(values)], [max(values) - median]],
                    capsize=3,
                    error_kw={"linewidth": 1},
                )
        axis.set_title(title)
        axis.set_xticks(range(len(WORKLOADS)), [case.name for case in WORKLOADS])
        if logarithmic:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        f"{manifest['policy']['profile']} · Qwen3-30B-A3B-FP8 · 2 Attention + 2 FFN",
        fontsize=17,
    )
    fig.text(
        0.5,
        0.025,
        f"{manifest['policy']['repeats']} balanced repeats · bars: repeat medians · "
        "whiskers: observed ranges\n"
        "Natural serving paths. Startup/capture excluded; graph replay included. "
        "One-token prefill has no TPOT.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
