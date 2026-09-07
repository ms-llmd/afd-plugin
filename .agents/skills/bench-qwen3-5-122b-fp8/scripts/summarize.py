#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Aggregate `vllm bench serve` result JSONs into a mean +/- std table.

Usage:
    summarize.py --label afd-eager-2a2f /tmp/results/afd/run-[1-5].json
    summarize.py --label afd-eager-2a2f --baseline native.json run-*.json

Writes the aggregate to <label>-summary.json next to the first input so a later
run can diff against it via --baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# (result-json key, display label, unit, higher_is_better)
METRICS = (
    ("request_throughput", "Request throughput", "req/s", True),
    ("output_throughput", "Output throughput", "tok/s", True),
    ("mean_ttft_ms", "TTFT", "ms", False),
    ("mean_tpot_ms", "TPOT", "ms", False),
    ("mean_itl_ms", "ITL", "ms", False),
    ("mean_e2el_ms", "E2E latency", "ms", False),
)


def load(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        with path.open() as handle:
            run = json.load(handle)
        run["_path"] = str(path)
        runs.append(run)
    return runs


def check_completion(runs: list[dict]) -> list[str]:
    """Return one warning per run that did not complete every request."""
    warnings = []
    for run in runs:
        completed = run.get("completed")
        expected = run.get("num_prompts")
        failed = run.get("failed") or 0
        if completed != expected or failed:
            warnings.append(
                f"{run['_path']}: completed {completed}/{expected}, failed {failed}"
            )
    return warnings


def fmt(value: float, unit: str) -> str:
    return f"{value:.4f}" if unit == "req/s" else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--label", default="run", help="Scenario name.")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="A previous <label>-summary.json to diff against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the summary JSON. Defaults to "
        "<first result dir>/<label>-summary.json.",
    )
    args = parser.parse_args()

    runs = load(args.results)
    if len(runs) < 2:
        print("need at least two runs to report a standard deviation", file=sys.stderr)
        return 2

    warnings = check_completion(runs)

    # Per-run table.
    print(f"\n=== {args.label}: {len(runs)} runs ===\n")
    header = f"{'Run':<28}" + "".join(f"{label:>20}" for _, label, _, _ in METRICS)
    print(header)
    print("-" * len(header))
    for run in runs:
        row = f"{Path(run['_path']).name:<28}"
        for key, _, unit, _ in METRICS:
            value = run.get(key)
            row += f"{'n/a' if value is None else fmt(value, unit):>20}"
        print(row)

    # Aggregate.
    summary = {"label": args.label, "num_runs": len(runs), "metrics": {}}
    print(f"\n=== {args.label}: mean +/- std ===\n")
    for key, label, unit, higher_is_better in METRICS:
        values = [run[key] for run in runs if run.get(key) is not None]
        if not values:
            print(f"{label:<20} n/a  (missing {key}; pass e2el in --percentile-metrics)")
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        summary["metrics"][key] = {
            "mean": mean,
            "std": std,
            "unit": unit,
            "higher_is_better": higher_is_better,
            "values": values,
        }
        line = f"{label:<20} {fmt(mean, unit)} +/- {fmt(std, unit)} {unit}"
        if args.baseline:
            line += "  " + delta(key, mean, args.baseline, higher_is_better)
        print(line)

    if warnings:
        print("\nINCOMPLETE RUNS - these numbers are not valid:")
        for warning in warnings:
            print(f"  {warning}")

    out = args.output or args.results[0].parent / f"{args.label}-summary.json"
    with out.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {out}")
    return 1 if warnings else 0


_BASELINE_CACHE: dict[Path, dict] = {}


def delta(key: str, mean: float, baseline_path: Path, higher_is_better: bool) -> str:
    if baseline_path not in _BASELINE_CACHE:
        with baseline_path.open() as handle:
            _BASELINE_CACHE[baseline_path] = json.load(handle)
    base = _BASELINE_CACHE[baseline_path].get("metrics", {}).get(key)
    if not base or not base["mean"]:
        return ""
    pct = (mean - base["mean"]) / base["mean"] * 100
    # Overlapping error bars are parity, not a win.
    if abs(mean - base["mean"]) <= max(base.get("std", 0.0), 0.0):
        return f"({pct:+.2f}% vs baseline, within 1 std - parity)"
    better = (pct > 0) == higher_is_better
    return f"({pct:+.2f}% vs baseline, {'better' if better else 'worse'})"


if __name__ == "__main__":
    raise SystemExit(main())
