#!/usr/bin/env python3
"""Compare all-rank GPU events; sums are normalized per rank, not wall time."""

import argparse
import collections
import gzip
import json
import re
from pathlib import Path


def union_us(events):
    end = float("-inf")
    active = 0
    for e in sorted(events, key=lambda e: e["ts"]):
        stop = e["ts"] + e["dur"]
        active += max(0, stop - max(e["ts"], end))
        end = max(end, stop)
    return active


def family(name):
    if "humming<" in name:
        shape = re.search(r"Shape<0u, (\d+)u, (\d+)u>", name)
        if shape:
            if int(shape[2]) == 5120:
                return "Humming W13"
            if int(shape[1]) == 5120:
                return "Humming W2"
        return "Humming other"
    if "sm90_fp8_fused_mega_moe_impl" in name:
        return "Fused MegaMoE (compute and communication)"
    if "all_gather_pull_kernel" in name:
        return "Custom all-gather pull (including synchronization)"
    if "reduce_scatter_pull_kernel" in name:
        return "Custom reduce-scatter pull (including synchronization)"
    for label, pattern in [
        ("CP barrier", "symmetric_memory::barrier_kernel"),
        ("Sparse attention", "sparse_attn_fwd"),
        ("DeepGEMM", "deep_gemm::"),
        ("mHC", "hc_"),
        ("Custom allreduce", "all_reduce_"),
        ("NCCL", "ncclDevKernel"),
        ("Engram", "engram"),
        ("Block32 quant/reorder", "block32"),
    ]:
        if pattern in name:
            return label
    return "Other"


def summarize(root):
    ranks = []
    named = collections.defaultdict(lambda: [0, 0, 0])
    grouped = collections.defaultdict(lambda: [0, 0])
    for file in sorted(Path(root).glob("*.trace.json.gz")):
        match = re.search(r"TP-(\d+)", file.name)
        if not match:
            raise RuntimeError(f"Missing TP rank in trace name: {file}")
        with gzip.open(file, "rt") as f:
            trace = json.load(f)
        events = [
            e
            for e in trace["traceEvents"]
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
            and e.get("dur", 0) > 0
        ]
        kernels = [e for e in events if e["cat"] == "kernel"]
        if not kernels:
            raise RuntimeError(
                f"No GPU kernels in {file}; require GPU activity profiling"
            )
        barriers = sorted(
            [e for e in kernels if "symmetric_memory::barrier_kernel" in e["name"]],
            key=lambda e: e["ts"],
        )
        first = min(e["ts"] for e in events)
        wall = max(e["ts"] + e["dur"] for e in events) - first
        for e in kernels:
            v = named[e["name"]]
            v[0] += e["dur"] / 1000
            v[1] += 1
            v[2] = max(v[2], e["dur"] / 1000)
            g = grouped[family(e["name"])]
            g[0] += e["dur"] / 1000
            g[1] += 1
        ranks.append(
            {
                "rank": int(match[1]),
                "wall_ms": wall / 1000,
                "gpu_active_union_ms": union_us(events) / 1000,
                "gpu_idle_ms": (wall - union_us(events)) / 1000,
                "kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000,
                "barrier_sum_ms": sum(e["dur"] for e in barriers) / 1000,
                "kernel_count": len(kernels),
                "barrier_count": len(barriers),
                "barriers": [
                    {
                        "ordinal": i,
                        "start_ms": (e["ts"] - first) / 1000,
                        "duration_ms": e["dur"] / 1000,
                        "stream": e.get("args", {}).get("stream"),
                    }
                    for i, e in enumerate(barriers)
                ],
            }
        )
    ranks.sort(key=lambda row: row["rank"])
    if [row["rank"] for row in ranks] != list(range(8)):
        raise RuntimeError(
            f"Expected exactly TP ranks 0..7, got {[r['rank'] for r in ranks]} in {root}"
        )
    return {
        "root": str(root),
        "ranks": ranks,
        "kernels": dict(named),
        "families": dict(grouped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    a, b = summarize(args.reference), summarize(args.candidate)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gpu_summary.json").write_text(
        json.dumps({"reference": a, "candidate": b}, indent=2)
    )
    lines = [
        "Times below are mean sums per TP rank; concurrent events can overlap.",
        "",
        "| Kernel family | Reference ms | Candidate ms | Delta ms | Calls ref/cand per rank |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(
        set(a["families"]) | set(b["families"]),
        key=lambda n: b["families"].get(n, [0])[0] - a["families"].get(n, [0])[0],
        reverse=True,
    ):
        av, bv = a["families"].get(name, [0, 0]), b["families"].get(name, [0, 0])
        lines.append(
            f"| {name} | {av[0] / 8:.3f} | {bv[0] / 8:.3f} | {(bv[0] - av[0]) / 8:+.3f} | {av[1] / 8:g}/{bv[1] / 8:g} |"
        )
    lines += [
        "",
        "| Rank | Reference GPU span ms | Candidate GPU span ms | Ref barrier ms | Candidate barrier ms | Ref idle ms | Candidate idle ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ar, br in zip(a["ranks"], b["ranks"]):
        lines.append(
            f"| {ar['rank']} | {ar['wall_ms']:.3f} | {br['wall_ms']:.3f} | {ar['barrier_sum_ms']:.3f} | {br['barrier_sum_ms']:.3f} | {ar['gpu_idle_ms']:.3f} | {br['gpu_idle_ms']:.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (out / "comparison.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
