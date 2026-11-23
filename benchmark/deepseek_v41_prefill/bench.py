#!/usr/bin/env python3
"""Exact-token, cache-off prefill measurement and separate all-rank GPU profiles."""

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def http(base, route, body, admin_key, timeout):
    headers = {"Content-Type": "application/json"}
    if admin_key:
        headers["Authorization"] = f"Bearer {admin_key}"
    request = urllib.request.Request(
        base + route,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
    )
    # Direct connection, independent of a developer shell's proxy variables.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{route}: HTTP {error.code}: {detail}") from error


def call(args, route, body=None, timeout=None):
    with http(
        args.base_url, route, body, args.admin_key, timeout or args.timeout
    ) as response:
        raw = response.read().decode()
    try:
        result = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(result, dict) and (
        result.get("error") or result.get("success") is False
    ):
        raise RuntimeError(f"{route}: {result}")
    return result


def wait_ready(args):
    deadline = time.monotonic() + args.wait_ready
    last_report = 0
    while True:
        if args.server_container:
            state = subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    args.server_container,
                ],
                text=True,
            ).strip()
            if state != "true":
                raise RuntimeError("Server container exited; inspect server.log")
        try:
            call(args, "/health", timeout=5)
            return
        except (OSError, RuntimeError) as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Server not ready at {args.base_url}") from error
            if time.monotonic() - last_report >= 30:
                print(f"Waiting for server: {args.base_url}", flush=True)
                last_report = time.monotonic()
            time.sleep(5)


def validate_server(info):
    server = info.get("server_args", info)
    expected = {
        "disable_radix_cache": True,
        "tp_size": 8,
        "attn_cp_size": 8,
        "ep_size": 1,
        "enable_prefill_cp": True,
        "cp_strategy": "interleave",
        "enable_cp_decode_attn_tp": True,
        "moe_runner_backend": "humming",
        "enable_decoder_swa_bounded_replay": True,
        "swa_full_tokens_ratio": 0.05,
        "speculative_algorithm": "DSPARK",
        "speculative_dspark_block_size": 5,
        "mem_fraction_static": 0.84,
        "max_running_requests": 64,
        "cuda_graph_max_bs_decode": 64,
        "chunked_prefill_size": 16384,
    }
    mismatch = {
        k: {"expected": v, "actual": server.get(k)}
        for k, v in expected.items()
        if server.get(k) != v
    }
    if mismatch:
        raise RuntimeError(f"Server differs from the Ant comparison: {mismatch}")


def generate(args, ids):
    payload = {
        "input_ids": ids,
        "stream": True,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True},
    }
    # Serialize before starting the clock, matching the historical strict client.
    # http() serializes its body, so pass the prebuilt request directly here.
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(
        args.base_url + "/generate", data=data, headers=headers
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    first = None
    last = None
    start = time.perf_counter()
    with opener.open(request, timeout=args.timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if value == b"[DONE]":
                continue
            event = json.loads(value)
            if event.get("error"):
                raise RuntimeError(f"Generation failed: {event}")
            meta = event.get("meta_info", {})
            if first is None and meta.get("completion_tokens", 0) > 0:
                first = time.perf_counter()
            last = event
    end = time.perf_counter()
    if first is None or last is None:
        raise RuntimeError(f"No token-bearing SSE event: {last}")
    meta = last["meta_info"]
    expected = {"prompt_tokens": len(ids), "cached_tokens": 0, "completion_tokens": 1}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Invalid benchmark sample: expected {expected}, got {meta}")
    return {
        "input_tokens": len(ids),
        "ttft_ms": (first - start) * 1000,
        "e2e_ms": (end - start) * 1000,
        "meta_info": meta,
    }


def wait_profiles(directory, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        files = sorted(directory.glob("*.trace.json.gz"))
        ranks = [int(m[1]) for p in files if (m := re.search(r"TP-(\d+)", p.name))]
        sizes = [(str(p), p.stat().st_size) for p in files]
        if sorted(ranks) == list(range(8)) and len(files) == 8 and sizes == previous:
            try:
                for path in files:
                    with gzip.open(path, "rb") as trace:
                        while trace.read(1024 * 1024):
                            pass
                return [str(p) for p in files]
            except (OSError, EOFError):
                pass
        previous = sizes
        time.sleep(2)
    raise RuntimeError(f"Eight complete TP rank traces did not appear in {directory}")


def profile(args, ids):
    length = str(len(ids))
    local_dir = args.profile_dir.resolve() / length
    local_dir.mkdir(parents=True, exist_ok=False)
    server_dir = Path(args.server_profile_dir or args.profile_dir.resolve()) / length
    call(
        args,
        "/start_profile",
        {
            "output_dir": str(server_dir),
            "activities": ["GPU"],
            "with_stack": False,
            "record_shapes": False,
        },
    )
    try:
        row = generate(args, ids)
    finally:
        call(args, "/stop_profile", {})
    row["traces"] = wait_profiles(local_dir, args.profile_timeout)
    return row


def write_summary(root, rows, args):
    summaries = []
    for length in args.lengths:
        samples = [row for row in rows if row["input_tokens"] == length]
        if not samples:
            continue
        times = [row["ttft_ms"] for row in samples]
        median = statistics.median(times)
        summaries.append(
            {
                "input_tokens": length,
                "samples": len(samples),
                "input_sha256": samples[0]["input_sha256"],
                "median_ttft_ms": median,
                "min_ttft_ms": min(times),
                "max_ttft_ms": max(times),
                "median_e2e_ms": statistics.median(row["e2e_ms"] for row in samples),
                "input_tokens_per_second": length * 1000 / median,
            }
        )
    save(root / "summary.json", summaries)
    lines = [
        "Serial, concurrency=1; input tokens / median TTFT. Output=1 token; radix cache off.",
        f"Warmups per length: {args.warmup}. Profile requests are excluded.",
        "",
        "| Input tokens | Samples | Median TTFT ms | Min/max ms | Input tok/s |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['input_tokens']} | {row['samples']} | {row['median_ttft_ms']:.3f} | "
            f"{row['min_ttft_ms']:.3f}/{row['max_ttft_ms']:.3f} | "
            f"{row['input_tokens_per_second']:.2f} |"
        )
    report = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30002")
    parser.add_argument(
        "--admin-key", default=os.getenv("SGLANG_ADMIN_API_KEY", "profile")
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[65536, 262144])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--server-profile-dir", help="Same directory as profile-dir, seen by server"
    )
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--wait-ready", type=float, default=0)
    parser.add_argument(
        "--server-container", help="Fail early if an owned Docker server exits"
    )
    parser.add_argument("--profile-timeout", type=float, default=300)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1 or min(args.lengths) < 1:
        parser.error("Require warmup >= 0, repeat >= 1 and positive input lengths")
    if len(set(args.lengths)) != len(args.lengths):
        parser.error("Input lengths must be unique")
    if args.server_profile_dir and not args.profile_dir:
        parser.error("--server-profile-dir requires --profile-dir")
    args.base_url = args.base_url.rstrip("/")
    return args


def main():
    args = parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(
        root / "protocol.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "lengths": args.lengths,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": 42,
            "token_id_range_inclusive": [10000, 100000],
            "max_new_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
            "concurrency": 1,
            "profile_activities": ["GPU"],
            "with_stack": False,
            "record_shapes": False,
            "ttft_clock": "after JSON serialization until first token-bearing SSE event",
        },
    )
    wait_ready(args)
    info = call(args, "/get_server_info")
    save(root / "server_info.json", info)
    validate_server(info)
    rng = random.Random(42)
    all_ids = [rng.randint(10000, 100000) for _ in range(max(args.lengths))]
    rows = []
    profile_rows = []
    for length in args.lengths:
        ids = all_ids[:length]
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        call(args, "/flush_cache", {"timeout": 60})
        for stage, count in (("warmup", args.warmup), ("sample", args.repeat)):
            for iteration in range(count):
                row = generate(args, ids)
                row.update(stage=stage, iteration=iteration, input_sha256=digest)
                print(json.dumps(row), flush=True)
                with (root / "requests.jsonl").open("a") as log:
                    log.write(json.dumps(row) + "\n")
                if stage == "sample":
                    rows.append(row)
                    save(root / "measurements.json", rows)
                    write_summary(root, rows, args)
        if args.profile_dir:
            print(f"Capturing GPU-only profile: {length} tokens", flush=True)
            row = profile(args, ids)
            row["input_sha256"] = digest
            profile_rows.append(row)
            save(root / "profile_requests.json", profile_rows)
    print(write_summary(root, rows, args), flush=True)
    save(root / "completed.json", {"success": True})


if __name__ == "__main__":
    main()
