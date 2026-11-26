import argparse
import asyncio
import bisect
import collections
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


async def api(session, args, route, body=None):
    headers = {"Authorization": "Bearer profile"}
    method = "GET" if body is None else "POST"
    async with session.request(
        method, args.url + route, json=body, headers=headers
    ) as response:
        response.raise_for_status()
        text = await response.text()
        try:
            result = json.loads(text)
        except ValueError:
            return text
        if isinstance(result, dict) and (
            result.get("error") or result.get("success") is False
        ):
            raise RuntimeError(result)
        return result


async def request(session, args, payload, result, index, started, all_started, start):
    try:
        async with session.post(
            args.url + "/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event)
                meta = event.get("meta_info", {})
                count = meta.get("completion_tokens", 0)
                if count > (
                    result[index]["events"][-1][1] if result[index]["events"] else 0
                ):
                    result[index]["events"].append([time.perf_counter() - start, count])
                    if index not in started:
                        started.add(index)
                        if len(started) == args.n:
                            all_started.set()
                result[index]["meta_info"] = meta
        meta = result[index]["meta_info"]
        assert meta.get("prompt_tokens") == args.input_len, meta
        assert meta.get("cached_tokens") == 0, meta
        assert meta.get("completion_tokens") == args.output_len, meta
        assert meta.get("finish_reason", {}).get("type") == "length", meta
    except BaseException as error:
        result[index]["error"] = repr(error)
        raise


def cumulative(events, timestamp):
    index = bisect.bisect_right([row[0] for row in events], timestamp)
    return events[index - 1][1] if index else 0


def summarize(result, total):
    first = max(row["events"][0][0] for row in result)
    last = min(row["events"][-1][0] for row in result)
    assert last > first, "No interval with all requests decoding"
    start = first + (last - first) * 0.15
    end = first + (last - first) * 0.85
    duration = end - start
    tokens = [
        cumulative(r["events"], end) - cumulative(r["events"], start) for r in result
    ]
    assert min(tokens) > 0
    event_count = sum(
        sum(start < event[0] <= end for event in row["events"]) for row in result
    )
    return {
        "total_s": total,
        "all_first_tokens_s": first,
        "all_active_start_s": first,
        "all_active_end_s": last,
        "steady_start_s": start,
        "steady_end_s": end,
        "steady_duration_s": duration,
        "steady_tokens": sum(tokens),
        "steady_tok_s": sum(tokens) / duration,
        "median_request_tpot_ms": statistics.median(
            duration * 1000 / n for n in tokens
        ),
        "tokens_per_positive_stream_event": sum(tokens) / event_count,
        "total_output_tokens": sum(r["events"][-1][1] for r in result),
        "completed_requests": len(result),
        "dp_distribution": dict(
            collections.Counter(r["meta_info"].get("dp_rank") for r in result)
        ),
    }


async def profile(session, args, all_started, requests_done, start):
    await asyncio.wait_for(all_started.wait(), timeout=args.timeout)
    await asyncio.sleep(0.5)
    if requests_done.done():
        raise RuntimeError("Requests ended before profile capture")
    output_name = getattr(args, "profile_output_name", None) or args.profile
    local = args.root.parent / "profiles" / output_name
    local.mkdir(parents=True, exist_ok=False)
    mapping = args.profile == "mapping"
    requested = time.perf_counter() - start
    await api(
        session,
        args,
        "/start_profile",
        {
            "output_dir": "/repro/results/profiles/" + output_name,
            "activities": ["CPU", "GPU"] if mapping else ["GPU"],
            "with_stack": mapping,
            "record_shapes": mapping,
        },
    )
    began = time.perf_counter() - start
    try:
        await asyncio.sleep(args.profile_seconds)
    finally:
        ended = time.perf_counter() - start
        await api(session, args, "/stop_profile", {})
    return {
        "start_requested_s": requested,
        "start_returned_s": began,
        "stop_requested_s": ended,
        "stop_returned_s": time.perf_counter() - start,
    }


async def wave(session, args, ids, wave_index, tag):
    objects = [
        {
            "input_ids": tokens,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.output_len,
                "ignore_eos": True,
            },
            "bootstrap_host": "2.2.2.2",
            "bootstrap_room": wave_index * 100000 + index,
        }
        for index, tokens in enumerate(ids)
    ]
    routing = getattr(args, "dp_routing", "natural")
    if routing == "balanced":
        assert args.n % args.dp == 0
        for index, value in enumerate(objects):
            value["routed_dp_rank"] = index % args.dp
    elif routing == "skewed":
        assert args.dp == 8 and args.n == 64
        ranks = [
            rank
            for rank, count in enumerate([9, 9, 9, 9, 7, 7, 7, 7])
            for _ in range(count)
        ]
        for value, rank in zip(objects, ranks):
            value["routed_dp_rank"] = rank
    payloads = [json.dumps(value).encode() for value in objects]
    result = [{"index": i, "events": []} for i in range(args.n)]
    started = set()
    all_started = asyncio.Event()
    start = time.perf_counter()
    pending = asyncio.gather(
        *[
            request(session, args, payload, result, index, started, all_started, start)
            for index, payload in enumerate(payloads)
        ]
    )
    try:
        if args.profile:
            _, captured = await asyncio.gather(
                pending, profile(session, args, all_started, pending, start)
            )
        else:
            await pending
        summary = summarize(result, time.perf_counter() - start)
        if routing == "balanced":
            assert summary["dp_distribution"] == {
                rank: args.n // args.dp for rank in range(args.dp)
            }
        elif routing == "skewed":
            assert summary["dp_distribution"] == dict(
                enumerate([9, 9, 9, 9, 7, 7, 7, 7])
            )
        if args.profile:
            summary["profile_control"] = captured
        summary["tag"] = tag
        save(args.root / f"{tag}.json", {"summary": summary, "requests": result})
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException:
        save(args.root / f"{tag}-failed.json", {"requests": result})
        pending.cancel()
        raise


async def main(args):
    args.root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    ids = [
        [rng.randrange(1, 129280) for _ in range(args.input_len)] for _ in range(args.n)
    ]
    save(
        args.root / "protocol.json",
        {
            "n": args.n,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": 42,
            "fake_prefill": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "profile": args.profile,
            "input_sha256": [
                hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in ids
            ],
            "dp_routing": getattr(args, "dp_routing", "natural"),
            "steady_window": "central 70% of interval between all first tokens and first completion",
            "token_count": "cumulative completion_tokens at end minus at start; no stream-chunk-as-token approximation",
        },
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=128)
    ) as session:
        info = await api(session, args, "/get_server_info")
        save(args.root / "server_info_before.json", info)
        server = info.get("server_args", info)
        assert server["disaggregation_mode"] == "decode", (
            "Real decode-only server required"
        )
        assert server["disaggregation_transfer_backend"] == args.backend
        assert server["disable_radix_cache"]
        for key, expected in {
            "tp_size": 8,
            "ep_size": args.ep,
            "attn_dp_size": args.dp,
        }.items():
            assert server[key] == expected, (key, server[key], expected)
        if args.reference_config:
            # Legacy --dp-size=8 --enable-dp-attention is normalized to the
            # attention-DP dimension; canonical dp_size is then 1.
            expected = {
                "mem_fraction_static": 0.85,
                "max_running_requests": 256,
                "cuda_graph_max_bs_decode": 32,
                "tokenizer_worker_num": 8,
                "load_balance_method": "total_tokens",
            }
            for key, value in expected.items():
                assert server[key] == value, (key, server.get(key), value)
        results = []
        for index in range(args.warmup + args.repeat):
            tag = (
                f"warmup-{index}"
                if index < args.warmup
                else f"sample-{index - args.warmup}"
            )
            row = await wave(session, args, ids, index + 1, tag)
            if index >= args.warmup:
                results.append(row)
        save(
            args.root / "server_info_after.json",
            await api(session, args, "/get_server_info"),
        )
    summary = {
        "samples": results,
        "median_steady_tok_s": statistics.median(r["steady_tok_s"] for r in results),
    }
    save(args.root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30002")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=65536)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--backend", choices=["fake", "mooncake"], default="fake")
    parser.add_argument("--reference-config", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--profile", choices=["gpu", "mapping"])
    parser.add_argument("--profile-seconds", type=float, default=0.8)
    parser.add_argument(
        "--dp-routing", choices=["natural", "balanced", "skewed"], default="natural"
    )
    parser.add_argument("--profile-output-name")
    asyncio.run(main(parser.parse_args()))
