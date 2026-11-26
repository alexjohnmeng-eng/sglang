"""DP8/EP8 fake-prefill throughput A/B, GPU profile and real-prefill GSM8K.

Run from the host; Python packages for clients execute in the serving image.
All source and tools are exported/copied before starting GPU work.
"""

import argparse
import json
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--revision", default="HEAD")
parser.add_argument("--baseline", default="82bebb1d34")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--prepare-only",
    action="store_true",
    help="Export source/tools/manifests only; do not query GPUs or start a server",
)
parser.add_argument(
    "--runtime", type=Path, default=Path("/data1/fjmeng2/dsv41-runtime")
)
parser.add_argument(
    "--cases",
    nargs="+",
    choices=["control", "candidate", "accuracy"],
    default=["control", "candidate", "accuracy"],
    help="Cases to run, in order; accuracy uses real prefill without simulation",
)
parser.add_argument(
    "--unified-deepgemm-site",
    type=Path,
    help="Use the single H20 wheel for candidate and accuracy; control uses the image",
)
args = parser.parse_args()
assets = Path(__file__).resolve().parent
repo = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=assets, text=True
    ).strip()
)
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "recipe.json").write_text(
    json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        indent=2,
    )
    + "\n"
)
image = "sha256:f3a303d4028dc4597e312c96ada28fa19509f46559601a94e89dc630ac1ef152"
if args.unified_deepgemm_site:
    subprocess.run(
        [
            "python3",
            str(assets.parent / "dependencies/unified/verify.py"),
            "site",
            str(args.unified_deepgemm_site.resolve()),
        ],
        check=True,
    )
if not args.prepare_only:
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs occupied")
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 8089))


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def capture(command):
    return subprocess.check_output(command, stderr=subprocess.STDOUT)


def ready(container):
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if (
            capture(
                ["docker", "inspect", "--format", "{{.State.Running}}", container]
            ).strip()
            != b"true"
        ):
            raise RuntimeError("Serving container exited")
        result = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "curl",
                "--noproxy",
                "*",
                "--fail",
                "--silent",
                "http://127.0.0.1:8089/health",
            ],
            stdout=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("Server readiness")


for name in args.cases:
    revision = args.baseline if name == "control" else args.revision
    mode = "accuracy" if name == "accuracy" else "decode"
    root = args.output / name
    root.mkdir()
    source = root / "source"
    source.mkdir()
    tree = (
        capture(["git", "-C", str(repo), "rev-parse", revision + "^{tree}"])
        .decode()
        .strip()
    )
    with (root / "source.tar").open("wb") as archive:
        run(["git", "-C", str(repo), "archive", tree], stdout=archive)
    with tarfile.open(root / "source.tar") as archive:
        archive.extractall(source, filter="data")
    (root / "source.tar").unlink()
    tools = root / "tools"
    shutil.copytree(assets, tools)
    for script in ("manifest.py", "verify_gsm8k.py"):
        shutil.copy2(assets.parent / script, tools / script)
    run(["python3", str(tools / "manifest.py"), "source", str(root), tree])
    if args.prepare_only:
        print("Prepared isolated source and tools:", root, flush=True)
        continue
    container = "dsv41-dp8ep8-audit-" + name + "-" + str(int(time.time()))
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--gpus",
        "all",
        "--ipc=host",
        "--network=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--cap-add",
        "IPC_LOCK",
        "-e",
        "DECODE_IMPLEMENTATION=local",
        "-e",
        "DECODE_EP=8",
        "-e",
        "DECODE_MODE=" + mode,
    ]
    if args.unified_deepgemm_site and name != "control":
        site = args.unified_deepgemm_site.resolve()
        if not (site / "deep_gemm/__init__.py").is_file():
            raise RuntimeError(f"Invalid unified DeepGEMM site: {site}")
        command += [
            "-v",
            str(site) + ":/repro/unified_deepgemm:ro",
            "-e",
            "SGLANG_OPT_DSV41_UNIFIED_DEEPGEMM=1",
        ]
    if Path("/dev/infiniband").is_dir():
        command += ["--device", "/dev/infiniband:/dev/infiniband"]
    for host, target in [
        (source, "/repro/source:ro"),
        (tools, "/repro/tools:ro"),
        (root, "/repro/results"),
        (args.runtime / "fa3-pylibs", "/repro/deps:ro"),
        (args.runtime / "eval-pylibs", "/repro/eval-pylibs:ro"),
        (args.runtime / "datasets/gsm8k/main", "/repro/gsm8k-data:ro"),
        (args.runtime / "sglang-cache-migrated", "/repro/cache"),
        (
            Path("/data1/fjmeng2/models/DeepSeek-V4.1-Flash-1017"),
            "/models/DeepSeek-V4.1-Flash-1017:ro",
        ),
    ]:
        command += ["-v", str(host) + ":" + target]
    command += [image, "bash", "/repro/tools/launch_server.sh"]
    (root / "container_id.txt").write_bytes(capture(command))
    try:
        (root / "container.json").write_bytes(capture(["docker", "inspect", container]))
        ready(container)
        info = json.loads(
            capture(
                [
                    "docker",
                    "exec",
                    container,
                    "curl",
                    "--noproxy",
                    "*",
                    "--fail",
                    "--silent",
                    "http://127.0.0.1:8089/get_server_info",
                ]
            )
        )
        (root / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        server = info.get("server_args", info)
        for key, expected in {
            "tp_size": 8,
            "attn_dp_size": 8,
            "dp_size": 1,
            "ep_size": 8,
            "enable_prefill_cp": False,
            "disable_radix_cache": True,
            "speculative_algorithm": "DSPARK",
        }.items():
            if server.get(key) != expected:
                raise RuntimeError(f"{key}: {server.get(key)!r} != {expected!r}")
        expected_role = "null" if mode == "accuracy" else "decode"
        if server["disaggregation_mode"] != expected_role:
            raise RuntimeError(f"Unexpected role: {server['disaggregation_mode']}")
        environment = json.loads((root / "runtime_manifest.json").read_text())[
            "environment"
        ]
        if mode == "accuracy" and any(
            k in environment
            for k in (
                "SGLANG_SIMULATE_ACC_LEN",
                "SGLANG_SIMULATE_ACC_GREEDY",
                "SGLANG_DSPARK_SIMULATE_FOLD",
            )
        ):
            raise RuntimeError("Acceptance simulation must be disabled for GSM8K")
        prefix = [
            "docker",
            "exec",
            "-e",
            "PYTHONPATH=/repro/eval-pylibs",
            container,
            "/opt/sglang/bin/python3",
        ]
        if mode == "decode":
            for label, extra in [
                ("timing", []),
                ("profile", ["--warmup", "0", "--repeat", "1", "--profile", "gpu"]),
            ]:
                with (root / (label + ".log")).open("w") as log:
                    run(
                        prefix
                        + [
                            "/repro/tools/strict_decode.py",
                            "--url",
                            "http://127.0.0.1:8089",
                            "--root",
                            "/repro/results/" + label,
                            "--ep",
                            "8",
                            "--backend",
                            "mooncake",
                            "--reference-config",
                            "--timeout",
                            "600",
                            *extra,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            if len(list((root / "profiles/gpu").glob("*.trace.json.gz"))) != 8:
                raise RuntimeError("Missing GPU profiles")
        else:
            dataset = {
                "gsm8k": {
                    "local_path": "/repro/gsm8k-data",
                    "subset_list": ["default"],
                    "eval_split": "test",
                }
            }
            with (root / "gsm8k.log").open("w") as log:
                run(
                    prefix
                    + [
                        "/repro/eval-pylibs/bin/evalscope",
                        "eval",
                        "--model",
                        "/models/DeepSeek-V4.1-Flash-1017",
                        "--model-id",
                        "dp8ep8-regression",
                        "--api-url",
                        "http://127.0.0.1:8089/v1",
                        "--api-key",
                        "EMPTY",
                        "--eval-type",
                        "openai_api",
                        "--model-task",
                        "text_generation",
                        "--datasets",
                        "gsm8k",
                        "--dataset-args",
                        json.dumps(dataset),
                        "--eval-batch-size",
                        "64",
                        "--generation-config",
                        '{"temperature":0,"max_tokens":2048,"timeout":1800}',
                        "--work-dir",
                        "/repro/results/gsm8k",
                        "--no-timestamp",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            run(
                [
                    "python3",
                    str(tools / "verify_gsm8k.py"),
                    str(root / "gsm8k"),
                    "dp8ep8-regression",
                    "--summary",
                    str(root / "gsm8k-summary.json"),
                ]
            )
        (root / "completed.json").write_text(
            json.dumps({"success": True, "tree": tree}) + "\n"
        )
    finally:
        (root / "server.log").write_bytes(capture(["docker", "logs", container]))
        run(["docker", "rm", "--force", container], stdout=subprocess.DEVNULL)
    print("Completed", name, flush=True)
