#!/usr/bin/env python3
"""Save exact experiment source and serving dependencies (standard library CLI)."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root, revision):
    files = sorted(p for p in (root / "source").rglob("*") if p.is_file())
    return {
        "source_revision": revision,
        "source_files_sha256": {
            str(p.relative_to(root / "source")): sha256(p) for p in files
        },
        "reproduction_files_sha256": {
            str(p.relative_to(root / "tools")): sha256(p)
            for p in sorted((root / "tools").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        },
    }


def runtime_manifest():
    import humming
    import sgl_fa3
    import sgl_kernel
    import torch

    expected = {
        "torch": "2.14.1+cu130",
        "humming-kernels": "0.1.12",
        "sglang-kernel": "0.4.11",
        "sgl-fa3": "0.1.0",
        "sgl-deep-gemm": "0.2.3.post2",
        "flashinfer-python": "0.7.2",
        "transformers": "5.19.0",
        "triton": "3.8.0",
        "tilelang": "0.1.12",
    }
    versions = {k: importlib.metadata.version(k) for k in expected}
    mismatches = {k: [expected[k], v] for k, v in versions.items() if v != expected[k]}
    result = {
        "python": platform.python_version(),
        "versions": versions,
        "expected_version_mismatches": mismatches,
        "cuda_runtime": torch.version.cuda,
        "modules": {m.__name__: m.__file__ for m in (humming, sgl_fa3, sgl_kernel)},
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("SGLANG_", "PYTORCH_", "CUDA_", "NCCL_"))
            and not any(word in k for word in ("KEY", "TOKEN", "SECRET"))
        },
        "pythonpath": os.environ["PYTHONPATH"],
    }
    if mismatches:
        raise RuntimeError(f"Serving dependency drift: {json.dumps(result, indent=2)}")
    if os.environ.get("SGLANG_OPT_DSV41_MEGA_MHC_PREFILL") == "1":
        import deep_gemm_mhc

        result["modules"]["deep_gemm_mhc"] = deep_gemm_mhc.__file__
        result["versions"]["deepgemm_mega_mhc"] = deep_gemm_mhc.__version__
    return result


def main():
    mode, output, *args = sys.argv[1:]
    root = Path(output)
    if mode == "source":
        result = source_manifest(root, args[0])
    elif mode == "runtime":
        result = runtime_manifest()
    else:
        raise ValueError(mode)
    (root / f"{mode}_manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
