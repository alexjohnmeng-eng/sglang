"""Verify patched source or isolated wheel structure without importing CUDA."""

import hashlib
import json
import sys
from pathlib import Path

mode, directory = sys.argv[1:]
root = Path(directory)
pins = json.loads(Path(__file__).with_name("manifest.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if mode == "source":
    for name, expected in pins["source_files_sha256"].items():
        assert sha(root / name) == expected, name
elif mode == "site":
    assert not (root / "deep_gemm_mega").exists()
    assert not (root / "deep_gemm_mhc").exists()
    extensions = list((root / "deep_gemm").glob("_C*.so"))
    assert len(extensions) == 1, extensions
    (metadata,) = root.glob("deep_gemm_h20-*.dist-info/METADATA")
    assert "Version: " + pins["version"] in metadata.read_text()
    assert (root / "deep_gemm/H20_PROVENANCE.json").is_file()
    print(json.dumps({str(p.relative_to(root)): sha(p) for p in extensions}, indent=2))
else:
    raise ValueError(mode)
