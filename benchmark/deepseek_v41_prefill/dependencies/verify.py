"""Check custom dependency provenance without importing CUDA extensions."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("mode", choices=["source", "site"])
parser.add_argument("kind", choices=["dense", "mega", "mhc"])
parser.add_argument("directory", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())[args.kind]
expected = manifest["source_files_sha256" if args.mode == "source" else "binary_sha256"]
actual = {
    p: hashlib.sha256((args.directory / p).read_bytes()).hexdigest() for p in expected
}
if args.report:
    args.report.write_text(
        json.dumps(
            {
                "actual": actual,
                "historical": expected,
                "matches_historical": actual == expected,
            },
            indent=2,
        )
        + "\n"
    )
if args.mode == "source" and actual != expected:
    raise SystemExit("Patched dependency differs from the tested source")
print(
    json.dumps(
        {"kind": args.kind, "mode": args.mode, "matches_historical": actual == expected}
    )
)
