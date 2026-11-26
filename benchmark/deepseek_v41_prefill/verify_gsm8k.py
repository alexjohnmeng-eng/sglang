"""Verify and summarize a full EvalScope GSM8K run."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("tag")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    report_path = args.output / "reports" / args.tag / "gsm8k.json"
    reviews_path = args.output / "reviews" / args.tag / "gsm8k_default.jsonl"
    report = json.loads(report_path.read_text())
    execution = report["execution_summary"]
    total = 1319
    expected = {
        "requested": total,
        "succeeded": total,
        "errored": 0,
        "incomplete": False,
    }
    if any(execution.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"Incomplete GSM8K results: {execution}")

    seen = {}
    correct = 0
    with reviews_path.open() as file:
        for line in file:
            review = json.loads(line)
            item = review["sample_score"]
            score = item["score"]
            sample_id = item["sample_id"]
            value = score["value"]["accuracy"]
            if sample_id in seen or score["status"] != "success" or value not in (0, 1):
                raise SystemExit(f"Invalid GSM8K review for {sample_id}")
            prompt = [
                [message["role"], message["content"]]
                for message in review["messages"][:-1]
            ]
            seen[sample_id] = [prompt, review["target"]]
            correct += int(value)
    totals = [
        (metric["num_correct"], metric["num"])
        for metric in report["metrics"]
        if metric["identity"]["name"] == "accuracy"
    ]
    if len(seen) != total or totals != [(correct, total)]:
        raise SystemExit(
            f"Reviews and report disagree: {correct}/{len(seen)}, {totals}"
        )

    canonical = json.dumps(sorted(seen.items()), ensure_ascii=False, sort_keys=True)
    summary = {
        "correct": correct,
        "total": total,
        "accuracy_percent": 100 * correct / total,
        "request_errors": execution["errored"],
        "prompt_target_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "report": str(report_path),
        "reviews": str(reviews_path),
    }
    output = json.dumps(summary, indent=2) + "\n"
    if args.summary is not None:
        args.summary.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
