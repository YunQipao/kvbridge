#!/usr/bin/env python3
"""Compare externally supplied reproduction rows with paper rows."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open() as f:
            return list(csv.DictReader(f))
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else [value]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paper", required=True); p.add_argument("--reproduction", required=True)
    p.add_argument("--output", default="results/paper_comparison.json")
    args = p.parse_args()
    paper, reproduction = read(args.paper), read(args.reproduction)
    result = []
    for left, right in zip(paper, reproduction):
        row = {"paper": left, "reproduction": right, "absolute_difference": {}}
        for key, value in left.items():
            try:
                row["absolute_difference"][key] = abs(float(right[key]) - float(value))
            except (KeyError, TypeError, ValueError):
                pass
        result.append(row)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
