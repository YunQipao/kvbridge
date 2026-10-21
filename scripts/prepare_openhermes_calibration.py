"""Convert the cached OpenHermes dataset to calibration text JSONL, offline."""

import argparse
import json
import os
from pathlib import Path

# Set these before importing datasets: conversion must not download anything.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path.home() / "luotao" / "openhermes_calibration.jsonl"),
    )
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    output = Path(os.path.expandvars(args.output)).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    # Exclusive creation avoids overwriting an existing calibration file.
    with output.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            text = "\n".join(
                f"{item.get('from', 'user')}: {item.get('value', '')}"
                for item in row.get("conversations", [])
            ).strip()
            if text:
                handle.write(json.dumps({"id": index, "text": text},
                                        ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} records: {output}")


if __name__ == "__main__":
    main()
