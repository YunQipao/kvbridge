#!/usr/bin/env python3
"""Print target-layer-specific source-layer rankings from an artifact."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapper", required=True)
    args = p.parse_args()
    path = Path(args.mapper) / "layer_selection.json"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text())
    for row in rows:
        scores = row.get("scores", [])
        ranked = row.get("ranked", [])
        print(f"target layer {row['target_layer']}")
        print(f"  selected: {row.get('selected', [])}")
        print("  ranked: " + ", ".join(
            f"{layer}:{scores[layer]:.6f}" if layer < len(scores)
            else str(layer) for layer in ranked))


if __name__ == "__main__":
    main()
