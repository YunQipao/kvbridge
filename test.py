from datasets import load_dataset
import json
ds = load_dataset("teknium/OpenHermes-2.5", split="train")
with open("$HOME/luotao/openhermes_calibration.jsonl", "w") as f:
    for row in ds:
        conv = row.get("conversations", [])
        text = "\n".join(
            f"{x.get('from', 'user')}: {x.get('value', '')}" for x in conv
        ).strip()
        if text:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
