"""Opt-in compatibility recipes; paper defaults live in their original paths.

The external DistKV c2c scripts are not bundled. This module implements the
explicit recipe supplied with this experiment, without claiming bitwise parity.
"""
import json
import re
from pathlib import Path

import torch


def resolve_attn_repair(method, mode="auto"):
    if mode not in ("auto", "paper", "off"):
        raise ValueError("unknown attn repair mode")
    return mode == "paper" or (mode == "auto" and method == "cachebridge")


def pack_jsonl(path, target_tokenizer, source_tokenizer, n_calib, seq_len):
    """Pack unmodified target token IDs; retain per-row token-slice provenance."""
    if n_calib < 1 or seq_len < 2:
        raise ValueError("require positive n_calib and seq_len >= 2")
    records, rows, pending, provenance = [], [], [], []
    with Path(path).open(encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if text is None:
                text = "\n".join(turn["value"] for turn in row["conversations"])
            if not isinstance(text, str):
                raise ValueError(f"{path}:{offset + 1}: text must be a string")
            ids = target_tokenizer(text, add_special_tokens=False).input_ids
            source_ids = source_tokenizer(text, add_special_tokens=False).input_ids
            if ids != source_ids:
                raise RuntimeError(f"source and target tokenizer IDs differ at JSONL line {offset + 1}")
            start = 0
            while start < len(ids):
                take = min(seq_len - len(pending), len(ids) - start)
                pending.extend(ids[start:start + take])
                provenance.append({"line": offset + 1, "id": row.get("id"),
                                   "token_start": start, "token_end": start + take,
                                   "text": text})
                start += take
                if len(pending) == seq_len:
                    records.append({"offset": len(rows), "id": len(rows),
                                    "token_ids": pending, "provenance": provenance,
                                    "token_length": seq_len})
                    rows.append(torch.tensor(pending, dtype=torch.long))
                    pending, provenance = [], []
                    if len(rows) == n_calib:
                        return records, rows
    raise RuntimeError(f"only {len(rows)} complete calibration blocks; needed {n_calib}; "
                       f"discarded {len(pending)} trailing tokens")


def validate_packed_records(records, source_tokenizer, target_tokenizer):
    """Recheck parent token blocks with the current tokenizers, without decode/re-encode."""
    for index, record in enumerate(records):
        reconstructed = []
        for part in record["provenance"]:
            text = part["text"]
            source = source_tokenizer(text, add_special_tokens=False).input_ids
            target = target_tokenizer(text, add_special_tokens=False).input_ids
            if source != target:
                raise RuntimeError(f"packed manifest tokenizer IDs differ in block {index}")
            reconstructed.extend(target[part["token_start"]:part["token_end"]])
        if reconstructed != record["token_ids"]:
            raise ValueError(f"packed manifest token block {index} differs from provenance")


def selection_indices(n, subsample=4096, seed=42):
    if n < 1 or subsample < 1:
        raise ValueError("selection positions and subsample must be positive")
    # Isolate RNG side effects while reproducing reference manual_seed/randperm.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.randperm(n)[:min(n, subsample)]


def ridge_probe_r2(x, y, ridge_lambda, stats_dtype=torch.float64):
    x, y = x.to(stats_dtype), y.to(stats_dtype)
    x, y = x - x.mean(0), y - y.mean(0)
    w = torch.linalg.solve(x.T @ x + ridge_lambda * torch.eye(x.shape[1], dtype=x.dtype), x.T @ y)
    return 1 - (y - x @ w).square().sum() / y.square().sum().clamp_min(1e-18)


def select_layers_ridge_subsample(source_dir, target_dir, n_source_layers,
                                  n_target_layers, n_heads, top_k, head_map,
                                  ridge_lambda=0.01, subsample=4096, seed=42,
                                  stats_dtype=torch.float64):
    """One seeded global position subset shared by every K/V/head/layer probe."""
    if ridge_lambda <= 0 or not 1 <= top_k <= n_source_layers:
        raise ValueError("ridge subsample requires positive lambda and valid top_k")
    files = sorted(Path(target_dir).glob("target_*.pt"))
    lengths = [len(torch.load(p, map_location="cpu", weights_only=False)["keys"][0]) for p in files]
    indices = selection_indices(sum(lengths), subsample, seed)
    source = {k: [] for k in ("keys", "values")}
    target = {k: [] for k in source}
    offset = 0
    collected_indices = []
    for path, length in zip(files, lengths):
        local = indices[(indices >= offset) & (indices < offset + length)] - offset
        collected_indices.extend((local + offset).tolist())
        offset += length
        if not len(local):
            continue
        suffix = path.stem.split("_")[-1]
        src = torch.load(Path(source_dir) / f"source_{suffix}.pt", map_location="cpu", weights_only=False)
        tgt = torch.load(path, map_location="cpu", weights_only=False)
        for component in source:
            source[component].append(torch.stack([x[local] for x in src[component]]))
            target[component].append(torch.stack([x[local] for x in tgt[component]]))
    source = {k: torch.cat(v, dim=1) for k, v in source.items()}
    target = {k: torch.cat(v, dim=1) for k, v in target.items()}
    # Shards were read in file order; restore the reference randperm row order.
    location = {value: i for i, value in enumerate(collected_indices)}
    order = torch.tensor([location[value] for value in indices.tolist()])
    source = {k: v[:, order] for k, v in source.items()}
    target = {k: v[:, order] for k, v in target.items()}
    rankings = []
    for layer in range(n_target_layers):
        scores = []
        for candidate in range(n_source_layers):
            values = []
            for component in source:
                for head in range(n_heads):
                    x = source[component][candidate][:, head_map[head], :].flatten(1)
                    y = target[component][layer, :, head, :]
                    values.append(ridge_probe_r2(x, y, ridge_lambda, stats_dtype))
            scores.append(float(torch.stack(values).mean()))
        order = sorted(range(n_source_layers), key=lambda i: (-scores[i], i))
        rankings.append({"target_layer": layer, "ranked": order,
                         "scores": scores, "selected": order[:top_k],
                         "selection_indices": indices.tolist()})
        print(f"ridge subsample selection target layer {layer + 1}/{n_target_layers}", flush=True)
    return rankings


def greedy_prompt(row):
    choices = "\n".join(f"{chr(65 + i)}. {text}" for i, text in enumerate(row["choices"]))
    return (f"Accurately answer the following question:\n{row['question']}\n"
            f"Choices:\n{choices}\nInstructions:\n"
            "- Carefully read the question and all options.\n"
            "- Select the single most correct answer.\n"
            '- Respond ONLY in the following format: "The correct answer is A/B/C/D/E".\n'
            "- Do not include any explanations, additional text, or punctuation besides the answer.\n"
            "The correct answer is")


def parse_greedy_answer(text):
    match = re.search(r"the correct answer is\s*([A-E])\b", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"\b([A-E])\b", text, re.IGNORECASE)
    return ord(match.group(1).upper()) - 65 if match else -1


@torch.no_grad()
def greedy_decode(model, tokenizer, prompt, cache_factory, device, max_new_tokens=64):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    ctx = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ctx.shape[1] < 2:
        raise ValueError("greedy prompt must contain at least two tokens")
    prefix = ctx.shape[1] - 1
    cache = cache_factory(ctx[:, :-1])
    token, generated = ctx[:, -1:], []
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (tuple, list)) else [eos])
    for step in range(max_new_tokens):
        pos = torch.tensor([prefix + step], device=device)
        output = model(token, past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones(1, prefix + step + 1, dtype=torch.long, device=device),
                       position_ids=pos[None], cache_position=pos)
        next_id = int(output.logits[0, -1].argmax())
        cache = output.past_key_values
        generated.append(next_id)
        del output
        if next_id in eos:
            break
        token = torch.tensor([[next_id]], device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_greedy_answer(text), text, int(prefix), len(generated)
