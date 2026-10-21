"""Opt-in DistKV sample protocol and cache continuation compatibility.

Reference: DistKV origin/c2c-cachebridge examples/c2c/eval_cachebridge.py.
Does not replace the historical current-profile evaluator.
"""
import hashlib
import json
import re

import torch

PROMPT = (
    "Accurately answer the following question:\n{question}\nChoices:\n{choices}\n"
    "Instructions:\n- Carefully read the question and all options.\n"
    "- Select the single most correct answer.\n"
    '- Respond ONLY in the following format: "The correct answer is A/B/C/D/E".\n'
    "- Do not include any explanations, additional text, or punctuation besides the answer.\n"
    "The correct answer is"
)
_EXPLICIT_ANSWER = re.compile(r"the\s+correct\s+answer\s+is\s*\(?\s*([A-E])\s*\)?", re.I)
_STANDALONE_ANSWER = re.compile(r"\b([A-E])\b", re.I)


def choice_text(value):
    if isinstance(value, dict):
        labels, texts = value.get("label"), value.get("text")
        if labels and texts:
            return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=True))
        value = texts or value
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, dict) for item in value):
            lines = []
            for index, item in enumerate(value):
                label = item.get("label") or chr(65 + index)
                text = item.get("text")
                if text is None:
                    raise ValueError("Choice mapping is missing its 'text' field")
                lines.append(f"{label}. {text}")
            return "\n".join(lines)
        return "\n".join(f"{chr(65 + index)}. {text}" for index, text in enumerate(value))
    raise ValueError(f"Unsupported choices value: {type(value).__name__}")


def letter(value):
    text = str(value).strip()
    if text.isdigit():
        index = int(text)
        return chr(65 + index) if 0 <= index <= 4 else None
    match = _EXPLICIT_ANSWER.search(text) or _STANDALONE_ANSWER.search(text)
    return match.group(1).upper() if match else None


def prepare_samples(dataset, config_name=None, *, global_offset=0):
    samples = []
    for index, row in enumerate(dataset):
        if str(row.get("error_type", "")).lower() == "no correct answer":
            continue
        choices = choice_text(row["choices"])
        prompt = PROMPT.format(question=row["question"], choices=choices)
        answer = letter(row["answer"])
        if answer:
            sample = dict(original_index=index, prompt=prompt, answer=answer,
                          question=row["question"], choices=choices)
            if config_name is not None:
                sample.update(original_index=f"{config_name}:{index}",
                              sample_id=f"{config_name}:{index}",
                              subject=config_name, config_name=config_name,
                              row_index=index, original_row_index=index,
                              global_row_index=global_offset + index)
            samples.append(sample)
    return samples


def load_sample_groups(dataset_path=None, dataset="edinburgh-dawg/mmlu-redux-2.0", split="test"):
    """Return ordered raw groups plus provenance; never discover configs online."""
    if dataset_path is not None:
        from datasets import load_from_disk
        from pathlib import Path
        if not str(dataset_path).strip():
            raise ValueError("--dataset-local-path is empty; omit it to use cached configs")
        groups = [(None, load_from_disk(dataset_path))]
        metadata = dict(dataset_loader="load_from_disk", dataset_source="datasets.load_from_disk",
                        dataset_local_path=str(Path(dataset_path).resolve()))
    else:
        from kvbridge.offline_dataset import cached_dataset_splits
        groups = sorted(cached_dataset_splits(dataset, split), key=lambda item: item[0])
        metadata = dict(dataset_loader="hf_cached_configs", dataset_source="hf_cached_configs",
                        dataset=dataset, split=split,
                        config_order=[name for name, _ in groups])
    return groups, metadata


def prepare_group_samples(groups):
    # Offsets count ALL raw rows, including rows later rejected by the protocol.
    # Filtering each ordered group is equivalent to filtering their concatenation.
    samples = []
    offset = 0
    for name, rows in groups:
        samples.extend(prepare_samples(rows, name, global_offset=offset))
        offset += len(rows)
    return samples


def sample_sha256(samples):
    ordered = [[s["original_index"], s["prompt"], s["answer"]] for s in samples]
    return hashlib.sha256(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def continuation_mode(profile, explicit=None):
    default = "distkv_duplicate_last" if profile == "distkv_exact" else "bridge_once"
    if profile == "current" and explicit not in (None, "bridge_once"):
        raise ValueError("current profile only supports its historical bridge_once semantics")
    return explicit or default


def profile_rows(dataset_path, profile, manifest=None, count=None, seed=42, sampling="systematic",
                 *, dataset="edinburgh-dawg/mmlu-redux-2.0", split="test"):
    from kvbridge.protocol import make_evaluation_manifest, validate_evaluation_manifest
    groups, source_metadata = load_sample_groups(dataset_path, dataset, split)
    samples = prepare_group_samples(groups)
    fingerprint = sample_sha256(samples)
    metadata = dict(**source_metadata,
                    raw_rows=sum(len(rows) for _, rows in groups),
                    config_count=sum(name is not None for name, _ in groups),
                    filtered_rows=sum(len(rows) for _, rows in groups) - len(samples),
                    final_rows=len(samples),
                    answer_field="answer", invalid_filter={"error_type": "no correct answer"},
                    eval_profile=profile, cache_continuation_mode=continuation_mode(profile),
                    protocol="distkv_greedy", population_sha256=fingerprint)
    if manifest is None:
        if sampling == "first":
            n = len(samples) if count is None else int(count)
            if not 0 <= n <= len(samples):
                raise ValueError("first sampling requires 0 <= count <= population")
            manifest = make_evaluation_manifest(len(samples), n, seed, "systematic")
            manifest.update(sampling="first", indices=list(range(n)), seed_affects_selection=False,
                            provenance="first_after_distkv_filter")
        else:
            manifest = make_evaluation_manifest(len(samples), count, seed, sampling)
        indices = manifest["indices"]
        manifest.update(metadata, samples=[samples[i] for i in indices])
    else:
        # canonical/exact deliberately share samples; only continuation differs.
        if manifest.get("eval_profile") not in ("distkv_canonical", "distkv_exact"):
            raise ValueError("manifest is not a DistKV profile manifest")
        for key in ("dataset_source", "answer_field", "invalid_filter", "population_sha256"):
            if manifest.get(key) != metadata[key]:
                raise ValueError(f"DistKV manifest mismatch: {key}")
        if manifest.get("dataset_loader", "load_from_disk") != metadata["dataset_loader"]:
            raise ValueError("DistKV manifest mismatch: dataset_loader")
        indices = validate_evaluation_manifest(manifest, len(samples))
        if manifest.get("samples") != [samples[i] for i in indices]:
            raise ValueError("DistKV manifest sample order/content mismatch")
    indices = validate_evaluation_manifest(manifest, len(samples))
    rows = [{**samples[i], "subject": samples[i].get("config_name", "mmlu_redux"), "answer": ord(samples[i]["answer"]) - 65}
            for i in indices]
    return rows, manifest


@torch.no_grad()
def decode(model, tokenizer, prompt, cache_factory, device, max_new_tokens=64,
           mode="bridge_once"):
    if mode not in ("bridge_once", "distkv_duplicate_last") or max_new_tokens < 1:
        raise ValueError("invalid continuation mode or max_new_tokens")
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    n = ids.shape[1]
    if n < 2:
        raise ValueError("prompt must have at least two tokens")
    prefix = n if mode == "distkv_duplicate_last" else n - 1
    cache = cache_factory(ids[:, :prefix])
    # Exact reference does not pass attention_mask or position_ids.
    output = model(input_ids=ids[:, -1:], past_key_values=cache,
                   cache_position=torch.tensor([n - 1], device=device),
                   use_cache=True, logits_to_keep=1)
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    cache = output.past_key_values
    del output
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    tokens = []
    for step in range(max_new_tokens):
        value = token.item()
        tokens.append(value)
        if value in eos:
            break
        output = model(input_ids=token.to(device), past_key_values=cache,
                       cache_position=torch.tensor([n + step], device=device),
                       use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        cache = output.past_key_values
        del output
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    answer = letter(text)
    return (ord(answer) - 65 if answer else -1), text, prefix, len(tokens)
