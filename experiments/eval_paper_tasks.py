"""Paper-task multiple-choice evaluator with mapped-cache accounting.

This is a small harness-independent scorer for the four paper task families.
Dataset schemas are normalized locally and all mapped modes use source
prefill -> transfer -> target suffix scoring; target prefix prefill is asserted
to remain zero in the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kvbridge.mapper import Mapper
from kvbridge.model_utils import load_causal_lm, load_tokenizer, model_input_device
from kvbridge.model_utils import assert_text_tokenizer_alignment
from kvbridge.model_utils import get_model_layer_devices
from kvbridge.protocol import (assert_mapper_method, make_evaluation_manifest,
                               validate_evaluation_manifest,
                               validate_mapped_evaluation_accounting)
from kvbridge.task_protocol import load_lm_eval_task, normalize_task_doc
from kvbridge.transfer import transfer
from experiments.eval_mmlu_redux import _score


TASKS = {
    "HellaSwag": ("Rowan/hellaswag", None, "validation"),
    "ARC-Challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "WinoGrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "MMLU": ("cais/mmlu", None, "test"),
}


def _mapped_prefix_prefill_guard(accounting):
    def guard(_module, args, kwargs):
        if kwargs.get("past_key_values") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            accounting["target_prefix_prefill_tokens"] += (
                int(input_ids.shape[-1]) if input_ids is not None else 0
            )
            raise ValueError("mapped evaluation attempted target prefix prefill")
    return guard


def _lm_eval_version():
    try:
        import lm_eval
        return getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        return None


def load_rows_simple(task, dataset=None, config=None, split="test"):
    from datasets import get_dataset_config_names, load_dataset
    dataset = dataset or TASKS[task][0]
    config = config or TASKS[task][1]
    split = split if split != "test" else TASKS[task][2]
    configs = [config] if config else get_dataset_config_names(dataset)
    rows = []
    for name in configs:
        for row in load_dataset(dataset, name, split=split):
            if task == "HellaSwag":
                choices = row["endings"]
                prompt = row["ctx"]
                answer = int(row["label"])
            elif task == "ARC-Challenge":
                labels, choices0 = row["choices"]["label"], row["choices"]["text"]
                choices = choices0
                answer = labels.index(row["answerKey"]) if row["answerKey"] in labels else int(row["answerKey"])
                prompt = row["question"]
            elif task == "WinoGrande":
                choices = [row["option1"], row["option2"]]
                answer = int(row["answer"])
                prompt = row["sentence"].replace("_", "{}")
            else:
                choices = row["choices"]
                answer = int(row["answer"])
                prompt = row["question"]
            rows.append({"prompt": prompt, "choices": choices, "answer": answer, "config": name})
    return rows


def load_rows(task, protocol="lm_eval"):
    if protocol == "lm_eval":
        names = {"HellaSwag": "hellaswag", "ARC-Challenge": "arc_challenge",
                 "WinoGrande": "winogrande", "MMLU": "mmlu"}
        lm_task, docs = load_lm_eval_task(names[task])
        return [normalize_task_doc(names[task], doc, lm_task) for doc in docs]
    return load_rows_simple(task)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--mode", choices=["target_native", "full_head", "head_local", "cachebridge"], required=True)
    p.add_argument("--source", required=True); p.add_argument("--target", required=True)
    p.add_argument("--target-tokenizer",
                   help="explicit target tokenizer reference when target snapshot omits tokenizer files")
    p.add_argument("--mapper"); p.add_argument("--num-eval-samples", type=int)
    p.add_argument("--sampling", choices=["systematic", "random"], default="systematic")
    p.add_argument("--sampling-seed", type=int, default=42); p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", choices=["none", "auto", "balanced", "balanced_low_0", "sequential"],
                   default=None, help="HF/Accelerate model sharding; default keeps single-device loading")
    p.add_argument("--output", default=None); p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--evaluation-manifest")
    p.add_argument("--evaluation-manifest-output")
    p.add_argument("--protocol", choices=["lm_eval", "simple"], default="lm_eval")
    args = p.parse_args()
    if args.device_map == "none":
        args.device_map = None
    if args.mode in ("full_head", "cachebridge", "head_local") and not args.mapper:
        raise ValueError("--mapper required for mapped modes")
    rows = load_rows(args.task, args.protocol)
    if args.evaluation_manifest:
        manifest = json.loads(Path(args.evaluation_manifest).read_text())
    else:
        manifest = make_evaluation_manifest(len(rows), args.num_eval_samples,
                                            args.sampling_seed, args.sampling)
    if args.evaluation_manifest_output:
        manifest_path = Path(args.evaluation_manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    rows = [rows[i] for i in validate_evaluation_manifest(manifest, len(rows))]
    target_tok = load_tokenizer(
        args.target_tokenizer or args.target,
        local_files_only=args.local_files_only,
    )
    target = load_causal_lm(args.target, device=args.device, dtype="auto",
                            device_map=args.device_map,
                            local_files_only=args.local_files_only)
    source_tok = load_tokenizer(args.source, local_files_only=args.local_files_only)
    source = None
    mapper = Mapper.load(args.mapper, device="cpu" if args.device_map else args.device,
                         dtype=torch.float32) if args.mapper else None
    if mapper is not None:
        assert_mapper_method(mapper, "cachebridge" if args.mode == "cachebridge" else args.mode)
    if args.mode != "target_native":
        source = load_causal_lm(args.source, device=args.device, dtype="auto",
                                device_map=args.device_map,
                                local_files_only=args.local_files_only)
    device = model_input_device(target)
    source_device = model_input_device(source) if source is not None else None
    layer_devices = get_model_layer_devices(target) if args.device_map and mapper is not None else None
    accounting = {"source_prefix_prefill_tokens": 0, "target_prefix_prefill_tokens": 0}
    guard_handle = None
    if args.mode != "target_native":
        guard_handle = target.register_forward_pre_hook(
            _mapped_prefix_prefill_guard(accounting), with_kwargs=True)
    correct = 0
    try:
        for row in rows:
            def factory(ids):
                if args.mode == "target_native":
                    accounting["target_prefix_prefill_tokens"] += int(ids.shape[1])
                    return target(ids, use_cache=True).past_key_values
                accounting["source_prefix_prefill_tokens"] += int(ids.shape[1])
                cache = source(ids.to(source_device), use_cache=True).past_key_values
                if layer_devices is not None:
                    return transfer(cache, mapper, layer_devices=layer_devices)
                return transfer(cache, mapper)
            prompt = row["prompt"]
            choices = row["choices"]
            if "{}" in prompt:
                prefix, suffix = prompt.split("{}", 1)
                prompt, choices = prefix, [choice + suffix for choice in choices]
            elif args.task == "WinoGrande" and "_" in prompt:
                prefix, suffix = prompt.split("_", 1)
                prompt, choices = prefix, [choice + suffix for choice in choices]
            if args.mode != "target_native":
                assert_text_tokenizer_alignment(source_tok, target_tok, prompt)
            pred, _, _, _ = _score(target, target_tok, prompt, choices,
                                   factory, device)
            correct += int(pred == row["answer"])
    finally:
        if guard_handle is not None:
            guard_handle.remove()
    if args.mode != "target_native":
        validate_mapped_evaluation_accounting(accounting)
    result = {"task": args.task, "mode": args.mode, "total": len(rows),
              "accuracy": correct / max(len(rows), 1), "correct": correct,
              "evaluation_manifest": manifest, "protocol": args.protocol,
              "lm_eval_version": _lm_eval_version() if args.protocol == "lm_eval" else None,
              **accounting}
    out = Path(args.output or f"results/{args.task.lower().replace('-', '_')}_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
