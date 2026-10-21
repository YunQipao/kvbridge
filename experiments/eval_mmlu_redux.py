#!/usr/bin/env python3
"""MMLU-Redux conditional-likelihood evaluator.

The mapped mode uses source-prefill -> CacheBridge transfer and scores only
answer continuations with the target. It never prefills the shared prompt on
the target in mapped mode. Dataset field names are normalized for the common
MMLU-Redux releases and unknown schemas fail explicitly.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import torch
from kvbridge.cache_compat import cache_layer_kv, cache_num_layers
from kvbridge.mapper import Mapper
from kvbridge.model_utils import (load_causal_lm, load_tokenizer, model_input_device,
    get_model_layer_devices, assert_text_tokenizer_alignment)
from kvbridge.transfer import transfer
from kvbridge.protocol import (assert_mapper_method, make_evaluation_manifest,
                               validate_evaluation_manifest, validate_mapped_evaluation_accounting)
from kvbridge.task_protocol import load_lm_eval_task, normalize_task_doc, protocol_metadata
from transformers import DynamicCache
from kvbridge.distkv_compat import greedy_prompt, greedy_decode


def _clone(cache):
    out = DynamicCache()
    for l in range(cache_num_layers(cache)):
        k, v = cache_layer_kv(cache, l); out.update(k.clone(), v.clone(), l)
    return out


def _expand(cache, batch):
    out = DynamicCache()
    for l in range(cache_num_layers(cache)):
        k, v = cache_layer_kv(cache, l)
        out.update(k.expand(batch, -1, -1, -1).contiguous(),
                   v.expand(batch, -1, -1, -1).contiguous(), l)
    return out


def _rows(dataset, split, limit, config=None, *, valid_only=False, local_files_only=False):
    from datasets import load_dataset
    offline = local_files_only or any(os.environ.get(key, "").upper() in ("1", "TRUE", "YES", "ON")
                                     for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))
    if offline:
        from kvbridge.offline_dataset import cached_dataset_splits
        datasets = cached_dataset_splits(dataset, split, config)
    else:
        if config:
            configs = [config]
        else:
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names(dataset)
        datasets = ((name, load_dataset(dataset, name, split=split)) for name in configs)
    out = []
    for name, ds in datasets:
        for row in ds:
            question = row.get("question") or row.get("Question")
            choices = row.get("choices") or row.get("options")
            answer = row.get("answer", row.get("correct_answer"))
            if valid_only:
                # Redux corrections supersede the original label; annotations
                # without a usable corrected answer cannot be scored.
                correction = row.get("correct_answer")
                if correction is not None and str(correction).strip():
                    answer = correction
                try:
                    parsed = (ord(answer.strip().upper()) - 65 if isinstance(answer, str)
                              and len(answer.strip()) == 1 and answer.strip().upper() in "ABCDE"
                              else int(answer))
                except (ValueError, TypeError):
                    continue
                if (not isinstance(question, str) or not isinstance(choices, (list, tuple))
                        or not 2 <= len(choices) <= 5 or not 0 <= parsed < len(choices)):
                    continue
                answer = parsed
            if question is None or choices is None or answer is None:
                raise RuntimeError(f"unsupported MMLU-Redux schema: {list(row)}")
            if isinstance(answer, str):
                answer = ord(answer.upper()) - ord("A") if len(answer) == 1 else int(answer)
            out.append({"subject": name, "question": question,
                        "choices": list(choices), "answer": int(answer)})
            if limit and len(out) >= limit:
                return out
    return out


def _prompt(row, fewshot_rows=()):
    if "prompt" in row:
        # Harness-normalized rows already carry the official task prompt and
        # answer continuations; do not reformat these as simple debug rows.
        examples = "".join(example["prompt"] + " " +
                           example["choices"][example["answer"]] + "\n\n"
                           for example in fewshot_rows)
        return examples + row["prompt"], row["choices"]
    labels = ["A", "B", "C", "D"]
    examples = []
    for example in fewshot_rows:
        examples.append(example["question"] + "\n" + "\n".join(
            f"{labels[i]}. {x}" for i, x in enumerate(example["choices"]))
            + f"\nAnswer: {labels[example['answer']]}\n")
    text = "".join(examples) + row["question"] + "\n" + "\n".join(
        f"{labels[i]}. {x}" for i, x in enumerate(row["choices"])) + "\nAnswer:"
    return text, labels


def _rows_lm_eval():
    task, docs = load_lm_eval_task("mmlu_redux", split="test")
    return [{"subject": doc.get("subject", ""), **normalize_task_doc(
        "mmlu", doc, task)} for doc in docs]


def _lm_eval_fewshot_rows(task, count):
    """Get demonstrations from the task's official non-test document source."""
    if count <= 0:
        return []
    for name in ("fewshot_docs", "training_docs"):
        getter = getattr(task, name, None)
        if getter is None:
            continue
        try:
            docs = getter() if name == "training_docs" else getter()
        except TypeError:
            docs = getter(k=count)
        docs = list(docs or [])
        if docs:
            return [{"subject": doc.get("subject", ""), **normalize_task_doc(
                "mmlu", doc, task)} for doc in docs[:count]]
    raise NotImplementedError(
        "lm_eval task does not expose official few-shot/training documents; "
        "refusing to use evaluation rows as demonstrations"
    )

@torch.no_grad()
def _chunked_continuation_scores(logits, targets, *, logit_starts=None, chunk_size=1):
    """Sum only continuation log-probs, with at most chunk_size FP32 rows.

    logits is [choice, returned_position, vocabulary]. logit_starts indexes
    the row predicting each choice's first token, NOT that token's own input
    position. With a full prompt this is prompt_length - 1; _score's cached
    prefix and bridge token make it zero in the returned suffix logits.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if logits.ndim != 3 or len(targets) != logits.shape[0]:
        raise ValueError("expected one target sequence per choice in [choice, position, vocab] logits")
    if logit_starts is None:
        logit_starts = [0] * len(targets)
    if len(logit_starts) != len(targets):
        raise ValueError("expected one logit start per choice")
    scores = []
    for choice, (ids, start) in enumerate(zip(targets, logit_starts)):
        if not ids or start < 0 or start + len(ids) > logits.shape[1]:
            raise ValueError("continuation scoring range is empty or outside returned logits")
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        for offset in range(0, len(ids), chunk_size):
            end = min(offset + chunk_size, len(ids))
            # Slice FIRST, including the choice dimension. No full FP32 logits
            # or log-probability tensor, and no padded/prompt rows are scored.
            selected = logits[choice, start + offset:start + end, :]
            rows_fp32 = selected.float()
            labels = torch.tensor(ids[offset:end], device=logits.device, dtype=torch.long)
            target_logits = rows_fp32.gather(-1, labels[:, None]).squeeze(-1)
            normalizers = torch.logsumexp(rows_fp32, dim=-1)
            token_log_probs = target_logits - normalizers
            total.add_(token_log_probs.sum())
            del selected, rows_fp32, labels, target_logits, normalizers, token_log_probs
        scores.append(float(total))
        del total
    return scores


@torch.no_grad()
def _score(model, tokenizer, prompt, choices, cache_factory, device, *, score_chunk_size=1):
    ctx = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    # Keep ctx[:-1] in cache; feeding its last token makes logits align with
    # the first candidate token, including a one-token answer.
    cache = cache_factory(ctx[:, :-1])
    rows, targets = [], []
    for choice in choices:
        ids = tokenizer(" " + choice, add_special_tokens=False).input_ids
        if not ids: raise RuntimeError("empty candidate tokenization")
        rows.append([int(ctx[0, -1])] + ids[:-1]); targets.append(ids)
    L = max(map(len, rows)); inp = torch.full((len(rows), L), tokenizer.eos_token_id or 0,
        dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), ctx.shape[1]-1+L), dtype=torch.long, device=device)
    mask[:, :ctx.shape[1]-1] = 1
    for i, row in enumerate(rows):
        inp[i, :len(row)] = torch.tensor(row, device=device); mask[i, ctx.shape[1]-1:ctx.shape[1]-1+len(row)] = 1
    pos = (ctx.shape[1]-1 + torch.arange(L, device=device))[None].expand(len(rows), -1)
    out = model(inp, attention_mask=mask, position_ids=pos, past_key_values=_expand(cache, len(rows)))
    # Returned position 0 is the final prompt (bridge) token: its logits
    # predict targets[i][0]. Prefix logits are absent because of the KV cache.
    vals = _chunked_continuation_scores(out.logits, targets, chunk_size=score_chunk_size)
    return max(range(len(vals)), key=vals.__getitem__), vals, int(ctx.shape[1]-1), sum(map(len, targets))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode", choices=["target_native","source_native","full_head","head_local","cachebridge"], required=True)
    p.add_argument("--source", required=True); p.add_argument("--target", required=True); p.add_argument("--mapper")
    p.add_argument("--dataset", default="edinburgh-dawg/mmlu-redux-2.0"); p.add_argument("--config", default=None); p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int); p.add_argument("--fewshot", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", choices=["none", "auto", "balanced", "balanced_low_0", "sequential"], default=None)
    p.add_argument("--output", default=None); p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--num-eval-samples", type=int); p.add_argument("--sampling",
              choices=["systematic", "random", "first"], default="systematic")
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--evaluation-manifest")
    p.add_argument("--evaluation-manifest-output")
    p.add_argument("--protocol", choices=["lm_eval", "simple", "distkv_greedy"], default="lm_eval")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-profile", choices=["current", "distkv_canonical", "distkv_exact"], default="current")
    p.add_argument("--dataset-local-path")
    p.add_argument("--cache-continuation-mode", choices=["bridge_once", "distkv_duplicate_last"])
    args=p.parse_args()
    from kvbridge.distkv_profiles import continuation_mode, profile_rows, decode as profile_decode
    args.cache_continuation_mode = continuation_mode(args.eval_profile, args.cache_continuation_mode)
    compat = args.eval_profile != "current"
    if compat:
        args.protocol = "distkv_greedy"
    elif args.sampling == "first":
        p.error("--sampling first requires a DistKV evaluation profile")
    if args.protocol == "distkv_greedy" and (args.fewshot != 0 or args.max_new_tokens < 1):
        raise ValueError("distkv_greedy requires fewshot=0 and positive max-new-tokens")
    if args.device_map == "none":
        args.device_map = None
    mapped = args.mode in ("cachebridge", "full_head", "head_local")
    if mapped and not args.mapper: raise ValueError("--mapper is required for mapped modes")
    lm_task = None
    if compat:
        supplied = json.loads(Path(args.evaluation_manifest).read_text()) if args.evaluation_manifest else None
        rows, manifest = profile_rows(args.dataset_local_path, args.eval_profile, supplied,
                                     args.num_eval_samples or args.limit, args.sampling_seed, args.sampling,
                                     dataset=args.dataset, split=args.split)
        if supplied is None:
            manifest["cache_continuation_mode"] = args.cache_continuation_mode
    elif args.protocol == "lm_eval":
        try:
            lm_task, lm_docs = load_lm_eval_task("mmlu_redux", split=args.split)
            rows = [{"subject": doc.get("subject", ""), **normalize_task_doc(
                "mmlu", doc, lm_task)} for doc in lm_docs]
        except (KeyError, RuntimeError) as exc:
            # lm-eval task registrations are optional and differ between
            # harness releases.  MMLU-Redux's canonical HF dataset has the
            # same question/choices/answer semantics, so use it locally when
            # the task YAML is absent. This keeps the evaluation population
            # MMLU-Redux while avoiding an unrelated harness-version failure.
            print(f"warning: lm_eval mmlu_redux task unavailable ({exc}); "
                  "using the local MMLU-Redux dataset adapter", flush=True)
            if args.fewshot:
                raise RuntimeError("--fewshot requires a registered lm_eval mmlu_redux task") from exc
            rows = _rows(args.dataset, args.split, None, args.config, local_files_only=args.local_files_only)
    elif args.protocol == "distkv_greedy":
        rows = _rows(args.dataset, args.split, None, args.config, valid_only=True, local_files_only=args.local_files_only)
    else:
        rows = _rows(args.dataset, args.split, None, args.config, local_files_only=args.local_files_only)
    population_fingerprint = None
    if args.protocol == "distkv_greedy" and not compat:
        population_fingerprint = hashlib.sha256(
            json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if compat:
        if not args.evaluation_manifest:
            manifest_path = Path(args.output).parent / "evaluation_manifest.json" if args.output else Path("results/evaluation_manifest.json")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
    elif args.evaluation_manifest:
        manifest = json.loads(Path(args.evaluation_manifest).read_text())
        if args.protocol == "distkv_greedy" and manifest.get("population_sha256") != population_fingerprint:
            raise ValueError("distkv_greedy requires a matching greedy population manifest; create a new manifest")
    else:
        manifest = make_evaluation_manifest(
            len(rows), args.num_eval_samples or args.limit,
            args.sampling_seed, args.sampling)
        if args.protocol == "distkv_greedy":
            manifest.update(population_sha256=population_fingerprint,
                            dataset=args.dataset, split=args.split, config=args.config,
                            protocol=args.protocol)
        manifest_path = Path(args.output).parent / "evaluation_manifest.json" if args.output else Path("results/evaluation_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if args.evaluation_manifest_output:
        manifest_path = Path(args.evaluation_manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if not compat:
        sampled_indices = validate_evaluation_manifest(manifest, len(rows))
        rows = [rows[i] for i in sampled_indices]
    if args.protocol == "lm_eval":
        fewshot_rows = _lm_eval_fewshot_rows(lm_task, args.fewshot)
    else:
        # The simple protocol is explicitly a debug formatter.  Keep its
        # historical behavior while making the non-formal status visible.
        fewshot_rows = [] if args.fewshot <= 0 else [
            rows[i] for i in range(min(args.fewshot, len(rows)))
        ]
    target_tok=load_tokenizer(args.target, local_files_only=args.local_files_only)
    target = (load_causal_lm(args.target, device=args.device, device_map=args.device_map,
              dtype="auto", local_files_only=args.local_files_only)
              if args.mode != "source_native" else None)
    source_tok=load_tokenizer(args.source, local_files_only=args.local_files_only); source=None
    mapper=Mapper.load(args.mapper, device="cpu" if args.device_map else args.device,
                       dtype=torch.float32) if args.mapper else None
    if mapper is not None:
        assert_mapper_method(mapper, args.mode)
    if mapped or args.mode == "source_native":
        source=load_causal_lm(args.source, device=args.device, device_map=args.device_map,
                              dtype="auto", local_files_only=args.local_files_only)
    source_device = model_input_device(source) if source is not None else None
    layer_devices = get_model_layer_devices(target) if args.device_map and mapped else None
    ti=0; per={}; generations=[]; t0=time.time(); accounting={"source_prefill_tokens":0,"mapped_tokens":0,"target_prefix_prefill_tokens":0,"target_suffix_tokens":0}
    eval_model = source if args.mode == "source_native" else target
    eval_tok = source_tok if args.mode == "source_native" else target_tok
    if compat:
        # Reference encodes/decodes teacher and student with the target tokenizer.
        eval_tok = target_tok
    eval_device = model_input_device(eval_model)
    prefill_options = {"logits_to_keep": 1} if compat else {}
    for row in rows:
        if compat:
            prompt, labels = row["prompt"], None
        elif args.protocol == "distkv_greedy":
            prompt, labels = greedy_prompt(row), None
        else:
            prompt,labels=_prompt(row, fewshot_rows)
        if args.mode == "target_native":
            def factory(ids): accounting["target_prefix_prefill_tokens"] += int(ids.shape[1]); return target(ids, use_cache=True, **prefill_options).past_key_values
        elif args.mode == "source_native":
            def factory(ids): accounting["source_prefill_tokens"] += int(ids.shape[1]); return source(ids, use_cache=True, **prefill_options).past_key_values
        elif args.mode in ("cachebridge", "full_head", "head_local"):
            def factory(ids):
                accounting["source_prefill_tokens"] += int(ids.shape[1])
                c=source(ids.to(source_device), use_cache=True, **prefill_options).past_key_values
                m = (transfer(c, mapper, layer_devices=layer_devices)
                     if layer_devices is not None else transfer(c, mapper))
                accounting["mapped_tokens"] += int(ids.shape[1])
                return m
        else:
            raise ValueError(f"unsupported mode: {args.mode}")
        if args.mode == "source_native" and args.protocol != "distkv_greedy":
            prompt, _ = _prompt(row)
        if mapped or args.protocol == "distkv_greedy":
            assert_text_tokenizer_alignment(source_tok, target_tok, prompt)
        if compat:
            pred, text, prefix_n, suffix_n = profile_decode(
                eval_model, eval_tok, prompt, factory, eval_device, args.max_new_tokens,
                args.cache_continuation_mode)
            generations.append({"original_index": row["original_index"], "prediction": pred,
                                "answer": row["answer"], "text": text,
                                **{key: row[key] for key in ("sample_id", "subject", "config_name", "row_index",
                                                           "original_row_index", "global_row_index") if key in row}})
        elif args.protocol == "distkv_greedy":
            pred, text, prefix_n, suffix_n = greedy_decode(
                eval_model, eval_tok, prompt, factory, eval_device, args.max_new_tokens)
            generations.append({"prediction": pred, "answer": row["answer"], "text": text})
        else:
            pred,_,prefix_n,suffix_n=_score(eval_model, eval_tok, prompt, labels, factory, eval_device)
        ti += pred == row["answer"]; per[row["subject"]]=per.get(row["subject"], [0,0]); per[row["subject"]][0]+=pred==row["answer"]; per[row["subject"]][1]+=1; accounting["target_suffix_tokens"] += suffix_n
        if mapped:
            accounting["shared_prefix_tokens"] = accounting.get("shared_prefix_tokens", 0) + prefix_n
    if mapped:
        validate_mapped_evaluation_accounting(accounting)
    result={"mode":args.mode,"source":args.source,"target":args.target,"mapper":args.mapper,
            "dataset":args.dataset,"limit":len(rows),"accuracy":ti/max(len(rows),1),
            "correct":ti,"total":len(rows),
            "per_subject_accuracy":{k:v[0]/v[1] for k,v in per.items()},
            "runtime":time.time()-t0,"evaluation_manifest":manifest,
            **protocol_metadata(args.protocol, "mmlu_redux", args.fewshot),
            **accounting}
    if args.protocol == "distkv_greedy":
        result.update(max_new_tokens=args.max_new_tokens, decoding="greedy_argmax",
                      add_special_tokens=False, generations=generations,
                      compatibility_reference="user-specified DistKV recipe; external c2c scripts unavailable")
    result.update(eval_profile=args.eval_profile, cache_continuation_mode=args.cache_continuation_mode)
    if compat:
        result.update(dataset_source=manifest["dataset_source"],
                      dataset_loader=manifest.get("dataset_loader", "load_from_disk"),
                      dataset_local_path=args.dataset_local_path,
                      answer_field="answer", invalid_filter={"error_type": "no correct answer"},
                      compatibility_reference="DistKV origin/c2c-cachebridge examples/c2c/eval_cachebridge.py")
    out=Path(args.output or f"results/mmlu_redux_{args.mode}.json"); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result,indent=2))
    summary_path = out.parent / "mmlu_redux_summary.csv"; summary_path.parent.mkdir(parents=True, exist_ok=True)
    old=[]
    if summary_path.exists():
        with summary_path.open() as f: old=list(csv.DictReader(f))
    old=[r for r in old if r["mode"] != args.mode]
    baseline=next((float(r["accuracy"]) for r in old if r["mode"]=="target_native"),None)
    row_out={"mode":args.mode,"accuracy":result["accuracy"],
             "absolute_drop_vs_target":("" if baseline is None else baseline-result["accuracy"]),
             "retention_vs_target":("" if baseline in (None,0) else result["accuracy"]/baseline),
             "runtime":result["runtime"],"mapping_time":""}
    old.append(row_out)
    with summary_path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(row_out)); writer.writeheader(); writer.writerows(old)
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
