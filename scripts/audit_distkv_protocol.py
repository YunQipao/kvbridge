#!/usr/bin/env python3
"""CPU-only audit against functions extracted from the local DistKV Git ref.

Does not import DistKV's model/mapper code, modify its worktree or use a GPU.
"""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvbridge.distkv_profiles import load_sample_groups, prepare_group_samples, sample_sha256


def reference_samples(repo, dataset_path=None, groups=None):
    ref = "origin/c2c-cachebridge"
    script = "examples/c2c/eval_cachebridge.py"
    text = subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{script}"], text=True)
    tree = ast.parse(text)
    wanted = {"BENCHMARK", "_EXPLICIT_ANSWER", "_STANDALONE_ANSWER"}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.FunctionDef) and node.name in ("_choice_text", "_letter", "prepare_samples")) or
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))]
    if groups is not None:
        # Feed the same in-memory rows into the extracted reference function.
        # No datasets monkey-patching, temporary exports or DistKV edits.
        for node in nodes:
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_samples":
                node.body = [statement for statement in node.body
                             if not (isinstance(statement, ast.ImportFrom) and statement.module == "datasets")]
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), script, "exec"), namespace)
    namespace["BENCHMARK"]["local_path"] = str(dataset_path)
    samples = []
    for name, data in (groups if groups is not None else [(None, None)]):
        if groups is not None:
            namespace["load_from_disk"] = lambda path, data=data: data
        rows = namespace["prepare_samples"](0)
        samples.extend(dict(original_index=f"{name}:{row['id']}" if name is not None else row["id"],
                            prompt=row["prompt"], answer=row["answer"]) for row in rows)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", ref], text=True).strip()
    return samples, commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distkv-repo", required=True)
    p.add_argument("--dataset-local-path")
    p.add_argument("--dataset", default="edinburgh-dawg/mmlu-redux-2.0")
    p.add_argument("--split", default="test")
    p.add_argument("--manifest", required=True)
    p.add_argument("--create-manifest", action="store_true", help="create only if absent; never overwrite")
    p.add_argument("--num-eval-samples", type=int)
    p.add_argument("--sampling", choices=["systematic", "random", "first"], default="systematic")
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--eval-profile", choices=["distkv_canonical", "distkv_exact"], default="distkv_canonical")
    args = p.parse_args()
    groups, source_metadata = load_sample_groups(args.dataset_local_path, args.dataset, args.split)
    samples = prepare_group_samples(groups)
    manifest_path = Path(args.manifest)
    if args.create_manifest and not manifest_path.exists():
        from kvbridge.distkv_profiles import profile_rows
        _, created = profile_rows(args.dataset_local_path, args.eval_profile, count=args.num_eval_samples,
                                  seed=args.sampling_seed, sampling=args.sampling,
                                  dataset=args.dataset, split=args.split)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x") as stream:
            json.dump(created, stream, indent=2)
    manifest = json.loads(manifest_path.read_text())
    from kvbridge.protocol import validate_evaluation_manifest
    indices = validate_evaluation_manifest(manifest, len(samples))
    selected = [samples[i] for i in indices]
    reference, commit = reference_samples(args.distkv_repo, args.dataset_local_path, groups)
    ours_hash, ref_hash = sample_sha256(samples), sample_sha256(reference)
    reference_selected = [reference[i] for i in indices] if len(reference) == len(samples) else []
    metadata_ok = (manifest.get("dataset_source") == source_metadata["dataset_source"] and
                   manifest.get("dataset_loader", "load_from_disk") == source_metadata["dataset_loader"] and
                   manifest.get("answer_field") == "answer" and
                   manifest.get("invalid_filter") == {"error_type": "no correct answer"} and
                   manifest.get("eval_profile") in ("distkv_canonical", "distkv_exact"))
    match = (metadata_ok and ours_hash == ref_hash and manifest.get("population_sha256") == ours_hash
             and manifest.get("samples") == selected
             and sample_sha256(selected) == sample_sha256(reference_selected))
    raw_rows = sum(len(ds) for _, ds in groups)
    print(json.dumps(dict(**source_metadata, raw_dataset_rows=raw_rows,
        raw_rows=raw_rows, filtered_rows=raw_rows - len(samples), final_rows=len(samples),
        config_count=sum(name is not None for name, _ in groups),
        invalid_no_correct_answer_rows=sum(str(row.get("error_type", "")).lower() == "no correct answer"
                                          for _, ds in groups for row in ds),
        parseable_rows=len(samples), final_sample_count=len(selected),
        first_5_original_indices=[s["original_index"] for s in selected[:5]],
        last_5_original_indices=[s["original_index"] for s in selected[-5:]],
        population_sha256=ours_hash, reference_population_sha256=ref_hash,
        selected_sha256=sample_sha256(selected), reference_selected_sha256=sample_sha256(reference_selected),
        ordered_sample_sha256=sample_sha256(selected),
        answer_distribution={a: Counter(s["answer"] for s in selected)[a] for a in "ABCDE"},
        reference_commit=commit), indent=2))
    if source_metadata["dataset_loader"] == "hf_cached_configs":
        print("Reference rules audited on the same sorted cached configs; this does not certify the external dataset order/version.")
    print(f"DISTKV_SAMPLE_PROTOCOL_MATCH={str(match).lower()}")
    if not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
