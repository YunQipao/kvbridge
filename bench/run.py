"""CLI: run context-retention tasks across cache strategies.

    python -m bench.run --pair tiny --mapper mappers/tiny-k8 \
        --tasks kvr,niah,recall --lengths 1024,2048 \
        --strategies mapped,hybrid:sink=4+window=128

`real` (ceiling) and `none` (floor) are always run, because a raw score means
nothing here -- what matters is **retention**, the fraction of the context
benefit that survived the mapper:

    accuracy tasks:  (score - none) / (real - none)
    NLL tasks:       (none - nll)   / (none - real)

Both are 1.0 for a perfect mapper and 0.0 for a cache that carries no
information, which makes numbers comparable across tasks, lengths, and pairs.

Note the strategy grammar uses `+` between parameters on the command line
(`hybrid:sink=4+window=128`) so that a comma can keep separating strategies.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from .engine import Engine, clone_cache, parse_strategy
from .pairs import PAIRS, get_pair
from .tasks import TASKS


def _fmt(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def score_sample(engine, strategy, sample, tok, device):
    ctx_ids = tok(sample.context, return_tensors="pt").input_ids.to(device)
    query_ids = tok(sample.query, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    cache = engine.build_cache(strategy, ctx_ids)

    if sample.mode == "choice":
        opts = [tok(o, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                for o in sample.options]
        nlls = [engine.token_nll(clone_cache(cache), query_ids, o) for o in opts]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        return {"correct": float(pred == sample.gold), "nll": nlls[sample.gold]}

    ans_ids = tok(sample.answer, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)
    return {"nll": engine.token_nll(cache, query_ids, ans_ids)}


def aggregate(records, key):
    vals = [r[key] for r in records if key in r]
    return statistics.fmean(vals) if vals else float("nan")


def retention(mode, score, real, none):
    span = (real - none) if mode == "choice" else (none - real)
    if abs(span) < 1e-9:
        return float("nan")
    delta = (score - none) if mode == "choice" else (none - score)
    return delta / span


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="tiny", choices=sorted(PAIRS))
    ap.add_argument("--mapper", default=None, help="path or HF repo id of a fitted mapper")
    ap.add_argument("--tasks", default="kvr,niah,recall")
    ap.add_argument("--strategies", default="mapped",
                    help="comma-separated; params joined with '+', e.g. hybrid:sink=4+window=128")
    ap.add_argument("--lengths", default="1024", help="context lengths in tokens")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--depths", type=int, default=8)
    ap.add_argument("--haystack", default=None, help="text file for filler / continuation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fidelity", action="store_true",
                    help="also report per-layer / per-position KV cosine")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args(argv)

    pair = get_pair(args.pair)
    lengths = [int(x) for x in args.lengths.split(",")]
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    user_strats = [parse_strategy(s.replace("+", ",")) for s in args.strategies.split(",") if s]
    strategies = [parse_strategy("none"), parse_strategy("real")]
    strategies += [s for s in user_strats if str(s) not in {"none", "real"}]

    engine = Engine(pair, mapper_path=args.mapper, device=args.device)
    tok, device = engine.tok, engine.device
    print(f"pair={pair.name} {pair.source} -> {pair.target}  device={device} dtype={engine.dtype}")

    results = []
    t0 = time.time()
    for n_ctx in lengths:
        for task_name in task_names:
            samples = TASKS[task_name](tok, n_ctx=n_ctx, n_samples=args.samples,
                                       n_depths=args.depths, seed=args.seed,
                                       haystack=args.haystack)
            mode = samples[0].mode
            for strategy in strategies:
                if strategy.needs_mapper and args.mapper is None:
                    print(f"  skip {strategy}: no --mapper")
                    continue
                recs, by_depth = [], defaultdict(list)
                for i, s in enumerate(samples):
                    r = score_sample(engine, strategy, s, tok, device)
                    recs.append(r)
                    by_depth[s.meta.get("depth")].append(r)
                    if (i + 1) % 16 == 0:
                        print(f"  [{task_name} L={n_ctx} {strategy}] {i+1}/{len(samples)}",
                              flush=True)
                key = "correct" if mode == "choice" else "nll"
                results.append({
                    "task": task_name, "mode": mode, "n_ctx": n_ctx,
                    "strategy": str(strategy), "score": aggregate(recs, key),
                    "n": len(recs),
                    "by_depth": {str(d): aggregate(v, key) for d, v in sorted(
                        by_depth.items(), key=lambda kv: (kv[0] is None, kv[0]))},
                })
                print(f"  {task_name} L={n_ctx} {strategy}: {key}={_fmt(results[-1]['score'])}")

    # ---- retention table ----
    index = {(r["task"], r["n_ctx"], r["strategy"]): r for r in results}
    for r in results:
        real = index.get((r["task"], r["n_ctx"], "real"))
        none = index.get((r["task"], r["n_ctx"], "none"))
        r["retention"] = (retention(r["mode"], r["score"], real["score"], none["score"])
                          if real and none else None)

    print("\n| task | n_ctx | strategy | score | retention |")
    print("|---|---:|---|---:|---:|")
    for r in results:
        print(f"| {r['task']} | {r['n_ctx']} | {r['strategy']} | "
              f"{_fmt(r['score'])} | {_fmt(r['retention'])} |")

    fid = None
    if args.fidelity:
        from .fidelity import attn_cosine, kv_fidelity
        probe = tok("\n\nQuestion:", return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
        fid = {}
        for n_ctx in lengths:
            ctx = TASKS[task_names[0]](tok, n_ctx=n_ctx, n_samples=1,
                                       n_depths=1, seed=args.seed,
                                       haystack=args.haystack)[0].context
            ctx_ids = tok(ctx, return_tensors="pt").input_ids.to(device)
            for strategy in strategies:
                if not strategy.needs_mapper:
                    continue
                f = kv_fidelity(engine, ctx_ids, strategy)
                f["attn"] = attn_cosine(engine, ctx_ids, probe, strategy)
                fid[f"{n_ctx}|{strategy}"] = f
                print(f"\nfidelity {strategy} @ {n_ctx}: K={_fmt(f['mean_k'])} "
                      f"V={_fmt(f['mean_v'])} attn={_fmt(f['attn']['mean'])}")
                print("  K by position:  " + " ".join(_fmt(x) for x in f["k_by_position"]))
                print("  attn by layer:  " + " ".join(_fmt(x, 2) for x in f["attn"]["per_layer"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{pair.name}_{'-'.join(task_names)}_{'-'.join(map(str, lengths))}"
    path = out_dir / f"bench_{stamp}.json"
    with open(path, "w") as f:
        json.dump({"pair": pair.name, "source": pair.source, "target": pair.target,
                   "mapper": args.mapper, "device": device, "dtype": str(engine.dtype),
                   "args": vars(args), "results": results, "fidelity": fid,
                   "wall_seconds": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
