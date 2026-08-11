"""Regenerate the README tables from the raw run summaries in this directory.

    python results/make_tables.py

Every number in the README comes out of here, so a rerun that changes a
result changes the README rather than drifting silently away from it.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SHORT = HERE / "summary_Qwen3-1.7B_to_Qwen3-4B_n500_L1024_lam0.01.json"
LONG = HERE / "summary_Qwen3-1.7B_to_Qwen3-4B_n500_L1024_lam0.01_ctx8192-16384.json"
COSINE = HERE / "cosine_long_Qwen3-1.7B_to_Qwen3-4B_k12_b8.json"

BENCHMARKS = (("hellaswag", "HellaSwag"), ("arc_challenge", "ARC-C"), ("mmlu", "MMLU"))


def load(path):
    with open(path) as f:
        return json.load(f)


def score(entry, name):
    mc = entry["mc"][name]
    return mc[mc["metric"]] * 100


def benchmark_table(long_run):
    """Accuracy under a fully mapped cache, and retention vs the 4B standalone.

    Floor-normalized retention credits the mapper only for accuracy above
    chance, which is the honest denominator on multiple choice.
    """
    bench = long_run["bench"]
    target, source = bench["baseline_target"], bench["baseline_source"]
    rows = []
    for k in sorted(bench["per_k"], key=int):
        entry = bench["per_k"][k]
        accs = [score(entry, n) for n, _ in BENCHMARKS]
        rets = [score(entry, n) / score(target, n) for n, _ in BENCHMARKS]
        floors = [entry["mc"][n]["floor"] * 100 for n, _ in BENCHMARKS]
        fn = [(a - f) / (score(target, n) - f)
              for a, f, (n, _) in zip(accs, floors, BENCHMARKS)]
        rows.append({"k": int(k), "accs": accs, "rets": rets,
                     "mean_acc": sum(accs) / len(accs),
                     "fn": sum(fn) / len(fn),
                     "wt2": entry["wikitext"]["ppl"]})

    out = ["| top_k | HellaSwag | ARC-C | MMLU | mean acc | retention | floor-norm | WikiText-2 ppl |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        star = "**" if r["k"] == best_k(rows) else ""
        out.append(
            f"| {star}{r['k']}{star} | " + " | ".join(f"{a:.1f}" for a in r["accs"])
            + f" | {r['mean_acc']:.2f} | {100 * sum(r['rets']) / 3:.1f}% "
              f"| {100 * r['fn']:.1f}% | {r['wt2']:.2f} |")
    ref = (f"| _4B standalone_ | " + " | ".join(f"{score(target, n):.1f}" for n, _ in BENCHMARKS)
           + f" | {sum(score(target, n) for n, _ in BENCHMARKS) / 3:.2f} | 100% | 100% "
             f"| {target['wikitext']['ppl']:.2f} |")
    src = (f"| _1.7B standalone_ | " + " | ".join(f"{score(source, n):.1f}" for n, _ in BENCHMARKS)
           + f" | {sum(score(source, n) for n, _ in BENCHMARKS) / 3:.2f} | -- | -- "
             f"| {source['wikitext']['ppl']:.2f} |")
    return "\n".join(out + [ref, src])


def best_k(rows):
    """The paper's selection criterion: argmax mean log-likelihood accuracy."""
    return max(rows, key=lambda r: r["mean_acc"])["k"]


def retention_table(short_run, long_run):
    """Context benefit retained on held-out continuations, by prefix length."""
    runs = [(short_run, ["512", "1024", "2048", "4096"]),
            (long_run, ["8192", "16384"])]
    ks = sorted(short_run["results"], key=int)
    header = [f"{int(c) // 1024}K" if int(c) >= 1024 else c
              for _, ctxs in runs for c in ctxs]
    out = ["| top_k | " + " | ".join(header) + " |",
           "|---:|" + "---:|" * len(header)]
    for k in ks:
        cells = []
        for run, ctxs in runs:
            for c in ctxs:
                q = run["results"][k]["quality"][c]
                cells.append(f"{100 * q['benefit_retained']:.1f}%")
        out.append(f"| {int(k)} | " + " | ".join(cells) + " |")
    n = [short_run["n_heldout"], long_run["n_heldout"]]
    return "\n".join(out) + f"\n\nHeld-out blocks: {n[0]} (512-4K), {n[1]} (8K/16K)."


def timing_table(long_run):
    """Mapping cost vs the prefill it replaces, same GPU, median of 5."""
    timing = long_run["timing"]
    ks = sorted(next(iter(timing.values()))["mapping"], key=int)
    out = ["| prefix | 1.7B prefill | 4B prefill | " +
           " | ".join(f"map k={k}" for k in ks) + " | speedup at k=12 |",
           "|---:|---:|---:|" + "---:|" * (len(ks) + 1)]
    for ctx in sorted(timing, key=int):
        t = timing[ctx]
        maps = " | ".join(f"{1000 * t['mapping'][k]:.0f} ms" for k in ks)
        speedup = t["target_prefill"] / t["mapping"]["12"]
        out.append(f"| {int(ctx) // 1024}K | {1000 * t['source_prefill']:.0f} ms "
                   f"| {1000 * t['target_prefill']:.0f} ms | {maps} | {speedup:.1f}x |")
    return "\n".join(out) + f"\n\nMeasured on {long_run['gpu']}, median of 5 trials."


def cosine_table(cos):
    out = ["| prefix | overall | " +
           " | ".join(f"b{i}" for i in range(cos["n_buckets"])) + " |",
           "|---:|---:|" + "---:|" * cos["n_buckets"]]
    for prefix in sorted(cos["results"], key=int):
        r = cos["results"][prefix]
        out.append(f"| {int(prefix) // 1024}K | {r['overall']:.3f} | "
                   + " | ".join(f"{b:.3f}" for b in r["buckets"]) + " |")
    return "\n".join(out) + (f"\n\ntop_k={cos['k']}, {cos['n_blocks']} blocks, "
                             f"buckets run early -> late.")


if __name__ == "__main__":
    short, long, cos = load(SHORT), load(LONG), load(COSINE)
    print("## Benchmarks\n")
    print(benchmark_table(long))
    print("\n## Context benefit retained\n")
    print(retention_table(short, long))
    print("\n## Timing\n")
    print(timing_table(long))
    print("\n## Attention-output cosine by position\n")
    print(cosine_table(cos))
