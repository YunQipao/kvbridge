#!/usr/bin/env python3
"""Measure mapper application only; no fabricated values are supplied."""
import argparse, json, statistics, time
from pathlib import Path
import torch
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.mapper import Mapper
from kvbridge.model_utils import load_causal_lm, model_input_device
from kvbridge.transfer import transfer


def _stats(values):
    values = sorted(values)
    q = lambda x: values[min(len(values)-1, int(x * len(values)))]
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "p25": q(.25), "p75": q(.75)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--target",required=True)
    p.add_argument("--mapper"); p.add_argument("--full-head-mapper"); p.add_argument("--cachebridge-mapper")
    p.add_argument("--prefix-lengths",default="128,512,1024,2048,4096,8192,16384")
    p.add_argument("--warmup",type=int,default=3); p.add_argument("--repeat",type=int,default=10)
    p.add_argument("--device",default="cuda"); p.add_argument("--output",default="results/transfer_benchmark.json")
    p.add_argument("--local-files-only", action="store_true"); args=p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"): raise RuntimeError("CUDA requested but unavailable")
    cb_path = args.cachebridge_mapper or args.mapper
    fh_path = args.full_head_mapper
    if not cb_path and not fh_path:
        raise ValueError("provide --cachebridge-mapper/--full-head-mapper")
    mappers = {}
    if cb_path: mappers["cachebridge"] = Mapper.load(cb_path, device=args.device, dtype=torch.float32)
    if fh_path: mappers["full_head"] = Mapper.load(fh_path, device=args.device, dtype=torch.float32)
    model=load_causal_lm(args.source, device=args.device, dtype="auto",
                         local_files_only=args.local_files_only)
    device = model_input_device(model)
    lengths=[int(x) for x in args.prefix_lengths.split(",")]; results=[]
    for n in lengths:
        ids=torch.randint(0, model.config.vocab_size, (1,n), device=device)
        with torch.no_grad(): cache=model(ids,use_cache=True).past_key_values
        for mapper in mappers.values():
            for _ in range(args.warmup): transfer(cache,mapper)
        if device.type == "cuda": torch.cuda.synchronize()
        vals = {name: [] for name in mappers}
        for _ in range(args.repeat):
            for name, mapper in mappers.items():
                if device.type == "cuda":
                    torch.cuda.synchronize(); start=torch.cuda.Event(enable_timing=True)
                    end=torch.cuda.Event(enable_timing=True); start.record()
                else: start_time=time.perf_counter()
                transfer(cache,mapper)
                if device.type == "cuda":
                    end.record(); torch.cuda.synchronize(); vals[name].append(start.elapsed_time(end))
                else: vals[name].append((time.perf_counter()-start_time)*1000)
        row = {"prefix": n}
        for name, samples in vals.items(): row[f"{name}_ms"] = _stats(samples)
        if "full_head" in vals and "cachebridge" in vals:
            row["speedup"] = _stats(vals["full_head"])["median"] / _stats(vals["cachebridge"])["median"]
        results.append(row)
        del cache
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))

if __name__ == "__main__": main()
