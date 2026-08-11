# kvbridge — transfer context between models, without prefill!

[![tests](https://github.com/Sidd03192/kvbridge/actions/workflows/tests.yml/badge.svg)](https://github.com/Sidd03192/kvbridge/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![mapper](https://img.shields.io/badge/%F0%9F%A4%97%20mapper-qwen3--1.7b%E2%86%924b-yellow)](https://huggingface.co/Siddharth85/kvbridge-qwen3-1.7b-to-qwen3-4b)

**Prefill with the small model. Answer with the big one.**

Main Idea: AI is big. Big is expensive. Small model cheap. Use small model where possible.

A 1.7B model reads the prompt. A fitted linear map converts its KV cache into
the format a 4B expects. The 4B then answers a document it never read — at 16K
context that costs **290 ms instead of a 1270 ms prefill**, and keeps **83.7%**
of the 4B's own benchmark accuracy.

```python
from kvbridge import Mapper, transfer

mapper = Mapper.from_pretrained("Siddharth85/kvbridge-qwen3-1.7b-to-qwen3-4b")

small_cache = small(prompt_ids, use_cache=True).past_key_values   # 1.7B reads the prompt
big_cache = transfer(small_cache, mapper)                          # ~1 matmul per layer
big.generate(input_ids=question_ids, past_key_values=big_cache)    # 4B never sees the prompt
```

```bash
pip install git+https://github.com/Sidd03192/kvbridge     # the library

git clone https://github.com/Sidd03192/kvbridge           # or, to run the demo
cd kvbridge && pip install -e .
python examples/handoff.py
```

Real output of that example, on an M5 Mac (MPS). The 4B is answering a memo
that only the 1.7B ever read:

```
Q: Why were the gearbox housings failing?
  mapped  The gearbox housings were failing due to a hairline crack in the ceramic
          gearbox housing that appeared after about 4,000 duty cycles of sustained
          load above 60% of rated torque.
  real    The housings failed due to a hairline crack that developed after
          approximately 4,000 duty cycles under sustained load exceeding 60% of
          the rated torque.
  none    The gearbox housings were failing due to the high temperatures generated
          by the friction between the gears...                      [hallucinated]

Q: What is the part number for the revised housing?
  mapped  The part number for the revised housing is HR-2291.       [truth: HR-2291-B]
  real    The part number for the revised housing is HR-2291-B.
  none    The part number for the revised housing is 1234567890.    [hallucinated]
```

That is the method's character in eight lines: **semantic content transfers
almost intact; verbatim identifiers fray at the last token.** The benchmarks
below quantify exactly that split.

## What this is

KV caches are model-specific: a cache built by one model is noise to another,
so every model in a serving stack pays for its own prefill of the same prompt.
*Cross-Model KV Cache Transfer in LLM Families* (Heo et al., NVIDIA,
[arXiv:2608.03893](https://arxiv.org/abs/2608.03893)) showed that within a model
family the relationship between two models' caches is close enough to **linear**
that a per-layer ridge regression, fit once offline, can convert one into the
other. A small model can then do the reading and a big model the answering.

kvbridge is an independent implementation of that method  packaged as something you can actually call, plus a fitted mapper on the
Hub and the measurements below. Everything runs on HuggingFace `DynamicCache`,
so there is no serving engine to adopt: if you can call `model.generate`, you
can use this.

**Requirement:** a *matched-KV pair* — same `num_key_value_heads` and same
`head_dim` in the two configs. Every dense Qwen3 model qualifies (8 KV heads ×
128); Llama-3.1 8B and 70B match on paper but are untested here. Layer counts
may differ — that is the interesting part (28 → 36 in the pair below).
`fit_mapper` checks the geometry and refuses mismatched pairs rather than
fitting something meaningless.

## Results

Qwen3-1.7B → Qwen3-4B, 28 → 36 layers. Mapper fit on 500 FineWeb-Edu sequences
× 1024 tokens. Every table below is generated from the raw run JSONs in
[`results/`](results/) by `python results/make_tables.py`.

### Accuracy under a fully mapped cache

lm-eval conventions (HellaSwag/ARC-C `acc_norm`, MMLU 5-shot `acc`), ~2000
samples per benchmark. The *entire* prompt — few-shots, document and question —
goes through the mapper; nothing is re-prefilled. Retention is the mapped score
over the 4B's own score; floor-norm credits only accuracy above chance.

| top_k | HellaSwag | ARC-C | MMLU | mean acc | retention | floor-norm | WikiText-2 ppl |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 60.8 | 41.6 | 55.1 | 52.48 | 81.7% | 69.1% | 15.66 |
| 8 | 62.6 | 42.7 | 55.2 | 53.53 | 83.4% | 72.0% | 13.78 |
| **12** | 63.3 | 42.7 | 55.1 | 53.73 | **83.7%** | 72.4% | 13.25 |
| 16 | 63.3 | 42.4 | 54.5 | 53.42 | 83.2% | 71.6% | 13.10 |
| 20 | 63.6 | 42.2 | 54.0 | 53.27 | 83.0% | 71.1% | 12.97 |
| 24 | 63.9 | 42.7 | 54.1 | 53.60 | 83.5% | 72.1% | 12.89 |
| _4B standalone_ | 68.7 | 53.9 | 69.2 | 63.92 | 100% | 100% | 11.59 |
| _1.7B standalone_ | 60.2 | 42.9 | 60.8 | 54.62 | — | — | 14.77 |

`top_k` is how many source layers feed each target layer. Accuracy **plateaus at
k=8–12 and never improves past it**, while perplexity keeps improving to k=24 —
see [Selecting k](#selecting-k-the-metric-that-lies).

### Context retained on held-out text

Fraction of the context benefit that survives the mapper, on unseen FineWeb-Edu
blocks: `(nll_no_context − nll_mapped) / (nll_no_context − nll_real_prefill)`.
1.0 is a perfect mapper, 0.0 is a cache carrying nothing.

| top_k | 512 | 1K | 2K | 4K | 8K | 16K |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 55.0% | 45.2% | 31.6% | 37.1% | 32.7% | 24.6% |
| 8 | 77.0% | 65.8% | 58.5% | 63.8% | 56.0% | 48.4% |
| 12 | 83.2% | 71.6% | 66.7% | 70.3% | 64.9% | 57.8% |
| 16 | 85.9% | 75.0% | 69.6% | 73.1% | 68.1% | 61.5% |
| 20 | 88.3% | 77.0% | 72.8% | 74.9% | 70.2% | 64.4% |
| 24 | 89.8% | 78.1% | 73.1% | 76.0% | 71.2% | 65.6% |

Retention decays with length — the mapper is fit on 1024-token sequences and
evaluated at up to 16× that. Held-out blocks: 16 (512–4K), 48 (8K/16K).

### Speed

| prefix | 1.7B prefill | 4B prefill | map k=8 | map k=12 | map k=24 | speedup at k=12 |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 250 ms | 545 ms | 120 ms | 158 ms | 254 ms | 3.5× |
| 16K | 568 ms | 1270 ms | 220 ms | 290 ms | 504 ms | 4.4× |

A100, median of 5 trials. The advantage grows with context length, because
mapping is linear in tokens while attention is not.

Two ways to read this, both true:

- **Warm** (the small model's cache already exists — multi-turn, a router that
  always runs the small model first, a cache server): you pay 290 ms instead of
  1270 ms at 16K. **4.4×**.
- **Cold** (nothing cached, you choose which path to run): 568 ms of 1.7B
  prefill + 290 ms of mapping = 858 ms vs 1270 ms. **1.48×**, and the crossover
  is around 8K — below that, cold-start mapping is not worth it.

## Honestly it's not perfect...

- **Mapping the whole prompt is not strictly better than just using the small
  model.** At k=12 the mapped 4B beats the 1.7B alone on HellaSwag (63.3 vs
  60.2) and WikiText-2 (13.25 vs 14.77), ties on ARC-C, and *loses badly on
  MMLU* (55.1 vs 60.8). Knowledge-heavy multiple choice is where routing the
  question itself through the mapper hurts. The deployable shape is a handoff:
  map the *document*, let the target prefill the question and generate.
- **Long context degrades.** 83% retained at 512 tokens, 58% at 16K (k=12).
  Position-bucketed diagnostics show per-token fidelity is nearly flat with
  length (0.825 at 2K → 0.820 at 16K) — the damage comes from *more* mapped
  keys competing in each softmax, not from positional extrapolation.
- **Matched-KV pairs only.** Different `num_key_value_heads` or `head_dim`
  between source and target and the closed-form map does not apply.
- **Batch size 1**, dense models only, no MLP fallback for hard pairs (paper
  §4.4), no vLLM/SGLang connector.
- **The artifact is large.** k=12 for this pair is 1.8 GB in bfloat16 — cheaper
  than a model, not free. k=8 is 1.2 GB and within noise on accuracy.
- One pair is measured end to end. The paper reports 2 of 6 pairs degrading
  sharply; that failure mode is real and this repo has not reproduced it yet.
  
## But we can make it better :)

## Fit your own pair

```python
from kvbridge import fit_mapper

mapper = fit_mapper("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", top_k=12)
mapper.save("qwen3-1.7b-to-4b")
```

One GPU, both models loaded one at a time, ~20 minutes and a few dollars on a
rented A100 for the defaults (500 × 1024 calibration). Fitting is **memory
bound**, not compute bound: both models' captured activations sit in RAM at
once (~30 GB for this pair). The ridge solve runs fp64 on CPU, so Apple Silicon
works too.

The pitfall worth naming: each target layer fits `top_k × n_kv_heads ×
head_dim` parameters, so a small `n_calib` gives you an interpolating fit that
reports R² ≈ 0.999 and transfers nothing. `python -m bench.fit_local` checks
rows-per-parameter and RAM before it loads anything, and refuses to run a fit
that would lie to you.

## Benchmark it yourself

[`bench/`](bench/) is a local, device-agnostic harness — same numbers on a Mac
(MPS), a CUDA box, or CPU, because everything in it is quality, not timing.

```bash
python -m bench.run --pair prod --mapper Siddharth85/kvbridge-qwen3-1.7b-to-qwen3-4b \
    --tasks kvr,niah,recall --lengths 1024,4096 --strategies mapped --fidelity
```

Two independent axes, so a new idea is a config rather than a rewrite:

| cache strategy | what the target's prefix cache is built from |
|---|---|
| `real` / `none` | full prefill (ceiling) / no context (floor) |
| `mapped` | every context token comes from the mapper |
| `hybrid:sink=S+window=W` | target natively prefills the first S and last W tokens; mapper covers the middle |
| `layers:real=first8` | mapped everywhere except those target layers (diagnostic) |

| task | probes |
|---|---|
| `kvr` | verbatim key→value lookup at a controlled depth |
| `niah` | a natural-language fact at a controlled depth |
| `recall` | verbatim reproduction of a tagged span — most sensitive to small fidelity loss |
| `continuation` | held-out continuation NLL — the metric in the tables above |

Three of the four score by *ranking* rather than generating, so a small model's
inability to follow instructions never gets scored as a cache failure.

A run of that harness on a mapper **fit entirely on a laptop** (Qwen3-0.6B →
1.7B, top_k=6, ~15 min fit on an M5 Mac), 1024-token contexts, retention
against the same ceiling and floor:

| task | mapped | hybrid (sink=4, window=128) |
|---|---:|---:|
| `niah` — semantic fact | 100% | 100% |
| `recall` — verbatim span NLL | 80.0% | 84.8% |
| `kvr` — verbatim uuid lookup | 19.2% | 15.4% |

Fidelity for the same run: K cosine 0.929, **V cosine 0.671**, attention-output
cosine 0.784 → 0.841 under `hybrid`. Three things worth stealing from this:
**V is the bottleneck, not K**; the recency window is where a hybrid strategy
earns its keep (the in-window bucket hits the real-prefill ceiling exactly);
and verbatim uuid retrieval decays with depth even where per-position K
fidelity is flat — the same mechanism the 16K diagnostic found on the big
pair, reproduced at 1/16 the length and $0. Raw JSON in
[`results/`](results/).

## How it works

1. Run a calibration corpus through both models, capture per-layer K and V.
2. **Strip RoPE from the keys.** Mapping in content space is what makes a
   mapper fit at 1024 tokens valid at any position (rotations are orthogonal,
   so this is exact).
3. For each target layer, rank source layers by per-head ridge R² and keep the
   top k. This selection is the single largest quality contributor.
4. Fit **separate** centered ridge regressions for K and V, features being all
   KV heads of the chosen source layers concatenated. Bias comes from the means
   and is not regularized.
5. At runtime: strip source RoPE → one bf16 matmul per target layer per cache
   type → re-apply target RoPE → hand back a `DynamicCache`.

Five deviations from the paper's protocol, once corrected, moved 4K retention
from 46.1% to 70.3% on this pair:

| fix | before → after |
|---|---|
| calibration length and decorrelation | 200 × 512 @ stride 1 → 500 × 1024 @ stride 4 |
| source layers per target layer | k ≤ 5 → k = 12 |
| layer-ranking sample | first ~32 documents → strided across the whole corpus |
| selection metric | fit R² → attention-output cosine |
| transfer precision | fp32 eager → one bf16 `addmm` per layer |

The sixth, an MLP fallback for pairs where the linear map fails (§4.4), is not
implemented.

## Repository layout

```
kvbridge/       the package: mapper, transfer, fit, rope, cache_compat
bench/          local context-retention benchmarks (Mac/CUDA/CPU)
experiments/    Modal pipeline that produced the tables above
results/        raw run JSONs + the script that generates the README tables
examples/       runnable handoff demo
tests/          unit tests (no model downloads, ~3 s)
```

```bash
pip install -e . pytest && pytest        # 27 tests, ~3 s, CPU only
```

The tests pin the invariants the numbers depend on: RoPE strip/apply are exact
inverses and match the rotation `transformers` applies, the streaming ridge
solve agrees with the direct solve and its moment-based R² with the residual
one, `transfer` round-trips an identity mapper bit-for-bit, and saved mappers
reload identically (with unknown format versions refused). CI runs them on
Python 3.10 and 3.12.

Reproducing the numbers needs a Modal account:

```bash
modal run experiments/fit_eval_modal.py --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B
modal run experiments/export_mapper.py --k 12          # volume fit -> Hub artifact
```

## Citation

The method is the paper's; the implementation, the k-selection finding and the
measurements are this repo's.

```bibtex
@article{heo2026crossmodel,
  title  = {Cross-Model KV Cache Transfer in LLM Families},
  author = {Heo, et al.},
  journal= {arXiv preprint arXiv:2608.03893},
  year   = {2026}
}
```

MIT licensed. Issues and pairs welcome — especially failures.
