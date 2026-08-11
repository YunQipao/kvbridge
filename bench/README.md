# bench — local context-retention benchmarks

Everything here runs identically on a Mac (MPS), a CUDA box, or CPU. The split
with the Modal pipeline is deliberate and worth stating once:

| | runs where | why |
|---|---|---|
| **quality** (retention, fidelity, ablations) | anywhere | device-independent by construction — same numbers on MPS and A100 |
| **timing** (speedup vs re-prefill) | Modal only | only comparable on identical hardware; MPS numbers would be misleading in a README |

So: iterate on ideas locally, confirm the winners on Modal, and never quote a
local millisecond.

## Setup

```bash
uv venv --python 3.12 .venv-local
uv pip install --python .venv-local/bin/python torch transformers safetensors huggingface-hub accelerate datasets
uv pip install --python .venv-local/bin/python -e kvbridge
```

## Fit a mapper locally

```bash
.venv-local/bin/python -m bench.fit_local --pair tiny --out mappers/tiny-k6
```

Fitting is memory-bound: both models' calibration KV live in RAM at once. The
command prints its estimate and the design conditioning before loading
anything, and refuses to run if either is out of range. Two failure modes it
guards against:

- **RAM**: capture scales as `n_calib × (seq_len/stride) × n_layers × H × d`.
- **Conditioning**: each target layer fits `top_k × H × d` parameters. Below
  ~4 rows per parameter the ridge interpolates and reports R² ≈ 0.999 for a
  mapper that transfers nothing. This is the single easiest way to waste a day
  here — an R² near 1.0 on a small `--n-calib` is a bug report, not a result.

`--texts FILE` calibrates on a local file instead of streaming FineWeb-Edu.

## Run benchmarks

```bash
.venv-local/bin/python -m bench.run --pair tiny --mapper mappers/tiny-k6 \
    --tasks kvr,niah,recall --lengths 1024,4096 --samples 64 \
    --strategies mapped,hybrid:sink=4+window=256 --fidelity
```

`real` (ceiling) and `none` (floor) are always run; the reported **retention**
is the fraction of the context benefit that survived the mapper, so 1.0 is a
perfect mapper and 0.0 is a cache carrying no information.

## The two axes

**Cache strategies** — how the target's prefix cache gets built:

| strategy | meaning |
|---|---|
| `real` | target prefills the context (ceiling) |
| `none` | no context (floor) |
| `mapped` | every context token comes from the mapper |
| `hybrid:sink=S+window=W` | target natively prefills the first S and last W tokens; mapper covers the middle |
| `layers:real=first8` | mapped everywhere except those target layers, which get the real prefill |

Parameters join with `+` on the command line so `,` can keep separating
strategies.

`hybrid` is the deployable one: it costs `(S+W)/T` of a target prefill and
targets the three weaknesses the long-context diagnostic found (recency
precision, sink-token
K/V, long-context tail droop) at once. `layers` is a diagnostic — splicing in
a *suffix* of layers is not something you can buy in production, since
computing layer L natively requires layers 0..L-1 — but it tells you which
layers the error actually hurts in, which is what per-layer capacity
allocation needs to know.

**Tasks** — what the context has to retain:

| task | mode | probes |
|---|---|---|
| `kvr` | choice | verbatim key→value lookup, position-resolved (chance = 1/n_options) |
| `niah` | choice | a natural-language fact at a controlled depth |
| `recall` | NLL | verbatim reproduction of a tagged random span — most sensitive to small fidelity loss |
| `continuation` | NLL | held-out continuation; the metric the published tables report. Needs `--haystack FILE` |

Three of the four score by ranking rather than generating, so a small target's
inability to follow instructions doesn't get scored as a cache failure.

Read `recall`/`continuation` against `kvr`/`niah` rather than alone: NLL keeps
improving past the `k` where task accuracy saturates, so a
config that wins on NLL and not on the retrieval tasks probably has not
improved anything a user would notice.

## Diagnostics (`--fidelity`)

Per-layer and per-position-bucket K/V cosine, plus attention-output cosine
(which tracked benchmark accuracy at r=+0.92 where NLL-type metrics did not).
Both cost one prefill. Run them before committing to an expensive eval — they
usually say whether a change is worth benchmarking at all, and the per-layer
attention profile is what tells you where extra mapping capacity should go.
