"""Named model pairs, with the memory each one needs to fit locally.

All Qwen3 dense models share 8 KV heads x 128 head_dim, so every combination
below is a matched-KV pair. Layer counts: 0.6B/1.7B = 28, 4B = 36, 8B = 36.

`fit_ram_gb` is the calibration-capture peak for `n_calib` documents at
seq_len 1024 / stride 4, i.e. 256 observations per document:

    bytes = n_obs * n_layers * n_kv_heads * head_dim * 2 (fp16) * 2 (K and V)

Both models' captures are held at once, so the two rows add up. The number
below is that sum plus the larger model's fp16 weights.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pair:
    name: str
    source: str
    target: str
    n_calib: int          # calibration documents for a local fit
    top_k: int            # default source-layer count
    notes: str = ""
    tags: tuple = field(default_factory=tuple)


PAIRS = {
    # ---- local (Apple Silicon, <= 24 GB unified) ----
    "tiny": Pair(
        name="tiny",
        source="Qwen/Qwen3-0.6B",
        target="Qwen/Qwen3-1.7B",
        n_calib=192,
        top_k=6,
        notes="28->28 layers, ~11 GB capture, ~8 rows per fitted parameter. "
              "Same depth on both sides, so it validates plumbing rather than "
              "depth remapping -- do not read quality conclusions off it.",
        tags=("local",),
    ),
    "small": Pair(
        name="small",
        source="Qwen/Qwen3-0.6B",
        target="Qwen/Qwen3-4B",
        n_calib=160,
        top_k=6,
        notes="28->36 layers, ~19 GB peak. Cheapest local pair that actually "
              "remaps depth, so ablation *orderings* found here are the ones "
              "worth spending Modal time to confirm on prod.",
        tags=("local",),
    ),
    "prod": Pair(
        name="prod",
        source="Qwen/Qwen3-1.7B",
        target="Qwen/Qwen3-4B",
        n_calib=500,
        top_k=12,
        notes="The pair the README numbers come from; n_calib=500/top_k=12 is "
              "the published fit. Evaluating that mapper locally is cheap "
              "(two models, no capture). A local *fit* of this pair needs "
              "--n-calib 160 --top-k 8 to stay both in RAM and well-determined.",
        tags=("local-eval", "modal"),
    ),
    "big": Pair(
        name="big",
        source="Qwen/Qwen3-4B",
        target="Qwen/Qwen3-8B",
        n_calib=500,
        top_k=12,
        notes="Modal only.",
        tags=("modal",),
    ),
}


def get_pair(name):
    if name not in PAIRS:
        raise SystemExit(f"unknown pair {name!r}; known: {', '.join(PAIRS)}")
    return PAIRS[name]
