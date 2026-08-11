"""Context-retention tasks.

Every task here is *synthetic and self-contained* -- no dataset downloads, no
network, deterministic under a seed. That is deliberate: the questions these
answer ("did position 14K survive the mapper?", "can the target still look up
a fact it never prefilled?") are exactly the ones a fixed generator answers
better than a scraped corpus, and it means a Mac can run the same benchmark a
Modal A100 does, byte for byte.

Three of the four score by *ranking* rather than generating. A 0.6B target
asked to emit a UUID will fail for reasons that have nothing to do with the
cache; ranking the true continuation against distractors drawn from the same
context isolates retention from instruction-following, and gives a floor
(chance) that makes retention percentages meaningful.

Task -> Sample -> (context, query, scoring). The context is what goes through
the cache strategy; the query is always prefilled natively by the target.
"""

import random
from dataclasses import dataclass, field

# Common-word filler. Kept small and boring on purpose: haystack text should
# be plausible English that carries no information the needle competes with.
_FILLER_SUBJ = ("the committee", "our team", "the report", "the department",
                "the study", "the council", "the group", "the project",
                "the review", "the office", "the program", "the survey")
_FILLER_VERB = ("noted", "reviewed", "considered", "approved", "discussed",
                "summarized", "revised", "published", "circulated", "endorsed")
_FILLER_OBJ = ("the proposal", "the schedule", "the budget", "the findings",
               "the guidelines", "the timeline", "the minutes", "the draft",
               "the agenda", "the recommendations", "the estimates")
_FILLER_TAIL = ("earlier this year", "without further comment", "at some length",
                "in the usual way", "before the deadline", "on several occasions",
                "as a matter of routine", "during the last session")

# Distinctive tokens for needles and recall sequences -- concrete nouns that a
# language model will not confidently predict from filler context.
_RARE = ("saffron", "obsidian", "kestrel", "lantern", "harbor", "trellis",
         "marmot", "quartz", "juniper", "cobalt", "fennel", "gargoyle",
         "monsoon", "pewter", "sable", "thistle", "vellum", "wombat",
         "zephyr", "alcove", "bramble", "cinder", "dolomite", "ember",
         "flint", "granite", "hazel", "ivory", "jasper", "kelp")


@dataclass
class Sample:
    """One benchmark item.

    context: text routed through the cache strategy (the "prefilled document")
    query:   text the target always prefills itself (the "question")
    mode:    'choice' -> rank `options`, correct is `gold`
             'nll'    -> mean NLL of `answer`
    meta:    per-sample facts to slice results by (depth, n_ctx_tokens, ...)
    """
    context: str
    query: str
    mode: str
    options: list = field(default_factory=list)
    gold: int = 0
    answer: str = ""
    meta: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _sentence(rng):
    return (f"{rng.choice(_FILLER_SUBJ)} {rng.choice(_FILLER_VERB)} "
            f"{rng.choice(_FILLER_OBJ)} {rng.choice(_FILLER_TAIL)}.").capitalize()


def _filler(rng, tok, n_tokens, haystack_words=None):
    """Return filler text of exactly n_tokens tokens.

    Depth is only meaningful if every context is the length it claims to be,
    so this grows the draft until it really tokenizes long enough rather than
    trusting a chars-per-token estimate.
    """
    if n_tokens <= 0:
        return ""
    parts, ids = [], []
    while len(ids) < n_tokens:
        if haystack_words:
            span = 2 * (n_tokens + 16)
            start = rng.randrange(max(1, len(haystack_words) - span))
            parts.append(" ".join(haystack_words[start:start + span]))
        else:
            parts.extend(_sentence(rng) for _ in range(max(8, n_tokens // 8)))
        ids = tok(" ".join(parts), add_special_tokens=False).input_ids
    return tok.decode(ids[:n_tokens])


def _uuid(rng):
    hexd = "0123456789abcdef"
    return "-".join("".join(rng.choice(hexd) for _ in range(4)) for _ in range(2))


def _depths(n):
    """n evenly spaced depths in [0, 1], endpoints included."""
    return [0.0] if n == 1 else [i / (n - 1) for i in range(n)]


def _load_haystack(path):
    if path is None:
        return None
    with open(path) as f:
        return f.read().split()


# -----------------------------------------------------------------------------
# tasks
# -----------------------------------------------------------------------------

def kv_retrieval(tok, n_ctx=2048, n_samples=32, n_depths=8, n_options=4,
                 seed=0, **_):
    """Key/value lookup (Liu et al., "Lost in the Middle").

    The context is a list of uuid -> uuid pairs; the query names one key. The
    target ranks the true value against other values from the same list, so
    chance is 1/n_options and the only way to win is to have retrieved from
    the mapped cache. This is the sharpest position-resolved probe available:
    no world knowledge, no reasoning, no partial credit.
    """
    rng = random.Random(seed)
    header = "Extract the value corresponding to the specified key in the JSON object below.\n\n"
    pair_tokens = len(tok(f'  "{_uuid(rng)}": "{_uuid(rng)}",\n',
                          add_special_tokens=False).input_ids)
    # 32 tokens reserved for the instruction header and the JSON braces
    n_pairs = max(n_options + 1, (n_ctx - 32) // max(1, pair_tokens))

    samples = []
    depths = _depths(n_depths)
    for i in range(n_samples):
        keys = [_uuid(rng) for _ in range(n_pairs)]
        vals = [_uuid(rng) for _ in range(n_pairs)]
        body = "".join(f'  "{k}": "{v}",\n' for k, v in zip(keys, vals))
        depth = depths[i % len(depths)]
        idx = min(n_pairs - 1, round(depth * (n_pairs - 1)))
        distractors = rng.sample([j for j in range(n_pairs) if j != idx],
                                 n_options - 1)
        options = [vals[j] for j in distractors]
        gold = rng.randrange(n_options)
        options.insert(gold, vals[idx])
        samples.append(Sample(
            context=header + "{\n" + body + "}\n",
            query=f'\nThe value for key "{keys[idx]}" is "',
            mode="choice",
            options=[f'{o}"' for o in options],
            gold=gold,
            meta={"depth": round(depth, 3), "pair_index": idx, "n_pairs": n_pairs},
        ))
    return samples


def niah(tok, n_ctx=2048, n_samples=32, n_depths=8, n_options=4, seed=0,
         haystack=None, **_):
    """Needle in a haystack, scored by ranking.

    A single fact is buried at a controlled depth in filler text. Unlike
    kv_retrieval the needle is a natural sentence, so this also exercises
    whether the mapped cache preserves ordinary semantic content rather than
    just verbatim strings.
    """
    rng = random.Random(seed + 1)
    words = _load_haystack(haystack)
    samples = []
    depths = _depths(n_depths)
    for i in range(n_samples):
        secret = rng.choice(_RARE)
        others = [w for w in _RARE if w != secret]
        options = rng.sample(others, n_options - 1)
        gold = rng.randrange(n_options)
        options.insert(gold, secret)
        city = rng.choice(("Lisbon", "Osaka", "Nairobi", "Calgary", "Perth"))
        needle = f" The hidden object stored in {city} is a {secret}. "

        depth = depths[i % len(depths)]
        needle_len = len(tok(needle, add_special_tokens=False).input_ids)
        budget = max(64, n_ctx - needle_len)
        before = int(round(depth * budget))
        context = (_filler(rng, tok, before, words) + needle
                   + _filler(rng, tok, budget - before, words))
        samples.append(Sample(
            context=context,
            query=f"\n\nQuestion: What is the hidden object stored in {city}?\nAnswer: a",
            mode="choice",
            options=[f" {o}." for o in options],
            gold=gold,
            meta={"depth": round(depth, 3)},
        ))
    return samples


def recall(tok, n_ctx=2048, n_samples=32, n_depths=8, seq_len=8, seed=0,
           haystack=None, **_):
    """Verbatim recall, scored by NLL of the exact continuation.

    A tagged random word sequence sits at a controlled depth; the query
    re-presents the tag. Reproducing the sequence requires copying it out of
    the cache token by token, so the score degrades smoothly with cache
    fidelity instead of falling off a cliff -- which makes it the most
    sensitive of these tasks to small mapping error, and the one to watch when
    comparing two nearly equal configurations.
    """
    rng = random.Random(seed + 2)
    words = _load_haystack(haystack)
    samples = []
    depths = _depths(n_depths)
    for i in range(n_samples):
        tag = rng.randrange(1000, 9999)
        seq = " ".join(rng.choice(_RARE) for _ in range(seq_len))
        block = f" Sequence {tag}: {seq}. "
        depth = depths[i % len(depths)]
        block_len = len(tok(block, add_special_tokens=False).input_ids)
        budget = max(64, n_ctx - block_len)
        before = int(round(depth * budget))
        context = (_filler(rng, tok, before, words) + block
                   + _filler(rng, tok, budget - before, words))
        samples.append(Sample(
            context=context,
            query=f"\n\nSequence {tag}:",
            mode="nll",
            answer=f" {seq}.",
            meta={"depth": round(depth, 3)},
        ))
    return samples


def continuation(tok, n_ctx=2048, n_samples=32, cont_len=256, seed=0,
                 haystack=None, **_):
    """Held-out continuation NLL -- the metric the published tables report.

    Included so local runs stay comparable to the Modal numbers, and because
    it is the one task here that rewards *diffuse* context use rather than
    retrieval of a single span. Read it beside the retrieval tasks: NLL keeps
    improving past the k where task accuracy saturates, so a config that wins
    here and nowhere else has probably not improved anything a user notices.

    Needs real text: pass --haystack FILE.
    """
    rng = random.Random(seed + 3)
    words = _load_haystack(haystack)
    if not words:
        raise SystemExit("task `continuation` needs --haystack FILE of real text")
    ids = tok(" ".join(words), add_special_tokens=False).input_ids
    need = n_ctx + cont_len
    if len(ids) < need * 2:
        raise SystemExit(f"--haystack too short: need ~{need * 2} tokens, got {len(ids)}")
    starts = rng.sample(range(0, len(ids) - need), min(n_samples, (len(ids) - need)))
    samples = []
    for s in starts[:n_samples]:
        block = ids[s:s + need]
        samples.append(Sample(
            context=tok.decode(block[:n_ctx]),
            query=tok.decode(block[n_ctx:n_ctx + 1]),
            mode="nll",
            answer=tok.decode(block[n_ctx + 1:]),
            meta={"start": s},
        ))
    return samples


TASKS = {
    "kvr": kv_retrieval,
    "niah": niah,
    "recall": recall,
    "continuation": continuation,
}
