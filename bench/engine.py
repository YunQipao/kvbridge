"""Model loading, cache-construction strategies, and scoring primitives.

The one abstraction that matters here is the **cache strategy**: a way of
producing the target model's prefix cache for a context it never prefilled.
Every idea worth testing (map everything, keep a native recency window, pin
the attention sinks, splice per layer) is a point in that space, so tasks
never learn about mapping and strategies never learn about tasks.

Scoring always follows the handoff shape the product actually has: the
*context* goes through the strategy, and the *query* is always prefilled
natively by the target on top of it. That isolates "did the context survive"
from "can the model use a question written into a mapped cache".
"""

import re
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvbridge.cache_compat import cache_layer_kv, cache_num_layers
from kvbridge.mapper import Mapper
from kvbridge.transfer import transfer


# -----------------------------------------------------------------------------
# device / dtype
# -----------------------------------------------------------------------------

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device):
    # MPS runs bf16 natively; fp16 there is prone to overflow in long softmaxes.
    return torch.float32 if device == "cpu" else torch.bfloat16


# -----------------------------------------------------------------------------
# DynamicCache surgery (version-independent: we only ever read via
# cache_layer_kv and write via update on a fresh cache)
# -----------------------------------------------------------------------------

def cache_len(cache):
    return cache_layer_kv(cache, 0)[0].shape[2]


def cache_map(cache, fn):
    """Build a new DynamicCache by applying fn(k, v, layer_idx) per layer."""
    out = DynamicCache()
    for layer in range(cache_num_layers(cache)):
        k, v = cache_layer_kv(cache, layer)
        nk, nv = fn(k, v, layer)
        out.update(nk, nv, layer)
    return out


def cache_slice(cache, start, end):
    return cache_map(cache, lambda k, v, _: (k[:, :, start:end].contiguous(),
                                             v[:, :, start:end].contiguous()))


def cache_concat(a, b):
    """Concatenate two caches along the sequence axis, layer by layer."""
    if a is None or cache_len(a) == 0:
        return b
    if b is None or cache_len(b) == 0:
        return a

    def join(k, v, layer):
        bk, bv = cache_layer_kv(b, layer)
        return torch.cat([k, bk], dim=2), torch.cat([v, bv], dim=2)

    return cache_map(a, join)


def cache_splice_layers(base, other, layer_ids):
    """Copy of `base` with the given layer indices taken from `other`.

    Used by the `layers` strategy to ask which target layers actually need a
    faithful cache -- the answer decides where mapping capacity should go.
    """
    ids = set(layer_ids)

    def pick(k, v, layer):
        if layer in ids:
            return cache_layer_kv(other, layer)
        return k, v

    return cache_map(base, pick)


# -----------------------------------------------------------------------------
# strategies
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    kind: str
    params: dict = field(default_factory=dict)

    def __str__(self):
        if not self.params:
            return self.kind
        body = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.kind}:{body}"

    @property
    def needs_mapper(self):
        return self.kind in ("mapped", "hybrid", "layers")


_KINDS = {"real", "none", "mapped", "hybrid", "layers"}


def parse_layer_spec(spec, n_layers):
    """'0-11' | '0,3,5' | '0-5,20-27' | 'first8' | 'last8' -> sorted list."""
    spec = str(spec)
    if m := re.fullmatch(r"first(\d+)", spec):
        return list(range(min(int(m.group(1)), n_layers)))
    if m := re.fullmatch(r"last(\d+)", spec):
        return list(range(max(0, n_layers - int(m.group(1))), n_layers))
    out = set()
    for part in spec.split("+"):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n_layers)


def parse_strategy(spec):
    """'hybrid:sink=4,window=256' -> Strategy('hybrid', {...}).

    Recognised forms:
      real                      target prefills the context itself (ceiling)
      none                      no context at all (floor)
      mapped                    every context token comes from the mapper
      hybrid:sink=S,window=W    target natively prefills the first S and last W
                                context tokens; the mapper covers the middle
      layers:real=first8        mapped everywhere, except those target layers
                                take the real prefill (diagnostic splice)
    """
    kind, _, rest = spec.partition(":")
    if kind not in _KINDS:
        raise SystemExit(f"unknown strategy {kind!r}; known: {', '.join(sorted(_KINDS))}")
    params = {}
    for item in filter(None, rest.split(",")):
        key, _, val = item.partition("=")
        params[key] = int(val) if val.lstrip("-").isdigit() else val
    return Strategy(kind, params)


# -----------------------------------------------------------------------------
# engine
# -----------------------------------------------------------------------------

class Engine:
    """Holds the two models plus a mapper, and turns strategies into caches.

    Models load lazily: `none`/`real` never touch the source model, so a
    ceiling/floor run costs one model's worth of memory. On a 24 GB Mac that
    is the difference between running and swapping.
    """

    def __init__(self, pair, mapper_path=None, device=None, dtype=None,
                 verbose=True):
        self.pair = pair
        self.device = device or default_device()
        self.dtype = dtype or default_dtype(self.device)
        self.mapper_path = mapper_path
        self.verbose = verbose
        self._source = None
        self._target = None
        self._mapper = None
        self.tok = AutoTokenizer.from_pretrained(pair.source)
        self.target_config = AutoConfig.from_pretrained(pair.target)
        self.n_target_layers = self.target_config.num_hidden_layers

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _load(self, model_id):
        self._log(f"loading {model_id} ({self.dtype}, {self.device})")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype)
        return model.to(self.device).eval()

    @property
    def source(self):
        if self._source is None:
            self._source = self._load(self.pair.source)
        return self._source

    @property
    def target(self):
        if self._target is None:
            self._target = self._load(self.pair.target)
        return self._target

    @property
    def mapper(self):
        if self._mapper is None:
            if self.mapper_path is None:
                raise SystemExit(
                    "this strategy needs a mapper; pass --mapper PATH "
                    "(fit one with `python -m bench.fit_local`)")
            self._mapper = Mapper.from_pretrained(self.mapper_path,
                                                  device=self.device)
            if self._mapper.source != self.pair.source or self._mapper.target != self.pair.target:
                raise SystemExit(
                    f"mapper is {self._mapper.source} -> {self._mapper.target} "
                    f"but pair {self.pair.name} is {self.pair.source} -> {self.pair.target}")
            self._log(f"loaded {self._mapper}")
        return self._mapper

    # ---- cache construction ----

    @torch.no_grad()
    def prefill(self, model, ids, cache=None):
        """Run `model` over ids on top of `cache`, returning (cache, logits)."""
        if cache is None:
            cache = DynamicCache()
        out = model(ids, past_key_values=cache, use_cache=True)
        return out.past_key_values, out.logits

    @torch.no_grad()
    def build_cache(self, strategy, ctx_ids):
        """Produce the target's prefix cache for `ctx_ids` [1, T]."""
        T = ctx_ids.shape[1]
        kind, p = strategy.kind, strategy.params

        if kind == "none":
            return DynamicCache()
        if kind == "real":
            return self.prefill(self.target, ctx_ids)[0]

        if kind == "mapped":
            src_cache = self.prefill(self.source, ctx_ids)[0]
            return transfer(src_cache, self.mapper)

        if kind == "hybrid":
            sink = int(p.get("sink", 0))
            window = int(p.get("window", 0))
            sink = max(0, min(sink, T))
            window = max(0, min(window, T - sink))
            mid_end = T - window
            # The source reads the whole prompt either way -- that is the
            # product. We simply decline to use its mapping for the two
            # regions the target can afford to compute exactly.
            src_cache = self.prefill(self.source, ctx_ids)[0]
            cache = None
            if sink:
                cache = self.prefill(self.target, ctx_ids[:, :sink])[0]
            if mid_end > sink:
                positions = torch.arange(sink, mid_end, device=self.device)
                mid = transfer(cache_slice(src_cache, sink, mid_end), self.mapper,
                               positions=positions)
                cache = cache_concat(cache, mid) if cache is not None else mid
            if window:
                cache = self.prefill(self.target, ctx_ids[:, mid_end:], cache)[0]
            return cache if cache is not None else DynamicCache()

        if kind == "layers":
            real_layers = parse_layer_spec(p.get("real", ""), self.n_target_layers)
            src_cache = self.prefill(self.source, ctx_ids)[0]
            mapped = transfer(src_cache, self.mapper)
            if not real_layers:
                return mapped
            real = self.prefill(self.target, ctx_ids)[0]
            return cache_splice_layers(mapped, real, real_layers)

        raise AssertionError(f"unhandled strategy {kind}")

    # ---- scoring ----
    #
    # Caches are consumed in place by a forward, so every scorer works on a
    # cache it is free to mutate: `score_*` take a *builder* when they need
    # more than one forward over the same context.

    @torch.no_grad()
    def token_nll(self, cache, query_ids, answer_ids, model=None):
        """Mean NLL (nats/token) of `answer_ids` given cache + query.

        `query_ids` must be non-empty: the first answer token is predicted
        from the last query position.
        """
        model = model or self.target
        assert query_ids.shape[1] >= 1, "query must contain at least one token"
        ids = torch.cat([query_ids, answer_ids], dim=1)
        n_ans = answer_ids.shape[1]
        logits = model(ids, past_key_values=cache, use_cache=True).logits
        logits = logits[0, -n_ans - 1:-1].float()
        loss = torch.nn.functional.cross_entropy(logits, answer_ids[0],
                                                 reduction="mean")
        return float(loss)

    @torch.no_grad()
    def rank_options(self, make_cache, query_ids, option_ids, model=None):
        """Length-normalised NLL of each option; returns (argmin, [nll,...]).

        `make_cache` is called once per option because the forward consumes
        the cache. For `mapped`/`hybrid` that re-runs the mapping, which is
        the honest cost but slow -- callers that care pass a cheap builder
        that clones a cached prefix instead.
        """
        nlls = [self.token_nll(make_cache(), query_ids, opt, model=model)
                for opt in option_ids]
        return int(min(range(len(nlls)), key=lambda i: nlls[i])), nlls

    @torch.no_grad()
    def generate(self, cache, query_ids, max_new_tokens=32, model=None):
        """Greedy decode on top of a prefix cache. Returns the new token ids."""
        model = model or self.target
        cache, logits = self.prefill(model, query_ids, cache)
        out = []
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        eos = model.config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        for _ in range(max_new_tokens):
            out.append(int(nxt))
            if int(nxt) in eos:
                break
            cache, logits = self.prefill(model, nxt, cache)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        return out


def clone_cache(cache):
    """Deep copy so a cache can be scored against several continuations."""
    return cache_map(cache, lambda k, v, _: (k.clone(), v.clone()))
