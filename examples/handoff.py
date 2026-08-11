"""The whole idea in one script: the 4B answers a document only the 1.7B read.

    python examples/handoff.py
    python examples/handoff.py --context my_document.txt --question "..."

For each question it prints three answers -- mapped cache, real prefill
(ceiling), no context (floor). Two questions are asked by default because the
two failure modes look different: a semantic question the mapped cache
usually gets, and a verbatim identifier, which is the hardest thing for a
lossy cache to carry.

Runs on CUDA, Apple Silicon (MPS) or CPU; downloads ~6 GB of models and a
1.8 GB mapper on first use.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvbridge import Mapper, transfer
from kvbridge.cache_compat import cache_layer_kv, cache_num_layers

DOCUMENT = """\
Internal memo, Halcyon Robotics, 14 March.

The Meridian-7 actuator recall covers units manufactured between January and
August of last year at the Tarrow Falls plant. The defect is a hairline crack
in the ceramic gearbox housing that appears only after roughly 4,000 duty
cycles under sustained load above 60 percent of rated torque.

Field replacement is authorised for all affected units. Replacement housings
ship from the Kestrel Bay depot, and the reference part number for the revised
housing is HR-2291-B. Units with serial prefixes MA7 and MA9 are not affected;
the crack was traced to a single mould line used only for the MB series.

Customers should not attempt to inspect the housing without removing the
actuator from service, as partial disassembly under load has caused three
reported injuries.
"""

QUESTIONS = [
    "\n\nQuestion: Why were the gearbox housings failing?\nAnswer:",
    "\n\nQuestion: What is the part number for the revised housing?\nAnswer:",
]


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def clone(cache):
    """A forward consumes the cache it is given, so each answer needs its own."""
    out = DynamicCache()
    for layer in range(cache_num_layers(cache)):
        k, v = cache_layer_kv(cache, layer)
        out.update(k.clone(), v.clone(), layer)
    return out


def generate(model, cache, query_ids, max_new_tokens=40):
    out_ids = []
    ids = query_ids
    for _ in range(max_new_tokens):
        out = model(ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        token = int(ids)
        if token == model.config.eos_token_id:
            break
        out_ids.append(token)
    return out_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapper", default="Siddharth85/kvbridge-qwen3-1.7b-to-qwen3-4b")
    ap.add_argument("--context", default=None, help="text file to use as the document")
    ap.add_argument("--question", default=None, help="ask one question instead of the two defaults")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    device, dtype = device_and_dtype()
    document = open(args.context).read() if args.context else DOCUMENT
    questions = [args.question] if args.question else QUESTIONS
    print(f"device={device} dtype={dtype}\n")

    mapper = Mapper.from_pretrained(args.mapper, device=device)
    print(mapper)
    tok = AutoTokenizer.from_pretrained(mapper.source)
    small = AutoModelForCausalLM.from_pretrained(mapper.source, dtype=dtype).to(device).eval()
    big = AutoModelForCausalLM.from_pretrained(mapper.target, dtype=dtype).to(device).eval()

    ctx_ids = tok(document, return_tensors="pt").input_ids.to(device)
    T = ctx_ids.shape[1]
    print(f"context: {T} tokens\n")

    with torch.no_grad():
        t0 = time.perf_counter()
        small_cache = small(ctx_ids, use_cache=True).past_key_values
        t1 = time.perf_counter()
        mapped = transfer(small_cache, mapper)
        t2 = time.perf_counter()
        real = big(ctx_ids, use_cache=True).past_key_values
        t3 = time.perf_counter()

        for question in questions:
            q_ids = tok(question, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(device)
            print("=" * 72)
            print(question.strip())
            print("=" * 72)
            for label, cache in (("mapped   (1.7B read it, 4B answers)", mapped),
                                 ("real     (4B read it)", real),
                                 ("none     (4B guesses)", DynamicCache())):
                ids = generate(big, clone(cache), q_ids, args.max_new_tokens)
                print(f"{label}: {tok.decode(ids).strip()}\n")

    print(f"1.7B prefill   {t1 - t0:6.3f}s")
    print(f"cache mapping  {t2 - t1:6.3f}s   -> handoff total {t2 - t0:6.3f}s")
    print(f"4B prefill     {t3 - t2:6.3f}s")
    if T < 4096:
        print(f"\n(At {T} tokens mapping is not a speed win -- the 4B prefill is cheap and "
              f"\n the mapper is a fixed cost per token. The crossover is around 8K; see the "
              f"\n timing table in the README.)")


if __name__ == "__main__":
    main()
