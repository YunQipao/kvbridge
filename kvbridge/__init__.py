"""kvbridge: cross-model KV cache transfer for LLM families.

Prefill a prompt with a small model, map its KV cache with a fitted linear
mapper, and let a bigger family member generate from it without re-reading
the prompt. Implementation of arXiv:2608.03893 (Heo et al., NVIDIA).

    from kvbridge import Mapper, transfer

    mapper = Mapper.from_pretrained("you/kvbridge-qwen3-1.7b-to-4b")
    small_cache = small(prompt_ids, use_cache=True).past_key_values
    big_cache = transfer(small_cache, mapper)
    big.generate(input_ids=prompt_ids, past_key_values=big_cache, ...)
"""

from .fit import fit_mapper
from .mapper import Mapper
from .transfer import transfer

__version__ = "0.1.0"
__all__ = ["Mapper", "fit_mapper", "transfer", "__version__"]
