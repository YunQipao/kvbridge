"""Device-agnostic context-retention benchmarks for kvbridge.

Two independent axes, so every experiment is a config rather than a rewrite:

- **cache strategies** (`bench.engine`): how the target's prefix cache is
  built -- real prefill, fully mapped, hybrid sink+window, per-layer splices.
- **tasks** (`bench.tasks`): what the prefix has to retain -- needle
  retrieval, key/value lookup, verbatim recall, continuation NLL.

Everything here is quality-only and runs identically on CUDA, MPS, and CPU;
timing lives in the Modal pipeline because only that hardware is comparable.
"""

from .engine import Engine, Strategy, parse_strategy
from .pairs import PAIRS, Pair

__all__ = ["Engine", "Strategy", "parse_strategy", "PAIRS", "Pair"]
