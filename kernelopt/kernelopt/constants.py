"""Shared numeric constants for KernelOPT.

These values have been tuned empirically and are gathered here so they can be
adjusted without hunting through the source tree.
"""

# Maximum characters of Inductor/PyTorch source to include in LLM prompts.
# Larger values improve context but increase token cost and latency.
MAX_SOURCE_CHARS = 3_000

# Kernel-group fusion threshold: graphs with this many groups or fewer are
# merged into a single fused kernel; larger graphs use the type-aware strategy.
MAX_GROUPS_FOR_SINGLE_FUSE = 6

# Hard timeout (seconds) for Vertex AI LLM API calls.
# Prevents infinite hangs on Vertex; 15 min covers the slowest Sonnet calls.
VERTEX_API_TIMEOUT_S = 900.0
