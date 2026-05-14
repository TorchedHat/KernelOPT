"""PyTorch-to-Triton synthesis via torch.compile.

Converts naive PyTorch code (nn.Module or standalone functions) into
standalone Triton kernel files that kernelopt can profile and optimise.

Zero LLM calls -- the entire pipeline is deterministic:
  detect -> torch.compile (subprocess) -> extract @triton.jit -> wrap standalone
"""

from .compile_extract import extract_triton_from_pytorch
from .wrap_kernel import wrap_inductor_output

__all__ = ["extract_triton_from_pytorch", "wrap_inductor_output"]
