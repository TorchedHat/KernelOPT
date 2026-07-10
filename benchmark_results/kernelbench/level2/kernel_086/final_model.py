"""KernelOPT output: torch.compile baseline.

KernelOPT could not produce a faster kernel for this model.
This file wraps the original model with torch.compile.
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        x = self.linear(x)
        x = x / self.divisor
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]

# ── torch.compile baseline ──
import torch as _torch
_torch.manual_seed(42)
_torch.cuda.manual_seed(42)
_init = get_init_inputs() if "get_init_inputs" in dir() else []
_model = Model(*_init).cuda().eval()
_compiled = _torch.compile(_model)
# Warmup
with _torch.no_grad():
    _compiled(*[t.cuda() if isinstance(t, _torch.Tensor) else t for t in get_inputs()])

def kernel_function(*inputs):
    with _torch.no_grad():
        return _compiled(*inputs)
